import type { PoolConnection, RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import logger from "../../utils/logger";
import { CONCURRENCY_LIMITS } from "../../utils/constants";
import type { ModelType } from "./models.schema";
import { buildModelExecutionLockName, buildModelSlotLockName } from "./model-runtime";

const MAX_CONCURRENT_MODEL_JOBS = CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS;

interface AdvisoryLockRow extends RowDataPacket {
  acquired: 0 | 1 | null;
}

export interface AdvisoryLockHandle {
  connection: PoolConnection;
  name: string;
  released?: boolean;
  releaseAdmission?: () => void;
}

export interface ModelJobLockHandle {
  slotLock: AdvisoryLockHandle;
  artifactLocks: AdvisoryLockHandle[];
}

export type ModelJobLockAcquisition =
  | { status: "acquired"; locks: ModelJobLockHandle }
  | { status: "capacity" }
  | { status: "conflict" };

export type ModelExecutionLockAcquisition =
  | { status: "acquired"; lock: AdvisoryLockHandle }
  | { status: "capacity" }
  | { status: "conflict" };

// The DB pool reserves exactly MAX_CONCURRENT_MODEL_JOBS execution-lock
// connections. Keep short artifact-only operations inside the same bound too;
// otherwise a burst of distinct experiments could consume the query headroom
// while each request still needs another connection to finish its work.
let activeExecutionAdmissions = 0;

function tryAcquireExecutionAdmission(): (() => void) | null {
  if (activeExecutionAdmissions >= MAX_CONCURRENT_MODEL_JOBS) {
    return null;
  }

  activeExecutionAdmissions += 1;
  let released = false;
  return () => {
    if (released) return;
    released = true;
    activeExecutionAdmissions -= 1;
  };
}

function safeRelease(conn: PoolConnection) {
  try { conn.release(); } catch { /* connection may already be destroyed */ }
}

function safeDestroy(conn: PoolConnection) {
  try { conn.destroy(); } catch { /* destruction is best-effort */ }
}

async function acquireAdvisoryLock(
  name: string,
  releaseAdmission?: () => void,
): Promise<AdvisoryLockHandle | null> {
  let connection: PoolConnection | null = null;

  try {
    connection = await pool.getConnection();
    const [rows] = await connection.query<AdvisoryLockRow[]>(
      "SELECT GET_LOCK(?, 0) AS acquired",
      [name],
    );

    if (rows[0]?.acquired === 1) {
      return { connection, name, releaseAdmission };
    }

    safeRelease(connection);
    releaseAdmission?.();
    return null;
  } catch (error) {
    // GET_LOCK may have succeeded before the response was lost. Never return
    // a connection with unknown lock ownership to the pool.
    if (connection) safeDestroy(connection);
    releaseAdmission?.();
    throw error;
  }
}

async function releaseAdvisoryLock(handle: AdvisoryLockHandle | null | undefined) {
  if (!handle) {
    return;
  }
  if (handle.released) {
    return;
  }
  handle.released = true;

  let connectionReusable = true;
  try {
    await handle.connection.query("SELECT RELEASE_LOCK(?)", [handle.name]);
  } catch (error) {
    connectionReusable = false;
    logger.warn({ err: error, lock: handle.name }, "释放数据库 advisory lock 失败");
    // It must never be returned to the pool while it may still own a named lock.
    safeDestroy(handle.connection);
  } finally {
    if (connectionReusable) {
      safeRelease(handle.connection);
    }
    handle.releaseAdmission?.();
  }
}

export const acquireModelSlot = async (): Promise<AdvisoryLockHandle | null> => {
  const connection = await pool.getConnection();

  try {
    for (let slot = 1; slot <= MAX_CONCURRENT_MODEL_JOBS; slot += 1) {
      const name = buildModelSlotLockName(slot);
      const [rows] = await connection.query<AdvisoryLockRow[]>(
        "SELECT GET_LOCK(?, 0) AS acquired",
        [name],
      );

      if (rows[0]?.acquired === 1) {
        return { connection, name };
      }
    }

    safeRelease(connection);
    return null;
  } catch (error) {
    // One of the slot attempts may have succeeded before the response failed.
    safeDestroy(connection);
    throw error;
  }
};

export const releaseModelSlot = async (handle: AdvisoryLockHandle | null | undefined) => {
  await releaseAdvisoryLock(handle);
};

async function acquireModelExecutionLockWithAdmission(
  experimentId: number,
  modelType: ModelType,
  releaseAdmission: () => void,
): Promise<Exclude<ModelExecutionLockAcquisition, { status: "capacity" }>> {
  const lock = await acquireAdvisoryLock(
    buildModelExecutionLockName(experimentId, modelType),
    releaseAdmission,
  );
  return lock
    ? { status: "acquired", lock }
    : { status: "conflict" };
}

export const acquireModelExecutionLockWithStatus = async (
  experimentId: number,
  modelType: ModelType,
): Promise<ModelExecutionLockAcquisition> => {
  const releaseAdmission = tryAcquireExecutionAdmission();
  if (!releaseAdmission) {
    return { status: "capacity" };
  }
  return await acquireModelExecutionLockWithAdmission(
    experimentId,
    modelType,
    releaseAdmission,
  );
};

export const acquireModelExecutionLock = async (
  experimentId: number,
  modelType: ModelType,
): Promise<AdvisoryLockHandle | null> => {
  const acquisition = await acquireModelExecutionLockWithStatus(experimentId, modelType);
  return acquisition.status === "acquired" ? acquisition.lock : null;
};

export const releaseModelExecutionLock = async (handle: AdvisoryLockHandle | null | undefined) => {
  await releaseAdvisoryLock(handle);
};

export const acquireModelArtifactMutationLocks = async (
  experimentId: number,
  modelType: ModelType,
): Promise<AdvisoryLockHandle[] | null> => {
  const lock = await acquireModelExecutionLock(experimentId, modelType);
  return lock ? [lock] : null;
};

export const releaseModelArtifactMutationLocks = async (
  handles: AdvisoryLockHandle[] | null | undefined,
) => {
  if (!handles) return;
  await Promise.all(handles.map((handle) => releaseAdvisoryLock(handle)));
};

/**
 * Acquire compute resources in the only safe order: global capacity first,
 * then the experiment-wide mutation lock. Capacity losers therefore never
 * retain execution-lock connections while waiting for a second pool slot.
 */
export const acquireModelJobLocks = async (
  experimentId: number,
  modelType: ModelType,
): Promise<ModelJobLockAcquisition> => {
  // Local admission is connection-free and prevents a retry storm from using
  // the query headroom merely to discover that all DB-wide slots are busy.
  const releaseAdmission = tryAcquireExecutionAdmission();
  if (!releaseAdmission) {
    return { status: "capacity" };
  }

  let slotLock: AdvisoryLockHandle | null = null;
  try {
    slotLock = await acquireModelSlot();
    if (!slotLock) {
      releaseAdmission();
      return { status: "capacity" };
    }

    const executionAcquisition = await acquireModelExecutionLockWithAdmission(
      experimentId,
      modelType,
      releaseAdmission,
    );
    if (executionAcquisition.status === "conflict") {
      await releaseModelSlot(slotLock);
      return { status: "conflict" };
    }
    const artifactLocks = [executionAcquisition.lock];

    return {
      status: "acquired",
      locks: { slotLock, artifactLocks },
    };
  } catch (error) {
    releaseAdmission();
    await releaseModelSlot(slotLock);
    throw error;
  }
};

export const releaseModelJobLocks = async (
  locks: ModelJobLockHandle | null | undefined,
) => {
  if (!locks) return;
  // Reverse acquisition order. Releasing the execution admission before the
  // slot also lets a waiting short mutation make progress without increasing
  // the number of active Python jobs.
  await releaseModelArtifactMutationLocks(locks.artifactLocks);
  await releaseModelSlot(locks.slotLock);
};
