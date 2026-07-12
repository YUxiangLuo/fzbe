import type { PoolConnection, RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import logger from "../../utils/logger";
import { CONCURRENCY_LIMITS } from "../../utils/constants";
import type { ModelType } from "./models.schema";
import { buildModelExecutionLockName, buildModelSlotLockName } from "./model-runtime";
import { BASE_MODEL_TYPES, ENSEMBLE_MODEL_TYPES } from "./model-types";

const MAX_CONCURRENT_MODEL_JOBS = CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS;

interface AdvisoryLockRow extends RowDataPacket {
  acquired: 0 | 1 | null;
}

export interface AdvisoryLockHandle {
  connection: PoolConnection;
  name: string;
}

function safeRelease(conn: PoolConnection) {
  try { conn.release(); } catch { /* connection may already be destroyed */ }
}

async function acquireAdvisoryLock(name: string): Promise<AdvisoryLockHandle | null> {
  const connection = await pool.getConnection();

  try {
    const [rows] = await connection.query<AdvisoryLockRow[]>(
      "SELECT GET_LOCK(?, 0) AS acquired",
      [name],
    );

    if (rows[0]?.acquired === 1) {
      return { connection, name };
    }

    safeRelease(connection);
    return null;
  } catch (error) {
    safeRelease(connection);
    throw error;
  }
}

async function releaseAdvisoryLock(handle: AdvisoryLockHandle | null | undefined) {
  if (!handle) {
    return;
  }

  try {
    await handle.connection.query("SELECT RELEASE_LOCK(?)", [handle.name]);
  } catch (error) {
    logger.warn({ err: error, lock: handle.name }, "释放数据库 advisory lock 失败");
  } finally {
    safeRelease(handle.connection);
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
    safeRelease(connection);
    throw error;
  }
};

export const releaseModelSlot = async (handle: AdvisoryLockHandle | null | undefined) => {
  await releaseAdvisoryLock(handle);
};

export const acquireModelExecutionLock = async (
  experimentId: number,
  modelType: ModelType,
): Promise<AdvisoryLockHandle | null> => {
  return await acquireAdvisoryLock(buildModelExecutionLockName(experimentId, modelType));
};

export const releaseModelExecutionLock = async (handle: AdvisoryLockHandle | null | undefined) => {
  await releaseAdvisoryLock(handle);
};

const getArtifactMutationLockTypes = (modelType: ModelType): ModelType[] => {
  const lockTypes = BASE_MODEL_TYPES.has(modelType)
    ? [modelType, ...ENSEMBLE_MODEL_TYPES]
    : [modelType];

  return [...new Set(lockTypes)].sort();
};

export const acquireModelArtifactMutationLocks = async (
  experimentId: number,
  modelType: ModelType,
): Promise<AdvisoryLockHandle[] | null> => {
  const acquiredLocks: AdvisoryLockHandle[] = [];

  try {
    for (const lockModelType of getArtifactMutationLockTypes(modelType)) {
      const lock = await acquireModelExecutionLock(experimentId, lockModelType);
      if (!lock) {
        await Promise.all(acquiredLocks.map((handle) => releaseAdvisoryLock(handle)));
        return null;
      }
      acquiredLocks.push(lock);
    }

    return acquiredLocks;
  } catch (error) {
    await Promise.all(acquiredLocks.map((handle) => releaseAdvisoryLock(handle)));
    throw error;
  }
};

export const releaseModelArtifactMutationLocks = async (
  handles: AdvisoryLockHandle[] | null | undefined,
) => {
  if (!handles) return;
  await Promise.all(handles.map((handle) => releaseAdvisoryLock(handle)));
};
