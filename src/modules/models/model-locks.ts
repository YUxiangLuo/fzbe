import type { PoolConnection, RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import logger from "../../utils/logger";
import { CONCURRENCY_LIMITS } from "../../utils/constants";
import type { ModelType } from "./models.schema";
import { buildModelExecutionLockName, buildModelSlotLockName } from "./model-runtime";
import { ModelServiceUnavailableError } from "./model-errors";

const MAX_CONCURRENT_MODEL_JOBS = CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS;
const LOCK_HEARTBEAT_INTERVAL_MS = 10_000;

interface AdvisoryLockRow extends RowDataPacket {
  acquired: 0 | 1 | null;
  connection_id: number;
}

interface AdvisoryLockOwnerRow extends RowDataPacket {
  owner_id: number | null;
}

export interface AdvisoryLockHandle {
  connection: PoolConnection;
  name: string;
  ownerConnectionId?: number;
  signal?: AbortSignal;
  stopHeartbeat?: () => void;
}

function safeRelease(conn: PoolConnection) {
  try { conn.release(); } catch { /* connection may already be destroyed */ }
}

async function acquireAdvisoryLock(name: string): Promise<AdvisoryLockHandle | null> {
  const connection = await pool.getConnection();

  try {
    const [rows] = await connection.query<AdvisoryLockRow[]>(
      "SELECT GET_LOCK(?, 0) AS acquired, CONNECTION_ID() AS connection_id",
      [name],
    );

    if (rows[0]?.acquired === 1) {
      return createWatchedLockHandle(connection, name, Number(rows[0].connection_id));
    }

    safeRelease(connection);
    return null;
  } catch (error) {
    safeRelease(connection);
    throw error;
  }
}

const createWatchedLockHandle = (
  connection: PoolConnection,
  name: string,
  ownerConnectionId: number,
): AdvisoryLockHandle => {
  const controller = new AbortController();
  let stopped = false;
  let heartbeatInFlight = false;

  const markLost = (cause?: unknown) => {
    if (stopped || controller.signal.aborted) return;
    const error = new ModelServiceUnavailableError(
      "数据库模型锁已失效，任务已安全终止，请重试",
      "MODEL_LOCK_LOST",
    );
    if (cause !== undefined) {
      (error as Error & { cause?: unknown }).cause = cause;
    }
    controller.abort(error);
    logger.error({ err: cause, lock: name, ownerConnectionId }, "数据库 advisory lock 所有权已丢失");
  };

  const heartbeat = async () => {
    if (stopped || heartbeatInFlight) return;
    heartbeatInFlight = true;
    try {
      const [rows] = await connection.query<AdvisoryLockOwnerRow[]>(
        "SELECT IS_USED_LOCK(?) AS owner_id",
        [name],
      );
      if (Number(rows[0]?.owner_id) !== ownerConnectionId) {
        markLost();
      }
    } catch (error) {
      markLost(error);
    } finally {
      heartbeatInFlight = false;
    }
  };

  const timer = setInterval(() => {
    void heartbeat();
  }, LOCK_HEARTBEAT_INTERVAL_MS);
  timer.unref?.();

  return {
    connection,
    name,
    ownerConnectionId,
    signal: controller.signal,
    stopHeartbeat: () => {
      if (stopped) return;
      stopped = true;
      clearInterval(timer);
    },
  };
};

export const assertModelLockOwned = (handle: AdvisoryLockHandle | null | undefined) => {
  if (!handle?.signal?.aborted) return;
  const reason = handle.signal.reason;
  if (reason instanceof Error) throw reason;
  throw new ModelServiceUnavailableError(
    "数据库模型锁已失效，任务已安全终止，请重试",
    "MODEL_LOCK_LOST",
  );
};

export const combineModelLockSignals = (
  ...handles: Array<AdvisoryLockHandle | null | undefined>
): AbortSignal | undefined => {
  const signals = handles
    .map((handle) => handle?.signal)
    .filter((signal): signal is AbortSignal => Boolean(signal));
  if (signals.length === 0) return undefined;
  if (signals.length === 1) return signals[0];
  return AbortSignal.any(signals);
};

async function releaseAdvisoryLock(handle: AdvisoryLockHandle | null | undefined) {
  if (!handle) {
    return;
  }

  handle.stopHeartbeat?.();
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
        "SELECT GET_LOCK(?, 0) AS acquired, CONNECTION_ID() AS connection_id",
        [name],
      );

      if (rows[0]?.acquired === 1) {
        return createWatchedLockHandle(connection, name, Number(rows[0].connection_id));
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
