import type { PoolConnection, RowDataPacket } from "mysql2/promise";
import type { ModelType } from "./models.schema";
import { buildModelExecutionLockName, buildModelSlotLockName } from "./model-runtime";
import { ModelServiceUnavailableError } from "./model-errors";

interface AdvisoryLockRow extends RowDataPacket {
  acquired: 0 | 1 | null;
  connection_id: number;
}

interface AdvisoryLockOwnerRow extends RowDataPacket {
  owner_id: number | null;
}

interface AdvisoryLockReleaseRow extends RowDataPacket {
  released: 0 | 1 | null;
}

export interface AdvisoryLockHandle {
  connection: PoolConnection;
  name: string;
  ownerConnectionId: number;
  signal: AbortSignal;
  stopHeartbeat: () => void;
}

interface ModelLockPool {
  getConnection(): Promise<PoolConnection>;
}

interface ModelLockLogger {
  error(bindings: Record<string, unknown>, message: string): void;
  warn(bindings: Record<string, unknown>, message: string): void;
}

export interface ModelLockManagerOptions {
  pool: ModelLockPool;
  maxConcurrentJobs: number;
  connectionAcquireTimeoutMs: number;
  queryTimeoutMs: number;
  heartbeatIntervalMs?: number;
  logger?: ModelLockLogger;
}

interface AdvisoryLockControl {
  isConnectionUsable(): boolean;
  isOwnershipQueryInFlight(): boolean;
  queryOwnerConnectionId(): Promise<number | null>;
  discardConnection(): void;
  markLost(
    error: ModelServiceUnavailableError,
    cause?: unknown,
    destroyConnection?: boolean,
  ): void;
}

const noopLogger: ModelLockLogger = {
  error: () => {},
  warn: () => {},
};

function safeRelease(connection: PoolConnection) {
  try {
    connection.release();
  } catch {
    // The connection may already have been destroyed.
  }
}

function safeDestroy(connection: PoolConnection) {
  try {
    connection.destroy();
  } catch {
    // Destruction is best-effort; a broken connection must never be returned to the pool.
  }
}

const withCause = <T extends Error>(error: T, cause: unknown): T => {
  (error as T & { cause?: unknown }).cause = cause;
  return error;
};

const getErrorCode = (error: unknown): string | undefined => {
  if (!error || typeof error !== "object" || !("code" in error)) return undefined;
  const code = (error as { code?: unknown }).code;
  return typeof code === "string" ? code : undefined;
};

const getAbortReason = (signal: AbortSignal): Error => signal.reason instanceof Error
  ? signal.reason
  : new Error("模型锁获取已终止");

export function createModelLockManager(options: ModelLockManagerOptions) {
  if (!Number.isInteger(options.maxConcurrentJobs) || options.maxConcurrentJobs <= 0) {
    throw new Error(
      `maxConcurrentJobs must be a positive integer, got: ${options.maxConcurrentJobs}`,
    );
  }
  for (const [name, value] of [
    ["connectionAcquireTimeoutMs", options.connectionAcquireTimeoutMs],
    ["queryTimeoutMs", options.queryTimeoutMs],
    ["heartbeatIntervalMs", options.heartbeatIntervalMs ?? 10_000],
  ] as const) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`${name} must be positive, got: ${value}`);
    }
  }

  const heartbeatIntervalMs = options.heartbeatIntervalMs ?? 10_000;
  const lockLogger = options.logger ?? noopLogger;
  const lockStats = {
    active: 0,
    acquired: 0,
    acquisitionFailures: 0,
    ownershipLosses: 0,
    releaseFailures: 0,
  };
  const lockControls = new WeakMap<AdvisoryLockHandle, AdvisoryLockControl>();

  const getModelLockStats = () => ({ ...lockStats });

  /**
   * mysql2 does not expose cancellation for Pool#getConnection. This wrapper
   * bounds the caller and releases a late connection that has not run SQL.
   */
  const getLockConnection = async (signal?: AbortSignal): Promise<PoolConnection> =>
    await new Promise<PoolConnection>((resolve, reject) => {
      if (signal?.aborted) {
        reject(getAbortReason(signal));
        return;
      }

      let settled = false;
      let abortHandler: (() => void) | undefined;
      const cleanup = () => {
        clearTimeout(timer);
        if (signal && abortHandler) signal.removeEventListener("abort", abortHandler);
      };
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        cleanup();
        lockStats.acquisitionFailures += 1;
        reject(new ModelServiceUnavailableError(
          "数据库模型锁连接获取超时，请稍后重试",
          "MODEL_LOCK_CONNECTION_TIMEOUT",
        ));
      }, options.connectionAcquireTimeoutMs);
      timer.unref?.();

      if (signal) {
        abortHandler = () => {
          if (settled) return;
          settled = true;
          cleanup();
          reject(getAbortReason(signal));
        };
        signal.addEventListener("abort", abortHandler, { once: true });
        if (signal.aborted) abortHandler();
      }

      void options.pool.getConnection().then(
        (connection) => {
          if (settled) {
            safeRelease(connection);
            return;
          }
          settled = true;
          cleanup();
          resolve(connection);
        },
        (cause) => {
          if (settled) return;
          settled = true;
          cleanup();
          lockStats.acquisitionFailures += 1;
          reject(withCause(new ModelServiceUnavailableError(
            "数据库模型锁连接不可用，请稍后重试",
            "MODEL_LOCK_CONNECTION_UNAVAILABLE",
          ), cause));
        },
      );
    });

  const queryLock = async <T extends RowDataPacket[]>(
    connection: PoolConnection,
    sql: string,
    values: unknown[],
  ) => await connection.query<T>({ sql, timeout: options.queryTimeoutMs }, values);

  /**
   * mysql2 cannot cancel an active query. During acquisition, destroying this
   * dedicated connection is safe and makes the operation deadline settle
   * immediately; any late GET_LOCK result is released by connection teardown.
   */
  const queryLockForAcquisition = async <T extends RowDataPacket[]>(
    connection: PoolConnection,
    sql: string,
    values: unknown[],
    signal?: AbortSignal,
  ): Promise<[T, unknown]> => {
    if (!signal) return await queryLock<T>(connection, sql, values);
    if (signal.aborted) {
      throw getAbortReason(signal);
    }

    return await new Promise<[T, unknown]>((resolve, reject) => {
      let settled = false;
      const cleanup = () => signal.removeEventListener("abort", abortHandler);
      const abortHandler = () => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(getAbortReason(signal));
      };
      signal.addEventListener("abort", abortHandler, { once: true });
      if (signal.aborted) abortHandler();
      if (settled) return;

      void queryLock<T>(connection, sql, values).then(
        (result) => {
          if (settled) return;
          settled = true;
          cleanup();
          resolve(result);
        },
        (error) => {
          if (settled) return;
          settled = true;
          cleanup();
          reject(error);
        },
      );
    });
  };

  const buildLockAcquisitionError = (cause: unknown) => {
    lockStats.acquisitionFailures += 1;
    const timedOut = getErrorCode(cause) === "PROTOCOL_SEQUENCE_TIMEOUT";
    return withCause(new ModelServiceUnavailableError(
      timedOut
        ? "数据库模型锁查询超时，请稍后重试"
        : "数据库模型锁获取失败，请稍后重试",
      timedOut ? "MODEL_LOCK_QUERY_TIMEOUT" : "MODEL_LOCK_ACQUISITION_FAILED",
    ), cause);
  };

  const createWatchedLockHandle = (
    connection: PoolConnection,
    name: string,
    ownerConnectionId: number,
  ): AdvisoryLockHandle => {
    const controller = new AbortController();
    let stopped = false;
    let heartbeatInFlight = false;
    let connectionUsable = true;
    let ownershipQueryPromise: Promise<number | null> | null = null;

    const discardConnection = () => {
      if (!connectionUsable) return;
      connectionUsable = false;
      safeDestroy(connection);
    };

    const queryOwnerConnectionId = (): Promise<number | null> => {
      if (ownershipQueryPromise) return ownershipQueryPromise;
      const pendingQuery = queryLock<AdvisoryLockOwnerRow[]>(
        connection,
        "SELECT IS_USED_LOCK(?) AS owner_id",
        [name],
      ).then(([rows]) => {
        const ownerId = rows[0]?.owner_id;
        return ownerId === null || ownerId === undefined ? null : Number(ownerId);
      });
      ownershipQueryPromise = pendingQuery;
      void pendingQuery.then(
        () => {
          if (ownershipQueryPromise === pendingQuery) ownershipQueryPromise = null;
        },
        () => {
          if (ownershipQueryPromise === pendingQuery) ownershipQueryPromise = null;
        },
      );
      return pendingQuery;
    };

    const markLost = (
      error: ModelServiceUnavailableError,
      cause?: unknown,
      destroyConnection = false,
    ) => {
      if (destroyConnection) {
        // mysql2 query timeouts do not cancel the active command. Destroying
        // prevents RELEASE_LOCK from queuing indefinitely behind it.
        discardConnection();
      }
      if (stopped || controller.signal.aborted) return;
      lockStats.ownershipLosses += 1;
      if (cause !== undefined) {
        (error as Error & { cause?: unknown }).cause = cause;
      }
      controller.abort(error);
      lockLogger.error(
        { err: cause, lock: name, ownerConnectionId },
        "数据库 advisory lock 所有权已丢失",
      );
    };

    lockStats.active += 1;
    lockStats.acquired += 1;

    const heartbeat = async () => {
      if (stopped || controller.signal.aborted || heartbeatInFlight) return;
      heartbeatInFlight = true;
      try {
        const reportedOwnerConnectionId = await queryOwnerConnectionId();
        if (reportedOwnerConnectionId !== ownerConnectionId) {
          markLost(new ModelServiceUnavailableError(
            "数据库模型锁已失效，任务已安全终止，请重试",
            "MODEL_LOCK_LOST",
          ));
        }
      } catch (error) {
        markLost(new ModelServiceUnavailableError(
          "数据库模型锁心跳失败，任务已安全终止，请重试",
          "MODEL_LOCK_LOST",
        ), error, true);
      } finally {
        heartbeatInFlight = false;
      }
    };

    const timer = setInterval(() => {
      void heartbeat();
    }, heartbeatIntervalMs);
    timer.unref?.();

    const handle: AdvisoryLockHandle = {
      connection,
      name,
      ownerConnectionId,
      signal: controller.signal,
      stopHeartbeat: () => {
        if (stopped) return;
        stopped = true;
        clearInterval(timer);
        lockStats.active = Math.max(0, lockStats.active - 1);
      },
    };
    lockControls.set(handle, {
      isConnectionUsable: () => connectionUsable,
      isOwnershipQueryInFlight: () => ownershipQueryPromise !== null,
      queryOwnerConnectionId,
      discardConnection,
      markLost,
    });
    return handle;
  };

  const assertModelLockOwned = (handle: AdvisoryLockHandle | null | undefined) => {
    if (!handle?.signal.aborted) return;
    const reason = handle.signal.reason;
    if (reason instanceof Error) throw reason;
    throw new ModelServiceUnavailableError(
      "数据库模型锁已失效，任务已安全终止，请重试",
      "MODEL_LOCK_LOST",
    );
  };

  const verifyModelLockOwned = async (
    handle: AdvisoryLockHandle | null | undefined,
  ): Promise<void> => {
    if (!handle) return;
    assertModelLockOwned(handle);
    const control = lockControls.get(handle);
    try {
      const reportedOwnerConnectionId = control
        ? await control.queryOwnerConnectionId()
        : Number((await queryLock<AdvisoryLockOwnerRow[]>(
          handle.connection,
          "SELECT IS_USED_LOCK(?) AS owner_id",
          [handle.name],
        ))[0][0]?.owner_id);
      assertModelLockOwned(handle);
      if (reportedOwnerConnectionId !== handle.ownerConnectionId) {
        const error = new ModelServiceUnavailableError(
          "数据库模型锁已失效，任务已安全终止，请重试",
          "MODEL_LOCK_LOST",
        );
        control?.markLost(error);
        throw error;
      }
    } catch (error) {
      if (error instanceof ModelServiceUnavailableError) throw error;
      const verifyError = withCause(new ModelServiceUnavailableError(
        "数据库模型锁所有权无法确认，任务已安全终止，请重试",
        "MODEL_LOCK_VERIFY_FAILED",
      ), error);
      control?.markLost(verifyError, error, true);
      throw verifyError;
    }
  };

  const combineModelLockSignals = (
    ...handles: Array<AdvisoryLockHandle | null | undefined>
  ): AbortSignal | undefined => {
    const signals = handles
      .map((handle) => handle?.signal)
      .filter((signal): signal is AbortSignal => Boolean(signal));
    if (signals.length === 0) return undefined;
    if (signals.length === 1) return signals[0];
    return AbortSignal.any(signals);
  };

  const releaseAdvisoryLock = async (
    handle: AdvisoryLockHandle | null | undefined,
  ) => {
    if (!handle) return;

    handle.stopHeartbeat();
    const control = lockControls.get(handle);
    if (control?.isOwnershipQueryInFlight()) {
      // mysql2 commands on one connection are serialized. RELEASE_LOCK would
      // otherwise sit behind a heartbeat/verification query whose timeout has
      // already consumed the cleanup budget. Closing the dedicated connection
      // releases its advisory lock without adding another queued command.
      control.discardConnection();
      lockLogger.warn(
        { lock: handle.name, ownerConnectionId: handle.ownerConnectionId },
        "锁所有权查询仍在执行，销毁专用连接以释放 advisory lock",
      );
      lockControls.delete(handle);
      return;
    }
    if (control && !control.isConnectionUsable()) {
      lockControls.delete(handle);
      return;
    }
    try {
      const [rows] = await queryLock<AdvisoryLockReleaseRow[]>(
        handle.connection,
        "SELECT RELEASE_LOCK(?) AS released",
        [handle.name],
      );
      if (rows[0]?.released !== 0 && rows[0]?.released !== 1) {
        throw new Error("RELEASE_LOCK did not return a valid ownership result");
      }
      safeRelease(handle.connection);
    } catch (error) {
      lockStats.releaseFailures += 1;
      lockLogger.warn(
        { err: error, lock: handle.name },
        "释放数据库 advisory lock 失败，销毁锁连接",
      );
      safeDestroy(handle.connection);
    } finally {
      lockControls.delete(handle);
    }
  };

  const acquireAdvisoryLock = async (
    name: string,
    signal?: AbortSignal,
  ): Promise<AdvisoryLockHandle | null> => {
    const connection = await getLockConnection(signal);
    try {
      const [rows] = await queryLockForAcquisition<AdvisoryLockRow[]>(
        connection,
        "SELECT GET_LOCK(?, 0) AS acquired, CONNECTION_ID() AS connection_id",
        [name],
        signal,
      );
      if (rows[0]?.acquired === 1) {
        return createWatchedLockHandle(connection, name, Number(rows[0].connection_id));
      }
      safeRelease(connection);
      return null;
    } catch (error) {
      safeDestroy(connection);
      if (signal?.aborted) throw getAbortReason(signal);
      throw buildLockAcquisitionError(error);
    }
  };

  const acquireModelSlot = async (
    signal?: AbortSignal,
  ): Promise<AdvisoryLockHandle | null> => {
    const connection = await getLockConnection(signal);
    try {
      for (let slot = 1; slot <= options.maxConcurrentJobs; slot += 1) {
        const name = buildModelSlotLockName(slot);
        const [rows] = await queryLockForAcquisition<AdvisoryLockRow[]>(
          connection,
          "SELECT GET_LOCK(?, 0) AS acquired, CONNECTION_ID() AS connection_id",
          [name],
          signal,
        );
        if (rows[0]?.acquired === 1) {
          return createWatchedLockHandle(connection, name, Number(rows[0].connection_id));
        }
      }
      safeRelease(connection);
      return null;
    } catch (error) {
      safeDestroy(connection);
      if (signal?.aborted) throw getAbortReason(signal);
      throw buildLockAcquisitionError(error);
    }
  };

  const releaseModelSlot = async (handle: AdvisoryLockHandle | null | undefined) => {
    await releaseAdvisoryLock(handle);
  };

  const acquireModelExecutionLock = async (
    experimentId: number,
    modelType: ModelType,
    signal?: AbortSignal,
  ): Promise<AdvisoryLockHandle | null> =>
    await acquireAdvisoryLock(buildModelExecutionLockName(experimentId, modelType), signal);

  const releaseModelExecutionLock = async (
    handle: AdvisoryLockHandle | null | undefined,
  ) => {
    await releaseAdvisoryLock(handle);
  };

  return {
    acquireModelExecutionLock,
    acquireModelSlot,
    combineModelLockSignals,
    getModelLockStats,
    releaseModelExecutionLock,
    releaseModelSlot,
    verifyModelLockOwned,
  };
}
