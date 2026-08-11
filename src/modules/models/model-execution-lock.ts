import logger from "../../utils/logger";
import type { ModelType } from "./models.schema";
import { ModelServiceUnavailableError } from "./model-errors";

export interface ModelExecutionLockHandle {
  readonly key: string;
  readonly token: symbol;
}

interface ModelExecutionLockLogger {
  error(bindings: Record<string, unknown>, message: string): void;
}

export interface ModelExecutionLockManagerOptions {
  logger?: ModelExecutionLockLogger;
}

const noopLogger: ModelExecutionLockLogger = {
  error: () => {},
};

const executionKey = (experimentId: number, modelType: ModelType) =>
  `${experimentId}:${modelType}`;

/**
 * Process-local, fail-fast exclusion for one experiment/model pair.
 *
 * The deployment contract permits exactly one Bun backend process. JavaScript
 * executes this check-and-set synchronously on that process's event loop, so a
 * Map is the authoritative execution mutex; no database session needs to stay
 * pinned while Python is running.
 */
export function createModelExecutionLockManager(
  options: ModelExecutionLockManagerOptions = {},
) {
  const activeLocks = new Map<string, symbol>();
  const lockLogger = options.logger ?? noopLogger;
  const stats = {
    acquired: 0,
    conflicts: 0,
    ownershipLosses: 0,
    releaseFailures: 0,
  };

  const tryAcquire = (
    experimentId: number,
    modelType: ModelType,
  ): ModelExecutionLockHandle | null => {
    const key = executionKey(experimentId, modelType);
    if (activeLocks.has(key)) {
      stats.conflicts += 1;
      return null;
    }

    const token = Symbol(key);
    activeLocks.set(key, token);
    stats.acquired += 1;
    return { key, token };
  };

  const verifyOwned = (handle: ModelExecutionLockHandle | null | undefined): void => {
    if (!handle) return;
    if (activeLocks.get(handle.key) === handle.token) return;

    stats.ownershipLosses += 1;
    throw new ModelServiceUnavailableError(
      "进程内模型执行锁已失效，任务已安全终止，请重试",
      "MODEL_EXECUTION_LOCK_LOST",
    );
  };

  const release = (handle: ModelExecutionLockHandle | null | undefined): void => {
    if (!handle) return;
    const owner = activeLocks.get(handle.key);
    if (owner === undefined) return;
    if (owner !== handle.token) {
      stats.releaseFailures += 1;
      lockLogger.error(
        { executionKey: handle.key },
        "拒绝释放不属于当前调用的进程内模型执行锁",
      );
      return;
    }
    activeLocks.delete(handle.key);
  };

  const getStats = () => ({
    scope: "process" as const,
    active: activeLocks.size,
    ...stats,
  });

  return {
    getStats,
    release,
    tryAcquire,
    verifyOwned,
  };
}

const manager = createModelExecutionLockManager({ logger });

export const getModelExecutionLockStats = manager.getStats;
export const releaseModelExecutionLock = manager.release;
export const tryAcquireModelExecutionLock = manager.tryAcquire;
export const verifyModelExecutionLockOwned = manager.verifyOwned;
