import { pool } from "../../config";
import logger from "../../utils/logger";
import { CONCURRENCY_LIMITS, TIMEOUTS } from "../../utils/constants";
import { createModelLockManager } from "./model-lock-manager";

export type { AdvisoryLockHandle } from "./model-lock-manager";

const manager = createModelLockManager({
  pool,
  maxConcurrentJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS,
  connectionAcquireTimeoutMs: TIMEOUTS.MODEL_LOCK_CONNECTION_ACQUIRE,
  queryTimeoutMs: TIMEOUTS.MODEL_LOCK_QUERY,
  logger,
});

export const acquireModelExecutionLock = manager.acquireModelExecutionLock;
export const acquireModelSlot = manager.acquireModelSlot;
export const combineModelLockSignals = manager.combineModelLockSignals;
export const getModelLockStats = manager.getModelLockStats;
export const releaseModelExecutionLock = manager.releaseModelExecutionLock;
export const releaseModelSlot = manager.releaseModelSlot;
export const verifyModelLockOwned = manager.verifyModelLockOwned;
