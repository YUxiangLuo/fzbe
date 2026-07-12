import { Semaphore } from "../../utils/semaphore";
import { KeyedLock } from "../../utils/keyed-lock";
import { CONCURRENCY_LIMITS } from "../../utils/constants";
import type { ModelType } from "./models.schema";
import { buildModelExecutionLockName } from "./model-runtime";

/**
 * 模型任务并发控制（进程内）。
 *
 * 训练、生产准备、预测三种行为各有独立的信号量池，互不挤占；任一池满载只影响该行为，
 * 其余行为不受影响。引导训练（guided-training）与批量训练共用 train 池。
 *
 * 与旧实现的差异：不再使用 MySQL advisory lock（GET_LOCK/RELEASE_LOCK），
 * 因此模型任务不再占用数据库连接，连接池大小与并发上限解耦。单实例部署下，
 * 进程内信号量 + 按键锁即可提供等价的限流与去重保证。
 *
 * - 槽位：非阻塞 tryAcquire，满则返回 null（上层 429）。
 * - 执行去重：按 `(experimentId, modelType)` 的进程内按键锁，占用中返回 null（上层 409）。
 */

export type ModelPoolKind = "train" | "prepare" | "predict";

const modelSlotPools: Record<ModelPoolKind, Semaphore> = {
  train: new Semaphore(CONCURRENCY_LIMITS.MAX_CONCURRENT_TRAIN_JOBS),
  prepare: new Semaphore(CONCURRENCY_LIMITS.MAX_CONCURRENT_PREPARE_JOBS),
  predict: new Semaphore(CONCURRENCY_LIMITS.MAX_CONCURRENT_PREDICT_JOBS),
};

const executionLock = new KeyedLock();

export interface ModelSlotHandle {
  kind: ModelPoolKind;
}

export interface ModelExecutionLockHandle {
  key: string;
}

/**
 * 非阻塞获取指定行为的模型槽位。有空位返回句柄，满则返回 null（调用方应返回 429）。
 */
export const acquireModelSlot = async (
  kind: ModelPoolKind,
): Promise<ModelSlotHandle | null> => {
  return modelSlotPools[kind].tryAcquire() ? { kind } : null;
};

export const releaseModelSlot = async (
  handle: ModelSlotHandle | null | undefined,
) => {
  if (!handle) {
    return;
  }
  modelSlotPools[handle.kind].release();
};

/**
 * 非阻塞获取单实验单模型执行锁。占用中返回 null（调用方应返回 409）。
 */
export const acquireModelExecutionLock = async (
  experimentId: number,
  modelType: ModelType,
): Promise<ModelExecutionLockHandle | null> => {
  const key = buildModelExecutionLockName(experimentId, modelType);
  return executionLock.tryAcquire(key) ? { key } : null;
};

export const releaseModelExecutionLock = async (
  handle: ModelExecutionLockHandle | null | undefined,
) => {
  if (!handle) {
    return;
  }
  executionLock.release(handle.key);
};

/**
 * 各模型行为池的瞬时状态，供 /api/v1/runtime-info 暴露（与 getPdfQueueStats 对称）。
 */
export function getModelPoolStats() {
  const snapshot = (sem: Semaphore) => ({
    activeJobs: sem.active,
    pendingJobs: sem.pending,
    maxConcurrentJobs: sem.capacity,
  });
  return {
    train: snapshot(modelSlotPools.train),
    prepare: snapshot(modelSlotPools.prepare),
    predict: snapshot(modelSlotPools.predict),
  };
}
