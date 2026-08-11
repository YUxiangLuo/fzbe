import logger from "../../utils/logger";
import { CONCURRENCY_LIMITS, TIMEOUTS } from "../../utils/constants";
import { Semaphore } from "../../utils/semaphore";
import type { ModelType } from "./models.schema";
import {
  ModelCapacityBusyError,
  ModelConflictError,
  ModelOperationDeadlineError,
} from "./model-errors";
import {
  releaseModelExecutionLock,
  tryAcquireModelExecutionLock,
  verifyModelExecutionLockOwned,
  type ModelExecutionLockHandle,
} from "./model-execution-lock";

export type ModelOperation =
  | "guided_training_session"
  | "guided_training_step"
  | "guided_training_discard"
  | "production_preparation"
  | "prediction"
  | "adf";

export interface ModelOperationContext {
  readonly operation: ModelOperation;
  readonly deadlineSignal: AbortSignal;
  throwIfTerminated(): void;
}

export interface ModelOperationLease {
  readonly terminationSignal: AbortSignal;
  throwIfTerminated(): void;
  verifyOwned(): Promise<void>;
}

interface LeaseOptions {
  execution?: {
    experimentId: number;
    modelType: ModelType;
  };
}

export interface ModelExecutionLockAdapter {
  tryAcquire(
    experimentId: number,
    modelType: ModelType,
  ): ModelExecutionLockHandle | null;
  release(handle: ModelExecutionLockHandle | null | undefined): void;
  verifyOwned(handle: ModelExecutionLockHandle | null | undefined): void;
}

interface ModelOperationLogger {
  info(bindings: Record<string, unknown>, message: string): void;
  warn(bindings: Record<string, unknown>, message: string): void;
}

export interface ModelOperationLeaseManagerOptions {
  capacity: number;
  deadlineMs: number;
  executionLocks?: ModelExecutionLockAdapter;
  logger?: ModelOperationLogger;
}

const defaultExecutionLocks: ModelExecutionLockAdapter = {
  tryAcquire: tryAcquireModelExecutionLock,
  release: releaseModelExecutionLock,
  verifyOwned: verifyModelExecutionLockOwned,
};

const abortReason = (signal: AbortSignal): Error => {
  if (signal.reason instanceof Error) return signal.reason;
  return new Error("模型操作已终止");
};

export function createModelOperationLeaseManager(
  options: ModelOperationLeaseManagerOptions,
) {
  if (!Number.isInteger(options.capacity) || options.capacity <= 0) {
    throw new Error(`capacity must be a positive integer, got: ${options.capacity}`);
  }
  if (!Number.isFinite(options.deadlineMs) || options.deadlineMs <= 0) {
    throw new Error(`deadlineMs must be positive, got: ${options.deadlineMs}`);
  }

  const admission = new Semaphore(options.capacity);
  const executionLocks = options.executionLocks ?? defaultExecutionLocks;
  const operationLogger = options.logger ?? logger;
  const operationStats = {
    started: 0,
    admitted: 0,
    completed: 0,
    failed: 0,
    deadlineExceeded: 0,
    capacityRejected: 0,
    modelBusyRejected: 0,
    activeLeases: 0,
  };
  const operationActive = new Map<ModelOperation, number>();

  const incrementActive = (operation: ModelOperation, delta: 1 | -1) => {
    const next = Math.max(0, (operationActive.get(operation) ?? 0) + delta);
    if (next === 0) operationActive.delete(operation);
    else operationActive.set(operation, next);
  };

  const getModelOperationStats = () => ({
    ...operationStats,
    admission: {
      scope: "process" as const,
      capacity: admission.capacity,
      active: admission.active,
      pending: admission.pending,
    },
    activeByOperation: Object.fromEntries(operationActive),
  });

  const withModelOperationContext = async <T>(
    operation: ModelOperation,
    callback: (context: ModelOperationContext) => Promise<T>,
    deadlineMs: number = options.deadlineMs,
  ): Promise<T> => {
    if (!Number.isFinite(deadlineMs) || deadlineMs <= 0) {
      throw new Error(`operation deadlineMs must be positive, got: ${deadlineMs}`);
    }
    if (deadlineMs > options.deadlineMs) {
      throw new Error(
        `operation deadlineMs cannot exceed the manager deadline ${options.deadlineMs}, got: ${deadlineMs}`,
      );
    }
    const startedAt = Date.now();
    const timeoutMs = deadlineMs;
    const controller = new AbortController();
    const deadlineError = new ModelOperationDeadlineError(timeoutMs, operation);
    const timer = setTimeout(() => controller.abort(deadlineError), timeoutMs);
    timer.unref?.();
    operationStats.started += 1;

    const context: ModelOperationContext = {
      operation,
      deadlineSignal: controller.signal,
      throwIfTerminated: () => {
        if (controller.signal.aborted) throw abortReason(controller.signal);
      },
    };

    try {
      const result = await callback(context);
      context.throwIfTerminated();
      operationStats.completed += 1;
      operationLogger.info({ operation, durationMs: Date.now() - startedAt }, "模型操作完成");
      return result;
    } catch (error) {
      const finalError = controller.signal.aborted ? abortReason(controller.signal) : error;
      operationStats.failed += 1;
      if (finalError instanceof ModelOperationDeadlineError) operationStats.deadlineExceeded += 1;
      operationLogger.warn(
        {
          err: finalError,
          operation,
          durationMs: Date.now() - startedAt,
          deadlineMs: timeoutMs,
        },
        "模型操作失败",
      );
      throw finalError;
    } finally {
      clearTimeout(timer);
    }
  };

  const withModelOperationLease = async <T>(
    context: ModelOperationContext,
    leaseOptions: LeaseOptions,
    callback: (lease: ModelOperationLease) => Promise<T>,
  ): Promise<T> => {
    context.throwIfTerminated();
    if (!admission.tryAcquire()) {
      operationStats.capacityRejected += 1;
      throw new ModelCapacityBusyError();
    }

    operationStats.admitted += 1;
    operationStats.activeLeases += 1;
    incrementActive(context.operation, 1);

    let executionLock: ModelExecutionLockHandle | null = null;
    try {
      if (leaseOptions.execution) {
        executionLock = executionLocks.tryAcquire(
          leaseOptions.execution.experimentId,
          leaseOptions.execution.modelType,
        );
        if (!executionLock) {
          operationStats.modelBusyRejected += 1;
          throw new ModelConflictError("同一模型正在训练或预测，请稍后再试", "MODEL_BUSY");
        }
      }

      // Deliberately internal-only: never merge the HTTP Request.signal here.
      // Once admitted, closing/offlining the browser must not cancel backend work.
      const terminationSignal = context.deadlineSignal;
      const throwIfTerminated = () => {
        if (terminationSignal.aborted) throw abortReason(terminationSignal);
      };
      const lease: ModelOperationLease = {
        terminationSignal,
        throwIfTerminated,
        verifyOwned: async () => {
          throwIfTerminated();
          executionLocks.verifyOwned(executionLock);
          throwIfTerminated();
        },
      };

      const result = await callback(lease);
      throwIfTerminated();
      return result;
    } finally {
      // The execution mutex is nested inside admission and must be released first.
      try {
        executionLocks.release(executionLock);
      } finally {
        admission.release();
        operationStats.activeLeases = Math.max(0, operationStats.activeLeases - 1);
        incrementActive(context.operation, -1);
      }
    }
  };

  return {
    getModelOperationStats,
    withModelOperationContext,
    withModelOperationLease,
  };
}

const defaultManager = createModelOperationLeaseManager({
  capacity: CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS,
  deadlineMs: TIMEOUTS.MODEL_SINGLE_OPERATION,
});

export const getModelOperationStats = defaultManager.getModelOperationStats;
export const withModelOperationContext = defaultManager.withModelOperationContext;
export const withModelOperationLease = defaultManager.withModelOperationLease;
