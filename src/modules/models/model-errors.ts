import type { ModelType } from "./models.schema";

export const MODEL_TRAINING_RUN_REQUIRED_CODE = "MODEL_TRAINING_RUN_REQUIRED";
export const MODEL_RETRAINING_REQUIRED_CODE = "MODEL_RETRAINING_REQUIRED";

export const buildModelRetrainingRequiredDetails = (
  modelType: ModelType,
  code = MODEL_RETRAINING_REQUIRED_CODE,
) => ({
  code,
  required_action: "retrain" as const,
  model_type: modelType,
});

export const buildMissingModelTrainingRunDetails = (modelType: ModelType) =>
  buildModelRetrainingRequiredDetails(modelType, MODEL_TRAINING_RUN_REQUIRED_CODE);

export class ModelValidationError extends Error {
  constructor(message: string, public details?: Record<string, unknown>) {
    super(message);
  }
}

export class ModelNotFoundError extends Error {}

export class ModelArtifactError extends Error {
  readonly code = "MODEL_ARTIFACT_INVALID";

  constructor(message = "模型产物损坏或格式不兼容") {
    super(message);
    this.name = "ModelArtifactError";
  }
}

export class ModelConflictError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: Record<string, unknown>,
  ) {
    super(message);
  }
}

export class ModelServiceUnavailableError extends Error {
  constructor(message: string, public code: string) {
    super(message);
  }
}

/**
 * The server may have committed even though the client never received the
 * acknowledgement. Callers must preserve every artifact that the transaction
 * could now reference and let an idempotent retry/reconciler determine the
 * durable outcome.
 */
export class ModelCommitOutcomeUnknownError extends ModelServiceUnavailableError {
  readonly retryable = true;

  constructor(message = "数据库提交结果暂时无法确认，请重试以恢复最新状态", cause?: unknown) {
    super(message, "MODEL_COMMIT_OUTCOME_UNKNOWN");
    this.name = "ModelCommitOutcomeUnknownError";
    if (cause !== undefined) {
      (this as Error & { cause?: unknown }).cause = cause;
    }
  }
}

export const isModelCommitOutcomeUnknownError = (
  error: unknown,
): error is ModelCommitOutcomeUnknownError =>
  error instanceof ModelCommitOutcomeUnknownError;

export class ModelCapacityBusyError extends Error {
  readonly code = "MODEL_CAPACITY_BUSY";
  readonly retryable = true;

  constructor(
    message = "模型服务繁忙，请稍后再试",
    public readonly retryAfterSeconds = 2,
  ) {
    super(message);
    this.name = "ModelCapacityBusyError";
  }
}

export class ModelOperationDeadlineError extends Error {
  readonly code = "MODEL_OPERATION_DEADLINE_EXCEEDED";
  readonly retryable = true;

  constructor(
    public readonly timeoutMs: number,
    public readonly operation: string,
  ) {
    super(`模型操作超过服务端时限 (${timeoutMs}ms)`);
    this.name = "ModelOperationDeadlineError";
  }
}

export type ModelTimeoutSource = "python_process" | "python_subprocess";

export class ModelTimeoutError extends Error {
  readonly code = "MODEL_TIMEOUT";

  constructor(
    message: string,
    public readonly timeoutMs: number,
    public readonly source: ModelTimeoutSource,
  ) {
    super(message);
    this.name = "ModelTimeoutError";
  }
}

export class ModelAccessError extends Error {
  constructor(message: string, public status: 400 | 403 | 404) {
    super(message);
  }
}
