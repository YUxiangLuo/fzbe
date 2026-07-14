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
