export class ModelValidationError extends Error {
  constructor(message: string, public details?: Record<string, unknown>) {
    super(message);
  }
}

export class ModelNotFoundError extends Error {}
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
