export class ModelValidationError extends Error {
  constructor(message: string, public details?: Record<string, unknown>) {
    super(message);
  }
}

export class ModelNotFoundError extends Error {}
export class ModelAccessError extends Error {
  constructor(message: string, public status: 400 | 403 | 404) {
    super(message);
  }
}

export class ModelConflictError extends Error {
  readonly code = "MODEL_CONFLICT";
}

export class ModelCapacityError extends Error {
  readonly code = "MODEL_CAPACITY";
}

export class ModelStaleAttemptError extends Error {
  readonly code = "MODEL_STALE_ATTEMPT";
}

