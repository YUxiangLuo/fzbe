export interface RetryableDatabaseError {
  code?: string;
  errno?: number;
  sqlState?: string;
  message?: string;
}

export interface DeadlockRetryInfo {
  error: unknown;
  attempt: number;
  nextAttempt: number;
  maxAttempts: number;
  delayMs: number;
}

export interface DeadlockRetryOptions {
  maxAttempts?: number;
  retryDelaysMs?: readonly number[];
  onRetry?: (info: DeadlockRetryInfo) => void;
}

const DEFAULT_MAX_ATTEMPTS = 3;
const DEFAULT_RETRY_DELAYS_MS = [50, 150, 300] as const;

export function isRetryableTransactionError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;

  const maybeError = error as RetryableDatabaseError;
  const message = maybeError.message ?? "";

  return (
    maybeError.code === "ER_LOCK_DEADLOCK" ||
    maybeError.errno === 1213 ||
    maybeError.sqlState === "40001" ||
    message.includes("Deadlock found") ||
    maybeError.code === "ER_LOCK_WAIT_TIMEOUT" ||
    maybeError.errno === 1205 ||
    message.includes("Lock wait timeout exceeded")
  );
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function withDeadlockRetry<T>(
  operation: () => Promise<T>,
  options: DeadlockRetryOptions = {},
): Promise<T> {
  const maxAttempts = options.maxAttempts ?? DEFAULT_MAX_ATTEMPTS;
  const retryDelaysMs = options.retryDelaysMs ?? DEFAULT_RETRY_DELAYS_MS;

  for (let attemptIndex = 0; attemptIndex < maxAttempts; attemptIndex += 1) {
    try {
      return await operation();
    } catch (error) {
      const hasMoreAttempts = attemptIndex < maxAttempts - 1;
      if (!hasMoreAttempts || !isRetryableTransactionError(error)) {
        throw error;
      }

      const delayMs = retryDelaysMs[attemptIndex] ?? retryDelaysMs[retryDelaysMs.length - 1] ?? 0;
      options.onRetry?.({
        error,
        attempt: attemptIndex + 1,
        nextAttempt: attemptIndex + 2,
        maxAttempts,
        delayMs,
      });
      if (delayMs > 0) {
        await sleep(delayMs);
      }
    }
  }

  throw new Error("Deadlock retry loop exhausted unexpectedly");
}
