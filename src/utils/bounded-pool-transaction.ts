import type { PoolConnection } from "mysql2/promise";

export interface PoolConnectionSource {
  getConnection(): Promise<PoolConnection>;
}

export class BoundedPoolTransactionTimeoutError extends Error {
  readonly code = "BOUNDED_POOL_TRANSACTION_TIMEOUT";

  constructor(public readonly timeoutMs: number) {
    super(`Database transaction timed out after ${timeoutMs}ms`);
    this.name = "BoundedPoolTransactionTimeoutError";
  }
}

const safeRelease = (connection: PoolConnection) => {
  try {
    connection.release();
  } catch {
    // A late but otherwise healthy pool connection should be returned when possible.
  }
};

const safeDestroy = (connection: PoolConnection) => {
  try {
    connection.destroy();
  } catch {
    // Destruction is best-effort; a timed-out transaction must never return to the pool.
  }
};

/**
 * Runs a transaction under one wall-clock deadline, including pool acquisition,
 * rollback, and commit.
 *
 * mysql2 cannot cancel a pending Pool#getConnection waiter. If the deadline
 * expires before a connection arrives, the late connection is released without
 * running SQL. Once transaction work has started, timeout destroys the
 * connection so queued/active commands cannot outlive the bounded caller on a
 * reusable pool connection.
 */
export async function withBoundedPoolTransaction<T>(
  source: PoolConnectionSource,
  timeoutMs: number,
  callback: (connection: PoolConnection) => Promise<T>,
): Promise<T> {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new Error(`timeoutMs must be positive, got: ${timeoutMs}`);
  }

  const timeoutError = new BoundedPoolTransactionTimeoutError(timeoutMs);
  let connection: PoolConnection | null = null;
  let timedOut = false;

  const operation = (async (): Promise<T> => {
    try {
      const acquiredConnection = await source.getConnection();
      if (timedOut) {
        safeRelease(acquiredConnection);
        // The public promise has already rejected with timeout. This return is
        // observed only by the attached settlement handler below.
        return undefined as T;
      }

      connection = acquiredConnection;
      await connection.beginTransaction();
      if (timedOut) throw timeoutError;

      const result = await callback(connection);
      if (timedOut) throw timeoutError;

      await connection.commit();
      if (timedOut) throw timeoutError;
      return result;
    } catch (error) {
      if (connection && !timedOut) {
        // Rollback is covered by the same outer timer. If rollback stalls, the
        // timer destroys the connection and rejects the caller.
        await connection.rollback().catch(() => {});
      }
      throw error;
    } finally {
      if (connection) {
        const settledConnection = connection;
        connection = null;
        if (timedOut) safeDestroy(settledConnection);
        else safeRelease(settledConnection);
      }
    }
  })();

  return await new Promise<T>((resolve, reject) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      timedOut = true;
      if (connection) safeDestroy(connection);
      reject(timeoutError);
    }, timeoutMs);
    timer.unref?.();

    void operation.then(
      (result) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve(result);
      },
      (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}
