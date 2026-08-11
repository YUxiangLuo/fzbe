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

const safeDestroy = (connection: PoolConnection) => {
  try {
    connection.destroy();
  } catch {
    // Destruction is best-effort; a timed-out transaction must never return to the pool.
  }
};

const safeRelease = (connection: PoolConnection) => {
  try {
    connection.release();
  } catch {
    // If the pool refuses a late or otherwise healthy return, close the
    // connection rather than silently leaking the pool slot.
    safeDestroy(connection);
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
  let connectionReusable = true;
  let transactionPhase: "acquired" | "started" | "commit-attempted" | "committed" = "acquired";

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
      transactionPhase = "started";

      const result = await callback(connection);
      if (timedOut) throw timeoutError;

      transactionPhase = "commit-attempted";
      await connection.commit();
      if (timedOut) throw timeoutError;
      transactionPhase = "committed";
      return result;
    } catch (error) {
      if (connection && !timedOut) {
        if (transactionPhase === "started") {
          // Rollback is covered by the same outer timer. If it rejects or
          // stalls, this connection is not safe to return to the pool.
          try {
            await connection.rollback();
          } catch {
            connectionReusable = false;
          }
        } else if (transactionPhase !== "committed") {
          // BEGIN and COMMIT acknowledgement loss both leave transaction state
          // ambiguous; destroying the session lets the server settle it.
          connectionReusable = false;
        }
      }
      throw error;
    } finally {
      if (connection) {
        const settledConnection = connection;
        connection = null;
        if (timedOut || !connectionReusable) safeDestroy(settledConnection);
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
      if (connection) {
        const timedOutConnection = connection;
        connection = null;
        safeDestroy(timedOutConnection);
      }
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
