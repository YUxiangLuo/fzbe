import type {
  FieldPacket,
  Pool,
  PoolConnection,
  ResultSetHeader,
  RowDataPacket,
} from "mysql2/promise";
import { isExplicitConnectionQueryPool } from "../../config/database-pool-capabilities";
import { TIMEOUTS } from "../../utils/constants";
import {
  ModelCommitOutcomeUnknownError,
  ModelServiceUnavailableError,
} from "./model-errors";
import { getCurrentModelOperationSignal } from "./model-operation-scope";

/**
 * mysql2 has no default per-query or connection-acquire timeout. Model
 * operations run these while holding a process-local admission permit, so a
 * hung query or a starved pool would keep the concurrency slot occupied far
 * past the operation deadline (the deadline AbortSignal cannot cancel an
 * in-flight mysql2 call). These helpers bound both with a JS-level race, so the
 * awaiting operation always settles and the lease releases. The race preserves
 * the underlying `(sql, values)` call shape, so callers and query mocks are
 * unaffected.
 */

type QueryResult =
  | RowDataPacket[]
  | RowDataPacket[][]
  | ResultSetHeader
  | ResultSetHeader[];

const timeoutError = (code: string, message: string) =>
  new ModelServiceUnavailableError(message, code);

const raceWithTimeout = async <T>(
  work: Promise<T>,
  timeoutMs: number,
  buildError: () => Error,
  onTerminate?: () => void,
  terminationSignal: AbortSignal | undefined = getCurrentModelOperationSignal(),
): Promise<T> => {
  // Swallow a late rejection from the losing branch so a post-timeout failure
  // does not surface as an unhandled rejection.
  void work.then(undefined, () => {});
  const terminate = () => {
    try {
      onTerminate?.();
    } catch {
      // best effort; releasing the permit matters more than clean teardown
    }
  };
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      terminate();
      reject(buildError());
    }, timeoutMs);
    timer.unref?.();
  });
  let abortHandler: (() => void) | undefined;
  const aborted = new Promise<never>((_, reject) => {
    if (!terminationSignal) return;
    abortHandler = () => {
      terminate();
      reject(
        terminationSignal.reason instanceof Error
          ? terminationSignal.reason
          : new Error("模型数据库操作已终止"),
      );
    };
    terminationSignal.addEventListener("abort", abortHandler, { once: true });
    if (terminationSignal.aborted) abortHandler();
  });
  try {
    return await Promise.race([work, timeout, aborted]);
  } finally {
    if (timer) clearTimeout(timer);
    if (abortHandler) terminationSignal?.removeEventListener("abort", abortHandler);
  }
};

/** Run a pooled query with a hard timeout that releases the awaiting caller. */
export async function queryWithTimeout<T extends QueryResult = ResultSetHeader>(
  poolLike: Pick<Pool, "query"> & Partial<Pick<Pool, "getConnection">>,
  sql: string,
  values?: unknown[],
  timeoutMs: number = TIMEOUTS.MODEL_DB_QUERY,
): Promise<[T, FieldPacket[]]> {
  if (
    typeof poolLike.getConnection === "function"
    && isExplicitConnectionQueryPool(poolLike)
  ) {
    // Pool#query hides the borrowed connection, so a timed-out query cannot be
    // cancelled or evicted by its caller. Borrow explicitly: the wrapped query
    // destroys its connection on timeout and release is a no-op afterwards.
    const connection = await acquireConnectionWithTimeout(
      poolLike as unknown as Pool,
      TIMEOUTS.MODEL_DB_CONNECTION_ACQUIRE,
      timeoutMs,
    );
    try {
      return (values === undefined
        ? await connection.query<T>(sql)
        : await connection.query<T>(sql, values)) as [T, FieldPacket[]];
    } finally {
      connection.release();
    }
  }

  const work = (values === undefined
    ? poolLike.query<T>(sql)
    : poolLike.query<T>(sql, values)) as unknown as Promise<[T, FieldPacket[]]>;
  return await raceWithTimeout(work, timeoutMs, () =>
    timeoutError("MODEL_DB_QUERY_TIMEOUT", "数据库查询超时，请稍后重试"));
}

const wrapConnectionQueries = (
  connection: PoolConnection,
  queryTimeoutMs: number,
): PoolConnection => {
  let connectionDestroyed = false;
  const destroyConnection = () => {
    if (connectionDestroyed) return;
    connectionDestroyed = true;
    try {
      connection.destroy();
    } catch {
      // connection may already be gone
    }
  };
  const wrapQuery = (method: "query" | "execute") =>
    (sql: unknown, values?: unknown) => {
      const runner = connection[method] as (
        sqlArg: unknown,
        valuesArg?: unknown,
      ) => Promise<unknown>;
      const work = values === undefined
        ? runner.call(connection, sql)
        : runner.call(connection, sql, values);
      return raceWithTimeout(
        work,
        queryTimeoutMs,
        () => timeoutError("MODEL_DB_QUERY_TIMEOUT", "数据库查询超时，请稍后重试"),
        // Destroy the dedicated connection so the in-flight query is torn down
        // and the pool slot is reclaimed instead of lingering.
        destroyConnection,
      );
    };
  const wrapTransactionControl = (method: "beginTransaction" | "rollback") =>
    () => {
      try {
        const work = connection[method]();
        return raceWithTimeout(
          work,
          queryTimeoutMs,
          () => timeoutError(
            "MODEL_DB_TRANSACTION_TIMEOUT",
            method === "beginTransaction"
              ? "数据库事务启动超时，请稍后重试"
              : "数据库事务回滚超时，请稍后重试",
          ),
          // Transaction-control acknowledgement loss makes the session state
          // unsafe to reuse. Server-side rollback can lag behind socket close
          // while the current statement is still executing; this is not an
          // immediate server query/lock cancellation guarantee.
          destroyConnection,
        ).catch((error) => {
          destroyConnection();
          throw error;
        });
      } catch (error) {
        destroyConnection();
        return Promise.reject(error);
      }
    };
  const overrides: Record<string, unknown> = {
    query: wrapQuery("query"),
    execute: wrapQuery("execute"),
    beginTransaction: wrapTransactionControl("beginTransaction"),
    rollback: wrapTransactionControl("rollback"),
    destroy: destroyConnection,
    release: () => {
      if (connectionDestroyed) return;
      try {
        connection.release();
      } catch {
        // A release failure must not turn an acknowledged read/transaction into
        // an application failure, but the pool slot must not be leaked.
        destroyConnection();
      }
    },
  };
  return new Proxy(connection, {
    get(target, prop, receiver) {
      if (typeof prop === "string" && prop in overrides) {
        return overrides[prop];
      }
      const value = Reflect.get(target, prop, receiver);
      return typeof value === "function" ? value.bind(target) : value;
    },
  }) as PoolConnection;
};

/**
 * Acquire a pooled connection with a bounded wait, then wrap queries plus
 * BEGIN/ROLLBACK so each is time-bounded and destroys the connection on
 * timeout. COMMIT uses the separate outcome-unknown guard below. A connection
 * that arrives after the acquire timeout is released or destroyed, never leaked.
 */
export async function acquireConnectionWithTimeout(
  pool: Pool,
  acquireTimeoutMs: number = TIMEOUTS.MODEL_DB_CONNECTION_ACQUIRE,
  queryTimeoutMs: number = TIMEOUTS.MODEL_DB_QUERY,
): Promise<PoolConnection> {
  const terminationSignal = getCurrentModelOperationSignal();
  const connection = await new Promise<PoolConnection>((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      clearTimeout(timer);
      terminationSignal?.removeEventListener("abort", abortHandler);
    };
    const settleRejected = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };
    const abortHandler = () => settleRejected(
      terminationSignal?.reason instanceof Error
        ? terminationSignal.reason
        : new Error("模型数据库连接获取已终止"),
    );
    const timer = setTimeout(() => {
      settleRejected(timeoutError(
        "MODEL_DB_CONNECTION_TIMEOUT",
        "数据库连接获取超时，请稍后重试",
      ));
    }, acquireTimeoutMs);
    timer.unref?.();
    terminationSignal?.addEventListener("abort", abortHandler, { once: true });
    if (terminationSignal?.aborted) abortHandler();

    pool.getConnection().then(
      (conn) => {
        if (settled) {
          try {
            conn.release();
          } catch {
            try {
              conn.destroy();
            } catch {
              // best effort; the pool already rejected this waiter
            }
          }
          return;
        }
        settled = true;
        cleanup();
        resolve(conn);
      },
      (cause) => {
        settleRejected(cause instanceof Error
          ? cause
          : timeoutError("MODEL_DB_CONNECTION_UNAVAILABLE", "数据库连接不可用，请稍后重试"));
      },
    );
  });
  return wrapConnectionQueries(connection, queryTimeoutMs);
}

/**
 * Bounds COMMIT and fails closed when its acknowledgement is lost. A COMMIT
 * error cannot prove whether the server committed, so the dedicated connection
 * is destroyed and must not be rolled back or returned to the pool.
 */
export async function commitConnectionWithOutcomeGuard(
  connection: PoolConnection,
  timeoutMs: number = TIMEOUTS.MODEL_DB_QUERY,
): Promise<void> {
  let destroyed = false;
  const destroy = () => {
    if (destroyed) return;
    destroyed = true;
    try {
      connection.destroy();
    } catch {
      // The connection may already have been torn down by mysql2.
    }
  };
  try {
    const commitWork = connection.commit();
    await raceWithTimeout(
      commitWork,
      timeoutMs,
      () => new ModelCommitOutcomeUnknownError(),
      destroy,
    );
  } catch (error) {
    destroy();
    if (error instanceof ModelCommitOutcomeUnknownError) throw error;
    throw new ModelCommitOutcomeUnknownError(undefined, error);
  }
}
