import type {
  FieldPacket,
  Pool,
  PoolConnection,
  ResultSetHeader,
  RowDataPacket,
} from "mysql2/promise";
import { TIMEOUTS } from "../../utils/constants";
import {
  ModelCommitOutcomeUnknownError,
  ModelServiceUnavailableError,
} from "./model-errors";

/**
 * mysql2 has no default per-query or connection-acquire timeout. Model
 * operations run these while holding an admission permit and advisory locks, so
 * a hung query or a starved pool would keep the concurrency slot occupied far
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
  onTimeout?: () => void,
): Promise<T> => {
  // Swallow a late rejection from the losing branch so a post-timeout failure
  // does not surface as an unhandled rejection.
  void work.then(undefined, () => {});
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      try {
        onTimeout?.();
      } catch {
        // best effort; releasing the permit matters more than clean teardown
      }
      reject(buildError());
    }, timeoutMs);
    timer.unref?.();
  });
  try {
    return await Promise.race([work, timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
};

/** Run a pooled query with a hard timeout that releases the awaiting caller. */
export function queryWithTimeout<T extends QueryResult = ResultSetHeader>(
  poolLike: Pick<Pool, "query">,
  sql: string,
  values?: unknown[],
  timeoutMs: number = TIMEOUTS.MODEL_DB_QUERY,
): Promise<[T, FieldPacket[]]> {
  const work = (values === undefined
    ? poolLike.query<T>(sql)
    : poolLike.query<T>(sql, values)) as unknown as Promise<[T, FieldPacket[]]>;
  return raceWithTimeout(work, timeoutMs, () =>
    timeoutError("MODEL_DB_QUERY_TIMEOUT", "数据库查询超时，请稍后重试"));
}

const wrapConnectionQueries = (
  connection: PoolConnection,
  queryTimeoutMs: number,
): PoolConnection => {
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
        () => {
          try {
            connection.destroy();
          } catch {
            // connection may already be gone
          }
        },
      );
    };
  const overrides: Record<string, unknown> = {
    query: wrapQuery("query"),
    execute: wrapQuery("execute"),
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
 * Acquire a pooled connection with a bounded wait, then wrap it so every query
 * on it is time-bounded (and the connection is destroyed on query timeout). A
 * connection that arrives after the acquire timeout is released, never leaked.
 */
export async function acquireConnectionWithTimeout(
  pool: Pool,
  acquireTimeoutMs: number = TIMEOUTS.MODEL_DB_CONNECTION_ACQUIRE,
  queryTimeoutMs: number = TIMEOUTS.MODEL_DB_QUERY,
): Promise<PoolConnection> {
  const connection = await new Promise<PoolConnection>((resolve, reject) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      reject(timeoutError(
        "MODEL_DB_CONNECTION_TIMEOUT",
        "数据库连接获取超时，请稍后重试",
      ));
    }, acquireTimeoutMs);
    timer.unref?.();
    pool.getConnection().then(
      (conn) => {
        if (settled) {
          try {
            conn.release();
          } catch {
            // best effort
          }
          return;
        }
        settled = true;
        clearTimeout(timer);
        resolve(conn);
      },
      (cause) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(cause instanceof Error
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
