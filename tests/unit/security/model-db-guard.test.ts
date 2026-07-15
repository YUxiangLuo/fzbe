import { describe, expect, mock, test } from "bun:test";
import {
  acquireConnectionWithTimeout,
  queryWithTimeout,
} from "../../../src/modules/models/model-db-guard";

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

describe("model-db-guard queryWithTimeout", () => {
  test("resolves a fast query and preserves the (sql, values) call shape", async () => {
    const query = mock(async (_sql: string, _values?: unknown[]) => [[{ id: 1 }], []]);
    const pool = { query } as never;

    const [rows] = await queryWithTimeout(pool, "SELECT 1", [7], 1_000);

    expect(rows).toEqual([{ id: 1 }]);
    expect(query.mock.calls[0]).toEqual(["SELECT 1", [7]]);
  });

  test("rejects a hung query with a typed timeout instead of hanging", async () => {
    const query = mock(() => deferred<unknown>().promise); // never settles
    const pool = { query } as never;

    await expect(queryWithTimeout(pool, "SELECT SLEEP(9999)", undefined, 30))
      .rejects.toMatchObject({ code: "MODEL_DB_QUERY_TIMEOUT" });
  });
});

describe("model-db-guard acquireConnectionWithTimeout", () => {
  const fakeConnection = () => ({
    query: mock(async (_sql: string, _values?: unknown[]) => [[], []]),
    beginTransaction: mock(async () => undefined),
    commit: mock(async () => undefined),
    rollback: mock(async () => undefined),
    release: mock(() => undefined),
    destroy: mock(() => undefined),
  });

  test("rejects when the pool never yields a connection", async () => {
    const pool = { getConnection: mock(() => deferred().promise) } as never;

    await expect(acquireConnectionWithTimeout(pool, 30, 1_000))
      .rejects.toMatchObject({ code: "MODEL_DB_CONNECTION_TIMEOUT" });
  });

  test("releases a connection that arrives after the acquire timeout", async () => {
    const conn = fakeConnection();
    const late = deferred<typeof conn>();
    const pool = { getConnection: mock(() => late.promise) } as never;

    const acquire = acquireConnectionWithTimeout(pool, 20, 1_000);
    await expect(acquire).rejects.toMatchObject({ code: "MODEL_DB_CONNECTION_TIMEOUT" });

    late.resolve(conn); // arrives late
    await Promise.resolve();
    await Promise.resolve();
    expect(conn.release).toHaveBeenCalledTimes(1);
  });

  test("wraps queries with a timeout that destroys the connection, and passes other methods through", async () => {
    const conn = fakeConnection();
    conn.query = mock(() => deferred<unknown>().promise); // hangs
    const pool = { getConnection: mock(async () => conn) } as never;

    const wrapped = await acquireConnectionWithTimeout(pool, 1_000, 25);

    // passthrough
    await wrapped.beginTransaction();
    expect(conn.beginTransaction).toHaveBeenCalledTimes(1);

    // hung query -> typed timeout + connection destroyed
    await expect(wrapped.query("SELECT SLEEP(9999)")).rejects.toMatchObject({
      code: "MODEL_DB_QUERY_TIMEOUT",
    });
    expect(conn.destroy).toHaveBeenCalledTimes(1);
  });

  test("wrapped fast query preserves call shape", async () => {
    const conn = fakeConnection();
    const pool = { getConnection: async () => conn } as never;

    const wrapped = await acquireConnectionWithTimeout(pool, 1_000, 1_000);
    await wrapped.query("INSERT INTO t VALUES (?)", [5]);

    expect(conn.query.mock.calls[0]).toEqual(["INSERT INTO t VALUES (?)", [5]]);
  });
});
