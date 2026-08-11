import { describe, expect, mock, test } from "bun:test";
import { registerExplicitConnectionQueryPool } from "../../../src/config/database-pool-capabilities";
import {
  acquireConnectionWithTimeout,
  commitConnectionWithOutcomeGuard,
  queryWithTimeout,
} from "../../../src/modules/models/model-db-guard";
import { runWithModelOperationSignal } from "../../../src/modules/models/model-operation-scope";

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

  test("borrows and releases an explicit connection for real pool queries", async () => {
    const connection = {
      query: mock(async () => [[{ id: 1 }], []]),
      execute: mock(async () => [[], []]),
      beginTransaction: mock(async () => undefined),
      commit: mock(async () => undefined),
      rollback: mock(async () => undefined),
      release: mock(() => undefined),
      destroy: mock(() => undefined),
    };
    const pool = registerExplicitConnectionQueryPool({
      query: mock(() => { throw new Error("hidden Pool#query must not be used"); }),
      getConnection: mock(async () => connection),
    });

    const [rows] = await queryWithTimeout(pool, "SELECT 1", [7], 1_000);

    expect(rows).toEqual([{ id: 1 }]);
    expect(connection.query.mock.calls[0]).toEqual(["SELECT 1", [7]]);
    expect(connection.release).toHaveBeenCalledTimes(1);
  });

  test("destroys the borrowed pool connection when the operation is aborted", async () => {
    const controller = new AbortController();
    const deadlineError = new Error("model operation deadline exceeded");
    const connection = {
      query: mock(() => deferred<unknown>().promise),
      execute: mock(async () => [[], []]),
      beginTransaction: mock(async () => undefined),
      commit: mock(async () => undefined),
      rollback: mock(async () => undefined),
      release: mock(() => undefined),
      destroy: mock(() => undefined),
    };
    const pool = registerExplicitConnectionQueryPool({
      query: mock(() => { throw new Error("hidden Pool#query must not be used"); }),
      getConnection: mock(async () => connection),
    });

    const query = runWithModelOperationSignal(
      controller.signal,
      () => queryWithTimeout(pool as never, "SELECT SLEEP(9999)", undefined, 1_000),
    );
    await Promise.resolve();
    await Promise.resolve();
    controller.abort(deadlineError);

    await expect(query).rejects.toBe(deadlineError);
    expect(connection.destroy).toHaveBeenCalledTimes(1);
    expect(connection.release).not.toHaveBeenCalled();
  });

  test("destroys the borrowed pool connection when its query times out", async () => {
    const connection = {
      query: mock(() => deferred<unknown>().promise),
      execute: mock(async () => [[], []]),
      beginTransaction: mock(async () => undefined),
      commit: mock(async () => undefined),
      rollback: mock(async () => undefined),
      release: mock(() => undefined),
      destroy: mock(() => undefined),
    };
    const pool = registerExplicitConnectionQueryPool({
      query: mock(() => { throw new Error("hidden Pool#query must not be used"); }),
      getConnection: mock(async () => connection),
    });

    await expect(queryWithTimeout(pool as never, "SELECT SLEEP(9999)", undefined, 20))
      .rejects.toMatchObject({ code: "MODEL_DB_QUERY_TIMEOUT" });
    expect(connection.destroy).toHaveBeenCalledTimes(1);
    expect(connection.release).not.toHaveBeenCalled();
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

  test("aborts connection acquisition and releases the late connection", async () => {
    const controller = new AbortController();
    const deadlineError = new Error("model operation deadline exceeded");
    const conn = fakeConnection();
    const late = deferred<typeof conn>();
    const pool = { getConnection: mock(() => late.promise) } as never;

    const acquire = runWithModelOperationSignal(
      controller.signal,
      () => acquireConnectionWithTimeout(pool, 1_000, 1_000),
    );
    controller.abort(deadlineError);
    await expect(acquire).rejects.toBe(deadlineError);

    late.resolve(conn);
    await Promise.resolve();
    await Promise.resolve();
    expect(conn.release).toHaveBeenCalledTimes(1);
  });

  test("rejects when the pool never yields a connection", async () => {
    const pool = { getConnection: mock(() => deferred().promise) } as never;

    await expect(acquireConnectionWithTimeout(pool, 30, 1_000))
      .rejects.toMatchObject({ code: "MODEL_DB_CONNECTION_TIMEOUT" });
  });

  test("destroys a late connection when returning it to the pool fails", async () => {
    const conn = fakeConnection();
    conn.release.mockImplementationOnce(() => {
      throw new Error("pool release failed");
    });
    const late = deferred<typeof conn>();
    const pool = { getConnection: mock(() => late.promise) } as never;

    const acquire = acquireConnectionWithTimeout(pool, 20, 1_000);
    await expect(acquire).rejects.toMatchObject({ code: "MODEL_DB_CONNECTION_TIMEOUT" });

    late.resolve(conn);
    await Promise.resolve();
    await Promise.resolve();
    expect(conn.release).toHaveBeenCalledTimes(1);
    expect(conn.destroy).toHaveBeenCalledTimes(1);
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

  test("destroys a wrapped connection when a normal pool release fails", async () => {
    const conn = fakeConnection();
    conn.release.mockImplementationOnce(() => {
      throw new Error("pool release failed");
    });
    const pool = { getConnection: mock(async () => conn) } as never;
    const wrapped = await acquireConnectionWithTimeout(pool, 1_000, 1_000);

    expect(() => wrapped.release()).not.toThrow();
    expect(conn.destroy).toHaveBeenCalledTimes(1);
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

  test.each([
    "beginTransaction",
    "rollback",
  ] as const)("bounds a hung %s and destroys the transaction connection", async (method) => {
    const conn = fakeConnection();
    conn[method] = mock(() => deferred<void>().promise);
    const pool = { getConnection: mock(async () => conn) } as never;
    const wrapped = await acquireConnectionWithTimeout(pool, 1_000, 20);

    await expect(wrapped[method]()).rejects.toMatchObject({
      code: "MODEL_DB_TRANSACTION_TIMEOUT",
    });
    expect(conn.destroy).toHaveBeenCalledTimes(1);
  });

  test.each([
    "beginTransaction",
    "rollback",
  ] as const)("destroys the transaction connection when %s rejects", async (method) => {
    const conn = fakeConnection();
    conn[method].mockRejectedValueOnce(new Error(`${method} acknowledgement lost`));
    const pool = { getConnection: mock(async () => conn) } as never;
    const wrapped = await acquireConnectionWithTimeout(pool, 1_000, 1_000);

    await expect(wrapped[method]()).rejects.toThrow("acknowledgement lost");
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

describe("model-db-guard commitConnectionWithOutcomeGuard", () => {
  const fakeConnection = () => ({
    commit: mock(async () => undefined),
    destroy: mock(() => undefined),
  });

  test("returns normally after an acknowledged commit", async () => {
    const connection = fakeConnection();

    await commitConnectionWithOutcomeGuard(connection as never, 1_000);

    expect(connection.commit).toHaveBeenCalledTimes(1);
    expect(connection.destroy).not.toHaveBeenCalled();
  });

  test("destroys the connection and classifies a rejected commit as outcome-unknown", async () => {
    const connection = fakeConnection();
    const acknowledgementLoss = new Error("commit acknowledgement lost");
    connection.commit.mockRejectedValueOnce(acknowledgementLoss);

    await expect(commitConnectionWithOutcomeGuard(connection as never, 1_000))
      .rejects.toMatchObject({
        code: "MODEL_COMMIT_OUTCOME_UNKNOWN",
        retryable: true,
        cause: acknowledgementLoss,
      });
    expect(connection.destroy).toHaveBeenCalledTimes(1);
  });

  test("bounds a hung commit and destroys its ambiguous connection", async () => {
    const connection = fakeConnection();
    connection.commit = mock(() => deferred<void>().promise);

    await expect(commitConnectionWithOutcomeGuard(connection as never, 20))
      .rejects.toMatchObject({ code: "MODEL_COMMIT_OUTCOME_UNKNOWN" });
    expect(connection.destroy).toHaveBeenCalledTimes(1);
  });
});
