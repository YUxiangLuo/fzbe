import { describe, expect, mock, test } from "bun:test";
import type { PoolConnection } from "mysql2/promise";
import {
  BoundedPoolTransactionTimeoutError,
  withBoundedPoolTransaction,
} from "../../../src/utils/bounded-pool-transaction";

const makeConnection = () => {
  const beginTransaction = mock(async () => undefined);
  const commit = mock(async () => undefined);
  const rollback = mock(async () => undefined);
  const release = mock(() => undefined);
  const destroy = mock(() => undefined);
  return {
    connection: {
      beginTransaction,
      commit,
      rollback,
      release,
      destroy,
    } as unknown as PoolConnection,
    beginTransaction,
    commit,
    rollback,
    release,
    destroy,
  };
};

describe("bounded pool transaction", () => {
  test("commits and releases a successful transaction", async () => {
    const fake = makeConnection();
    const source = { getConnection: mock(async () => fake.connection) };
    const callback = mock(async () => "written");

    const result = await withBoundedPoolTransaction(source, 100, callback);

    expect(result).toBe("written");
    expect(fake.beginTransaction).toHaveBeenCalledTimes(1);
    expect(callback).toHaveBeenCalledWith(fake.connection);
    expect(fake.commit).toHaveBeenCalledTimes(1);
    expect(fake.rollback).not.toHaveBeenCalled();
    expect(fake.release).toHaveBeenCalledTimes(1);
    expect(fake.destroy).not.toHaveBeenCalled();
  });

  test("rolls back and releases when transaction work fails", async () => {
    const fake = makeConnection();
    const source = { getConnection: mock(async () => fake.connection) };
    const original = new Error("write failed");

    await expect(withBoundedPoolTransaction(source, 100, async () => {
      throw original;
    })).rejects.toBe(original);

    expect(fake.rollback).toHaveBeenCalledTimes(1);
    expect(fake.release).toHaveBeenCalledTimes(1);
    expect(fake.destroy).not.toHaveBeenCalled();
  });

  test("destroys rather than releases when rollback acknowledgement is lost", async () => {
    const fake = makeConnection();
    fake.rollback.mockRejectedValueOnce(new Error("rollback acknowledgement lost"));
    const source = { getConnection: mock(async () => fake.connection) };
    const original = new Error("write failed");

    await expect(withBoundedPoolTransaction(source, 100, async () => {
      throw original;
    })).rejects.toBe(original);

    expect(fake.release).not.toHaveBeenCalled();
    expect(fake.destroy).toHaveBeenCalledTimes(1);
  });

  test.each([
    "beginTransaction",
    "commit",
  ] as const)("destroys rather than releases when %s acknowledgement is lost", async (method) => {
    const fake = makeConnection();
    fake[method].mockRejectedValueOnce(new Error(`${method} acknowledgement lost`));
    const source = { getConnection: mock(async () => fake.connection) };

    await expect(withBoundedPoolTransaction(source, 100, async () => "written"))
      .rejects.toThrow("acknowledgement lost");

    expect(fake.rollback).not.toHaveBeenCalled();
    expect(fake.release).not.toHaveBeenCalled();
    expect(fake.destroy).toHaveBeenCalledTimes(1);
  });

  test("propagates pool acquisition errors without running transaction work", async () => {
    const original = new Error("pool unavailable");
    const source = { getConnection: mock(async () => { throw original; }) };
    const callback = mock(async () => undefined);

    await expect(withBoundedPoolTransaction(source, 100, callback)).rejects.toBe(original);

    expect(callback).not.toHaveBeenCalled();
  });

  test("releases a healthy connection that arrives after acquisition timeout", async () => {
    const fake = makeConnection();
    let resolveConnection!: (connection: PoolConnection) => void;
    const source = {
      getConnection: mock(() => new Promise<PoolConnection>((resolve) => {
        resolveConnection = resolve;
      })),
    };

    const result = withBoundedPoolTransaction(source, 10, async () => undefined);
    await expect(result).rejects.toBeInstanceOf(BoundedPoolTransactionTimeoutError);

    resolveConnection(fake.connection);
    await Bun.sleep(1);

    expect(fake.beginTransaction).not.toHaveBeenCalled();
    expect(fake.release).toHaveBeenCalledTimes(1);
    expect(fake.destroy).not.toHaveBeenCalled();
  });

  test("destroys a late connection when returning it to the pool fails", async () => {
    const fake = makeConnection();
    fake.release.mockImplementationOnce(() => {
      throw new Error("pool release failed");
    });
    let resolveConnection!: (connection: PoolConnection) => void;
    const source = {
      getConnection: mock(() => new Promise<PoolConnection>((resolve) => {
        resolveConnection = resolve;
      })),
    };

    const result = withBoundedPoolTransaction(source, 10, async () => undefined);
    await expect(result).rejects.toBeInstanceOf(BoundedPoolTransactionTimeoutError);
    resolveConnection(fake.connection);
    await Bun.sleep(1);

    expect(fake.release).toHaveBeenCalledTimes(1);
    expect(fake.destroy).toHaveBeenCalledTimes(1);
  });

  test("destroys an acquired connection when transaction work exceeds the deadline", async () => {
    const fake = makeConnection();
    const source = { getConnection: mock(async () => fake.connection) };
    let unblock!: () => void;
    const blocked = new Promise<void>((resolve) => {
      unblock = resolve;
    });

    const result = withBoundedPoolTransaction(source, 10, async () => {
      await blocked;
    });
    await expect(result).rejects.toMatchObject({
      code: "BOUNDED_POOL_TRANSACTION_TIMEOUT",
      timeoutMs: 10,
    });

    expect(fake.destroy).toHaveBeenCalled();
    expect(fake.release).not.toHaveBeenCalled();

    unblock();
    await Bun.sleep(1);
    expect(fake.commit).not.toHaveBeenCalled();
  });
});
