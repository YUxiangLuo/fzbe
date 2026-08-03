import { describe, expect, mock, test } from "bun:test";
import {
  buildManagedDeletionLockNames,
  tryAcquireManagedDeletionLocks,
} from "../../../src/utils/managed-deletion-lock";

const identity = (entityIds: number[]) => ({
  databaseName: "fz_unit",
  kind: "user-deletion",
  entityIds,
});

describe("managed deletion advisory locks", () => {
  test("places a stable kind-wide lock before entity locks when requested", () => {
    const identity = {
      databaseName: "fangzhen_e2e",
      kind: "dataset",
      entityIds: [8, 7],
      includeGlobalLock: true,
    };
    const names = buildManagedDeletionLockNames(identity);
    expect(names).toHaveLength(3);
    expect(names[0]).toBe(buildManagedDeletionLockNames({
      ...identity,
      entityIds: [99],
    })[0]);
    expect(new Set(names).size).toBe(3);
  });

  test("deduplicates entity IDs and builds lock names in numeric order", () => {
    const names = buildManagedDeletionLockNames(identity([20, 3, 20, 11]));

    expect(names).toEqual([
      ...buildManagedDeletionLockNames(identity([3])),
      ...buildManagedDeletionLockNames(identity([11])),
      ...buildManagedDeletionLockNames(identity([20])),
    ]);
    expect(names.every((name) => Buffer.byteLength(name) < 64)).toBe(true);
  });

  test("releases a partial acquisition in reverse order before returning the connection", async () => {
    const names = buildManagedDeletionLockNames(identity([3, 11]));
    const events: string[] = [];
    let getCalls = 0;
    const query = mock(async (sql: string, values: unknown[]) => {
      const lockName = String(values[0]);
      if (sql.includes("GET_LOCK")) {
        events.push(`get:${lockName}`);
        getCalls += 1;
        return [[{ acquired: getCalls === 1 ? 1 : 0 }]];
      }
      events.push(`release:${lockName}`);
      return [[{ released: 1 }]];
    });
    const release = mock(() => { events.push("connection.release"); });
    const destroy = mock(() => { events.push("connection.destroy"); });
    const pool = {
      getConnection: mock(async () => ({ query, release, destroy })),
    };

    const lease = await tryAcquireManagedDeletionLocks(pool as never, identity([11, 3]), 0);

    expect(lease).toBeNull();
    expect(events).toEqual([
      `get:${names[0]}`,
      `get:${names[1]}`,
      `release:${names[0]}`,
      "connection.release",
    ]);
    expect(destroy).not.toHaveBeenCalled();
  });

  test("destroys the connection when lock acquisition throws", async () => {
    const release = mock(() => {});
    const destroy = mock(() => {});
    const pool = {
      getConnection: mock(async () => ({
        query: mock(async () => { throw new Error("database disconnected"); }),
        release,
        destroy,
      })),
    };

    await expect(
      tryAcquireManagedDeletionLocks(pool as never, identity([7]), 0),
    ).rejects.toThrow("database disconnected");
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(release).not.toHaveBeenCalled();
  });

  test("releases all locks in reverse order and makes lease release idempotent", async () => {
    const names = buildManagedDeletionLockNames(identity([3, 11]));
    const events: string[] = [];
    const query = mock(async (sql: string, values: unknown[]) => {
      const lockName = String(values[0]);
      if (sql.includes("GET_LOCK")) {
        events.push(`get:${lockName}`);
        return [[{ acquired: 1 }]];
      }
      events.push(`release:${lockName}`);
      return [[{ released: 1 }]];
    });
    const release = mock(() => { events.push("connection.release"); });
    const destroy = mock(() => { events.push("connection.destroy"); });
    const pool = {
      getConnection: mock(async () => ({ query, release, destroy })),
    };
    const lease = await tryAcquireManagedDeletionLocks(pool as never, identity([11, 3]), 0);

    await lease!.release();
    await lease!.release();

    expect(events).toEqual([
      `get:${names[0]}`,
      `get:${names[1]}`,
      `release:${names[1]}`,
      `release:${names[0]}`,
      "connection.release",
    ]);
    expect(destroy).not.toHaveBeenCalled();
  });
});
