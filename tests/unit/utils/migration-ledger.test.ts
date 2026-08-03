import { describe, expect, test } from "bun:test";
import type { Connection } from "mysql2/promise";
import {
  createDatabaseMigrationCoordinator,
  DATABASE_MIGRATION_LEDGER_TABLE,
  getDatabaseMigrationLockName,
} from "../../../scripts/migration-ledger";

const databaseEnv = {
  DB_HOST: "127.0.0.1",
  DB_USER: "migration-user",
  DB_PASSWORD: "migration-password",
  DB_DATABASE: "migration-ledger-test",
};

const normalizeSql = (sql: string) => sql.replace(/\s+/g, " ").trim();

describe("database migration ledger coordinator", () => {
  test("holds an advisory lock around ledger creation, reads and successful records", async () => {
    const events: string[] = [];
    const rows = new Map<string, string>();
    const connection = {
      async query(sql: string, values?: unknown[]) {
        const normalized = normalizeSql(sql);
        if (normalized.includes("GET_LOCK")) {
          events.push(`lock:${String(values?.[0])}:${String(values?.[1])}`);
          return [[{ acquired: 1 }], []];
        }
        if (normalized.startsWith("CREATE TABLE")) {
          events.push("create-ledger");
          expect(normalized).toContain(DATABASE_MIGRATION_LEDGER_TABLE);
          return [[], []];
        }
        if (normalized.startsWith("SELECT migration_id")) {
          events.push("list");
          return [[...rows].map(([migration_id, script_path]) => ({
            migration_id,
            script_path,
          })), []];
        }
        if (normalized.includes("RELEASE_LOCK")) {
          events.push(`release:${String(values?.[0])}`);
          return [[{ acquired: 1 }], []];
        }
        throw new Error(`Unexpected query: ${normalized}`);
      },
      async execute(sql: string, values?: unknown[]) {
        const normalized = normalizeSql(sql);
        expect(normalized).toContain(`INSERT INTO \`${DATABASE_MIGRATION_LEDGER_TABLE}\``);
        const id = String(values?.[0]);
        const scriptPath = String(values?.[1]);
        rows.set(id, scriptPath);
        events.push(`record:${id}`);
        return [{ affectedRows: 1 }, []];
      },
      async end() {
        events.push("end");
      },
    } as unknown as Connection;

    const coordinator = createDatabaseMigrationCoordinator(
      databaseEnv,
      17,
      async () => connection,
    );
    await coordinator.runExclusive(async (ledger) => {
      expect(await ledger.listApplied()).toEqual(new Map());
      await ledger.recordApplied({
        id: "001-example",
        scriptPath: "scripts/example.ts",
      });
    });

    const lockName = getDatabaseMigrationLockName(databaseEnv.DB_DATABASE);
    expect(lockName.length).toBeLessThanOrEqual(64);
    expect(events).toEqual([
      `lock:${lockName}:17`,
      "create-ledger",
      "list",
      "record:001-example",
      `release:${lockName}`,
      "end",
    ]);
  });

  test("fails without running work when the advisory lock times out and still closes", async () => {
    const events: string[] = [];
    const connection = {
      async query(sql: string) {
        expect(normalizeSql(sql)).toContain("GET_LOCK");
        events.push("lock");
        return [[{ acquired: 0 }], []];
      },
      async end() {
        events.push("end");
      },
    } as unknown as Connection;
    const coordinator = createDatabaseMigrationCoordinator(
      databaseEnv,
      0,
      async () => connection,
    );

    let called = false;
    await expect(coordinator.runExclusive(async () => {
      called = true;
    })).rejects.toThrow("Timed out waiting 0s");
    expect(called).toBeFalse();
    expect(events).toEqual(["lock", "end"]);
  });

  test("releases the advisory lock when migration work throws", async () => {
    const events: string[] = [];
    const connection = {
      async query(sql: string) {
        const normalized = normalizeSql(sql);
        if (normalized.includes("GET_LOCK")) {
          events.push("lock");
          return [[{ acquired: 1 }], []];
        }
        if (normalized.startsWith("CREATE TABLE")) {
          events.push("create-ledger");
          return [[], []];
        }
        if (normalized.includes("RELEASE_LOCK")) {
          events.push("release");
          return [[{ acquired: 1 }], []];
        }
        throw new Error(`Unexpected query: ${normalized}`);
      },
      async end() {
        events.push("end");
      },
    } as unknown as Connection;
    const coordinator = createDatabaseMigrationCoordinator(
      databaseEnv,
      1,
      async () => connection,
    );

    await expect(coordinator.runExclusive(async () => {
      throw new Error("migration exploded");
    })).rejects.toThrow("migration exploded");
    expect(events).toEqual(["lock", "create-ledger", "release", "end"]);
  });
});
