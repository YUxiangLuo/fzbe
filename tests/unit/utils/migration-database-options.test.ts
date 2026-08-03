import { describe, expect, test } from "bun:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { DATABASE_MIGRATIONS } from "../../../scripts/migrate";
import { buildMigrationDatabaseOptions } from "../../../scripts/migration-database";

const projectRoot = path.resolve(import.meta.dir, "../../..");

const baseDatabaseEnv = {
  DB_HOST: "db.internal",
  DB_USER: "migration-user",
  DB_PASSWORD: "migration-password",
  DB_DATABASE: "migration-db",
};

describe("migration database options", () => {
  test("uses the same port, timeout, timezone, date and TLS semantics without application env", async () => {
    const temporaryDirectory = await mkdtemp(path.join(tmpdir(), "fzbe-migration-ca-"));
    const caPath = path.join(temporaryDirectory, "ca.pem");
    await writeFile(caPath, "test migration ca\n", "utf8");

    try {
      const options = buildMigrationDatabaseOptions({
        ...baseDatabaseEnv,
        DB_PORT: "4406",
        DB_CONNECT_TIMEOUT_MS: "12345",
        DB_TIMEZONE: "Z",
        DB_DATE_STRINGS: "true",
        DB_SSL_MODE: "verify_identity",
        DB_SSL_CA_PATH: caPath,
        DB_SSL_REJECT_UNAUTHORIZED: "false",
      });

      expect(options).toMatchObject({
        host: "db.internal",
        port: 4406,
        user: "migration-user",
        password: "migration-password",
        database: "migration-db",
        connectTimeout: 12_345,
        timezone: "Z",
        dateStrings: true,
        charset: "utf8mb4",
        enableKeepAlive: true,
        keepAliveInitialDelay: 0,
      });
      expect(options.ssl).toEqual({
        rejectUnauthorized: false,
        ca: "test migration ca\n",
      });
    } finally {
      await rm(temporaryDirectory, { recursive: true, force: true });
    }
  });

  test("defaults to the driver's port with TLS disabled and rejects invalid shared settings", () => {
    const options = buildMigrationDatabaseOptions(baseDatabaseEnv);
    expect(options.port).toBeUndefined();
    expect(options.ssl).toBeUndefined();

    expect(() => buildMigrationDatabaseOptions({
      ...baseDatabaseEnv,
      DB_PORT: "70000",
    })).toThrow("DB_PORT");
    expect(() => buildMigrationDatabaseOptions({
      ...baseDatabaseEnv,
      DB_SSL_MODE: "sometimes",
    })).toThrow("DB_SSL_MODE");
  });

  test("routes every ordered migration through the lightweight shared connection helper", async () => {
    for (const migration of DATABASE_MIGRATIONS) {
      const source = await readFile(path.join(projectRoot, migration.scriptPath), "utf8");
      expect(source).toContain('from "./migration-database"');
      expect(source).toContain("createMigrationConnection(");
      expect(source).not.toContain("mysql.createConnection(");
      expect(source).not.toMatch(/process\.env\.DB_(HOST|PORT|USER|PASSWORD|DATABASE|SSL)/);
    }
  });
});
