import mysql from "mysql2/promise";
import { buildDatabasePoolOptions } from "../src/config/database-options";
import { assertE2eDatabaseResetAllowed } from "./test-database-reset-guard";

export type E2eDatabaseBootstrapOutcome = "exists" | "created";

export interface E2eDatabaseBootstrapDeps {
  databaseExists: (database: string) => Promise<boolean>;
  createDatabase: (database: string) => Promise<void>;
}

type E2eDatabaseConnectionFactory = (
  options: mysql.ConnectionOptions,
) => Promise<mysql.Connection>;

/**
 * Playwright starts `webServer` before `globalSetup` (see
 * playwright/lib/runner/tasks.ts, where plugin setup tasks precede global
 * setups), so on a brand new disposable database nothing has run `db:reset`
 * by the time the E2E backend boots. The startup preflight then aborts with
 * "Unknown database" and the suite can never bootstrap itself.
 *
 * Creating the empty database here breaks that deadlock. Schema and seed data
 * remain owned by `globalSetup`; this only guarantees the backend can connect.
 *
 * The guard runs before even checking whether the database exists. Importing
 * the E2E server triggers startup cleanup before Playwright's `globalSetup`, so
 * connecting that server to a non-disposable database is itself a mutation
 * risk even when schema creation is unnecessary.
 */
export async function ensureE2eDatabase(
  database: string,
  deps: E2eDatabaseBootstrapDeps,
  env: Readonly<Record<string, string | undefined>> = process.env,
): Promise<E2eDatabaseBootstrapOutcome> {
  const trimmedDatabase = database.trim();
  if (!trimmedDatabase) {
    throw new Error("DB_DATABASE is required to start the E2E backend.");
  }
  if (database !== trimmedDatabase) {
    throw new Error("DB_DATABASE must not contain leading or trailing whitespace.");
  }

  try {
    // Validate the actual database used by the E2E server, rather than trusting
    // a potentially different DB_DATABASE value inherited through `env`.
    assertE2eDatabaseResetAllowed({ ...env, DB_DATABASE: database });
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Refusing to start the E2E backend with DB_DATABASE=${JSON.stringify(database)}. ${reason}`,
    );
  }

  if (await deps.databaseExists(database)) {
    return "exists";
  }

  await deps.createDatabase(database);
  return "created";
}

export async function ensureE2eDatabaseExists(
  env: Readonly<Record<string, string | undefined>> = process.env,
  createConnection: E2eDatabaseConnectionFactory = (options) => mysql.createConnection(options),
): Promise<E2eDatabaseBootstrapOutcome> {
  // This guard must run before parsing connection options or opening even a
  // server-level connection. Merely connecting an E2E backend can trigger
  // startup cleanup, so remote and production targets require the same
  // explicit overrides as destructive reset/seed commands.
  assertE2eDatabaseResetAllowed(env);

  const { database, ...serverOptions } = buildDatabasePoolOptions(env);
  const connection = await createConnection({
    ...serverOptions,
    // Connect to the server itself; the target schema may not exist yet.
    database: undefined,
    multipleStatements: false,
  });

  try {
    return await ensureE2eDatabase(
      String(database ?? ""),
      {
        databaseExists: async (name) => {
          const [rows] = await connection.query<mysql.RowDataPacket[]>(
            "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = ? LIMIT 1",
            [name],
          );
          return rows.length > 0;
        },
        // The name reached this point only after the disposable-name guard,
        // whose allow list excludes backticks and every other quoting hazard.
        createDatabase: async (name) => {
          await connection.query(
            `CREATE DATABASE IF NOT EXISTS \`${name}\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci`,
          );
        },
      },
      env,
    );
  } finally {
    await connection.end();
  }
}
