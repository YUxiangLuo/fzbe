import mysql, {
  type Connection,
  type ConnectionOptions,
} from "mysql2/promise";
import {
  buildDatabaseConnectionOptions,
  loadDatabaseEnvironment,
  type RuntimeDatabaseEnvSource,
} from "../src/config/database-connection-options";
import { configureDatabaseSession } from "../src/config/database-session";

export function buildMigrationDatabaseOptions(
  source: RuntimeDatabaseEnvSource = process.env,
  overrides: Pick<ConnectionOptions, "multipleStatements"> = {},
): ConnectionOptions {
  return {
    ...buildDatabaseConnectionOptions(loadDatabaseEnvironment(source)),
    ...overrides,
  };
}

export async function createMigrationConnection(
  overrides: Pick<ConnectionOptions, "multipleStatements"> = {},
  source: RuntimeDatabaseEnvSource = process.env,
): Promise<Connection> {
  const databaseEnv = loadDatabaseEnvironment(source);
  const connection = await mysql.createConnection({
    ...buildDatabaseConnectionOptions(databaseEnv),
    ...overrides,
  });
  try {
    await configureDatabaseSession(connection, databaseEnv.DB_TIMEZONE);
    return connection;
  } catch (error) {
    await connection.end().catch(() => {});
    throw error;
  }
}
