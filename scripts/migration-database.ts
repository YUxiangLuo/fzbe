import mysql, {
  type Connection,
  type ConnectionOptions,
} from "mysql2/promise";
import {
  buildDatabaseConnectionOptions,
  loadDatabaseEnvironment,
  type RuntimeDatabaseEnvSource,
} from "../src/config/database-connection-options";

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
  return await mysql.createConnection(buildMigrationDatabaseOptions(source, overrides));
}
