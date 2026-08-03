import type { PoolOptions } from "mysql2/promise";
import { DB_POOL_CONFIG } from "../utils/constants";
import {
  buildDatabaseConnectionOptions,
  loadDatabaseEnvironment,
  type RuntimeDatabaseEnvSource,
} from "./database-connection-options";

export function buildDatabasePoolOptions(
  source: RuntimeDatabaseEnvSource = process.env,
): PoolOptions {
  const databaseEnv = loadDatabaseEnvironment(source);
  const options: PoolOptions = {
    ...buildDatabaseConnectionOptions(databaseEnv),
    waitForConnections: true,
    connectionLimit: DB_POOL_CONFIG.CONNECTION_LIMIT,
    queueLimit: databaseEnv.DB_QUEUE_LIMIT,
  };
  return options;
}
