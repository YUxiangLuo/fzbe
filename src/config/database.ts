import mysql from "mysql2/promise";
import logger from "../utils/logger";
import { CONCURRENCY_LIMITS, DB_POOL_CONFIG } from "../utils/constants";
import { buildDatabasePoolOptions } from "./database-options";
import { configureDatabasePoolSessions } from "./database-session";
import { registerExplicitConnectionQueryPool } from "./database-pool-capabilities";

export const connectionLimit = DB_POOL_CONFIG.CONNECTION_LIMIT;
export const databasePoolOptions = buildDatabasePoolOptions();

export const pool = registerExplicitConnectionQueryPool(
  mysql.createPool(databasePoolOptions),
);
configureDatabasePoolSessions(
  pool,
  databasePoolOptions.timezone as string,
  (error) => logger.error(
    { err: { message: error.message, stack: error.stack } },
    "Failed to configure database session timezone; connection destroyed.",
  ),
);

logger.info(
  {
    deploymentMode: "single_instance",
    connectionLimit,
    queueLimit: databasePoolOptions.queueLimit,
    connectTimeout: databasePoolOptions.connectTimeout,
    timezone: databasePoolOptions.timezone,
    dateStrings: databasePoolOptions.dateStrings,
    ssl: Boolean(databasePoolOptions.ssl),
    maxConcurrentModelJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS,
  },
  "Database connection pool created.",
);

export type DbPool = typeof pool;
