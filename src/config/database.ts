import mysql from "mysql2/promise";
import logger from "../utils/logger";
import { CONCURRENCY_LIMITS, DB_POOL_CONFIG } from "../utils/constants";
import { buildDatabasePoolOptions } from "./database-options";

export const connectionLimit = DB_POOL_CONFIG.CONNECTION_LIMIT;
export const databasePoolOptions = buildDatabasePoolOptions();

export const pool = mysql.createPool(databasePoolOptions);

logger.info(
  {
    connectionLimit,
    queueLimit: databasePoolOptions.queueLimit,
    connectTimeout: databasePoolOptions.connectTimeout,
    timezone: databasePoolOptions.timezone,
    dateStrings: databasePoolOptions.dateStrings,
    ssl: Boolean(databasePoolOptions.ssl),
    maxConcurrentTrainJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_TRAIN_JOBS,
    maxConcurrentPrepareJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_PREPARE_JOBS,
    maxConcurrentPredictJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_PREDICT_JOBS,
    maxConcurrentPdfJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_PDF_JOBS,
  },
  "Database connection pool created.",
);

export type DbPool = typeof pool;
