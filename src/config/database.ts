import mysql from "mysql2/promise";
import { env } from "./env";
import logger from "../utils/logger";
import { CONCURRENCY_LIMITS, DB_POOL_CONFIG } from "../utils/constants";

export const connectionLimit = DB_POOL_CONFIG.CONNECTION_LIMIT;

export const pool = mysql.createPool({
  host: env.DB_HOST,
  user: env.DB_USER,
  password: env.DB_PASSWORD,
  database: env.DB_DATABASE,
  waitForConnections: true,
  connectionLimit,
  queueLimit: 100,
  charset: "utf8mb4",
});

logger.info(
  { connectionLimit, maxConcurrentModelJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS },
  "Database connection pool created.",
);

export type DbPool = typeof pool;
