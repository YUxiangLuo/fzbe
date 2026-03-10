import mysql from "mysql2/promise";
import { env } from "./env";
import logger from "../utils/logger";
import { CONCURRENCY_LIMITS } from "../utils/constants";

/**
 * 连接池大小必须 > 2 × MAX_CONCURRENT_MODEL_JOBS，否则会死锁：
 * 每个模型任务持有 2 个连接（执行锁 + 槽位锁），训练/预测逻辑还需要
 * 额外连接执行数据库查询。若连接池耗尽，查询等待连接释放，但连接被锁
 * 占用、锁又要等训练完成——形成循环等待。
 */
const LOCK_CONNECTIONS = 2 * CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS;
/**
 * 除了锁连接，其他模块（auth、experiments、reports 等 13 个 service）
 * 也共享同一连接池。18 留出足够余量让普通查询、事务与锁连接并行，
 * 避免所有连接都被锁占满后其他请求排队超时。
 */
const QUERY_HEADROOM = 18;
export const connectionLimit = LOCK_CONNECTIONS + QUERY_HEADROOM;

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
