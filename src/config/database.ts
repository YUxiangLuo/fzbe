import mysql from "mysql2/promise";
import { env } from "./env";
import logger from "../logger";

export const pool = mysql.createPool({
  host: env.DB_HOST,
  user: env.DB_USER,
  password: env.DB_PASSWORD,
  database: env.DB_DATABASE,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  charset: "utf8mb4",
});

logger.info("Database connection pool created.");

export type DbPool = typeof pool;
