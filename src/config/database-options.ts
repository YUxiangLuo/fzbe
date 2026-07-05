import { readFileSync } from "node:fs";
import type { PoolOptions } from "mysql2/promise";
import { env } from "./env";
import { DB_POOL_CONFIG } from "../utils/constants";

function buildSslOptions(): PoolOptions["ssl"] | undefined {
  if (env.DB_SSL_MODE === "disabled") {
    return undefined;
  }

  const sslOptions: Exclude<PoolOptions["ssl"], string | undefined> = {
    rejectUnauthorized: env.DB_SSL_REJECT_UNAUTHORIZED,
  };

  if (env.DB_SSL_CA_PATH) {
    sslOptions.ca = readFileSync(env.DB_SSL_CA_PATH, "utf8");
  }

  // Node's TLS stack performs hostname verification when a CA is trusted and
  // rejectUnauthorized is true. `verify_ca` and `verify_identity` are both
  // represented with the same PoolOptions shape here.
  return sslOptions;
}

export function buildDatabasePoolOptions(): PoolOptions {
  const options: PoolOptions = {
    host: env.DB_HOST,
    port: env.DB_PORT,
    user: env.DB_USER,
    password: env.DB_PASSWORD,
    database: env.DB_DATABASE,
    waitForConnections: true,
    connectionLimit: DB_POOL_CONFIG.CONNECTION_LIMIT,
    queueLimit: env.DB_QUEUE_LIMIT,
    charset: "utf8mb4",
    enableKeepAlive: true,
    keepAliveInitialDelay: 0,
  };

  if (env.DB_CONNECT_TIMEOUT_MS !== undefined) {
    options.connectTimeout = env.DB_CONNECT_TIMEOUT_MS;
  }

  if (env.DB_TIMEZONE) {
    options.timezone = env.DB_TIMEZONE;
  }

  if (env.DB_DATE_STRINGS) {
    options.dateStrings = true;
  }

  const ssl = buildSslOptions();
  if (ssl) {
    options.ssl = ssl;
  }

  return options;
}
