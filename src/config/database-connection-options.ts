import { readFileSync } from "node:fs";
import type { ConnectionOptions } from "mysql2/promise";
import { z } from "zod";

export const isUnsetEnvValue = (value: unknown) =>
  value === undefined || value === null || String(value).trim() === "";

export const createBooleanEnv = (defaultValue: boolean) => z.preprocess((value) => {
  if (isUnsetEnvValue(value)) return defaultValue;
  if (typeof value === "boolean") return value;

  const normalized = String(value).trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off"].includes(normalized)) return false;

  return value;
}, z.boolean());

const dbSslModeEnv = z.preprocess((value) => {
  if (isUnsetEnvValue(value)) {
    return "disabled";
  }

  const normalized = String(value).trim().toLowerCase().replace(/-/g, "_");
  if (["0", "false", "off", "disable"].includes(normalized)) return "disabled";
  if (["1", "true", "on"].includes(normalized)) return "required";
  return normalized;
}, z.enum(["disabled", "required", "verify_ca", "verify_identity"]));

const dbPortEnv = z.preprocess((value) => {
  if (isUnsetEnvValue(value)) return undefined;
  return value;
}, z.coerce.number().int().min(1).max(65535).optional());

const optionalPositiveIntegerEnv = z.preprocess((value) => {
  if (isUnsetEnvValue(value)) return undefined;
  return value;
}, z.coerce.number().int().positive().optional());

const databaseTimezoneEnv = z.preprocess((value) => {
  if (isUnsetEnvValue(value)) return "Z";
  const normalized = String(value).trim();
  return normalized.toUpperCase() === "Z" ? "Z" : normalized;
}, z.string().refine((value) => {
  if (value === "Z") return true;
  const match = /^([+-])(\d{2}):(\d{2})$/.exec(value);
  if (!match) return false;
  const sign = match[1];
  const hours = Number(match[2]);
  const minutes = Number(match[3]);
  return minutes <= 59
    && hours <= 14
    && !(hours === 14 && minutes !== 0)
    && !(sign === "-" && hours === 14);
}, "DB_TIMEZONE must be Z or a valid fixed offset such as +08:00"));

const dbQueueLimitEnv = z.preprocess((value) => {
  if (isUnsetEnvValue(value)) return 100;
  return value;
}, z.coerce.number().int().positive());

export const optionalTrimmedStringEnv = z.preprocess((value) => {
  if (isUnsetEnvValue(value)) return undefined;
  return value;
}, z.string().trim().min(1).optional());

export const databaseEnvironmentSchema = z.object({
  DB_HOST: z.string().min(1),
  DB_PORT: dbPortEnv,
  DB_USER: z.string().min(1),
  DB_PASSWORD: z.string(),
  DB_DATABASE: z.string().min(1),
  DB_CONNECT_TIMEOUT_MS: optionalPositiveIntegerEnv,
  DB_QUEUE_LIMIT: dbQueueLimitEnv,
  DB_TIMEZONE: databaseTimezoneEnv,
  DB_DATE_STRINGS: createBooleanEnv(false),
  DB_SSL_MODE: dbSslModeEnv,
  DB_SSL_CA_PATH: optionalTrimmedStringEnv,
  DB_SSL_REJECT_UNAUTHORIZED: createBooleanEnv(true),
});

export type RuntimeDatabaseEnvSource = Readonly<Record<string, string | undefined>>;
export type DatabaseEnvironment = z.infer<typeof databaseEnvironmentSchema>;

const formatEnvironmentError = (error: z.ZodError) => {
  const invalid = error.issues.map((issue) => issue.path.join(".")).join(", ");
  return new Error(`Missing or invalid environment variables: ${invalid}`);
};

export function loadDatabaseEnvironment(
  source: RuntimeDatabaseEnvSource = process.env,
): DatabaseEnvironment {
  const result = databaseEnvironmentSchema.safeParse(source);
  if (!result.success) {
    throw formatEnvironmentError(result.error);
  }
  return result.data;
}

function buildSslOptions(
  databaseEnv: DatabaseEnvironment,
): ConnectionOptions["ssl"] | undefined {
  if (databaseEnv.DB_SSL_MODE === "disabled") {
    return undefined;
  }

  const sslOptions: Exclude<ConnectionOptions["ssl"], string | undefined> = {
    rejectUnauthorized: databaseEnv.DB_SSL_REJECT_UNAUTHORIZED,
  };

  if (databaseEnv.DB_SSL_CA_PATH) {
    sslOptions.ca = readFileSync(databaseEnv.DB_SSL_CA_PATH, "utf8");
  }

  // mysql2 delegates CA and hostname verification to Node's TLS stack. The
  // application intentionally represents required/verify_ca/verify_identity
  // with the same options shape; migrations must make the identical choice.
  return sslOptions;
}

export function buildDatabaseConnectionOptions(
  databaseEnv: DatabaseEnvironment,
): ConnectionOptions {
  const options: ConnectionOptions = {
    host: databaseEnv.DB_HOST,
    port: databaseEnv.DB_PORT,
    user: databaseEnv.DB_USER,
    password: databaseEnv.DB_PASSWORD,
    database: databaseEnv.DB_DATABASE,
    charset: "utf8mb4",
    enableKeepAlive: true,
    keepAliveInitialDelay: 0,
  };

  if (databaseEnv.DB_CONNECT_TIMEOUT_MS !== undefined) {
    options.connectTimeout = databaseEnv.DB_CONNECT_TIMEOUT_MS;
  }
  if (databaseEnv.DB_TIMEZONE) {
    options.timezone = databaseEnv.DB_TIMEZONE;
  }
  if (databaseEnv.DB_DATE_STRINGS) {
    options.dateStrings = true;
  }

  const ssl = buildSslOptions(databaseEnv);
  if (ssl) {
    options.ssl = ssl;
  }

  return options;
}
