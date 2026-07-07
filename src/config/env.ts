import { z } from "zod";

const isUnsetEnvValue = (value: unknown) =>
  value === undefined || value === null || String(value).trim() === "";

const createBooleanEnv = (defaultValue: boolean) => z.preprocess((value) => {
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

const dbQueueLimitEnv = z.preprocess((value) => {
  if (isUnsetEnvValue(value)) return 100;
  return value;
}, z.coerce.number().int().positive());

const optionalTrimmedStringEnv = z.preprocess((value) => {
  if (isUnsetEnvValue(value)) return undefined;
  return value;
}, z.string().trim().min(1).optional());

const isProductionRuntime = () =>
  [process.env.NODE_ENV, process.env.BUN_ENV]
    .some((value) => String(value ?? "").toLowerCase() === "production");

const WEAK_JWT_SECRET_VALUES = new Set([
  "change_me",
  "changeme",
  "secret",
  "jwt_secret",
  "your-secret-key",
  "please-change-me",
  "replace_with_at_least_32_random_characters",
]);

function assertProductionJwtSecret(secret: string) {
  if (!isProductionRuntime()) return;

  const trimmedSecret = secret.trim();
  const normalized = trimmedSecret.toLowerCase();
  if (
    trimmedSecret.length < 32
    || trimmedSecret !== secret
    || WEAK_JWT_SECRET_VALUES.has(normalized)
  ) {
    throw new Error("JWT_SECRET must be at least 32 non-whitespace characters and must not use a placeholder value in production");
  }
}

const envSchema = z.object({
  DB_HOST: z.string().min(1),
  DB_PORT: dbPortEnv,
  DB_USER: z.string().min(1),
  DB_PASSWORD: z.string(),
  DB_DATABASE: z.string().min(1),
  DB_CONNECT_TIMEOUT_MS: optionalPositiveIntegerEnv,
  DB_QUEUE_LIMIT: dbQueueLimitEnv,
  DB_TIMEZONE: z.preprocess((value) => {
    if (isUnsetEnvValue(value)) return undefined;
    return value;
  }, z.string().trim().min(1).optional()),
  DB_DATE_STRINGS: createBooleanEnv(false),
  DB_SSL_MODE: dbSslModeEnv,
  DB_SSL_CA_PATH: optionalTrimmedStringEnv,
  DB_SSL_REJECT_UNAUTHORIZED: createBooleanEnv(true),
  JWT_SECRET: z.string().min(1),
  CORS_ALLOWED_ORIGINS: optionalTrimmedStringEnv,
  PORT: z.coerce.number().int().positive().default(3001),
});

function loadEnv() {
  const result = envSchema.safeParse(process.env);
  if (!result.success) {
    const missing = result.error.issues.map((i) => i.path.join(".")).join(", ");
    throw new Error(`Missing or invalid environment variables: ${missing}`);
  }
  assertProductionJwtSecret(result.data.JWT_SECRET);
  return result.data;
}

export const env = loadEnv();
