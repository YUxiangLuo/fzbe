import { z } from "zod";
import {
  createBooleanEnv,
  databaseEnvironmentSchema,
  optionalTrimmedStringEnv,
  type RuntimeDatabaseEnvSource,
} from "./database-connection-options";

type RuntimeEnvSource = RuntimeDatabaseEnvSource;

const isProductionRuntime = (source: RuntimeEnvSource) =>
  [source.NODE_ENV, source.BUN_ENV]
    .some((value) => String(value ?? "").toLowerCase() === "production");

const REMOVED_SINGLE_INSTANCE_ENV_NAMES = [
  "MODEL_LOCK_CONNECTION_ACQUIRE_TIMEOUT_MS",
  "MODEL_LOCK_QUERY_TIMEOUT_MS",
  "DB_QUERY_HEADROOM",
] as const;

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

const envSchema = databaseEnvironmentSchema.extend({
  JWT_SECRET: z.string().min(1),
  ALLOW_LEGACY_PRODUCTION_PLAN_WRITES: createBooleanEnv(false),
  E2E_ALLOW_DB_RESET: createBooleanEnv(false),
  CORS_ALLOWED_ORIGINS: optionalTrimmedStringEnv,
  PORT: z.coerce.number().int().positive().default(3001),
});

export function loadEnv(source: RuntimeEnvSource = process.env) {
  const result = envSchema.safeParse(source);
  if (!result.success) {
    const missing = result.error.issues.map((i) => i.path.join(".")).join(", ");
    throw new Error(`Missing or invalid environment variables: ${missing}`);
  }

  if (isProductionRuntime(source)) {
    assertProductionJwtSecret(result.data.JWT_SECRET);
    if (source.E2E_SERVER?.trim() === "1") {
      throw new Error("E2E_SERVER must not be enabled in production");
    }
    const removedConfiguration = REMOVED_SINGLE_INSTANCE_ENV_NAMES.filter(
      (name) => source[name]?.trim(),
    );
    if (removedConfiguration.length > 0) {
      throw new Error(
        `Removed advisory-lock-era configuration must be deleted: ${removedConfiguration.join(", ")}`,
      );
    }
    const productionHost = source.HOST?.trim() || "127.0.0.1";
    if (result.data.PORT !== 3001 || productionHost !== "127.0.0.1") {
      throw new Error(
        "Production single-instance binding must be exactly 127.0.0.1:3001",
      );
    }
  }
  return result.data;
}

export const env = loadEnv();
