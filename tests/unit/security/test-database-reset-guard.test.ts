import { describe, expect, test } from "bun:test";
import { assertE2eDatabaseResetAllowed } from "../../../scripts/test-database-reset-guard";

// Mirrors DISPOSABLE_DATABASE_CONTRACT in the frontend repository's
// tests/e2e/helpers/database-safety.ts. The frontend's tests/e2e/run.ts asserts
// both guards agree on this table before every E2E run, so any change here must
// be made on both sides.
const ALLOWED_DATABASES = [
  "fangzhen_e2e",
  "fangzhen_e2e_quiz",
  "fangzhen_e2e_pr_42",
  "fz_e2e_local",
  "fz_pr_42",
  "fz_test_browser",
  "fz_student_dogfood_20260730",
];

const REJECTED_DATABASES = [
  "",
  "fangzhen",
  "fangzhen001",
  "fangzhen_dev",
  "production",
  "mysql",
  "fz_development",
  "fangzhen_e2e_",
  "fangzhen_e2ex",
  "fz_e2e",
  "fz_e2e_",
  "fz_e2e__",
  "fz_pr",
  "fz_pr_",
  "fz_test",
  "fz_test_",
  "fz_student_dogfood",
  "fz_student_dogfood_",
  "fz_pr_candidate-with-dash",
];

describe("destructive system-test database guard", () => {
  test.each(ALLOWED_DATABASES)(
    "allows an explicitly approved disposable database: %s",
    (database) => {
      expect(() => assertE2eDatabaseResetAllowed({
        E2E_ALLOW_DB_RESET: "1",
        DB_DATABASE: database,
        DB_HOST: "127.0.0.1",
      })).not.toThrow();
    },
  );

  test("rejects missing approval even when the database name is disposable", () => {
    expect(() => assertE2eDatabaseResetAllowed({
      DB_DATABASE: "fz_pr_12345",
      DB_HOST: "127.0.0.1",
    })).toThrow("E2E_ALLOW_DB_RESET=1");
  });

  test.each(REJECTED_DATABASES)(
    "rejects a development, production, system, or malformed name: %s",
    (database) => {
      expect(() => assertE2eDatabaseResetAllowed({
        E2E_ALLOW_DB_RESET: "1",
        DB_DATABASE: database,
        DB_HOST: "127.0.0.1",
      })).toThrow("Refusing to reset");
    },
  );

  test.each(["localhost", "127.0.0.1", "::1", " LOCALHOST "])(
    "allows a local database host: %s",
    (host) => {
      expect(() => assertE2eDatabaseResetAllowed({
        E2E_ALLOW_DB_RESET: "1",
        DB_DATABASE: "fz_e2e_local",
        DB_HOST: host,
      })).not.toThrow();
    },
  );

  test("rejects a missing or remote host without an explicit override", () => {
    expect(() => assertE2eDatabaseResetAllowed({
      E2E_ALLOW_DB_RESET: "1",
      DB_DATABASE: "fz_e2e_local",
    })).toThrow("DB_HOST is required");

    expect(() => assertE2eDatabaseResetAllowed({
      E2E_ALLOW_DB_RESET: "1",
      DB_DATABASE: "fz_e2e_local",
      DB_HOST: "db.internal",
    })).toThrow(/remote host "db\.internal".*DB_RESET_ALLOW_REMOTE=1/);
  });

  test.each(["DB_RESET_ALLOW_REMOTE", "DB_SETUP_ALLOW_REMOTE"] as const)(
    "allows a remote host with the %s override",
    (override) => {
      expect(() => assertE2eDatabaseResetAllowed({
        E2E_ALLOW_DB_RESET: "1",
        DB_DATABASE: "fz_e2e_remote",
        DB_HOST: "db.internal",
        [override]: "1",
      })).not.toThrow();
    },
  );

  test.each(["NODE_ENV", "BUN_ENV"] as const)(
    "rejects production declared through %s without an explicit override",
    (runtimeVariable) => {
      expect(() => assertE2eDatabaseResetAllowed({
        E2E_ALLOW_DB_RESET: "1",
        DB_DATABASE: "fz_e2e_production",
        DB_HOST: "127.0.0.1",
        [runtimeVariable]: "PrOdUcTiOn",
      })).toThrow(/production.*DB_RESET_ALLOW_PROD=1/);
    },
  );

  test.each(["DB_RESET_ALLOW_PROD", "DB_SETUP_ALLOW_PROD"] as const)(
    "allows production with the %s override",
    (override) => {
      expect(() => assertE2eDatabaseResetAllowed({
        E2E_ALLOW_DB_RESET: "1",
        DB_DATABASE: "fz_e2e_production",
        DB_HOST: "127.0.0.1",
        NODE_ENV: "production",
        [override]: "1",
      })).not.toThrow();
    },
  );

  test("allows BUN_ENV production with the explicit production override", () => {
    expect(() => assertE2eDatabaseResetAllowed({
      E2E_ALLOW_DB_RESET: "1",
      DB_DATABASE: "fz_e2e_production",
      DB_HOST: "127.0.0.1",
      BUN_ENV: "production",
      DB_RESET_ALLOW_PROD: "1",
    })).not.toThrow();
  });

  test("requires both overrides for a remote production target", () => {
    expect(() => assertE2eDatabaseResetAllowed({
      E2E_ALLOW_DB_RESET: "1",
      DB_DATABASE: "fz_e2e_remote_production",
      DB_HOST: "db.internal",
      NODE_ENV: "production",
      DB_RESET_ALLOW_REMOTE: "1",
    })).toThrow(/production/);

    expect(() => assertE2eDatabaseResetAllowed({
      E2E_ALLOW_DB_RESET: "1",
      DB_DATABASE: "fz_e2e_remote_production",
      DB_HOST: "db.internal",
      NODE_ENV: "production",
      DB_RESET_ALLOW_REMOTE: "1",
      DB_RESET_ALLOW_PROD: "1",
    })).not.toThrow();
  });
});
