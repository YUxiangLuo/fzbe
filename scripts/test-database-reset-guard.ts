/**
 * Must stay identical to SAFE_E2E_DATABASE in the frontend repository's
 * `tests/e2e/helpers/database-safety.ts`. The two repositories cannot import
 * from each other, so the frontend's `tests/e2e/run.ts` asserts both guards
 * agree on a shared name table before every E2E run.
 */
const ALLOWED_E2E_DATABASE =
  /^(?:fangzhen_e2e(?:_[a-z0-9][a-z0-9_]*)?|fz_(?:e2e|pr|test|student_dogfood)_[a-z0-9][a-z0-9_]*)$/i;

const LOCAL_DATABASE_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);

export const ALLOWED_E2E_DATABASE_NAMES = "fangzhen_e2e, fangzhen_e2e_*, fz_e2e_*, fz_pr_*, fz_test_*, fz_student_dogfood_*";

function hasExplicitOverride(...values: Array<string | undefined>): boolean {
  return values.some((value) => value === "1");
}

function isProductionRuntime(
  env: Readonly<Record<string, string | undefined>>,
): boolean {
  return [env.NODE_ENV, env.BUN_ENV]
    .some((value) => String(value ?? "").toLowerCase() === "production");
}

export function assertE2eDatabaseResetAllowed(
  env: Readonly<Record<string, string | undefined>> = process.env,
) {
  if (env.E2E_ALLOW_DB_RESET !== "1") {
    throw new Error(
      "The API system test resets its database. Set E2E_ALLOW_DB_RESET=1 only for a disposable test database.",
    );
  }

  const database = env.DB_DATABASE?.trim() ?? "";
  if (!ALLOWED_E2E_DATABASE.test(database)) {
    throw new Error(
      `Refusing to reset DB_DATABASE="${database || "(unset)"}". `
      + `Allowed names: ${ALLOWED_E2E_DATABASE_NAMES}.`,
    );
  }

  if (
    isProductionRuntime(env)
    && !hasExplicitOverride(env.DB_RESET_ALLOW_PROD, env.DB_SETUP_ALLOW_PROD)
  ) {
    throw new Error(
      "Refusing to run E2E database mutations in production. Set DB_RESET_ALLOW_PROD=1 to override.",
    );
  }

  const databaseHost = env.DB_HOST?.trim() ?? "";
  if (!databaseHost) {
    throw new Error("DB_HOST is required for E2E database mutations.");
  }
  if (
    !LOCAL_DATABASE_HOSTS.has(databaseHost.toLowerCase())
    && !hasExplicitOverride(env.DB_RESET_ALLOW_REMOTE, env.DB_SETUP_ALLOW_REMOTE)
  ) {
    throw new Error(
      `Refusing to run E2E database mutations against remote host "${databaseHost}". `
      + "Set DB_RESET_ALLOW_REMOTE=1 to override.",
    );
  }
}
