import { describe, expect, test } from "bun:test";
import {
  ensureE2eDatabase,
  ensureE2eDatabaseExists,
} from "../../../scripts/ensure-e2e-database";

const ALLOWED_ENV = {
  E2E_ALLOW_DB_RESET: "1",
  DB_DATABASE: "fz_e2e_bootstrap",
  DB_HOST: "127.0.0.1",
} as const;

function trackingDeps(exists: boolean) {
  const queried: string[] = [];
  const created: string[] = [];
  return {
    queried,
    created,
    deps: {
      databaseExists: async (database: string) => {
        queried.push(database);
        return exists;
      },
      createDatabase: async (database: string) => {
        created.push(database);
      },
    },
  };
}

describe("E2E database bootstrap", () => {
  test("allows an explicitly opted-in existing disposable database", async () => {
    const { queried, created, deps } = trackingDeps(true);

    expect(await ensureE2eDatabase("fz_e2e_bootstrap", deps, ALLOWED_ENV)).toBe("exists");
    expect(queried).toEqual(["fz_e2e_bootstrap"]);
    expect(created).toEqual([]);
  });

  test("refuses an existing non-disposable database before querying it", async () => {
    const { queried, created, deps } = trackingDeps(true);

    await expect(
      ensureE2eDatabase("fangzhen", deps, {
        E2E_ALLOW_DB_RESET: "1",
        DB_DATABASE: "fangzhen",
        DB_HOST: "127.0.0.1",
      }),
    ).rejects.toThrow(
      /Refusing to start the E2E backend with DB_DATABASE="fangzhen".*Refusing to reset DB_DATABASE="fangzhen"/,
    );
    expect(queried).toEqual([]);
    expect(created).toEqual([]);
  });

  test("refuses an existing disposable database without explicit opt-in before querying it", async () => {
    const { queried, created, deps } = trackingDeps(true);

    await expect(
      ensureE2eDatabase("fz_e2e_bootstrap", deps, {
        DB_DATABASE: "fz_e2e_bootstrap",
        DB_HOST: "127.0.0.1",
      }),
    ).rejects.toThrow(
      /Refusing to start the E2E backend.*Set E2E_ALLOW_DB_RESET=1/,
    );
    expect(queried).toEqual([]);
    expect(created).toEqual([]);
  });

  test("creates an explicitly opted-in missing disposable database", async () => {
    const { queried, created, deps } = trackingDeps(false);

    expect(await ensureE2eDatabase("fz_e2e_bootstrap", deps, ALLOWED_ENV)).toBe("created");
    expect(queried).toEqual(["fz_e2e_bootstrap"]);
    expect(created).toEqual(["fz_e2e_bootstrap"]);
  });

  test("refuses a missing disposable database without explicit opt-in before querying it", async () => {
    const { queried, created, deps } = trackingDeps(false);

    await expect(
      ensureE2eDatabase("fz_e2e_bootstrap", deps, {
        DB_DATABASE: "fz_e2e_bootstrap",
        DB_HOST: "127.0.0.1",
      }),
    ).rejects.toThrow(/Refusing to start the E2E backend.*Set E2E_ALLOW_DB_RESET=1/);
    expect(queried).toEqual([]);
    expect(created).toEqual([]);
  });

  test("refuses to create a database whose name is not disposable", async () => {
    const { queried, created, deps } = trackingDeps(false);

    await expect(
      ensureE2eDatabase("fangzhen", deps, {
        E2E_ALLOW_DB_RESET: "1",
        DB_DATABASE: "fangzhen",
        DB_HOST: "127.0.0.1",
      }),
    ).rejects.toThrow(/Refusing to reset DB_DATABASE="fangzhen"/);
    expect(queried).toEqual([]);
    expect(created).toEqual([]);
  });

  test("validates the actual database argument instead of a different safe env name", async () => {
    const { queried, created, deps } = trackingDeps(false);

    await expect(
      ensureE2eDatabase("fangzhen", deps, ALLOWED_ENV),
    ).rejects.toThrow(/Refusing to reset DB_DATABASE="fangzhen"/);
    expect(queried).toEqual([]);
    expect(created).toEqual([]);
  });

  test("rejects an empty database name", async () => {
    const { queried, created, deps } = trackingDeps(false);

    await expect(ensureE2eDatabase("  ", deps, ALLOWED_ENV)).rejects.toThrow(
      /DB_DATABASE is required/,
    );
    expect(queried).toEqual([]);
    expect(created).toEqual([]);
  });

  test("rejects surrounding whitespace instead of approving a trimmed but different name", async () => {
    const { queried, created, deps } = trackingDeps(false);

    await expect(
      ensureE2eDatabase(" fz_e2e_bootstrap ", deps, ALLOWED_ENV),
    ).rejects.toThrow(/must not contain leading or trailing whitespace/);
    expect(queried).toEqual([]);
    expect(created).toEqual([]);
  });

  test.each([
    {
      name: "remote host",
      env: { ...ALLOWED_ENV, DB_HOST: "db.internal" },
      message: /remote host/,
    },
    {
      name: "production runtime",
      env: { ...ALLOWED_ENV, NODE_ENV: "production" },
      message: /production/,
    },
    {
      name: "Bun production runtime",
      env: { ...ALLOWED_ENV, BUN_ENV: "production" },
      message: /production/,
    },
  ])("refuses a $name before querying or creating", async ({ env, message }) => {
    const { queried, created, deps } = trackingDeps(false);

    await expect(
      ensureE2eDatabase("fz_e2e_bootstrap", deps, env),
    ).rejects.toThrow(message);
    expect(queried).toEqual([]);
    expect(created).toEqual([]);
  });

  test.each([
    {
      name: "remote host",
      env: {
        ...ALLOWED_ENV,
        DB_HOST: "db.internal",
        DB_USER: "agent",
        DB_PASSWORD: "secret",
      },
      message: /remote host/,
    },
    {
      name: "production runtime",
      env: {
        ...ALLOWED_ENV,
        DB_USER: "agent",
        DB_PASSWORD: "secret",
        NODE_ENV: "production",
      },
      message: /production/,
    },
    {
      name: "Bun production runtime",
      env: {
        ...ALLOWED_ENV,
        DB_USER: "agent",
        DB_PASSWORD: "secret",
        BUN_ENV: "production",
      },
      message: /production/,
    },
  ])("refuses a $name before opening a server connection", async ({ env, message }) => {
    let connectionAttempts = 0;

    await expect(
      ensureE2eDatabaseExists(env, async () => {
        connectionAttempts += 1;
        throw new Error("connection factory must not run");
      }),
    ).rejects.toThrow(message);
    expect(connectionAttempts).toBe(0);
  });
});
