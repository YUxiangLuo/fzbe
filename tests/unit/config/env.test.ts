import { describe, expect, test } from "bun:test";
import { loadEnv } from "../../../src/config/env";

const baseEnv = {
  NODE_ENV: "test",
  DB_HOST: "127.0.0.1",
  DB_USER: "test",
  DB_PASSWORD: "",
  DB_DATABASE: "test",
  JWT_SECRET: "test-jwt-secret",
};

describe("runtime environment", () => {
  test("defaults the old-page compatibility window off", () => {
    const parsed = loadEnv(baseEnv);

    expect(parsed.ALLOW_LEGACY_PRODUCTION_PLAN_WRITES).toBe(false);
  });

  test("does not require separate evidence credentials", () => {
    const { NODE_ENV: _testRuntime, ...runtimeEnv } = baseEnv;

    expect(loadEnv(runtimeEnv)).toMatchObject({
      JWT_SECRET: "test-jwt-secret",
      ALLOW_LEGACY_PRODUCTION_PLAN_WRITES: false,
    });
  });

  test("parses an explicitly enabled compatibility window", () => {
    const parsed = loadEnv({
      ...baseEnv,
      ALLOW_LEGACY_PRODUCTION_PLAN_WRITES: "true",
    });

    expect(parsed.ALLOW_LEGACY_PRODUCTION_PLAN_WRITES).toBe(true);
  });

  test("requires a strong JWT secret in production", () => {
    const productionEnv = {
      ...baseEnv,
      NODE_ENV: "production",
    };

    expect(() => loadEnv(productionEnv))
      .toThrow("JWT_SECRET");

    expect(loadEnv({
      ...productionEnv,
      JWT_SECRET: "production-jwt-secret-at-least-32-characters",
    }).JWT_SECRET).toBe("production-jwt-secret-at-least-32-characters");
  });

  test("pins production to the one nginx-facing backend listener", () => {
    const productionEnv = {
      ...baseEnv,
      NODE_ENV: "production",
      JWT_SECRET: "production-jwt-secret-at-least-32-characters",
    };

    expect(() => loadEnv({ ...productionEnv, PORT: "3002" }))
      .toThrow("127.0.0.1:3001");
    expect(() => loadEnv({ ...productionEnv, HOST: "0.0.0.0" }))
      .toThrow("127.0.0.1:3001");
    expect(loadEnv({ ...productionEnv, PORT: "3001", HOST: "127.0.0.1" }).PORT)
      .toBe(3001);
  });

  test("rejects removed advisory-lock-era settings in production", () => {
    expect(() => loadEnv({
      ...baseEnv,
      NODE_ENV: "production",
      JWT_SECRET: "production-jwt-secret-at-least-32-characters",
      DB_QUERY_HEADROOM: "40",
      MODEL_LOCK_QUERY_TIMEOUT_MS: "2000",
    })).toThrow("MODEL_LOCK_QUERY_TIMEOUT_MS, DB_QUERY_HEADROOM");
  });

  test("rejects E2E-only routes in production", () => {
    expect(() => loadEnv({
      ...baseEnv,
      NODE_ENV: "production",
      JWT_SECRET: "production-jwt-secret-at-least-32-characters",
      E2E_SERVER: "1",
    })).toThrow("E2E_SERVER must not be enabled in production");
  });
});
