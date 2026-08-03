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
  test("defaults the rolling-upgrade compatibility window off", () => {
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

  test("requires only a strong JWT secret in production", () => {
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
});
