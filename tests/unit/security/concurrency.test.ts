import { describe, expect, test } from "bun:test";
import { CONCURRENCY_LIMITS } from "../../../src/utils/constants";

describe("concurrency safeguards", () => {
  test("connectionLimit exceeds 2 × MAX_CONCURRENT_MODEL_JOBS to prevent advisory-lock deadlock", async () => {
    // 每个模型任务持有 2 个连接（执行锁 + 槽位锁）。
    // 若 connectionLimit ≤ 2 × MAX_CONCURRENT_MODEL_JOBS，
    // 并发训练中的 pool.query() 会等待连接释放，而连接被锁占用，形成死锁。
    process.env.DB_HOST ??= "127.0.0.1";
    process.env.DB_USER ??= "test";
    process.env.DB_PASSWORD ??= "test";
    process.env.DB_DATABASE ??= "test";
    process.env.JWT_SECRET ??= "test-jwt-secret";
    process.env.PORT ??= "3001";
    process.env.DEFAULT_PASSWORD_HASH ??= "test-default-hash";

    // Dynamic import to avoid env issues
    const { pool } = await import("../../../src/config/database");

    // mysql2 exposes pool config on pool.pool.config
    const poolInternal = pool.pool as { config?: { connectionLimit?: number } };
    const connLimit = poolInternal.config?.connectionLimit ?? 0;
    const minRequired = 2 * CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS;

    expect(connLimit).toBeGreaterThan(minRequired);
  });

  test("MAX_CONCURRENT_MODEL_JOBS is a positive integer", () => {
    expect(CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS).toBeGreaterThan(0);
    expect(Number.isInteger(CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS)).toBe(true);
  });
});
