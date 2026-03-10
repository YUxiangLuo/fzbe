import { describe, expect, test } from "bun:test";
import { CONCURRENCY_LIMITS, DB_POOL_CONFIG } from "../../../src/utils/constants";

describe("concurrency safeguards", () => {
  test("CONNECTION_LIMIT exceeds 2 × MAX_CONCURRENT_MODEL_JOBS to prevent advisory-lock deadlock", () => {
    // 每个模型任务持有 2 个连接（执行锁 + 槽位锁）。
    // 若 connectionLimit ≤ 2 × MAX_CONCURRENT_MODEL_JOBS，
    // 并发训练中的 pool.query() 会等待连接释放，而连接被锁占用，形成死锁。
    const minRequired = 2 * CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS;

    expect(DB_POOL_CONFIG.CONNECTION_LIMIT).toBeGreaterThan(minRequired);
  });

  test("MAX_CONCURRENT_MODEL_JOBS is a positive integer", () => {
    expect(CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS).toBeGreaterThan(0);
    expect(Number.isInteger(CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS)).toBe(true);
  });

  test("MAX_CONCURRENT_PDF_JOBS is a positive integer", () => {
    expect(CONCURRENCY_LIMITS.MAX_CONCURRENT_PDF_JOBS).toBeGreaterThan(0);
    expect(Number.isInteger(CONCURRENCY_LIMITS.MAX_CONCURRENT_PDF_JOBS)).toBe(true);
  });
});
