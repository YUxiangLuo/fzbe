import { describe, expect, test } from "bun:test";
import {
  FixedWindowRateLimiter,
  normalizeRateLimitToken,
} from "../../../src/utils/rate-limit";

describe("rate limit helpers", () => {
  test("blocks requests after the fixed window limit is reached", () => {
    const limiter = new FixedWindowRateLimiter({ limit: 2, windowMs: 1000 });

    expect(limiter.check("Student001", 1000).allowed).toBeTrue();
    expect(limiter.check(" student001 ", 1100).allowed).toBeTrue();

    const blocked = limiter.check("student001", 1200);
    expect(blocked.allowed).toBeFalse();
    expect(blocked.remaining).toBe(0);
    expect(blocked.retryAfterMs).toBe(800);
  });

  test("starts a new bucket after the window expires", () => {
    const limiter = new FixedWindowRateLimiter({ limit: 1, windowMs: 500 });

    expect(limiter.check("ip:127.0.0.1", 1000).allowed).toBeTrue();
    expect(limiter.check("ip:127.0.0.1", 1200).allowed).toBeFalse();
    expect(limiter.check("ip:127.0.0.1", 1500).allowed).toBeTrue();
  });

  test("caps unique buckets within an active window", () => {
    const limiter = new FixedWindowRateLimiter({ limit: 1, windowMs: 1000, maxKeys: 3 });

    expect(limiter.check("key-1", 0).allowed).toBeTrue();
    expect(limiter.check("key-2", 10).allowed).toBeTrue();
    expect(limiter.check("key-3", 20).allowed).toBeTrue();

    const blocked = limiter.check("key-4", 100);

    expect(blocked.allowed).toBeFalse();
    expect(blocked.retryAfterMs).toBe(900);
    expect(limiter.size()).toBe(3);
  });

  test("accepts new unique buckets after capped buckets expire", () => {
    const limiter = new FixedWindowRateLimiter({ limit: 1, windowMs: 1000, maxKeys: 2 });

    expect(limiter.check("key-1", 0).allowed).toBeTrue();
    expect(limiter.check("key-2", 10).allowed).toBeTrue();
    expect(limiter.check("key-3", 100).allowed).toBeFalse();
    expect(limiter.size()).toBe(2);

    expect(limiter.check("key-3", 1010).allowed).toBeTrue();
    expect(limiter.size()).toBe(1);
  });

  test("normalizes keys to avoid casing and whitespace bypasses", () => {
    expect(normalizeRateLimitToken("  Alice@Example.COM  ")).toBe("alice@example.com");
    expect(normalizeRateLimitToken("")).toBe("unknown");
    expect(normalizeRateLimitToken(null)).toBe("unknown");
  });
});
