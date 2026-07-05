import type { Context } from "hono";

interface RateLimitBucket {
  count: number;
  resetAt: number;
}

export interface FixedWindowRateLimiterOptions {
  limit: number;
  windowMs: number;
  maxKeys?: number;
}

export interface RateLimitCheckResult {
  allowed: boolean;
  limit: number;
  remaining: number;
  resetAt: number;
  retryAfterMs: number;
}

export class FixedWindowRateLimiter {
  private readonly buckets = new Map<string, RateLimitBucket>();
  private readonly limit: number;
  private readonly windowMs: number;
  private readonly maxKeys: number;
  private operationsSincePrune = 0;

  constructor(options: FixedWindowRateLimiterOptions) {
    if (!Number.isInteger(options.limit) || options.limit < 1) {
      throw new Error("rate limit must be a positive integer");
    }
    if (!Number.isInteger(options.windowMs) || options.windowMs < 1) {
      throw new Error("rate limit window must be a positive integer");
    }
    if (
      options.maxKeys !== undefined
      && (!Number.isInteger(options.maxKeys) || options.maxKeys < 1)
    ) {
      throw new Error("rate limit maxKeys must be a positive integer");
    }

    this.limit = options.limit;
    this.windowMs = options.windowMs;
    this.maxKeys = options.maxKeys ?? 20_000;
  }

  check(rawKey: string, now = Date.now()): RateLimitCheckResult {
    const key = normalizeRateLimitToken(rawKey);
    const existing = this.buckets.get(key);

    if (!existing || existing.resetAt <= now) {
      if (!existing) {
        const capacityResult = this.checkCapacityForNewKey(now);
        if (capacityResult) {
          return capacityResult;
        }
      }

      const resetAt = now + this.windowMs;
      this.buckets.set(key, { count: 1, resetAt });
      this.pruneIfNeeded(now);
      return {
        allowed: true,
        limit: this.limit,
        remaining: this.limit - 1,
        resetAt,
        retryAfterMs: 0,
      };
    }

    if (existing.count >= this.limit) {
      return {
        allowed: false,
        limit: this.limit,
        remaining: 0,
        resetAt: existing.resetAt,
        retryAfterMs: Math.max(1, existing.resetAt - now),
      };
    }

    existing.count += 1;
    return {
      allowed: true,
      limit: this.limit,
      remaining: this.limit - existing.count,
      resetAt: existing.resetAt,
      retryAfterMs: 0,
    };
  }

  reset(): void {
    this.buckets.clear();
    this.operationsSincePrune = 0;
  }

  size(): number {
    return this.buckets.size;
  }

  private pruneIfNeeded(now: number): void {
    this.operationsSincePrune += 1;
    if (this.buckets.size <= this.maxKeys && this.operationsSincePrune < 1000) {
      return;
    }

    this.operationsSincePrune = 0;
    this.pruneExpired(now);
  }

  private checkCapacityForNewKey(now: number): RateLimitCheckResult | null {
    if (this.buckets.size < this.maxKeys) {
      return null;
    }

    this.operationsSincePrune = 0;
    this.pruneExpired(now);
    if (this.buckets.size < this.maxKeys) {
      return null;
    }

    const resetAt = this.getEarliestResetAt(now + this.windowMs);
    return {
      allowed: false,
      limit: this.limit,
      remaining: 0,
      resetAt,
      retryAfterMs: Math.max(1, resetAt - now),
    };
  }

  private pruneExpired(now: number): void {
    for (const [key, bucket] of this.buckets) {
      if (bucket.resetAt <= now) {
        this.buckets.delete(key);
      }
    }
  }

  private getEarliestResetAt(fallback: number): number {
    let earliest = Number.POSITIVE_INFINITY;
    for (const bucket of this.buckets.values()) {
      earliest = Math.min(earliest, bucket.resetAt);
    }

    return Number.isFinite(earliest) ? earliest : fallback;
  }
}

export function normalizeRateLimitToken(value: string | null | undefined): string {
  const normalized = (value ?? "unknown").trim().toLowerCase();
  return (normalized || "unknown").slice(0, 256);
}

export function getClientIp(c: Context): string {
  const forwardedFor = c.req.header("x-forwarded-for")
    ?.split(",")
    .map((part) => part.trim())
    .find((part) => part.length > 0);

  return normalizeRateLimitToken(
    forwardedFor
      ?? c.req.header("cf-connecting-ip")
      ?? c.req.header("x-real-ip")
      ?? "unknown",
  );
}

export function applyRateLimitHeaders(c: Context, result: RateLimitCheckResult): void {
  c.header("X-RateLimit-Limit", String(result.limit));
  c.header("X-RateLimit-Remaining", String(result.remaining));
  c.header("X-RateLimit-Reset", String(Math.ceil(result.resetAt / 1000)));

  if (!result.allowed) {
    c.header("Retry-After", String(Math.ceil(result.retryAfterMs / 1000)));
  }
}
