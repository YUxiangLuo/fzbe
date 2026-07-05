import type { Context } from "hono";
import { RATE_LIMITS } from "../../utils/constants";
import {
  applyRateLimitHeaders,
  FixedWindowRateLimiter,
  getClientIp,
  normalizeRateLimitToken,
  type RateLimitCheckResult,
} from "../../utils/rate-limit";

const loginIpLimiter = new FixedWindowRateLimiter({
  limit: RATE_LIMITS.AUTH_LOGIN_MAX_PER_IP,
  windowMs: RATE_LIMITS.AUTH_LOGIN_WINDOW_MS,
});
const loginAccountLimiter = new FixedWindowRateLimiter({
  limit: RATE_LIMITS.AUTH_LOGIN_MAX_PER_ACCOUNT,
  windowMs: RATE_LIMITS.AUTH_LOGIN_WINDOW_MS,
});
const registerIpLimiter = new FixedWindowRateLimiter({
  limit: RATE_LIMITS.AUTH_REGISTER_MAX_PER_IP,
  windowMs: RATE_LIMITS.AUTH_REGISTER_WINDOW_MS,
});
const registerAccountLimiter = new FixedWindowRateLimiter({
  limit: RATE_LIMITS.AUTH_REGISTER_MAX_PER_ACCOUNT,
  windowMs: RATE_LIMITS.AUTH_REGISTER_WINDOW_MS,
});

const buildLoginIpKey = (c: Context) => `login:ip:${getClientIp(c)}`;
const buildLoginAccountKey = (username: string) => `login:account:${normalizeRateLimitToken(username)}`;
const buildRegisterIpKey = (c: Context) => `register:ip:${getClientIp(c)}`;
const isPlainRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const buildRegisterAccountKey = (body: unknown) =>
  `register:account:${normalizeRateLimitToken(
    isPlainRecord(body) && typeof body.username === "string" ? body.username : null,
  )}`;

function buildRateLimitFailure(result: RateLimitCheckResult, message: string) {
  return {
    error: message,
    retry_after_seconds: Math.ceil(result.retryAfterMs / 1000),
  };
}

export function enforceLoginRateLimit(
  c: Context,
  input: { username: string },
): Response | null {
  const ipResult = loginIpLimiter.check(buildLoginIpKey(c));
  applyRateLimitHeaders(c, ipResult);
  if (!ipResult.allowed) {
    return c.json(buildRateLimitFailure(ipResult, "登录尝试过于频繁，请稍后再试"), 429);
  }

  const accountResult = loginAccountLimiter.check(buildLoginAccountKey(input.username));
  applyRateLimitHeaders(c, accountResult);
  if (!accountResult.allowed) {
    return c.json(buildRateLimitFailure(accountResult, "该账号登录尝试过于频繁，请稍后再试"), 429);
  }

  return null;
}

export function enforceSelfRegistrationRateLimit(
  c: Context,
  body: unknown,
): Response | null {
  const ipResult = registerIpLimiter.check(buildRegisterIpKey(c));
  applyRateLimitHeaders(c, ipResult);
  if (!ipResult.allowed) {
    return c.json(buildRateLimitFailure(ipResult, "注册请求过于频繁，请稍后再试"), 429);
  }

  const accountResult = registerAccountLimiter.check(buildRegisterAccountKey(body));
  applyRateLimitHeaders(c, accountResult);
  if (!accountResult.allowed) {
    return c.json(buildRateLimitFailure(accountResult, "该用户名注册尝试过于频繁，请稍后再试"), 429);
  }

  return null;
}

export function resetAuthRateLimitersForTest(): void {
  loginIpLimiter.reset();
  loginAccountLimiter.reset();
  registerIpLimiter.reset();
  registerAccountLimiter.reset();
}
