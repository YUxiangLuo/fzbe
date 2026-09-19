import { beforeEach, describe, expect, test } from "bun:test";
import type { Context } from "hono";
import {
  enforceLoginRateLimit,
  enforceSelfRegistrationRateLimit,
  resetAuthRateLimitersForTest,
} from "../../../src/modules/auth/auth.rate-limit";
import { RATE_LIMITS } from "../../../src/utils/constants";
import { getClientIp } from "../../../src/utils/rate-limit";

interface FakeContext extends Context {
  capturedHeaders: Headers;
}

function createContext(ip: string, realIp?: string): FakeContext {
  const requestHeaders = new Headers({ "x-forwarded-for": ip });
  if (realIp) requestHeaders.set("x-real-ip", realIp);
  const capturedHeaders = new Headers();

  return {
    capturedHeaders,
    req: {
      header: (name: string) => requestHeaders.get(name) ?? undefined,
    },
    header: (name: string, value: string) => {
      capturedHeaders.set(name, value);
    },
    json: (body: unknown, status?: number) =>
      new Response(JSON.stringify(body), {
        status: status ?? 200,
        headers: { "content-type": "application/json" },
      }),
  } as unknown as FakeContext;
}

async function expectRateLimitResponse(response: Response | null, expectedError: string) {
  expect(response).toBeInstanceOf(Response);
  expect(response?.status).toBe(429);
  await expect(response?.json()).resolves.toEqual({
    error: expectedError,
    retry_after_seconds: expect.any(Number),
  });
}

describe("auth rate limit guards", () => {
  beforeEach(() => {
    resetAuthRateLimitersForTest();
  });

  test("uses nginx-authenticated X-Real-IP instead of spoofable forwarding prefixes", () => {
    expect(getClientIp(createContext("192.0.2.9, 198.51.100.10", "203.0.113.7")))
      .toBe("203.0.113.7");
    expect(getClientIp(createContext("192.0.2.9, 198.51.100.10")))
      .toBe("198.51.100.10");
  });

  test("allows 50 classroom accounts ten attempts each behind one NAT", async () => {
    expect(RATE_LIMITS.AUTH_LOGIN_MAX_PER_IP).toBe(500);
    for (let student = 1; student <= 50; student += 1) {
      for (let attempt = 0; attempt < 10; attempt += 1) {
        expect(enforceLoginRateLimit(
          createContext("203.0.113.50"),
          { username: `student-${student}` },
        )).toBeNull();
      }
    }

    const blockedContext = createContext("203.0.113.50");
    const response = enforceLoginRateLimit(blockedContext, { username: "student-51" });
    await expectRateLimitResponse(response, "登录尝试过于频繁，请稍后再试");
  });

  test("blocks repeated login attempts for the same account before service execution", async () => {
    for (let i = 0; i < RATE_LIMITS.AUTH_LOGIN_MAX_PER_ACCOUNT; i += 1) {
      const response = enforceLoginRateLimit(
        createContext(`198.51.100.${i + 1}`),
        { username: " Student001 " },
      );
      expect(response).toBeNull();
    }

    const blockedContext = createContext("198.51.100.200");
    const response = enforceLoginRateLimit(blockedContext, { username: "student001" });

    await expectRateLimitResponse(response, "该账号登录尝试过于频繁，请稍后再试");
    expect(blockedContext.capturedHeaders.get("retry-after")).not.toBeNull();
  });

  test("blocks repeated self-registration attempts for the same username", async () => {
    for (let i = 0; i < RATE_LIMITS.AUTH_REGISTER_MAX_PER_ACCOUNT; i += 1) {
      const response = enforceSelfRegistrationRateLimit(
        createContext(`203.0.113.${i + 1}`),
        { username: " Alice " },
      );
      expect(response).toBeNull();
    }

    const blockedContext = createContext("203.0.113.200");
    const response = enforceSelfRegistrationRateLimit(blockedContext, { username: "alice" });

    await expectRateLimitResponse(response, "该用户名注册尝试过于频繁，请稍后再试");
    expect(blockedContext.capturedHeaders.get("retry-after")).not.toBeNull();
  });

  test("handles non-object self-registration JSON bodies without throwing", () => {
    expect(() => {
      const response = enforceSelfRegistrationRateLimit(createContext("203.0.113.50"), null);
      expect(response).toBeNull();
    }).not.toThrow();
  });
});
