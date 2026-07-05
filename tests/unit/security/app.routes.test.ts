import { beforeEach, describe, expect, mock, test } from "bun:test";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { sign } from "hono/jwt";

type AppLike = {
  fetch: (request: Request) => Promise<Response>;
};

const loginMock = mock(async () => ({
  success: true as const,
  token: "test-token",
  user: { role: "Student" },
}));

const poolQueryMock = mock(async (sql?: string) => {
  if (String(sql).includes("FROM experiment_manuals")) {
    return [[{ manual_id: 1, is_active: 1 }]];
  }

  return [[]];
});

mock.module("../../../src/config", () => ({
  env: {
    DB_HOST: "127.0.0.1",
    DB_USER: "test",
    DB_PASSWORD: "test",
    DB_DATABASE: "test",
    JWT_SECRET: "test-jwt-secret",
    PORT: 3001,
  },
  connectionLimit: 64,
  pool: {
    query: poolQueryMock,
  },
}));

mock.module("../../../src/modules/auth/auth.service", () => ({
  login: loginMock,
}));

const ensureTestEnv = () => {
  process.env.DB_HOST ??= "127.0.0.1";
  process.env.DB_USER ??= "test";
  process.env.DB_PASSWORD ??= "test";
  process.env.DB_DATABASE ??= "test";
  process.env.JWT_SECRET = "test-jwt-secret";
  process.env.PORT ??= "3001";
};

let appPromise: Promise<AppLike> | null = null;

const getApp = async (): Promise<AppLike> => {
  if (!appPromise) {
    ensureTestEnv();
    appPromise = import("../../../src/index").then((module) => module.app as AppLike);
  }

  return appPromise;
};

describe("app route smoke tests", () => {
  beforeEach(() => {
    loginMock.mockClear();
    poolQueryMock.mockClear();
  });

  test("wraps successful JSON responses from mounted controllers in a data envelope", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/api/v1/sessions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username: "student001", password: "secret123" }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      data: {
        token: "test-token",
        user: { role: "Student" },
      },
    });
    expect(loginMock).toHaveBeenCalledTimes(1);
  });

  test("preserves non-2xx JSON errors without wrapping them in a data envelope", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/api/v1/sessions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username: "student001" }),
      }),
    );

    expect(response.status).toBe(400);
    const payload = await response.json();
    expect(payload).toHaveProperty("error");
    expect(payload).not.toHaveProperty("data");
  });

  test("keeps null self-registration JSON as a validation error", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/api/v1/users/register/Student", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: "null",
      }),
    );

    expect(response.status).toBe(400);
    const payload = await response.json();
    expect(payload).toHaveProperty("error", "验证失败");
    expect(payload).not.toHaveProperty("data");
  });

  test("returns the application-level 404 payload for unknown routes", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/does-not-exist"),
    );

    expect(response.status).toBe(404);
    await expect(response.json()).resolves.toEqual({
      error: "接口不存在",
    });
  });

  test("keeps auth guard behavior when requests enter through the real app", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/api/v1/manuals/active"),
    );

    expect(response.status).toBe(401);
    const payload = await response.json();
    expect(payload).toHaveProperty("error");
    expect(payload).not.toHaveProperty("data");
  });

  test("runtime info exposes PDF queue observability", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/api/v1/runtime-info"),
    );

    expect(response.status).toBe(200);
    const payload = await response.json() as {
      data?: {
        concurrency?: { maxConcurrentPdfJobs?: unknown };
        timeouts?: { pdfQueueMs?: unknown; pdfGenerationMs?: unknown };
        pdfQueue?: Record<string, unknown>;
      };
    };

    expect(payload.data?.pdfQueue).toEqual({
      activeJobs: 0,
      pendingJobs: 0,
      maxConcurrentJobs: payload.data?.concurrency?.maxConcurrentPdfJobs,
      queueTimeoutMs: payload.data?.timeouts?.pdfQueueMs,
      generationTimeoutMs: payload.data?.timeouts?.pdfGenerationMs,
    });
  });

  test("forbids assistants without assigned classes from reading the question bank", async () => {
    const app = await getApp();
    const token = await sign(
      {
        sub: 201,
        role: "Assistant",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );

    const response = await app.fetch(
      new Request("http://localhost/api/v1/question-bank/questions", {
        headers: { Authorization: `Bearer ${token}` },
      }),
    );

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({ error: "权限不足" });
    expect(poolQueryMock).toHaveBeenCalledWith(
      "SELECT 1 FROM class_assistants WHERE assistant_id = ? LIMIT 1",
      [201],
    );
    expect(poolQueryMock.mock.calls.some(([sql]) => String(sql).includes("FROM question_bank"))).toBe(false);
  });

  test("authenticated file responses include length metadata and support byte ranges", async () => {
    const app = await getApp();
    const uploadsDir = resolve(import.meta.dir, "../../../uploads/manuals");
    const filename = "unit-range-fixture.pdf";
    const filePath = resolve(uploadsDir, filename);
    const token = await sign(
      {
        sub: 123,
        role: "Student",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );

    await mkdir(uploadsDir, { recursive: true });
    await writeFile(filePath, "0123456789");

    try {
      const fullResponse = await app.fetch(
        new Request(`http://localhost/manuals/${filename}`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
      );

      expect(fullResponse.status).toBe(200);
      expect(fullResponse.headers.get("content-length")).toBe("10");
      expect(fullResponse.headers.get("accept-ranges")).toBe("bytes");
      expect(await fullResponse.text()).toBe("0123456789");

      const rangeResponse = await app.fetch(
        new Request(`http://localhost/manuals/${filename}`, {
          headers: {
            Authorization: `Bearer ${token}`,
            Range: "bytes=2-5",
          },
        }),
      );

      expect(rangeResponse.status).toBe(206);
      expect(rangeResponse.headers.get("content-length")).toBe("4");
      expect(rangeResponse.headers.get("content-range")).toBe("bytes 2-5/10");
      expect(await rangeResponse.text()).toBe("2345");

      const multiRangeResponse = await app.fetch(
        new Request(`http://localhost/manuals/${filename}`, {
          headers: {
            Authorization: `Bearer ${token}`,
            Range: "bytes=0-0,2-3",
          },
        }),
      );

      expect(multiRangeResponse.status).toBe(200);
      expect(multiRangeResponse.headers.get("content-length")).toBe("10");
      expect(multiRangeResponse.headers.get("content-range")).toBeNull();
      expect(await multiRangeResponse.text()).toBe("0123456789");
    } finally {
      await rm(filePath, { force: true });
    }
  });
});
