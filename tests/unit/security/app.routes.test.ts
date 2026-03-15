import { beforeEach, describe, expect, mock, test } from "bun:test";

type AppLike = {
  fetch: (request: Request) => Promise<Response>;
};

const loginMock = mock(async () => ({
  success: true as const,
  token: "test-token",
  user: { role: "Student" },
}));

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
    query: mock(async () => [[]]),
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
  process.env.JWT_SECRET ??= "test-jwt-secret";
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
});