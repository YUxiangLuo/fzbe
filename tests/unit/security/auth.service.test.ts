process.env.DB_HOST = "127.0.0.1";
process.env.DB_USER = "test";
process.env.DB_PASSWORD = "test";
process.env.DB_DATABASE = "test";
process.env.JWT_SECRET = "test-jwt-secret";
process.env.PORT = "3001";

import { beforeEach, describe, expect, mock, test } from "bun:test";
import { AUTH_CONSTANTS } from "../../../src/utils/constants";

mock.restore();

const queryMock = mock(async () => [[
  {
    user_id: 42,
    username: "student001",
    password_hash: "hash",
    role: "Student",
  },
]]);
const verifyPasswordMock = mock(async () => true);
const signMock = mock(async () => "signed-token");

mock.module("../../../src/config", () => ({
  env: {
    DB_HOST: "127.0.0.1",
    DB_USER: "test",
    DB_PASSWORD: "test",
    DB_DATABASE: "test",
    JWT_SECRET: "test-jwt-secret",
    PORT: 3001,
  },
  pool: {
    query: queryMock,
  },
}));

mock.module("../../../src/utils/password", () => ({
  verifyPassword: verifyPasswordMock,
}));

mock.module("hono/jwt", () => ({
  sign: signMock,
}));

const { login } = await import("../../../src/modules/auth/auth.service");

describe("auth service", () => {
  beforeEach(() => {
    queryMock.mockClear();
    queryMock.mockResolvedValue([[
      {
        user_id: 42,
        username: "student001",
        password_hash: "hash",
        role: "Student",
      },
    ]]);
    verifyPasswordMock.mockClear();
    verifyPasswordMock.mockResolvedValue(true);
    signMock.mockClear();
    signMock.mockResolvedValue("signed-token");
  });

  test("issues JWTs that expire in two hours", async () => {
    expect(AUTH_CONSTANTS.JWT_EXPIRATION_SECONDS).toBe(60 * 60 * 2);

    const before = Math.floor(Date.now() / 1000);

    const result = await login({
      username: "student001",
      password: "secret123",
      role: "Student",
    });

    const after = Math.floor(Date.now() / 1000);
    expect(result).toEqual({
      success: true,
      token: "signed-token",
      user: { role: "Student" },
    });
    expect(signMock).toHaveBeenCalledTimes(1);

    const [payload, secret] = signMock.mock.calls[0]!;
    expect(secret).toBe("test-jwt-secret");
    expect(payload.exp).toBeGreaterThanOrEqual(before + 60 * 60 * 2);
    expect(payload.exp).toBeLessThanOrEqual(after + 60 * 60 * 2);
  });
});
