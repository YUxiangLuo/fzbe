import { sign } from "hono/jwt";

type AppLike = {
  fetch: (request: Request) => Promise<Response>;
};

type UserRole = "Student" | "Teacher" | "Assistant" | "Admin";

const ensureTestEnv = () => {
  process.env.DB_HOST ??= "127.0.0.1";
  process.env.DB_USER ??= "test";
  process.env.DB_PASSWORD ??= "test";
  process.env.DB_DATABASE ??= "test";
  process.env.JWT_SECRET ??= "test-jwt-secret";
  process.env.PORT ??= "3001";
  process.env.DEFAULT_PASSWORD_HASH ??= "test-default-hash";
};

let appPromise: Promise<AppLike> | null = null;

const getApp = async (): Promise<AppLike> => {
  if (!appPromise) {
    ensureTestEnv();
    appPromise = import("../../../src/index").then((module) => module.app as AppLike);
  }
  return appPromise;
};

const createAuthHeader = async (role: UserRole): Promise<HeadersInit> => {
  ensureTestEnv();
  const token = await sign(
    {
      sub: 1001,
      username: `test-${role.toLowerCase()}`,
      role,
      exp: Math.floor(Date.now() / 1000) + 60 * 10,
    },
    process.env.JWT_SECRET!,
  );

  return {
    Authorization: `Bearer ${token}`,
  };
};

describe("Permission Guards", () => {
  test("GET /manuals/:filename requires authentication", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/manuals/example.pdf"),
    );

    expect(response.status).toBe(401);
  });

  test("GET /datasets/:filename requires authentication", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/datasets/example.csv"),
    );

    expect(response.status).toBe(401);
  });

  test("GET /api/v1/manuals/active requires authentication", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/api/v1/manuals/active"),
    );

    expect(response.status).toBe(401);
  });

  test("GET /api/v1/datasets forbids non-admin users", async () => {
    const app = await getApp();
    const headers = await createAuthHeader("Teacher");
    const response = await app.fetch(
      new Request("http://localhost/api/v1/datasets", { headers }),
    );

    expect(response.status).toBe(403);
  });

  test("GET /api/v1/datasets/:datasetId forbids non-admin users", async () => {
    const app = await getApp();
    const headers = await createAuthHeader("Assistant");
    const response = await app.fetch(
      new Request("http://localhost/api/v1/datasets/1", { headers }),
    );

    expect(response.status).toBe(403);
  });
});
