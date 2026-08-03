process.env.DB_HOST = "127.0.0.1";
process.env.DB_USER = "test";
process.env.DB_PASSWORD = "test";
process.env.DB_DATABASE = "test";
process.env.JWT_SECRET = "test-jwt-secret";
process.env.PORT = "3001";

import { beforeEach, describe, expect, mock, test } from "bun:test";
import { Hono } from "hono";
import { sign } from "hono/jwt";

mock.restore();

const deleteUser = mock(async () => ({ error: "file_cleanup_failed" as const }));
const batchDeleteUsers = mock(async () => ({ error: "file_cleanup_failed" as const }));
const unused = mock(async () => ({ data: [] }));

mock.module("../../../src/modules/users/users.service", () => ({
  registerSelfUser: unused,
  createAssistantSingle: unused,
  searchUsers: unused,
  createAssistantWithClasses: unused,
  createTeacher: unused,
  batchCreateTeachers: unused,
  batchCreateAssistants: unused,
  getTeacherAssistants: unused,
  getAllUsers: unused,
  getAllAssistants: unused,
  getUnenrolledStudents: unused,
  getCurrentUser: unused,
  updateCurrentUser: unused,
  changePassword: unused,
  getUserById: unused,
  adminChangePassword: unused,
  resetStudentPassword: unused,
  deleteUser,
  batchDeleteUsers,
}));

mock.module("../../../src/modules/academic-terms", () => ({
  resolveAcademicTermId: mock(async () => ({ termId: 1 })),
}));

mock.module("../../../src/modules/auth/auth.rate-limit", () => ({
  enforceSelfRegistrationRateLimit: mock(() => null),
}));

const { default: usersController } = await import(
  "../../../src/modules/users/users.controller"
);

const app = new Hono();
app.route("/", usersController);

const adminAuthorization = async () => `Bearer ${await sign({
  sub: 9001,
  username: "admin-test",
  role: "Admin",
  exp: Math.floor(Date.now() / 1000) + 600,
}, process.env.JWT_SECRET!)}`;

describe("user deletion controller cleanup failures", () => {
  beforeEach(() => {
    deleteUser.mockReset();
    deleteUser.mockResolvedValue({ error: "file_cleanup_failed" });
    batchDeleteUsers.mockReset();
    batchDeleteUsers.mockResolvedValue({ error: "file_cleanup_failed" });
  });

  test("single delete reports HTTP 500 and never claims the user was removed", async () => {
    const response = await app.request("/api/v1/users/77", {
      method: "DELETE",
      headers: { Authorization: await adminAuthorization() },
    });

    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({ error: "报告文件无法安全清理，用户未删除" });
  });

  test("bulk delete reports HTTP 500 and states that no user was removed", async () => {
    const response = await app.request("/api/v1/users", {
      method: "DELETE",
      headers: {
        Authorization: await adminAuthorization(),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ userIds: [77, 78] }),
    });

    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({ error: "部分报告文件无法安全清理，用户均未删除" });
  });
});
