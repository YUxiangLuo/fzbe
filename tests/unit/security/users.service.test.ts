import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const passwordModulePath = path.join(srcRoot, "utils", "password.ts");
const usersServiceModulePath = path.join(srcRoot, "modules", "users", "users.service.ts");
const freshUsersServiceModulePath = `${usersServiceModulePath}?users-service-test`;

const poolQuery = mock(async (query: string) => {
  if (query.includes("SELECT username, email, phone_number FROM users WHERE")) {
    return [[]];
  }
  return [{ insertId: 1 }];
});
const connectionQuery = mock(async (query: string) => {
  if (query.includes("SELECT user_id, username, email, phone_number FROM users WHERE")) {
    return [[]];
  }
  if (query.includes("SELECT class_id, teacher_id FROM classes WHERE")) {
    return [[{ class_id: 1, teacher_id: 999 }]];
  }
  return [{ insertId: 1 }];
});
const connectionBeginTransaction = mock(async () => {});
const connectionCommit = mock(async () => {});
const connectionRollback = mock(async () => {});
const connectionRelease = mock(() => {});
const getConnection = mock(async () => ({
  beginTransaction: connectionBeginTransaction,
  commit: connectionCommit,
  rollback: connectionRollback,
  release: connectionRelease,
  query: connectionQuery,
}));
const hashPassword = mock(async (_password: string) => "hashed-password");
const shouldChangePassword = mock(async () => false);
const verifyPassword = mock(async () => true);

mock.module(configModulePath, () => ({
  pool: {
    query: poolQuery,
    getConnection,
  },
}));

mock.module(passwordModulePath, () => ({
  hashPassword,
  shouldChangePassword,
  verifyPassword,
}));

const {
  createTeacher,
  createAssistantSingle,
  registerSelfUser,
  createAssistantWithClasses,
  updateCurrentUser,
  batchCreateTeachers,
  batchCreateAssistants,
  batchDeleteUsers,
} = await import(freshUsersServiceModulePath);

const gbkUserCsv = Buffer.from([
  0xd0, 0xd5, 0xc3, 0xfb, 0x2c, 0xca, 0xd6, 0xbb, 0xfa, 0xba, 0xc5, 0x0a,
  0xd5, 0xc5, 0xc8, 0xfd, 0x2c, 0x31, 0x33, 0x38, 0x30, 0x30, 0x31, 0x33,
  0x38, 0x30, 0x30, 0x30, 0x0a,
]);

describe("users service duplicate-key handling", () => {
  beforeEach(() => {
    poolQuery.mockReset();
    poolQuery.mockImplementation(async (query: string) => {
      if (query.includes("SELECT user_id, username, email, phone_number FROM users WHERE")) {
        return [[]];
      }
      if (query.includes("SELECT user_id, username, full_name, email, phone_number, role, created_at FROM users WHERE user_id = ?")) {
        return [[{ user_id: 1, username: "u1", full_name: "用户", email: "u1@test.com", phone_number: "13900001111", role: "Teacher", created_at: new Date() }]];
      }
      return [{ insertId: 1 }];
    });
    connectionQuery.mockReset();
    connectionQuery.mockImplementation(async (query: string) => {
      if (query.includes("SELECT user_id, username, email, phone_number FROM users WHERE")) {
        return [[]];
      }
      if (query.includes("SELECT class_id, teacher_id FROM classes WHERE")) {
        return [[{ class_id: 1, teacher_id: 999 }]];
      }
      return [{ insertId: 1 }];
    });
    hashPassword.mockClear();
    getConnection.mockClear();
    connectionBeginTransaction.mockClear();
    connectionCommit.mockClear();
    connectionRollback.mockClear();
    connectionRelease.mockClear();
  });

  test("maps duplicate username errors from MySQL key names when creating a teacher", async () => {
    poolQuery.mockImplementationOnce(async () => [[]]);
    poolQuery.mockRejectedValueOnce({
      code: "ER_DUP_ENTRY",
      message: "Duplicate entry 'dup001' for key 'username'",
    });

    const result = await createTeacher({
      username: "dup001",
      full_name: "重复教师",
      email: "dup001@test.com",
      phone: "13900001111",
      password: "Pass1234",
    });

    expect(result).toEqual({ error: "dup_username" });
  });

  test("returns duplicate username before insert when preflight lookup finds a conflict", async () => {
    poolQuery.mockImplementationOnce(async () => [[{ username: "dup001", email: "other@test.com", phone_number: "13900009999" }]]);

    const result = await createTeacher({
      username: "dup001",
      full_name: "重复教师",
      email: "dup001@test.com",
      phone: "13900001111",
      password: "Pass1234",
    });

    expect(result).toEqual({ error: "dup_username" });
    expect(hashPassword).not.toHaveBeenCalled();
  });

  test("returns duplicate phone before insert when preflight lookup finds a conflict", async () => {
    poolQuery.mockImplementationOnce(async () => [[{ username: "other001", email: "other@test.com", phone_number: "13900001111" }]]);

    const result = await createTeacher({
      username: "new001",
      full_name: "重复手机号教师",
      email: "new001@test.com",
      phone: "13900001111",
      password: "Pass1234",
    });

    expect(result).toEqual({ error: "dup_phone" });
  });

  test("maps duplicate phone errors from MySQL key names when creating an assistant", async () => {
    poolQuery.mockImplementationOnce(async () => [[]]);
    poolQuery.mockRejectedValueOnce({
      code: "ER_DUP_ENTRY",
      message: "Duplicate entry '13900001111' for key 'phone_number'",
    });

    const result = await createAssistantSingle({
      username: "assistantdup",
      full_name: "重复助教",
      email: "assistantdup@test.com",
      phone: "13900001111",
      password: "Pass1234",
    });

    expect(result).toEqual({ error: "dup_phone" });
  });

  test("maps duplicate email errors from prefixed index names during self registration", async () => {
    poolQuery.mockImplementationOnce(async () => [[]]);
    poolQuery.mockRejectedValueOnce({
      code: "ER_DUP_ENTRY",
      message: "Duplicate entry 'repeat@test.com' for key 'users.email'",
    });

    const result = await registerSelfUser("Teacher", {
      username: "repeat001",
      password: "Pass1234",
      name: "重复教师",
      email: "repeat@test.com",
      phone: "13900002222",
    });

    expect(result).toEqual({ error: "dup_email" });
  });

  test("returns duplicate phone before inserting assistant with classes", async () => {
    connectionQuery.mockImplementationOnce(async () => [[{ class_id: 1, teacher_id: 999 }]]);
    connectionQuery.mockImplementationOnce(async () => [[{ username: "other001", email: "other@test.com", phone_number: "13900001111" }]]);

    const result = await createAssistantWithClasses({
      username: "assistant001",
      password: "Pass1234",
      full_name: "重复手机号助教",
      email: "assistant001@test.com",
      phone_number: "13900001111",
      class_ids: [1],
    }, 999, "Admin");

    expect(result).toEqual({ error: "dup_phone" });
    expect(connectionRollback).toHaveBeenCalledTimes(1);
    expect(connectionCommit).not.toHaveBeenCalled();
  });

  test("returns duplicate phone when updating current user profile", async () => {
    poolQuery.mockImplementationOnce(async () => [[{ username: "other001", email: "other@test.com", phone_number: "13900002222" }]]);

    const result = await updateCurrentUser(1, {
      phone_number: "13900002222",
    });

    expect(result).toEqual({ error: "dup_phone" });
  });

  test("batch teacher import accepts Windows Excel CSV encoding", async () => {
    const result = await batchCreateTeachers(
      new File([gbkUserCsv], "teachers.csv", { type: "application/vnd.ms-excel" }),
      undefined,
    );

    expect(result).toEqual({
      data: {
        created: 1,
        teachers: [{
          user_id: 1,
          username: "prof_13800138000",
          full_name: "张三",
          phone_number: "13800138000",
          email: "prof_13800138000@teacher.edu",
        }],
      },
    });
    expect(hashPassword).toHaveBeenCalledWith("13800138000");
    expect(connectionCommit).toHaveBeenCalledTimes(1);
  });

  test("batch assistant import accepts Windows Excel CSV encoding", async () => {
    const result = await batchCreateAssistants(
      new File([gbkUserCsv], "assistants.csv", { type: "application/vnd.ms-excel" }),
      undefined,
    );

    expect(result).toEqual({
      data: {
        created: 1,
        assistants: [{
          user_id: 1,
          username: "ta_13800138000",
          full_name: "张三",
          phone_number: "13800138000",
          email: "ta_13800138000@assistant.edu",
        }],
      },
    });
    expect(hashPassword).toHaveBeenCalledWith("13800138000");
    expect(connectionCommit).toHaveBeenCalledTimes(1);
  });

  test("bulk delete rejects self deletion before opening a transaction", async () => {
    const result = await batchDeleteUsers([1, 2], 1);

    expect(result).toEqual({ error: "self_delete" });
    expect(getConnection).not.toHaveBeenCalled();
  });

  test("bulk delete removes multiple users in one transaction and aggregates cleanup counts", async () => {
    connectionQuery.mockImplementation(async (query: string, values?: unknown[]) => {
      if (query.includes("SELECT user_id, role FROM users WHERE user_id IN")) {
        const userIds = values as number[];
        return [[
          { user_id: userIds[0], role: "Teacher" },
          { user_id: userIds[1], role: "Assistant" },
        ]];
      }
      if (query.includes("SELECT class_id FROM classes WHERE teacher_id = ?")) {
        return [[{ class_id: 11 }, { class_id: 12 }]];
      }
      if (query.includes("SELECT class_id FROM class_assistants WHERE assistant_id = ?")) {
        return [[{ class_id: 21 }]];
      }
      return [{ affectedRows: 1 }];
    });

    const result = await batchDeleteUsers([101, 202], 999);

    expect(result).toEqual({
      success: true,
      deletedCount: 2,
      cleanup: {
        deletedClasses: 2,
        removedAssistantAssignments: 1,
      },
    });
    expect(connectionBeginTransaction).toHaveBeenCalledTimes(1);
    expect(connectionCommit).toHaveBeenCalledTimes(1);
    expect(connectionRollback).not.toHaveBeenCalled();
  });

  test("bulk delete removes student enrollments and reports cleanup counts", async () => {
    connectionQuery.mockImplementation(async (query: string, values?: unknown[]) => {
      if (query.includes("SELECT user_id, role FROM users WHERE user_id IN")) {
        const userIds = values as number[];
        return [[{ user_id: userIds[0], role: "Student" }]];
      }
      if (query.includes("SELECT class_id FROM enrollments WHERE student_id = ?")) {
        return [[{ class_id: 31 }, { class_id: 32 }]];
      }
      return [{ affectedRows: 1 }];
    });

    const result = await batchDeleteUsers([303], 999);

    expect(result).toEqual({
      success: true,
      deletedCount: 1,
      cleanup: {
        removedEnrollments: 2,
      },
    });
    expect(connectionBeginTransaction).toHaveBeenCalledTimes(1);
    expect(connectionCommit).toHaveBeenCalledTimes(1);
    expect(connectionRollback).not.toHaveBeenCalled();
  });
});
