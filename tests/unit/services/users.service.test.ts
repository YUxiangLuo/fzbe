import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";
import { rm } from "node:fs/promises";
import {
  ensureDatabaseManagedStorageRoot,
  getDatabaseManagedStorageRoot,
} from "../../../src/utils/managed-storage";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const passwordModulePath = path.join(srcRoot, "utils", "password.ts");
const usersServiceModulePath = path.join(srcRoot, "modules", "users", "users.service.ts");
const freshUsersServiceModulePath = `${usersServiceModulePath}?users-service-test`;
const TEST_DATABASE = "fz_test_users_service";
const TEST_STORAGE_ENV = {
  DB_HOST: "127.0.0.1",
  DB_PORT: 3306,
  DB_USER: "test",
  DB_PASSWORD: "test",
  DB_DATABASE: TEST_DATABASE,
  E2E_ALLOW_DB_RESET: true,
  JWT_SECRET: "test-jwt-secret",
  PORT: 3001,
};
const reportsRoot = path.resolve(srcRoot, "../uploads/reports");
const reportsStorageRoot = getDatabaseManagedStorageRoot(reportsRoot, TEST_STORAGE_ENV);

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
  env: TEST_STORAGE_ENV,
  connectionLimit: 64,
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
  searchUsers,
  buildBatchImportTempFilename,
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

  test("uses collision-resistant bounded names for same-millisecond batch uploads", () => {
    const originalNow = Date.now;
    Date.now = () => 1234567890;
    try {
      const first = buildBatchImportTempFilename("teachers", "名单.csv");
      const second = buildBatchImportTempFilename("teachers", "名单.csv");
      expect(first).not.toBe(second);
      expect(first).toStartWith("teachers-1234567890-");
      expect(Buffer.byteLength(first)).toBeLessThanOrEqual(255);
    } finally {
      Date.now = originalNow;
    }
  });

  test("bounds long CJK and ASCII batch-upload names by component bytes", () => {
    for (const [kind, originalName] of [
      ["teachers", `${"中".repeat(300)}.csv`],
      ["assistants", `${"a".repeat(600)}.csv`],
    ] as const) {
      const filename = buildBatchImportTempFilename(kind, originalName);
      expect(filename).toEndWith(".csv");
      expect(Buffer.byteLength(filename, "utf8")).toBeLessThanOrEqual(255);
    }
  });

  test("does not apply the database VARCHAR path budget to temporary CSV paths", () => {
    const longButValidDirectory = path.join("/tmp", "a".repeat(210));
    const filename = buildBatchImportTempFilename(
      "teachers",
      `${"教师名单".repeat(100)}.csv`,
    );

    expect(Buffer.byteLength(filename, "utf8")).toBeLessThanOrEqual(255);
    expect(path.join(longButValidDirectory, filename).length).toBeGreaterThan(255);
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

  test("maps duplicate phone errors from the explicit phone unique index", async () => {
    poolQuery.mockImplementationOnce(async () => [[]]);
    poolQuery.mockRejectedValueOnce({
      code: "ER_DUP_ENTRY",
      message: "Duplicate entry '13900001111' for key 'users_phone_number_unique'",
    });

    const result = await createTeacher({
      username: "phoneindex",
      full_name: "索引冲突教师",
      email: "phoneindex@test.com",
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

  test("clears the current user's phone with SQL NULL", async () => {
    await updateCurrentUser(1, { phone_number: null });

    expect(poolQuery.mock.calls[0]).toEqual([
      "UPDATE users SET phone_number = ? WHERE user_id = ?",
      [null, 1],
    ]);
  });

  test("searches long ASCII terms by username as well as full-text profile fields", async () => {
    poolQuery.mockResolvedValueOnce([[{ total: 1 }]]);
    poolQuery.mockResolvedValueOnce([[
      {
        user_id: 41,
        username: "admsrch260801",
        full_name: "边缘搜索教师",
        email: "distinct260801@example.test",
        role: "Teacher",
        created_at: new Date("2026-08-01T00:00:00Z"),
        relevance: 0,
      },
    ]]);

    const result = await searchUsers({ q: "admsrch260801", page: 1, limit: 10 });

    expect(result.data.data).toHaveLength(1);
    expect(result.data.data[0]?.username).toBe("admsrch260801");
    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("UNION");
    expect(String(poolQuery.mock.calls[1]?.[0])).toContain("JOIN");
    expect(poolQuery.mock.calls[0]?.[1]).toEqual(["admsrch260801%", "+admsrch260801*"]);
  });

  test("escapes username wildcard characters in the indexed prefix branch", async () => {
    poolQuery.mockResolvedValueOnce([[{ total: 0 }]]);
    poolQuery.mockResolvedValueOnce([[]]);

    await searchUsers({ q: "prof_138", page: 1, limit: 10 });

    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("ESCAPE '='");
    expect(poolQuery.mock.calls[0]?.[1]).toEqual(["prof=_138%", "+prof_138*"]);
  });

  test("searches short and non-ASCII terms by username in the LIKE branch", async () => {
    poolQuery.mockResolvedValueOnce([[{ total: 0 }]]);
    poolQuery.mockResolvedValueOnce([[]]);

    await searchUsers({ q: "短名", page: 1, limit: 10 });

    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("username LIKE ?");
    expect(poolQuery.mock.calls[0]?.[1]).toEqual(["%短名%", "%短名%", "%短名%", "%短名%"]);
  });

  test("treats LIKE fallback wildcard and escape characters as literal text", async () => {
    poolQuery.mockResolvedValueOnce([[{ total: 0 }]]);
    poolQuery.mockResolvedValueOnce([[]]);

    await searchUsers({ q: "%_=", page: 1, limit: 10 });

    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("username LIKE ? ESCAPE '='");
    expect(String(poolQuery.mock.calls[1]?.[0])).toContain("phone_number LIKE ? ESCAPE '='");
    expect(poolQuery.mock.calls[0]?.[1]).toEqual([
      "%=%=_==%",
      "%=%=_==%",
      "%=%=_==%",
      "%=%=_==%",
    ]);
    expect(poolQuery.mock.calls[1]?.[1]).toEqual([
      "%=%=_==%",
      "%=%=_==%",
      "%=%=_==%",
      "%=%=_==%",
      "%_=",
      10,
      0,
    ]);
  });

  test("keeps count/data parameters aligned for escaped full-text pagination", async () => {
    poolQuery.mockResolvedValueOnce([[{ total: 15 }]]);
    poolQuery.mockResolvedValueOnce([[]]);

    const result = await searchUsers({ q: "prof_x.260801", page: 3, limit: 7 });

    expect(poolQuery.mock.calls[0]?.[1]).toEqual([
      "prof=_x.260801%",
      "+prof_x.260801*",
    ]);
    expect(poolQuery.mock.calls[1]?.[1]).toEqual([
      "+prof_x.260801*",
      "prof=_x.260801%",
      "+prof_x.260801*",
      "prof_x.260801",
      7,
      14,
    ]);
    expect(String(poolQuery.mock.calls[1]?.[0])).toContain("(u.username = ?) DESC");
    expect(result.data.pagination).toEqual({
      currentPage: 3,
      pageSize: 7,
      totalItems: 15,
      totalPages: 3,
    });
  });

  test("rejects a whitespace-only search before querying the database", async () => {
    await expect(searchUsers({ q: "   ", page: 1, limit: 10 }))
      .resolves.toEqual({ error: "empty_search" });
    expect(poolQuery).not.toHaveBeenCalled();
  });

  test("batch teacher import accepts Windows Excel CSV encoding", async () => {
    const result = await batchCreateTeachers(
      new File(
        [gbkUserCsv],
        `${"教师名单".repeat(100)}.csv`,
        { type: "application/vnd.ms-excel" },
      ),
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
      new File(
        [gbkUserCsv],
        `${"assistant-import-".repeat(50)}.csv`,
        { type: "application/vnd.ms-excel" },
      ),
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

  test("maps a transactional duplicate phone race in teacher batch import to conflicts", async () => {
    connectionQuery.mockImplementation(async (query: string) => {
      if (query.includes("SELECT DISTINCT username, phone_number, email FROM users")) return [[]];
      if (query.includes("INSERT INTO users")) {
        throw {
          code: "ER_DUP_ENTRY",
          message: "Duplicate entry '13800138001' for key 'users_phone_number_unique'",
        };
      }
      return [{ insertId: 1 }];
    });

    const result = await batchCreateTeachers(
      new File(["姓名,手机号\n并发教师,13800138001"], "teachers-race.csv", { type: "text/csv" }),
      "Pass1234",
    );

    expect(result).toEqual({
      error: "conflicts",
      conflicts: [{
        username: "prof_13800138001",
        email: "prof_13800138001@teacher.edu",
        phone_number: "13800138001",
      }],
    });
    expect(connectionRollback).toHaveBeenCalledTimes(1);
    expect(connectionCommit).not.toHaveBeenCalled();
  });

  test("maps a transactional duplicate phone race in assistant batch import to conflicts", async () => {
    connectionQuery.mockImplementation(async (query: string) => {
      if (query.includes("SELECT DISTINCT username, phone_number, email FROM users")) return [[]];
      if (query.includes("INSERT INTO users")) {
        throw {
          code: "ER_DUP_ENTRY",
          message: "Duplicate entry '13800138002' for key 'users_phone_number_unique'",
        };
      }
      return [{ insertId: 1 }];
    });

    const result = await batchCreateAssistants(
      new File(["姓名,手机号\n并发助教,13800138002"], "assistants-race.csv", { type: "text/csv" }),
      "Pass1234",
    );

    expect(result).toEqual({
      error: "conflicts",
      conflicts: [{
        username: "ta_13800138002",
        email: "ta_13800138002@assistant.edu",
        phone_number: "13800138002",
      }],
    });
    expect(connectionRollback).toHaveBeenCalledTimes(1);
    expect(connectionCommit).not.toHaveBeenCalled();
  });

  test("returns a bounded conflict preview when MySQL omits the duplicate value", async () => {
    connectionQuery.mockImplementation(async (query: string) => {
      if (query.includes("SELECT DISTINCT username, phone_number, email FROM users")) return [[]];
      if (query.includes("INSERT INTO users")) {
        throw {
          code: "ER_DUP_ENTRY",
          message: "Duplicate entry for key 'users_phone_number_unique'",
        };
      }
      return [{ insertId: 1 }];
    });
    const csvRows = Array.from({ length: 12 }, (_, index) =>
      `批量教师${index + 1},1390000${String(index + 1).padStart(4, "0")}`
    );

    const result = await batchCreateTeachers(
      new File([
        `姓名,手机号\n${csvRows.join("\n")}`,
      ], "teachers-race-no-value.csv", { type: "text/csv" }),
      "Pass1234",
    );

    expect(result).toMatchObject({ error: "conflicts" });
    if (!("conflicts" in result)) throw new Error("expected conflict result");
    expect(result.conflicts).toHaveLength(10);
    expect(result.conflicts[0]?.username).toBe("prof_13900000001");
    expect(result.conflicts[9]?.username).toBe("prof_13900000010");
    expect(connectionRollback).toHaveBeenCalledTimes(1);
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
      if (query.includes("SELECT experiment_id") && query.includes("FROM experiment_status")) {
        return [[]];
      }
      if (query.includes("SELECT report_id, student_id, pdf_file_path")) {
        return [[]];
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
      if (query.includes("SELECT experiment_id") && query.includes("FROM experiment_status")) {
        return [[]];
      }
      if (query.includes("SELECT report_id, student_id, pdf_file_path")) {
        return [[]];
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

  test("bulk delete isolates and removes student report files before returning success", async () => {
    await ensureDatabaseManagedStorageRoot(reportsRoot, TEST_STORAGE_ENV);
    const reportPath = path.join(
      reportsStorageRoot,
      `unit-user-delete-${process.pid}-${Date.now()}.pdf`,
    );
    await Bun.write(reportPath, "%PDF-1.4 unit report");

    try {
      connectionQuery.mockImplementation(async (query: string, values?: unknown[]) => {
        if (query.includes("SELECT user_id, role FROM users WHERE user_id IN")) {
          const userIds = values as number[];
          return [[{ user_id: userIds[0], role: "Student" }]];
        }
        if (query.includes("SELECT experiment_id") && query.includes("FROM experiment_status")) {
          return [[{ experiment_id: 91 }]];
        }
        if (query.includes("SELECT report_id, student_id, pdf_file_path")) {
          return [[{ report_id: 71, student_id: 404, pdf_file_path: reportPath }]];
        }
        if (query.includes("SELECT class_id FROM enrollments WHERE student_id = ?")) {
          return [[]];
        }
        return [{ affectedRows: 1 }];
      });

      const result = await batchDeleteUsers([404], 999);

      expect(result).toEqual({
        success: true,
        deletedCount: 1,
        cleanup: { removedReportFiles: 1 },
      });
      expect(await Bun.file(reportPath).exists()).toBe(false);
      expect(connectionCommit).toHaveBeenCalledTimes(1);
      expect(connectionRelease).toHaveBeenCalledTimes(1);
      const lockingQueries = connectionQuery.mock.calls
        .map(([query]) => String(query))
        .filter((query) => query.includes("FOR UPDATE"));
      expect(lockingQueries.map((query) => (
        query.includes("FROM experiment_status")
          ? "experiment"
          : query.includes("FROM experiment_reports") ? "report" : "user"
      ))).toEqual(["user", "experiment", "report"]);
    } finally {
      await rm(reportPath, { force: true });
    }
  });
});
