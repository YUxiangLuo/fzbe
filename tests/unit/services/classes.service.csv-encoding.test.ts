import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const passwordModulePath = path.join(srcRoot, "utils", "password.ts");
const classesServiceModulePath = path.join(srcRoot, "modules", "classes", "classes.service.ts");
const freshClassesServiceModulePath = `${classesServiceModulePath}?classes-csv-encoding-test`;

const defaultConnectionQuery = async (query: string, params?: unknown[]) => {
  if (query.includes("FROM academic_terms") && query.includes("is_active = TRUE")) {
    return [[{
      term_id: 1,
      academic_year: "2025-2026",
      semester: 2,
      is_active: true,
      created_at: new Date("2026-01-01T00:00:00Z"),
    }]];
  }
  if (query.startsWith("INSERT INTO classes")) {
    return [{ insertId: 17 }];
  }
  if (query.includes("SELECT username, email FROM users WHERE username IN")) {
    return [[]];
  }
  if (query.startsWith("INSERT INTO users")) {
    return [{ insertId: 9001 }];
  }
  if (query.includes("SELECT user_id, username FROM users WHERE username IN")) {
    const usernames = (params ?? []) as string[];
    return [usernames.map((username, index) => ({ user_id: 9001 + index, username }))];
  }
  if (query.includes("u.full_name as teacher_name")) {
    return [[{
      class_id: 17,
      class_name: "GBK班级",
      class_code: "GBK001",
      term_id: 1,
      academic_year: "2025-2026",
      semester: 2,
      term_label: "2025-2026 第2学期",
      is_current_term: true,
      teacher_id: 99,
      teacher_name: "测试教师",
      created_at: new Date("2026-01-01T00:00:00Z"),
    }]];
  }
  return [{ insertId: 1, affectedRows: 1 }];
};
const connectionQuery = mock(defaultConnectionQuery);
const connectionBeginTransaction = mock(async () => {});
const connectionCommit = mock(async () => {});
const connectionRollback = mock(async () => {});
const connectionRelease = mock(() => {});
const hashPassword = mock(async (_password: string) => "hashed-password");

mock.module(configModulePath, () => ({
  env: {
    DB_HOST: process.env.DB_HOST ?? "127.0.0.1",
    DB_USER: process.env.DB_USER ?? "test",
    DB_PASSWORD: process.env.DB_PASSWORD ?? "test",
    DB_DATABASE: process.env.DB_DATABASE ?? "test",
    JWT_SECRET: "test-jwt-secret",
    PORT: Number(process.env.PORT ?? "3001"),
  },
  connectionLimit: 64,
  pool: {
    query: mock(async () => [[]]),
    getConnection: mock(async () => ({
      beginTransaction: connectionBeginTransaction,
      commit: connectionCommit,
      rollback: connectionRollback,
      release: connectionRelease,
      query: connectionQuery,
    })),
  },
}));

mock.module(passwordModulePath, () => ({
  hashPassword,
}));

const { createClass } = await import(freshClassesServiceModulePath);

const gbkStudentCsv = Buffer.from([
  0xd1, 0xa7, 0xba, 0xc5, 0x2c, 0xd0, 0xd5, 0xc3, 0xfb, 0x0a, 0x32, 0x30,
  0x32, 0x34, 0x30, 0x30, 0x39, 0x39, 0x2c, 0xc0, 0xee, 0xcb, 0xc4, 0x0a,
]);

describe("classes service CSV encoding", () => {
  beforeEach(() => {
    connectionQuery.mockReset();
    connectionQuery.mockImplementation(defaultConnectionQuery);
    connectionBeginTransaction.mockClear();
    connectionCommit.mockClear();
    connectionRollback.mockClear();
    connectionRelease.mockClear();
    hashPassword.mockClear();
  });

  test("createClass imports Windows Excel CSV student list", async () => {
    const result = await createClass(
      "GBK班级",
      "GBK001",
      99,
      new File([gbkStudentCsv], "students.csv", { type: "text/csv" }),
    );

    expect(result).toMatchObject({
      data: {
        class: {
          class_id: 17,
          created_at: "2026-01-01T00:00:00.000Z",
        },
        students_created: 1,
        students_enrolled: 1,
        students: [{
          user_id: 9001,
          username: "20240099",
          full_name: "李四",
          email: "20240099@student.edu",
        }],
      },
    });
    expect(hashPassword).toHaveBeenCalledWith("20240099");
    expect(connectionCommit).toHaveBeenCalledTimes(1);
    expect(connectionRollback).not.toHaveBeenCalled();
  });

  test("rolls back instead of reporting failure after a class was committed when the created row cannot be reloaded", async () => {
    connectionQuery.mockImplementation(async (query: string) => {
      if (query.includes("u.full_name as teacher_name")) return [[]];
      return defaultConnectionQuery(query);
    });

    await expect(createClass("空回读班级", "EMPTY001", 99)).rejects.toThrow(
      "Created class could not be reloaded before commit",
    );

    expect(connectionCommit).not.toHaveBeenCalled();
    expect(connectionRollback).toHaveBeenCalledTimes(1);
  });

  test("reports a generated student email conflict instead of a class-code conflict", async () => {
    connectionQuery.mockImplementation(async (query: string) => {
      if (query.includes("FROM academic_terms") && query.includes("is_active = TRUE")) {
        return [[{
          term_id: 1,
          academic_year: "2025-2026",
          semester: 2,
          is_active: true,
          created_at: new Date("2026-01-01T00:00:00Z"),
        }]];
      }
      if (query.startsWith("INSERT INTO classes")) return [{ insertId: 18 }];
      if (query.includes("SELECT username, email FROM users WHERE username IN")) {
        return [[{
          username: "different-existing-user",
          email: "20240099@student.edu",
        }]];
      }
      return [{ insertId: 1, affectedRows: 1 }];
    });

    const result = await createClass(
      "邮箱冲突班级",
      "EMAIL-CONFLICT-01",
      99,
      new File([gbkStudentCsv], "students.csv", { type: "text/csv" }),
    );

    expect(result).toEqual({
      error: "事务失败。以下学生账号或邮箱已存在于数据库中。",
      status: 409,
      details: { existing_students: ["20240099"] },
    });
    expect(connectionCommit).not.toHaveBeenCalled();
    expect(connectionRollback).toHaveBeenCalledTimes(1);
  });

  test("maps an email uniqueness race during student insertion to the student conflict", async () => {
    connectionQuery.mockImplementation(async (query: string) => {
      if (query.includes("FROM academic_terms") && query.includes("is_active = TRUE")) {
        return [[{
          term_id: 1,
          academic_year: "2025-2026",
          semester: 2,
          is_active: true,
          created_at: new Date("2026-01-01T00:00:00Z"),
        }]];
      }
      if (query.startsWith("INSERT INTO classes")) return [{ insertId: 19 }];
      if (query.includes("SELECT username, email FROM users WHERE username IN")) return [[]];
      if (query.startsWith("INSERT INTO users")) {
        throw {
          code: "ER_DUP_ENTRY",
          message: "Duplicate entry '20240099@student.edu' for key 'users.email'",
        };
      }
      return [{ insertId: 1, affectedRows: 1 }];
    });

    const result = await createClass(
      "邮箱并发冲突班级",
      "EMAIL-RACE-01",
      99,
      new File([gbkStudentCsv], "students.csv", { type: "text/csv" }),
    );

    expect(result).toEqual({
      error: "事务失败。以下学生账号或邮箱已存在于数据库中。",
      status: 409,
      details: { existing_students: ["20240099"] },
    });
    expect(connectionCommit).not.toHaveBeenCalled();
    expect(connectionRollback).toHaveBeenCalledTimes(1);
  });

  test("enrolls students by reloaded user_ids when auto_increment_increment > 1", async () => {
    let insertedEnrollments: unknown = null;
    connectionQuery.mockImplementation(async (query: string, params?: unknown[]) => {
      if (query.includes("SELECT user_id, username FROM users WHERE username IN")) {
        return [[
          { user_id: 9001, username: "20240101" },
          { user_id: 9003, username: "20240102" },
        ]];
      }
      if (query.startsWith("INSERT INTO enrollments")) {
        insertedEnrollments = params?.[0];
        return [{ affectedRows: 2 }];
      }
      return defaultConnectionQuery(query, params);
    });

    const result = await createClass(
      "步长班级",
      "STEP002",
      99,
      new File(
        ["学号,姓名\n20240101,学生甲\n20240102,学生乙"],
        "students-step.csv",
        { type: "text/csv" },
      ),
    );

    expect(insertedEnrollments).toEqual([
      [9001, 17],
      [9003, 17],
    ]);
    expect(result).toMatchObject({
      data: {
        students_created: 2,
        students_enrolled: 2,
        students: [
          { user_id: 9001, username: "20240101" },
          { user_id: 9003, username: "20240102" },
        ],
      },
    });
  });
});
