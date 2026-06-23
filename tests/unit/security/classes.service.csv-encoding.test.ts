import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const passwordModulePath = path.join(srcRoot, "utils", "password.ts");
const classesServiceModulePath = path.join(srcRoot, "modules", "classes", "classes.service.ts");
const freshClassesServiceModulePath = `${classesServiceModulePath}?classes-csv-encoding-test`;

const connectionQuery = mock(async (query: string) => {
  if (query.startsWith("INSERT INTO classes")) {
    return [{ insertId: 17 }];
  }
  if (query.includes("SELECT username FROM users WHERE username IN")) {
    return [[]];
  }
  if (query.startsWith("INSERT INTO users")) {
    return [{ insertId: 9001 }];
  }
  if (query.includes("SELECT c.*, u.full_name as teacher_name")) {
    return [[{
      class_id: 17,
      class_name: "GBK班级",
      class_code: "GBK001",
      teacher_id: 99,
      teacher_name: "测试教师",
      created_at: new Date("2026-01-01T00:00:00Z"),
    }]];
  }
  return [{ insertId: 1, affectedRows: 1 }];
});
const connectionBeginTransaction = mock(async () => {});
const connectionCommit = mock(async () => {});
const connectionRollback = mock(async () => {});
const connectionRelease = mock(() => {});
const hashPassword = mock(async (_password: string) => "hashed-password");

mock.module(configModulePath, () => ({
  pool: {
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
    connectionQuery.mockClear();
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
});
