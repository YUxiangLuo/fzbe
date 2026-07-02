import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const passwordModulePath = path.join(srcRoot, "utils", "password.ts");
const classesServiceModulePath = `${path.join(srcRoot, "modules", "classes", "classes.service.ts")}?academic-term-service-test`;
const usersServiceModulePath = `${path.join(srcRoot, "modules", "users", "users.service.ts")}?academic-term-service-test`;
const classGradesServiceModulePath = `${path.join(srcRoot, "modules", "class-grades", "class-grades.service.ts")}?academic-term-service-test`;

const teacherClassRows = [
  {
    class_id: 501,
    class_name: "供应链管理 2026 春",
    class_code: "SCM-2026-S",
    academic_year_start: 2025,
    semester: 2,
    teacher_id: 42,
    teacher_name: "王老师",
    created_at: new Date("2026-03-01T00:00:00Z"),
  },
];

const assistantRows = [
  {
    class_id: 501,
    user_id: 601,
    username: "assistant601",
    full_name: "助教甲",
    email: "assistant601@example.test",
    role: "Assistant",
  },
];

const teacherAssistantRows = [
  {
    user_id: 601,
    username: "assistant601",
    full_name: "助教甲",
    email: "assistant601@example.test",
    phone_number: "13800000001",
  },
];

const poolQuery = mock(async (query: string, params?: unknown[]) => {
  if (query.includes("SELECT role FROM users WHERE user_id = ? AND role = 'Teacher'")) {
    return [[{ role: "Teacher" }]];
  }

  if (query.includes("SELECT role FROM users WHERE user_id = ? AND role = 'Assistant'")) {
    return [[{ role: "Assistant" }]];
  }

  if (query.includes("SELECT ca.class_id") && query.includes("WHERE ca.class_id IN")) {
    const classIds = params as number[];
    return [assistantRows.filter((row) => classIds.includes(row.class_id))];
  }

  if (query.includes("SELECT c.*, u.full_name as teacher_name")) {
    return [teacherClassRows];
  }

  if (query.includes("SELECT DISTINCT u.user_id")) {
    return [teacherAssistantRows];
  }

  if (query.includes("SELECT class_id, class_name, academic_year_start, semester") && query.includes("FROM classes")) {
    return [[]];
  }

  if (query.includes("SELECT c.class_id, c.class_name, c.academic_year_start, c.semester")) {
    return [[]];
  }

  return [[]];
});

const connectionQuery = mock(async (query: string) => {
  if (query.startsWith("INSERT INTO classes")) {
    return [{ insertId: 777 }];
  }

  if (query.startsWith("INSERT INTO grade_weights")) {
    return [{ affectedRows: 1 }];
  }

  if (query.includes("SELECT c.*, u.full_name as teacher_name")) {
    return [[{
      class_id: 777,
      class_name: "需求预测 2026 秋",
      class_code: "FORECAST-2026-F",
      academic_year_start: 2026,
      semester: 1,
      teacher_id: 42,
      teacher_name: "王老师",
      created_at: new Date("2026-09-01T00:00:00Z"),
    }]];
  }

  return [[]];
});

const connectionBeginTransaction = mock(async () => {});
const connectionCommit = mock(async () => {});
const connectionRollback = mock(async () => {});
const connectionRelease = mock(() => {});
const poolGetConnection = mock(async () => ({
  beginTransaction: connectionBeginTransaction,
  commit: connectionCommit,
  rollback: connectionRollback,
  release: connectionRelease,
  query: connectionQuery,
}));

mock.module(configModulePath, () => ({
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
    query: poolQuery,
    getConnection: poolGetConnection,
  },
}));

mock.module(passwordModulePath, () => ({
  hashPassword: mock(async (password: string) => `hashed:${password}`),
  shouldChangePassword: mock(async () => false),
  verifyPassword: mock(async () => true),
}));

const {
  createClass,
  getClassesByTeacher,
  getClassesByTeacherAssistant,
  getClassesByAssistant,
} = await import(classesServiceModulePath);
const { getTeacherAssistants } = await import(usersServiceModulePath);
const {
  getTeacherClassGradeSummaries,
  getAssistantClassGradeSummaries,
} = await import(classGradesServiceModulePath);

const findPoolQuery = (predicate: (sql: string) => boolean) => {
  const call = poolQuery.mock.calls.find(([sql]) => predicate(String(sql)));
  expect(call).toBeDefined();
  return call as [string, unknown[] | undefined];
};

describe("academic term service filtering", () => {
  beforeEach(() => {
    poolQuery.mockClear();
    poolGetConnection.mockClear();
    connectionQuery.mockClear();
    connectionBeginTransaction.mockClear();
    connectionCommit.mockClear();
    connectionRollback.mockClear();
    connectionRelease.mockClear();
  });

  test("createClass inserts explicit academic term values instead of relying on database defaults", async () => {
    const result = await createClass("需求预测 2026 秋", "FORECAST-2026-F", 42, undefined, 2026, 1);

    expect(result).toMatchObject({
      data: {
        class: {
          class_id: 777,
          academic_year_start: 2026,
          semester: 1,
        },
      },
    });
    expect(connectionQuery).toHaveBeenCalledWith(
      "INSERT INTO classes (class_name, class_code, academic_year_start, semester, teacher_id) VALUES (?, ?, ?, ?, ?)",
      ["需求预测 2026 秋", "FORECAST-2026-F", 2026, 1, 42],
    );
    expect(connectionCommit).toHaveBeenCalledTimes(1);
    expect(connectionRollback).not.toHaveBeenCalled();
  });

  test("teacher class listing remains unfiltered when no academic term query is provided", async () => {
    await getClassesByTeacher(42, 42, "Teacher");

    const [sql, params] = findPoolQuery((text) =>
      text.includes("SELECT c.*, u.full_name as teacher_name") && text.includes("WHERE c.teacher_id = ?"),
    );
    expect(sql).not.toContain("c.academic_year_start = ?");
    expect(sql).not.toContain("c.semester = ?");
    expect(params).toEqual([42]);
  });

  test("teacher class listing applies academic year and semester filters", async () => {
    const result = await getClassesByTeacher(42, 42, "Teacher", {
      academic_year_start: 2025,
      semester: 2,
    });

    expect(result).toMatchObject({
      data: [{ class_id: 501, academic_year_start: 2025, semester: 2 }],
    });
    const [sql, params] = findPoolQuery((text) =>
      text.includes("SELECT c.*, u.full_name as teacher_name") && text.includes("WHERE c.teacher_id = ?"),
    );
    expect(sql).toContain("c.academic_year_start = ?");
    expect(sql).toContain("c.semester = ?");
    expect(params).toEqual([42, 2025, 2]);
  });

  test("teacher-assistant class listing filters only classes in the selected academic term", async () => {
    await getClassesByTeacherAssistant(42, 601, 42, "Teacher", {
      academic_year_start: 2025,
      semester: 2,
    });

    const [sql, params] = findPoolQuery((text) =>
      text.includes("JOIN class_assistants ca") && text.includes("c.teacher_id = ?") && text.includes("ca.assistant_id = ?"),
    );
    expect(sql).toContain("c.academic_year_start = ?");
    expect(sql).toContain("c.semester = ?");
    expect(params).toEqual([42, 601, 2025, 2]);
  });

  test("assistant class listing applies the same selected academic term", async () => {
    await getClassesByAssistant(601, 601, "Assistant", {
      academic_year_start: 2025,
      semester: 2,
    });

    const [sql, params] = findPoolQuery((text) =>
      text.includes("JOIN class_assistants ca") && text.includes("ca.assistant_id = ?") && !text.includes("c.teacher_id = ?"),
    );
    expect(sql).toContain("c.academic_year_start = ?");
    expect(sql).toContain("c.semester = ?");
    expect(params).toEqual([601, 2025, 2]);
  });

  test("teacher assistant listing is narrowed to assistants assigned in the selected term", async () => {
    const result = await getTeacherAssistants(42, {
      academic_year_start: 2025,
      semester: 2,
    });

    expect(result).toMatchObject({
      data: [{ user_id: 601, username: "assistant601" }],
    });
    const [sql, params] = findPoolQuery((text) => text.includes("SELECT DISTINCT u.user_id"));
    expect(sql).toContain("c.teacher_id = ?");
    expect(sql).toContain("c.academic_year_start = ?");
    expect(sql).toContain("c.semester = ?");
    expect(params).toEqual([42, 2025, 2]);
  });

  test("teacher grade summary class discovery filters by selected academic term", async () => {
    const result = await getTeacherClassGradeSummaries(42, 42, "Teacher", {
      academic_year_start: 2025,
      semester: 2,
    });

    expect(result).toEqual({ data: [] });
    const [sql, params] = findPoolQuery((text) =>
      text.includes("SELECT class_id, class_name, academic_year_start, semester") && text.includes("FROM classes"),
    );
    expect(sql).toContain("classes.academic_year_start = ?");
    expect(sql).toContain("classes.semester = ?");
    expect(params).toEqual([42, 2025, 2]);
  });

  test("assistant grade summary class discovery filters by selected academic term", async () => {
    const result = await getAssistantClassGradeSummaries(601, 601, "Assistant", {
      academic_year_start: 2025,
      semester: 2,
    });

    expect(result).toEqual({ data: [] });
    const [sql, params] = findPoolQuery((text) =>
      text.includes("SELECT c.class_id, c.class_name, c.academic_year_start, c.semester") && text.includes("FROM class_assistants ca"),
    );
    expect(sql).toContain("c.academic_year_start = ?");
    expect(sql).toContain("c.semester = ?");
    expect(params).toEqual([601, 2025, 2]);
  });
});
