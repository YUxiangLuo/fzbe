import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const classesServiceModulePath = path.join(srcRoot, "modules", "classes", "classes.service.ts");
const freshClassesServiceModulePath = `${classesServiceModulePath}?classes-service-assistants-test`;

const teacherClasses = [
  {
    class_id: 101,
    class_name: "一班",
    class_code: "C101",
    teacher_id: 7,
    teacher_name: "王老师",
    created_at: new Date("2026-01-01T00:00:00Z"),
  },
  {
    class_id: 102,
    class_name: "二班",
    class_code: "C102",
    teacher_id: 7,
    teacher_name: "王老师",
    created_at: new Date("2026-01-02T00:00:00Z"),
  },
];

const assistantClasses = [teacherClasses[0]];

const assistantRows = [
  {
    class_id: 101,
    user_id: 201,
    username: "assistant_a",
    full_name: "助教甲",
    email: "assistant-a@example.com",
    role: "Assistant",
  },
  {
    class_id: 101,
    user_id: 202,
    username: "assistant_b",
    full_name: "助教乙",
    email: "assistant-b@example.com",
    role: "Assistant",
  },
];

const poolQuery = mock(async (query: string, params?: unknown[]) => {
  if (query.includes("SELECT role FROM users") && query.includes("role = 'Teacher'")) {
    return [[{ role: "Teacher" }]];
  }

  if (query.includes("SELECT role FROM users") && query.includes("role = 'Assistant'")) {
    return [[{ role: "Assistant" }]];
  }

  if (query.includes("WHERE ca.class_id IN")) {
    return [assistantRows.filter((row) => (params as number[]).includes(row.class_id))];
  }

  if (query.includes("WHERE c.teacher_id = ? AND ca.assistant_id = ?")) {
    return [assistantClasses];
  }

  if (query.includes("WHERE ca.assistant_id = ?")) {
    return [assistantClasses];
  }

  if (query.includes("WHERE c.teacher_id = ?")) {
    return [teacherClasses];
  }

  return [[]];
});

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
    query: poolQuery,
  },
}));

const { getClassesByTeacher, getClassesByAssistant } = await import(freshClassesServiceModulePath);

describe("classes service assistant metadata", () => {
  beforeEach(() => {
    poolQuery.mockClear();
  });

  test("teacher class list includes assistants for each class", async () => {
    const result = await getClassesByTeacher(7, 7, "Teacher");

    expect(result).toMatchObject({
      data: [
        {
          class_id: 101,
          assistants: [
            { user_id: 201, username: "assistant_a", full_name: "助教甲" },
            { user_id: 202, username: "assistant_b", full_name: "助教乙" },
          ],
        },
        {
          class_id: 102,
          assistants: [],
        },
      ],
    });
  });

  test("assistant class list also includes assistant names for class management", async () => {
    const result = await getClassesByAssistant(201, 201, "Assistant");

    expect(result).toMatchObject({
      data: [
        {
          class_id: 101,
          assistants: [
            { user_id: 201, username: "assistant_a", full_name: "助教甲" },
            { user_id: 202, username: "assistant_b", full_name: "助教乙" },
          ],
        },
      ],
    });
  });
});
