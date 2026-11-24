import { expect, mock, test } from "bun:test";

const calls: string[] = [];
let enrollmentReads = 0;
const connection = {
  beginTransaction: async () => { calls.push("begin"); },
  rollback: async () => { calls.push("rollback"); },
  commit: mock(async () => {}),
  release: () => { calls.push("release"); },
  query: async (sql: string) => {
    if (sql.includes("FROM classes")) return [[{ teacher_id: 10 }]];
    if (sql.includes("FROM users")) return [[{ user_id: 1, role: "Student", full_name: "Student" }]];
    if (sql.includes("FROM enrollments")) {
      calls.push("read-enrollment");
      return ++enrollmentReads === 1 ? [[]] : [[{ class_id: 3, class_name: "Existing class" }]];
    }
    if (sql.startsWith("INSERT")) {
      throw Object.assign(new Error("concurrent enrollment won"), { code: "ER_DUP_ENTRY" });
    }
    throw new Error(`Unexpected SQL: ${sql}`);
  },
};
const borrow = mock(async () => connection);
const nestedQuery = mock(async () => { throw new Error("Pool exhausted while the caller still owns its connection"); });
mock.module("../../../src/config", () => ({
  env: { DB_TIMEZONE: "Z" }, pool: { getConnection: borrow, query: nestedQuery },
}));
const {
  createStudentInClass,
  enrollExistingStudent,
} = await import("../../../src/modules/class-members/class-members.service");

test("duplicate enrollment recovers with the owned connection when the pool is exhausted", async () => {
  expect(await enrollExistingStudent(2, 1, 10, "Teacher")).toEqual({
    error: "already_enrolled", currentClass: { class_id: 3, class_name: "Existing class" },
  });
  expect(borrow).toHaveBeenCalledTimes(1);
  expect(nestedQuery).not.toHaveBeenCalled();
  expect(connection.commit).not.toHaveBeenCalled();
  expect(calls.slice(-3)).toEqual(["rollback", "read-enrollment", "release"]);
});

test.each([
  { message: "Duplicate entry '20240001' for key 'username'", expected: "dup_username" },
  { message: "Duplicate entry '20240001' for key 'users.username'", expected: "dup_username" },
  { message: "Duplicate entry 'a@b.edu' for key 'email'", expected: "dup_email" },
  { message: "Duplicate entry 'a@b.edu' for key 'users.email'", expected: "dup_email" },
  { message: "Duplicate entry '13800138000' for key 'users_phone_number_unique'", expected: "dup_phone" },
  { message: "Duplicate entry '13800138000' for key 'users.users_phone_number_canonical_unique'", expected: "dup_phone" },
])("createStudentInClass maps MariaDB and MySQL 8 duplicate keys ($expected)", async ({ message, expected }) => {
  const originalQuery = connection.query;
  connection.query = async (sql: string) => {
    if (sql.includes("FROM classes")) return [[{ teacher_id: 10 }]];
    if (sql.startsWith("INSERT INTO users")) {
      throw Object.assign(new Error(message), { code: "ER_DUP_ENTRY" });
    }
    throw new Error(`Unexpected SQL: ${sql}`);
  };
  try {
    const result = await createStudentInClass(2, 10, "Teacher", {
      username: "20240001",
      full_name: "测试学生",
      email: "a@b.edu",
      phone_number: "13800138000",
    });
    expect(result).toEqual({ error: expected });
  } finally {
    connection.query = originalQuery;
  }
});
