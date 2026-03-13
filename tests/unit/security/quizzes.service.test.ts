import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const quizzesServiceModulePath = path.join(srcRoot, "modules", "quizzes", "quizzes.service.ts");
const freshQuizzesServiceModulePath = `${quizzesServiceModulePath}?quizzes-service-test`;

const connectionQuery = mock(async () => [[]]);
const beginTransaction = mock(async () => {});
const commit = mock(async () => {});
const rollback = mock(async () => {});
const release = mock(() => {});
const getConnection = mock(async () => ({
  query: connectionQuery,
  beginTransaction,
  commit,
  rollback,
  release,
}));

mock.module(configModulePath, () => ({
  pool: {
    getConnection,
    query: mock(async () => [[]]),
  },
}));

const { submitAnswers } = await import(freshQuizzesServiceModulePath);

describe("quizzes service", () => {
  beforeEach(() => {
    connectionQuery.mockReset();
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    release.mockClear();
    getConnection.mockClear();
  });

  test("blocks quiz submissions for completed experiments after locking the experiment row", async () => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT student_id, status FROM experiment_status")) {
        return [[{ student_id: 20240002, status: "Completed" }]];
      }

      throw new Error(`Unexpected SQL in completed-experiment test: ${sql}`);
    });

    const result = await submitAnswers(990002, 20240002, [
      { question_id: 1, submitted_answer: ["A"] },
    ]);

    expect(result).toEqual({
      error: "已完成的实验不允许继续提交测试答案",
      status: 400,
    });
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledTimes(1);
    expect(connectionQuery.mock.calls[0]?.[0]).toBe("SET TRANSACTION ISOLATION LEVEL READ COMMITTED");
  });

  test("rejects duplicate question ids before opening a transaction", async () => {
    const result = await submitAnswers(990003, 20240003, [
      { question_id: 1, submitted_answer: ["A"] },
      { question_id: 1, submitted_answer: ["B"] },
    ]);

    expect(result).toEqual({
      error: "测验提交包含重复题目",
      status: 400,
    });
    expect(getConnection).not.toHaveBeenCalled();
    expect(beginTransaction).not.toHaveBeenCalled();
  });

  test("retries the transaction when MySQL reports a deadlock", async () => {
    let insertAttempt = 0;

    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT student_id, status FROM experiment_status")) {
        return [[{ student_id: 20240004, status: "In Progress" }]];
      }

      if (sql.includes("FROM question_bank")) {
        return [[
          {
            question_id: 11,
            correct_answers: JSON.stringify(["A"]),
            knowledge_point: "模型基础",
          },
        ]];
      }

      if (sql.includes("DELETE FROM quiz_answers")) {
        return [[]];
      }

      if (sql.includes("INSERT INTO quiz_answers")) {
        insertAttempt += 1;
        if (insertAttempt === 1) {
          const error = new Error("Deadlock found when trying to get lock; try restarting transaction") as Error & {
            code?: string;
            errno?: number;
            sqlState?: string;
          };
          error.code = "ER_LOCK_DEADLOCK";
          error.errno = 1213;
          error.sqlState = "40001";
          throw error;
        }
        return [[]];
      }

      if (sql.includes("SUM(is_correct) AS correct_count")) {
        return [[{ correct_count: 1, total_count: 1 }]];
      }

      if (sql.includes("UPDATE experiment_grade")) {
        return [[]];
      }

      if (sql.includes("UPDATE experiment_status")) {
        return [[]];
      }

      throw new Error(`Unexpected SQL in deadlock-retry test: ${sql}`);
    });

    const result = await submitAnswers(990004, 20240004, [
      { question_id: 11, submitted_answer: ["A"] },
    ]);

    expect(result).toEqual({
      results: [
        {
          question_id: 11,
          is_correct: true,
          correct_answers: ["A"],
        },
      ],
    });
    expect(beginTransaction).toHaveBeenCalledTimes(2);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(release).toHaveBeenCalledTimes(2);
    expect(insertAttempt).toBe(2);
  });
});
