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

  test("rejects empty submissions before opening a transaction", async () => {
    const result = await submitAnswers(990001, 20240001, []);

    expect(result).toEqual({
      error: "未提供答案",
      status: 400,
    });
    expect(getConnection).not.toHaveBeenCalled();
  });

  test("returns not_found when the experiment row does not exist", async () => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT student_id, status FROM experiment_status")) {
        return [[]];
      }

      throw new Error(`Unexpected SQL in missing-experiment test: ${sql}`);
    });

    const result = await submitAnswers(990005, 20240005, [
      { question_id: 1, submitted_answer: ["A"] },
    ]);

    expect(result).toEqual({
      error: "实验不存在",
      status: 404,
    });
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
  });

  test("returns forbidden when the experiment belongs to another student", async () => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT student_id, status FROM experiment_status")) {
        return [[{ student_id: 20249999, status: "In Progress" }]];
      }

      throw new Error(`Unexpected SQL in forbidden-owner test: ${sql}`);
    });

    const result = await submitAnswers(990006, 20240006, [
      { question_id: 1, submitted_answer: ["A"] },
    ]);

    expect(result).toEqual({
      error: "权限不足：该实验不属于当前用户",
      status: 403,
    });
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
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

  test("updates knowledge score and completion flags for mixed model and plan submissions", async () => {
    connectionQuery.mockImplementation(async (sql: string, params?: unknown[]) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT student_id, status FROM experiment_status")) {
        return [[{ student_id: 20240007, status: "Not Started" }]];
      }

      if (sql.includes("SELECT question_id, correct_answers, knowledge_point")) {
        return [[
          {
            question_id: 101,
            correct_answers: JSON.stringify(["A"]),
            knowledge_point: "预测模型-ARIMA模型",
          },
          {
            question_id: 202,
            correct_answers: JSON.stringify(["B"]),
            knowledge_point: "生产计划-MPS基本原理",
          },
        ]];
      }

      if (sql.includes("COUNT(*) AS available_count")) {
        return [[{ available_count: 1 }]];
      }

      if (sql.includes("DELETE FROM quiz_answers")) {
        expect(params).toEqual([990007, 20240007, "quiz_about_model", "quiz_about_plan"]);
        return [[]];
      }

      if (sql.includes("INSERT INTO quiz_answers")) {
        return [[]];
      }

      if (sql.includes("SUM(is_correct) AS correct_count")) {
        return [[{ correct_count: 1, total_count: 2 }]];
      }

      if (sql.includes("UPDATE experiment_grade")) {
        expect(params).toEqual([50, 990007, 20240007]);
        return [[]];
      }

      if (sql.includes("UPDATE experiment_status")) {
        expect(params).toEqual([1, 1, 990007, 20240007]);
        return [[]];
      }

      throw new Error(`Unexpected SQL in mixed-submission test: ${sql}`);
    });

    const result = await submitAnswers(990007, 20240007, [
      { question_id: 101, submitted_answer: ["A"] },
      { question_id: 202, submitted_answer: ["A"] },
    ]);

    expect(result).toEqual({
      results: [
        {
          question_id: 101,
          is_correct: true,
        },
        {
          question_id: 202,
          is_correct: false,
        },
      ],
    });
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
  });

  test("grades multiple-choice answers correctly regardless of submitted order", async () => {
    connectionQuery.mockImplementation(async (sql: string, params?: unknown[]) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT student_id, status FROM experiment_status")) {
        return [[{ student_id: 20240008, status: "In Progress" }]];
      }

      if (sql.includes("SELECT question_id, correct_answers, knowledge_point")) {
        return [[
          {
            question_id: 301,
            correct_answers: JSON.stringify(["A", "C"]),
            knowledge_point: "预测模型-模型评估指标",
          },
        ]];
      }

      if (sql.includes("COUNT(*) AS available_count")) {
        return [[{ available_count: 1 }]];
      }

      if (sql.includes("DELETE FROM quiz_answers")) {
        expect(params).toEqual([990008, 20240008, "quiz_about_model"]);
        return [[]];
      }

      if (sql.includes("INSERT INTO quiz_answers")) {
        expect(params).toBeDefined();
        return [[]];
      }

      if (sql.includes("SUM(is_correct) AS correct_count")) {
        return [[{ correct_count: 1, total_count: 1 }]];
      }

      if (sql.includes("UPDATE experiment_grade")) {
        expect(params).toEqual([100, 990008, 20240008]);
        return [[]];
      }

      if (sql.includes("UPDATE experiment_status")) {
        expect(params).toEqual([1, 0, 990008, 20240008]);
        return [[]];
      }

      throw new Error(`Unexpected SQL in unordered-multi-choice test: ${sql}`);
    });

    const result = await submitAnswers(990008, 20240008, [
      { question_id: 301, submitted_answer: ["C", "A"] },
    ]);

    expect(result).toEqual({
      results: [
        {
          question_id: 301,
          is_correct: true,
        },
      ],
    });
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
  });

  test("marks answers incorrect when a multiple-choice submission is missing a correct option", async () => {
    connectionQuery.mockImplementation(async (sql: string, params?: unknown[]) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT student_id, status FROM experiment_status")) {
        return [[{ student_id: 20240010, status: "In Progress" }]];
      }

      if (sql.includes("SELECT question_id, correct_answers, knowledge_point")) {
        return [[
          {
            question_id: 501,
            correct_answers: JSON.stringify(["A", "C"]),
            knowledge_point: "预测模型-模型评估指标",
          },
        ]];
      }

      if (sql.includes("COUNT(*) AS available_count")) {
        return [[{ available_count: 1 }]];
      }

      if (sql.includes("DELETE FROM quiz_answers")) {
        expect(params).toEqual([990010, 20240010, "quiz_about_model"]);
        return [[]];
      }

      if (sql.includes("INSERT INTO quiz_answers")) {
        return [[]];
      }

      if (sql.includes("SUM(is_correct) AS correct_count")) {
        return [[{ correct_count: 0, total_count: 1 }]];
      }

      if (sql.includes("UPDATE experiment_grade")) {
        expect(params).toEqual([0, 990010, 20240010]);
        return [[]];
      }

      if (sql.includes("UPDATE experiment_status")) {
        expect(params).toEqual([1, 0, 990010, 20240010]);
        return [[]];
      }

      throw new Error(`Unexpected SQL in missing-choice grading test: ${sql}`);
    });

    const result = await submitAnswers(990010, 20240010, [
      { question_id: 501, submitted_answer: ["A"] },
    ]);

    expect(result).toEqual({
      results: [
        {
          question_id: 501,
          is_correct: false,
        },
      ],
    });
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
  });

  test("marks answers incorrect when a multiple-choice submission includes extra options", async () => {
    connectionQuery.mockImplementation(async (sql: string, params?: unknown[]) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT student_id, status FROM experiment_status")) {
        return [[{ student_id: 20240011, status: "In Progress" }]];
      }

      if (sql.includes("SELECT question_id, correct_answers, knowledge_point")) {
        return [[
          {
            question_id: 601,
            correct_answers: JSON.stringify(["B"]),
            knowledge_point: "生产计划-库存与缺货",
          },
        ]];
      }

      if (sql.includes("COUNT(*) AS available_count")) {
        return [[{ available_count: 1 }]];
      }

      if (sql.includes("DELETE FROM quiz_answers")) {
        expect(params).toEqual([990011, 20240011, "quiz_about_plan"]);
        return [[]];
      }

      if (sql.includes("INSERT INTO quiz_answers")) {
        return [[]];
      }

      if (sql.includes("SUM(is_correct) AS correct_count")) {
        return [[{ correct_count: 0, total_count: 1 }]];
      }

      if (sql.includes("UPDATE experiment_grade")) {
        expect(params).toEqual([0, 990011, 20240011]);
        return [[]];
      }

      if (sql.includes("UPDATE experiment_status")) {
        expect(params).toEqual([0, 1, 990011, 20240011]);
        return [[]];
      }

      throw new Error(`Unexpected SQL in extra-choice grading test: ${sql}`);
    });

    const result = await submitAnswers(990011, 20240011, [
      { question_id: 601, submitted_answer: ["B", "D"] },
    ]);

    expect(result).toEqual({
      results: [
        {
          question_id: 601,
          is_correct: false,
        },
      ],
    });
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
  });

  test("rounds knowledge score to two decimals when only part of the quiz is correct", async () => {
    connectionQuery.mockImplementation(async (sql: string, params?: unknown[]) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT student_id, status FROM experiment_status")) {
        return [[{ student_id: 20240009, status: "In Progress" }]];
      }

      if (sql.includes("SELECT question_id, correct_answers, knowledge_point")) {
        return [[
          {
            question_id: 401,
            correct_answers: JSON.stringify(["A"]),
            knowledge_point: "预测模型-ARIMA模型",
          },
          {
            question_id: 402,
            correct_answers: JSON.stringify(["B"]),
            knowledge_point: "预测模型-LSTM深度学习",
          },
          {
            question_id: 403,
            correct_answers: JSON.stringify(["C"]),
            knowledge_point: "预测模型-模型融合技术",
          },
        ]];
      }

      if (sql.includes("COUNT(*) AS available_count")) {
        return [[{ available_count: 3 }]];
      }

      if (sql.includes("DELETE FROM quiz_answers")) {
        expect(params).toEqual([990009, 20240009, "quiz_about_model"]);
        return [[]];
      }

      if (sql.includes("INSERT INTO quiz_answers")) {
        return [[]];
      }

      if (sql.includes("SUM(is_correct) AS correct_count")) {
        return [[{ correct_count: 2, total_count: 3 }]];
      }

      if (sql.includes("UPDATE experiment_grade")) {
        expect(params).toEqual([66.67, 990009, 20240009]);
        return [[]];
      }

      if (sql.includes("UPDATE experiment_status")) {
        expect(params).toEqual([1, 0, 990009, 20240009]);
        return [[]];
      }

      throw new Error(`Unexpected SQL in knowledge-score-rounding test: ${sql}`);
    });

    const result = await submitAnswers(990009, 20240009, [
      { question_id: 401, submitted_answer: ["A"] },
      { question_id: 402, submitted_answer: ["B"] },
      { question_id: 403, submitted_answer: ["D"] },
    ]);

    expect(result).toEqual({
      results: [
        {
          question_id: 401,
          is_correct: true,
        },
        {
          question_id: 402,
          is_correct: true,
        },
        {
          question_id: 403,
          is_correct: false,
        },
      ],
    });
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
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

  test("rejects unknown question ids instead of silently accepting empty results", async () => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT student_id, status FROM experiment_status")) {
        return [[{ student_id: 20240012, status: "In Progress" }]];
      }

      if (sql.includes("SELECT question_id, correct_answers, knowledge_point")) {
        return [[]];
      }

      throw new Error(`Unexpected SQL in unknown-question test: ${sql}`);
    });

    const result = await submitAnswers(990012, 20240012, [
      { question_id: 999999, submitted_answer: ["A"] },
    ]);

    expect(result).toEqual({
      error: "测验题目不存在或不可用于提交",
      status: 400,
    });
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
  });

  test("rejects incomplete quiz submissions when enough questions are available", async () => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT student_id, status FROM experiment_status")) {
        return [[{ student_id: 20240013, status: "In Progress" }]];
      }

      if (sql.includes("SELECT question_id, correct_answers, knowledge_point")) {
        return [[
          {
            question_id: 701,
            correct_answers: JSON.stringify(["A"]),
            knowledge_point: "预测模型-时间序列基础",
          },
        ]];
      }

      if (sql.includes("COUNT(*) AS available_count")) {
        return [[{ available_count: 8 }]];
      }

      throw new Error(`Unexpected SQL in incomplete-quiz test: ${sql}`);
    });

    const result = await submitAnswers(990013, 20240013, [
      { question_id: 701, submitted_answer: ["A"] },
    ]);

    expect(result).toEqual({
      error: "预测模型测验必须一次提交 4 道题",
      status: 400,
    });
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
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

      if (sql.includes("SELECT question_id, correct_answers, knowledge_point")) {
        return [[
          {
            question_id: 11,
            correct_answers: JSON.stringify(["A"]),
            knowledge_point: "模型基础",
          },
        ]];
      }

      if (sql.includes("COUNT(*) AS available_count")) {
        return [[{ available_count: 1 }]];
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
