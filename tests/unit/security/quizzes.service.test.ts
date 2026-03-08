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
    connectionQuery.mockResolvedValueOnce([
      [{ student_id: 20240002, status: "Completed" }],
    ]);

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
  });
});