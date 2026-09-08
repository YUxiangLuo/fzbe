import { afterAll, beforeEach, expect, mock, test } from "bun:test";
import { ModelServiceUnavailableError } from "../../../src/modules/models/model-errors";

const originalTimeout = process.env.MODEL_DB_QUERY_TIMEOUT_MS;
process.env.MODEL_DB_QUERY_TIMEOUT_MS = "25";
afterAll(() => {
  if (originalTimeout === undefined) delete process.env.MODEL_DB_QUERY_TIMEOUT_MS;
  else process.env.MODEL_DB_QUERY_TIMEOUT_MS = originalTimeout;
});

const closedConnectionError = new Error("Can't add new command when connection is in closed state");
const connection = {
  query: mock(async (_sql: string) => [[]]),
  beginTransaction: mock(async () => {}),
  commit: mock(async () => {}),
  rollback: mock(async () => {}),
  destroy: mock(() => {}),
  release: mock(() => {}),
};
const getConnection = mock(async () => connection);
mock.module("../../../src/config", () => ({ pool: { getConnection } }));

// Keep the real acquisition/query/transaction guards: a guard timeout destroys
// the connection before the store tries to roll back, just like mysql2 does.
const { commitModelTrainingRun } = await import("../../../src/modules/models/model-run-store");
const input = () => ({
  runId: "timeout-run", sessionId: null, experimentId: 42, studentId: 1,
  modelType: "ma" as const, requestHash: "not-reached", datasetRevision: "v1", runtimeRevision: "v1",
  body: { experiment_id: 42, moving_average_window: 6 },
  result: { status: "success", results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } } },
  artifactPath: "/models/not-created.pkl",
});

beforeEach(() => {
  connection.query.mockReset().mockResolvedValue([[]]);
  connection.beginTransaction.mockReset().mockResolvedValue(undefined);
  connection.commit.mockClear();
  connection.destroy.mockClear();
  connection.release.mockClear();
  connection.rollback.mockReset().mockImplementation(async () => {
    if (connection.destroy.mock.calls.length > 0) throw closedConnectionError;
  });
  getConnection.mockClear();
});

const expectDisposedWithoutRetry = () => {
  expect(getConnection).toHaveBeenCalledTimes(1);
  expect(connection.rollback).toHaveBeenCalledTimes(1);
  expect(connection.destroy).toHaveBeenCalledTimes(1);
  expect(connection.commit).not.toHaveBeenCalled();
  expect(connection.release).not.toHaveBeenCalled();
};

test.each([
  ["query", "MODEL_DB_QUERY_TIMEOUT"],
  ["begin", "MODEL_DB_TRANSACTION_TIMEOUT"],
] as const)("preserves the typed %s timeout after rollback on the closed connection fails", async (phase, code) => {
  if (phase === "query") {
    connection.query.mockImplementation((sql) => sql.includes("FROM users")
      ? new Promise<never>(() => {})
      : Promise.resolve([[]]));
  } else {
    connection.beginTransaction.mockImplementation(() => new Promise<never>(() => {}));
  }

  const error = await commitModelTrainingRun(input()).catch((caught) => caught);
  expectDisposedWithoutRetry();
  expect(error).toBeInstanceOf(ModelServiceUnavailableError);
  expect(error.code).toBe(code);
});

test.each(["ER_LOCK_DEADLOCK", "ER_LOCK_WAIT_TIMEOUT"])(
  "does not replay %s when rollback cannot be acknowledged", async (code) => {
    const originalError = Object.assign(new Error("retryable SQL failure"), { code });
    connection.query.mockRejectedValueOnce(originalError);
    connection.rollback.mockRejectedValueOnce(closedConnectionError);

    await expect(commitModelTrainingRun(input())).rejects.toMatchObject({
      message: "模型结果事务回滚失败", cause: closedConnectionError,
    });
    expectDisposedWithoutRetry();
  },
);
