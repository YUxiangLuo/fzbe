import path from "node:path";
import { beforeEach, describe, expect, mock, test } from "bun:test";

const query = mock(async (_sql: string, _values?: unknown[]): Promise<unknown[]> => [[]]);
const connection = {
  query, beginTransaction: mock(async () => {}), commit: mock(async () => {}),
  rollback: mock(async () => {}), release: mock(() => {}),
};
const poolQuery = mock(async () => { throw new Error("Use the transaction connection"); });
mock.module(path.resolve(import.meta.dir, "../../../src/config/index.ts"), () => ({
  env: { DB_DATABASE: "weights_unit", DB_HOST: "127.0.0.1", DB_TIMEZONE: "Z" },
  pool: { query: poolQuery, getConnection: mock(async () => connection) },
}));
const { updateGradingPolicy } = await import("../../../src/modules/class-grades/class-grades.service");
const { EXP_FLOW_FIELDS } = await import("../../../src/modules/class-grades/class-grades.schema");
const weights = {
  ...Object.fromEntries(EXP_FLOW_FIELDS.map((field, index) => [field, index === 0 ? 100 : 0])),
  exp_flow_weight: 40, knowledge_test_weight: 20, model_quality_weight: 20, report_quality_weight: 20,
};

describe("transactional grading policy updates", () => {
  beforeEach(() => {
    query.mockReset(); connection.beginTransaction.mockReset().mockResolvedValue(undefined); connection.commit.mockReset().mockResolvedValue(undefined);
    connection.rollback.mockReset().mockResolvedValue(undefined); connection.release.mockClear(); poolQuery.mockClear();
  });
  test("validates a locked snapshot and returns its own committed update", async () => {
    const updated = { ...weights, exp_flow_weight: 50, knowledge_test_weight: 10 };
    query.mockResolvedValueOnce([[weights]]).mockResolvedValueOnce([{ affectedRows: 1 }]).mockResolvedValueOnce([[updated]]);
    expect(await updateGradingPolicy(1, { exp_flow_weight: 50, knowledge_test_weight: 10 })).toEqual({ data: updated });
    expect(query.mock.calls[0]).toEqual(["SELECT * FROM grade_weights WHERE class_id = ? FOR UPDATE", [1]]);
    expect(connection.beginTransaction).toHaveBeenCalledTimes(1);
    expect(connection.commit).toHaveBeenCalledTimes(1);
    expect(connection.rollback).not.toHaveBeenCalled();
    expect(connection.release).toHaveBeenCalledTimes(1);
    expect(poolQuery).not.toHaveBeenCalled();
  });
  test("rejects a patch that was valid before another editor committed", async () => {
    query.mockResolvedValueOnce([[{ ...weights, exp_flow_weight: 50, knowledge_test_weight: 10 }]]);
    expect(await updateGradingPolicy(1, { knowledge_test_weight: 30, model_quality_weight: 10 }))
      .toMatchObject({ error: "invalid_weight_total", details: { topLevelTotal: 110 } });
    expect(query).toHaveBeenCalledTimes(1);
    expect(connection.rollback).toHaveBeenCalledTimes(1);
    expect(connection.commit).not.toHaveBeenCalled();
    expect(connection.release).toHaveBeenCalledTimes(1);
  });
  test("rolls back missing policies", async () => {
    query.mockResolvedValueOnce([[]]);
    expect(await updateGradingPolicy(1, { exp_flow_weight: 40 })).toEqual({ error: "not_found" });
    expect(connection.rollback).toHaveBeenCalledTimes(1);
    expect(connection.release).toHaveBeenCalledTimes(1);
  });
  test("preserves write failure when rollback also fails", async () => {
    query.mockResolvedValueOnce([[weights]]).mockRejectedValueOnce(new Error("write failed"));
    connection.rollback.mockRejectedValueOnce(new Error("rollback failed"));
    await expect(updateGradingPolicy(1, { exp_flow_weight: 40 })).rejects.toThrow("write failed");
    expect(connection.commit).not.toHaveBeenCalled();
    expect(connection.release).toHaveBeenCalledTimes(1);
  });
  test("does not report success when commit fails", async () => {
    query.mockResolvedValueOnce([[weights]]).mockResolvedValueOnce([{}]).mockResolvedValueOnce([[weights]]);
    connection.commit.mockRejectedValueOnce(new Error("commit failed"));
    await expect(updateGradingPolicy(1, { exp_flow_weight: 40 })).rejects.toThrow("commit failed");
    expect(connection.rollback).toHaveBeenCalledTimes(1);
    expect(connection.release).toHaveBeenCalledTimes(1);
  });
});
