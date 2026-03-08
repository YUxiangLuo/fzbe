import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const experimentsServiceModulePath = path.join(srcRoot, "modules", "experiments", "experiments.service.ts");
const freshExperimentsServiceModulePath = `${experimentsServiceModulePath}?experiments-service-test`;

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
const poolQuery = mock(async () => [[]]);

mock.module(configModulePath, () => ({
  pool: {
    getConnection,
    query: poolQuery,
  },
}));

const { createExperimentEvent } = await import(freshExperimentsServiceModulePath);

describe("experiments service", () => {
  beforeEach(() => {
    connectionQuery.mockReset();
    poolQuery.mockReset();
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    release.mockClear();
    getConnection.mockClear();
  });

  test("blocks step-event writes for completed experiments", async () => {
    connectionQuery.mockResolvedValueOnce([
      [{ student_id: 20240002, status: "Completed" }],
    ]);

    const result = await createExperimentEvent(990002, 20240002, {
      step_order: 5,
      event_type: "STARTED",
    });

    expect(result).toEqual({ error: "completed" });
    expect(beginTransaction).not.toHaveBeenCalled();
    expect(commit).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledTimes(1);
  });
});