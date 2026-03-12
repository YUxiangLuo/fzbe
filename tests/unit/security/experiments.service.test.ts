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

const {
  createExperimentEvent,
  updateExperiment,
  normalizeExperimentRow,
} = await import(freshExperimentsServiceModulePath);

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

  test("blocks step-event writes for completed experiments after locking the experiment row", async () => {
    connectionQuery.mockResolvedValueOnce([
      [{ student_id: 20240002, status: "Completed" }],
    ]);

    const result = await createExperimentEvent(990002, 20240002, {
      step_order: 5,
      event_type: "STARTED",
    });

    expect(result).toEqual({ error: "completed" });
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledTimes(1);
  });

  test("blocks experiment updates for completed experiments after locking the experiment row", async () => {
    connectionQuery.mockResolvedValueOnce([
      [{
        experiment_id: 990002,
        student_id: 20240002,
        status: "Completed",
        start_time: "2026-03-08T09:00:00.000Z",
        last_activity_at: "2026-03-08T09:30:00.000Z",
        total_active_duration_seconds: 120,
      }],
    ]);

    const result = await updateExperiment(990002, 20240002, {
      selected_company: "企业A",
    });

    expect(result).toEqual({ error: "completed" });
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledTimes(1);
  });

  test("normalizes legacy model identifiers in experiment rows", () => {
    const normalized = normalizeExperimentRow({
      selected_base_models: JSON.stringify(["ma", "exp", "arima", "lstm", "ma"]),
      selected_ensemble_models: JSON.stringify([
        "ensemble_weighted",
        "boosting_ensemble",
        "stacking",
        "weighted_avg",
      ]),
      ensemble_weighted_base_models: JSON.stringify(["ma", "es"]),
      ensemble_boosting_base_models: JSON.stringify(["exp", "arima"]),
      ensemble_stacking_base_models: JSON.stringify(["moving_average", "lstm"]),
    } as any);

    expect(normalized.selected_base_models).toEqual([
      "moving_average",
      "exponential_smoothing",
      "arima",
      "lstm",
    ]);
    expect(normalized.selected_ensemble_models).toEqual([
      "weighted_ensemble",
      "boosting_ensemble",
      "stacking_ensemble",
    ]);
    expect(normalized.ensemble_weighted_base_models).toEqual([
      "moving_average",
      "exponential_smoothing",
    ]);
    expect(normalized.ensemble_boosting_base_models).toEqual([
      "exponential_smoothing",
      "arima",
    ]);
    expect(normalized.ensemble_stacking_base_models).toEqual([
      "moving_average",
      "lstm",
    ]);
  });

  test("canonicalizes legacy model identifiers before saving experiment updates", async () => {
    connectionQuery
      .mockResolvedValueOnce([
        [{
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          start_time: "2026-03-08T09:00:00.000Z",
          last_activity_at: "2026-03-08T09:30:00.000Z",
          total_active_duration_seconds: 120,
        }],
      ])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([
        [{
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          selected_base_models: JSON.stringify(["moving_average", "lstm"]),
          selected_ensemble_models: JSON.stringify(["weighted_ensemble"]),
          ensemble_weighted_base_models: JSON.stringify([
            "moving_average",
            "exponential_smoothing",
          ]),
        }],
      ]);

    const result = await updateExperiment(990002, 20240002, {
      selected_base_models: ["ma", "lstm"],
      selected_ensemble_models: ["ensemble_weighted"],
      ensemble_weighted_base_models: ["ma", "es"],
    });

    const updateCall = connectionQuery.mock.calls[1];
    expect(updateCall?.[0]).toContain("UPDATE experiment_status SET");
    expect(updateCall?.[1]).toEqual(expect.arrayContaining([
      JSON.stringify(["moving_average", "lstm"]),
      JSON.stringify(["weighted_ensemble"]),
      JSON.stringify(["moving_average", "exponential_smoothing"]),
    ]));
    expect(result).toEqual({
      data: expect.objectContaining({
        selected_base_models: ["moving_average", "lstm"],
        selected_ensemble_models: ["weighted_ensemble"],
        ensemble_weighted_base_models: ["moving_average", "exponential_smoothing"],
      }),
    });
  });
});