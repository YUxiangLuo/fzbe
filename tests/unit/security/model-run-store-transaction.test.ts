import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";
import type { TrainInput } from "../../../src/modules/models/models.schema";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const storeModulePath = path.join(srcRoot, "modules", "models", "model-run-store.ts");
const freshStoreModulePath = `${storeModulePath}?transaction-test`;

const query = mock(async () => [[]]);
const poolQuery = mock(async () => [[]]);
const beginTransaction = mock(async () => {});
const commit = mock(async () => {});
const rollback = mock(async () => {});
const release = mock(() => {});
const connection = { query, beginTransaction, commit, rollback, release };

mock.module(configModulePath, () => ({
  pool: {
    getConnection: mock(async () => connection),
    query: poolQuery,
  },
}));

const {
  buildExperimentTrainingRevision,
  buildModelTrainingRequestHash,
  commitModelTrainingRun,
  loadCurrentExperimentStatePatch,
} = await import(freshStoreModulePath);

const baseExperiment = {
  experiment_id: 42,
  student_id: 20240002,
  status: "In Progress",
  state_version: 4,
  selected_industry: "制造业",
  selected_company: "示例企业",
  selected_product: "示例产品",
  data_window_train_start_index: 0,
  data_window_train_end_index: 11,
  data_window_evaluate_start_index: 12,
  data_window_evaluate_end_index: 17,
  moving_average_window: 6,
  exponential_smoothing_alpha: 0.3,
  ensemble_weighted_base_models: JSON.stringify(["moving_average", "exponential_smoothing"]),
};

const baseBody = {
  experiment_id: 42,
  selected_industry: "制造业",
  selected_company: "示例企业",
  selected_product: "示例产品",
  data_window_train_start_index: 0,
  data_window_train_end_index: 11,
  data_window_evaluate_start_index: 12,
  data_window_evaluate_end_index: 17,
} satisfies TrainInput;

const successResult = () => ({
  status: "success",
  results: {
    metrics: { rmse: 1.2, mae: 0.8, r2: 0.91 },
  },
});

describe("model run transactional activation", () => {
  beforeEach(() => {
    query.mockReset();
    poolQuery.mockReset();
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    release.mockClear();
  });

  test("loads a complete current experiment patch without rewriting the model run", async () => {
    poolQuery.mockResolvedValueOnce([[
      {
        ...baseExperiment,
        state_version: 11,
        moving_average_completed: 1,
        ensemble_weighted_completed: 1,
        ensemble_weighted_base_models: JSON.stringify(["moving_average", "exponential_smoothing"]),
        selected_best_model: "ensemble_weighted",
        production_plan_completed: 1,
      },
    ]]);

    const patch = await loadCurrentExperimentStatePatch(42, 20240002);

    expect(patch).toMatchObject({
      state_version: 11,
      moving_average_completed: true,
      ensemble_weighted_completed: true,
      ensemble_weighted_base_models: ["moving_average", "exponential_smoothing"],
      selected_best_model: "ensemble_weighted",
      production_plan_completed: true,
    });
    expect(poolQuery).toHaveBeenCalledWith(
      expect.stringContaining("FROM experiment_status"),
      [42, 20240002],
    );
    expect(beginTransaction).not.toHaveBeenCalled();
  });

  test("activates the immutable run and authoritative experiment patch in one transaction", async () => {
    const body: TrainInput = { ...baseBody, moving_average_window: 6 };
    const requestHash = buildModelTrainingRequestHash({
      modelType: "ma",
      body,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      experimentRevision: buildExperimentTrainingRevision("ma", baseExperiment, body),
      baseRunFingerprint: {},
    });
    query.mockImplementation(async (sql: unknown) => {
      const statement = String(sql);
      if (statement.includes("SELECT *") && statement.includes("FROM experiment_status")) {
        return [[baseExperiment]];
      }
      if (statement.includes("SELECT state_version")) {
        return [[{ state_version: 5 }]];
      }
      return [{ affectedRows: 1 }];
    });
    const resultPayload = successResult();

    const committed = await commitModelTrainingRun({
      runId: "run-ma-v1",
      sessionId: null,
      experimentId: 42,
      studentId: 20240002,
      modelType: "ma",
      requestHash,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      body,
      result: resultPayload,
      artifactPath: "/models/run-ma-v1.pkl",
    });

    expect(committed).toEqual({
      runId: "run-ma-v1",
      stateVersion: 5,
      reusedExistingRun: false,
    });
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledTimes(1);
    expect(query.mock.calls.some(([sql]) => String(sql).includes("INSERT INTO model_training_runs"))).toBe(true);
    expect(query.mock.calls.some(([sql]) => String(sql).includes("UPDATE experiment_status"))).toBe(true);
    expect(resultPayload.experiment_state_patch).toMatchObject({
      moving_average_completed: true,
      moving_average_window: 6,
      moving_average_metrics_rmse: 1.2,
      state_version: 5,
    });
  });

  test("returns the current authoritative experiment state when a completed session reuses its active run", async () => {
    const experiment = {
      ...baseExperiment,
      moving_average_completed: 1,
      moving_average_metrics_rmse: 1.2,
      moving_average_metrics_mae: 0.8,
      moving_average_metrics_r2: 0.91,
      ensemble_weighted_completed: 0,
      ensemble_weighted_base_models: null,
      selected_best_model: null,
      production_plan_completed: 0,
      production_mps_table: null,
    };
    query.mockImplementation(async (sql: unknown) => {
      const statement = String(sql);
      if (statement.includes("SELECT *") && statement.includes("FROM experiment_status")) {
        return [[experiment]];
      }
      if (statement.includes("SELECT *") && statement.includes("FROM model_training_runs")) {
        return [[{ run_id: "run-ma-v1", is_active: 1 }]];
      }
      return [{ affectedRows: 1 }];
    });
    const resultPayload = successResult();

    const committed = await commitModelTrainingRun({
      runId: "run-ma-v1",
      sessionId: "guided-ma-v1",
      experimentId: 42,
      studentId: 20240002,
      modelType: "ma",
      requestHash: "request-v1",
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      body: { ...baseBody, moving_average_window: 6 },
      result: resultPayload,
      artifactPath: "/models/run-ma-v1.pkl",
      guidedSession: {
        sessionId: "guided-ma-v1",
        currentStepId: null,
        completedStepIds: ["save_artifact"],
        stepOutputs: {},
        stepPlan: [],
        result: resultPayload,
      },
    });

    expect(committed).toEqual({
      runId: "run-ma-v1",
      stateVersion: 4,
      reusedExistingRun: true,
    });
    expect(resultPayload.experiment_state_patch).toMatchObject({
      moving_average_completed: true,
      moving_average_window: 6,
      moving_average_metrics_rmse: 1.2,
      moving_average_metrics_mae: 0.8,
      moving_average_metrics_r2: 0.91,
      ensemble_weighted_completed: false,
      ensemble_weighted_base_models: [],
      selected_best_model: null,
      production_plan_completed: false,
      production_mps_table: [],
      state_version: 4,
    });
    expect(query.mock.calls.some(([sql]) => String(sql).includes("UPDATE experiment_status"))).toBe(false);

    const sessionUpdate = query.mock.calls.find(([sql]) =>
      String(sql).includes("UPDATE model_guided_training_sessions")
    );
    expect(sessionUpdate).toBeDefined();
    const sessionUpdateValues = sessionUpdate?.[1] as unknown[];
    expect(JSON.parse(String(sessionUpdateValues[1])).experiment_state_patch).toEqual(
      resultPayload.experiment_state_patch,
    );
  });

  test("rolls back an ensemble commit if a base run changes during training", async () => {
    const body: TrainInput = { ...baseBody, models: "ma,es" };
    const requestHash = buildModelTrainingRequestHash({
      modelType: "weighted_avg",
      body,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      experimentRevision: buildExperimentTrainingRevision("weighted_avg", baseExperiment, body),
      baseRunFingerprint: { ma: "ma-run-old", es: "es-run-1" },
    });
    query.mockImplementation(async (sql: unknown, values?: unknown[]) => {
      const statement = String(sql);
      if (statement.includes("SELECT *") && statement.includes("FROM experiment_status")) {
        return [[baseExperiment]];
      }
      if (statement.includes("SELECT run_id") && statement.includes("model_type = ?")) {
        const modelType = values?.[2];
        return [[{ run_id: modelType === "ma" ? "ma-run-new" : "es-run-1" }]];
      }
      return [[]];
    });

    await expect(commitModelTrainingRun({
      runId: "ensemble-run-1",
      sessionId: null,
      experimentId: 42,
      studentId: 20240002,
      modelType: "weighted_avg",
      requestHash,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      body,
      result: successResult(),
      artifactPath: "/models/ensemble-run-1.pkl",
    })).rejects.toMatchObject({
      message: "训练期间实验配置或基础模型版本已变化，请重新创建训练会话",
    });

    expect(commit).not.toHaveBeenCalled();
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(query.mock.calls.some(([sql]) => String(sql).includes("INSERT INTO model_training_runs"))).toBe(false);
    expect(query.mock.calls.some(([sql]) => String(sql).includes("UPDATE experiment_status"))).toBe(false);
  });
});
