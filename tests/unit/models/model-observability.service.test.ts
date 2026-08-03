import { beforeEach, describe, expect, mock, test } from "bun:test";

process.env.DB_HOST ??= "127.0.0.1";
process.env.DB_USER ??= "test";
process.env.DB_PASSWORD ??= "test";
process.env.DB_DATABASE ??= "test";
process.env.JWT_SECRET ??= "test-jwt-secret";

let ownershipRows: Array<{ student_id: number; status?: string }> = [];
let activeRunRows: Array<Record<string, unknown>> = [];

const poolQueryMock = mock(async (sql: string) => {
  if (sql.includes("FROM experiment_status")) {
    return [ownershipRows, []];
  }
  if (sql.includes("FROM model_training_runs")) {
    return [activeRunRows, []];
  }
  return [[], []];
});

mock.module("../../../src/config", () => ({
  pool: { query: poolQueryMock },
}));

const {
  buildModelObservabilityMap,
  getExperimentModelObservability,
} = await import(
  "../../../src/modules/models/model-observability.service"
);

describe("model observability response", () => {
  beforeEach(() => {
    poolQueryMock.mockClear();
    ownershipRows = [{ student_id: 20240002, status: "Completed" }];
    activeRunRows = [{
      run_id: "run-ma",
      model_type: "ma",
      parameters: {
        observability: {
          method_name: "移动平均",
          uncertainty_evidence: { sources: ["empirical"] },
        },
      },
      metrics: {},
    }];
  });

  test("returns only active-run observability and recovers interval metrics", () => {
    const result = buildModelObservabilityMap([
      {
        run_id: "weighted-run",
        model_type: "weighted_avg",
        parameters: JSON.stringify({
          observability: {
            method_name: "加权平均",
            weights: [0.3, 0.7],
            outer_interval_evaluation: null,
          },
        }),
        metrics: {
          interval_evaluation: {
            scope: "outer_evaluation_descriptive",
            mean_width: 4.2,
          },
        },
      },
      {
        run_id: "ma-run",
        model_type: "ma",
        parameters: { moving_average_window: 3 },
        metrics: {},
      },
    ] as any);

    expect(result).toEqual({
      weighted_avg: {
        method_name: "加权平均",
        weights: [0.3, 0.7],
        outer_interval_evaluation: {
          scope: "outer_evaluation_descriptive",
          mean_width: 4.2,
        },
      },
    });
  });

  test("returns the persisted LSTM data contract needed by the student summary", () => {
    const lstmObservability = {
      method_name: "LSTM 历史多特征销量预测",
      forecast_strategy: "direct_multi_horizon",
      target_key: "销售数量",
      feature_keys: ["价格指数", "渠道"],
      feature_types: {
        销售数量: "numeric",
        价格指数: "numeric",
        渠道: "categorical",
      },
      normalization: "zscore",
      forecast_horizon: 6,
      training_regime: "standard",
      training_metadata: {
        look_back: 4,
        feature_keys: ["价格指数", "渠道"],
        seed_runs: 3,
      },
    };
    const result = buildModelObservabilityMap([{
      run_id: "lstm-run",
      model_type: "lstm",
      parameters: JSON.stringify({ observability: lstmObservability }),
      metrics: JSON.stringify({}),
    }] as any);

    expect(result).toEqual({ lstm: lstmObservability });
  });

  test("skips malformed and empty observability without leaking raw values", () => {
    const result = buildModelObservabilityMap([
      {
        run_id: "malformed-parameters",
        model_type: "ma",
        parameters: "{not-json",
        metrics: {},
      },
      {
        run_id: "malformed-observability",
        model_type: "es",
        parameters: JSON.stringify({ observability: "{not-json" }),
        metrics: {},
      },
      {
        run_id: "empty-observability",
        model_type: "arima",
        parameters: { observability: {} },
        metrics: {},
      },
      {
        run_id: "valid-observability",
        model_type: "lstm",
        parameters: JSON.stringify({
          observability: JSON.stringify({ method_name: "LSTM 模型" }),
        }),
        metrics: "{not-json",
      },
    ] as any);

    expect(result).toEqual({
      lstm: { method_name: "LSTM 模型" },
    });
    expect(JSON.stringify(result)).not.toContain("not-json");
  });

  test("deterministically keeps the last ordered active run for each model", () => {
    const result = buildModelObservabilityMap([
      {
        run_id: "run-a",
        model_type: "ma",
        parameters: { observability: { method_name: "旧运行" } },
        metrics: {},
      },
      {
        run_id: "run-b",
        model_type: "ma",
        parameters: { observability: { method_name: "新运行" } },
        metrics: {},
      },
    ] as any);

    expect(result).toEqual({
      ma: { method_name: "新运行" },
    });
  });

  test("allows an owner to read observability after the experiment is completed", async () => {
    await expect(
      getExperimentModelObservability(990002, 20240002),
    ).resolves.toEqual({
      models: {
        ma: {
          method_name: "移动平均",
          uncertainty_evidence: { sources: ["empirical"] },
        },
      },
    });
    expect(poolQueryMock.mock.calls).toHaveLength(2);
    expect(String(poolQueryMock.mock.calls[0]?.[0])).toContain("FROM experiment_status");
    expect(String(poolQueryMock.mock.calls[1]?.[0])).toContain("FROM model_training_runs");
    expect(String(poolQueryMock.mock.calls[1]?.[0])).toContain("is_active = true");
    expect(String(poolQueryMock.mock.calls[1]?.[0])).toContain(
      "ORDER BY activated_at ASC, run_id ASC",
    );
  });

  test("rejects a missing experiment before reading active runs", async () => {
    ownershipRows = [];

    await expect(
      getExperimentModelObservability(990002, 20240002),
    ).rejects.toMatchObject({
      message: "实验不存在",
      status: 404,
    });
    expect(poolQueryMock.mock.calls).toHaveLength(1);
  });

  test("rejects cross-student access before reading active runs", async () => {
    ownershipRows = [{ student_id: 20249999 }];

    await expect(
      getExperimentModelObservability(990002, 20240002),
    ).rejects.toMatchObject({
      message: "权限不足：该实验不属于当前用户",
      status: 403,
    });
    expect(poolQueryMock.mock.calls).toHaveLength(1);
  });
});
