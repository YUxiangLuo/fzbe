import { describe, expect, test } from "bun:test";
import type { TrainInput } from "../../../src/modules/models/models.schema";
import {
  buildAuthoritativeResult,
  buildCurrentExperimentStatePatch,
  buildExperimentTrainingRevision,
  buildModelTrainingRequestHash,
  getServerInvalidationPatch,
} from "../../../src/modules/models/model-run-store";
import type { GuidedModelType } from "../../../src/modules/models/model-types";

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

const resultFor = (modelType: GuidedModelType) => ({
  status: "success",
  results: {
    metrics: { rmse: 1.2, mae: 0.8, r2: 0.91 },
    ...(modelType === "arima" ? { best_order: { p: 2, d: 1, q: 3 } } : {}),
  },
});

describe("model run authoritative state", () => {
  test("maps each of the seven training results to server-owned experiment fields", () => {
    const cases: Array<{
      modelType: GuidedModelType;
      body: TrainInput;
      completedField: string;
      configAssertions: Record<string, unknown>;
    }> = [
      {
        modelType: "ma",
        body: { ...baseBody, moving_average_window: 6 },
        completedField: "moving_average_completed",
        configAssertions: { moving_average_window: 6 },
      },
      {
        modelType: "es",
        body: { ...baseBody, exponential_smoothing_alpha: 0.3 },
        completedField: "exponential_smoothing_completed",
        configAssertions: { exponential_smoothing_alpha: 0.3 },
      },
      {
        modelType: "arima",
        body: { ...baseBody, arimaD: 1 },
        completedField: "arima_completed",
        configAssertions: { arima_p: 2, arima_d: 1, arima_q: 3 },
      },
      {
        modelType: "lstm",
        body: {
          ...baseBody,
          lstmNormalization: "minmax",
          lstmFeatures: JSON.stringify(["价格指数", "产能利用率"]),
          lstmTargetFeature: "销售数量",
        },
        completedField: "lstm_completed",
        configAssertions: {
          lstm_normalization: "minmax",
          lstm_features: JSON.stringify(["价格指数", "产能利用率"]),
          lstm_target_field: "销售数量",
        },
      },
      {
        modelType: "weighted_avg",
        body: { ...baseBody, models: "ma,es" },
        completedField: "ensemble_weighted_completed",
        configAssertions: {
          ensemble_weighted_base_models: JSON.stringify(["moving_average", "exponential_smoothing"]),
        },
      },
      {
        modelType: "boosting",
        body: { ...baseBody, models: "ma,arima" },
        completedField: "ensemble_boosting_completed",
        configAssertions: {
          ensemble_boosting_base_models: JSON.stringify(["moving_average", "arima"]),
        },
      },
      {
        modelType: "stacking",
        body: { ...baseBody, models: "es,lstm" },
        completedField: "ensemble_stacking_completed",
        configAssertions: {
          ensemble_stacking_base_models: JSON.stringify(["exponential_smoothing", "lstm"]),
        },
      },
    ];

    for (const testCase of cases) {
      const authoritative = buildAuthoritativeResult(
        testCase.modelType,
        testCase.body,
        resultFor(testCase.modelType),
      );

      expect(authoritative.experimentPatch[testCase.completedField]).toBe(1);
      expect(authoritative.experimentPatch).toMatchObject(testCase.configAssertions);
      expect(authoritative.metrics).toEqual({ rmse: 1.2, mae: 0.8, r2: 0.91 });
      expect(authoritative.experimentPatch.selected_best_model).toBeNull();
      expect(authoritative.experimentPatch.production_plan_completed).toBe(0);
    }
  });

  test("preserves legacy comma-separated LSTM features in authoritative state", () => {
    const authoritative = buildAuthoritativeResult(
      "lstm",
      {
        ...baseBody,
        lstmNormalization: "minmax",
        lstmFeatures: "价格指数, 产能利用率",
        lstmTargetFeature: "销售数量",
      },
      resultFor("lstm"),
    );

    expect(authoritative.parameters.features).toEqual(["价格指数", "产能利用率"]);
    expect(authoritative.experimentPatch.lstm_features).toBe(
      JSON.stringify(["价格指数", "产能利用率"]),
    );
  });

  test("keeps JSON LSTM feature names containing commas unambiguous", () => {
    const authoritative = buildAuthoritativeResult(
      "lstm",
      {
        ...baseBody,
        lstmNormalization: "minmax",
        lstmFeatures: JSON.stringify(["价格指数", "外部,指标"]),
        lstmTargetFeature: "销售数量",
      },
      resultFor("lstm"),
    );

    expect(authoritative.parameters.features).toEqual(["价格指数", "外部,指标"]);
  });

  test("restores the complete current experiment state when reusing an active run", () => {
    const experiment = {
      state_version: 17,
      moving_average_completed: 1,
      moving_average_window: 6,
      moving_average_metrics_rmse: 1.1,
      moving_average_metrics_mae: 0.7,
      moving_average_metrics_r2: 0.92,
      exponential_smoothing_completed: 1,
      exponential_smoothing_alpha: 0.3,
      exponential_smoothing_metrics_rmse: 1.2,
      exponential_smoothing_metrics_mae: 0.8,
      exponential_smoothing_metrics_r2: 0.91,
      arima_completed: 1,
      arima_p: 2,
      arima_d: 1,
      arima_q: 3,
      arima_metrics_rmse: 1.3,
      arima_metrics_mae: 0.9,
      arima_metrics_r2: 0.9,
      lstm_completed: 1,
      lstm_normalization: "minmax",
      lstm_features: JSON.stringify(["价格指数"]),
      lstm_target_field: "销售数量",
      lstm_metrics_rmse: 1.4,
      lstm_metrics_mae: 1,
      lstm_metrics_r2: 0.89,
      ensemble_weighted_completed: 1,
      ensemble_weighted_base_models: JSON.stringify(["moving_average", "exponential_smoothing"]),
      ensemble_weighted_metrics_rmse: 1,
      ensemble_weighted_metrics_mae: 0.6,
      ensemble_weighted_metrics_r2: 0.94,
      ensemble_boosting_completed: 1,
      ensemble_boosting_base_models: JSON.stringify(["moving_average", "arima"]),
      ensemble_boosting_metrics_rmse: 0.9,
      ensemble_boosting_metrics_mae: 0.5,
      ensemble_boosting_metrics_r2: 0.95,
      ensemble_stacking_completed: 1,
      ensemble_stacking_base_models: JSON.stringify(["exponential_smoothing", "lstm"]),
      ensemble_stacking_metrics_rmse: 0.8,
      ensemble_stacking_metrics_mae: 0.4,
      ensemble_stacking_metrics_r2: 0.96,
      selected_best_model: "ensemble_stacking",
      production_plan_completed: 1,
    };

    const patch = buildCurrentExperimentStatePatch(experiment, 17);
    expect(patch).toMatchObject({
      moving_average_completed: true,
      exponential_smoothing_completed: true,
      arima_completed: true,
      lstm_completed: true,
      ensemble_weighted_completed: true,
      ensemble_boosting_completed: true,
      ensemble_stacking_completed: true,
      selected_best_model: "ensemble_stacking",
      production_plan_completed: true,
      state_version: 17,
    });
    expect(patch.lstm_features).toEqual(["价格指数"]);
    expect(patch.ensemble_weighted_base_models).toEqual(["moving_average", "exponential_smoothing"]);
  });

  test("invalidates every ensemble and production result after a base-model config change", () => {
    const invalidation = getServerInvalidationPatch(new Set(["moving_average_window"]));

    expect(new Set(invalidation.modelsToDeactivate)).toEqual(
      new Set(["ma", "weighted_avg", "boosting", "stacking"]),
    );
    expect(invalidation.patch).toMatchObject({
      moving_average_completed: 0,
      ensemble_weighted_completed: 0,
      ensemble_boosting_completed: 0,
      ensemble_stacking_completed: 0,
      selected_best_model: null,
      production_plan_completed: 0,
    });
    expect(invalidation.patch.exponential_smoothing_completed).toBeUndefined();
  });

  test("invalidates all seven active runs when dataset context or windows change", () => {
    const invalidation = getServerInvalidationPatch(new Set(["data_window_train_end_index"]));

    expect(new Set(invalidation.modelsToDeactivate)).toEqual(
      new Set(["ma", "es", "arima", "lstm", "weighted_avg", "boosting", "stacking"]),
    );
    expect(invalidation.patch).toMatchObject({
      moving_average_completed: 0,
      exponential_smoothing_completed: 0,
      arima_completed: 0,
      lstm_completed: 0,
      ensemble_weighted_completed: 0,
      ensemble_boosting_completed: 0,
      ensemble_stacking_completed: 0,
    });
  });
});

describe("model training request integrity", () => {
  const experiment = {
    ...baseBody,
    moving_average_window: 6,
    exponential_smoothing_alpha: 0.3,
    arima_d: 1,
    lstm_normalization: "minmax",
    lstm_features: JSON.stringify(["价格指数"]),
    lstm_target_field: "销售数量",
    ensemble_weighted_base_models: JSON.stringify(["moving_average", "exponential_smoothing"]),
  };

  test("changes the request hash when a relevant experiment config changes", () => {
    const body: TrainInput = { ...baseBody, moving_average_window: 6 };
    const initialRevision = buildExperimentTrainingRevision("ma", experiment, body);
    const changedRevision = buildExperimentTrainingRevision(
      "ma",
      { ...experiment, moving_average_window: 7 },
      body,
    );

    const initialHash = buildModelTrainingRequestHash({
      modelType: "ma",
      body,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      experimentRevision: initialRevision,
      baseRunFingerprint: {},
    });
    const changedHash = buildModelTrainingRequestHash({
      modelType: "ma",
      body,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      experimentRevision: changedRevision,
      baseRunFingerprint: {},
    });

    expect(changedHash).not.toBe(initialHash);
  });

  test("changes an ensemble request hash when any active base run changes", () => {
    const body: TrainInput = { ...baseBody, models: "ma,es" };
    const experimentRevision = buildExperimentTrainingRevision("weighted_avg", experiment, body);
    const buildHash = (maRun: string) => buildModelTrainingRequestHash({
      modelType: "weighted_avg",
      body,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      experimentRevision,
      baseRunFingerprint: { ma: maRun, es: "es-run-1" },
    });

    expect(buildHash("ma-run-2")).not.toBe(buildHash("ma-run-1"));
  });

  test("ignores unrelated workflow fields in the training revision", () => {
    const body: TrainInput = { ...baseBody, moving_average_window: 6 };
    expect(buildExperimentTrainingRevision("ma", experiment, body)).toBe(
      buildExperimentTrainingRevision("ma", { ...experiment, current_step: 7 }, body),
    );
  });
});
