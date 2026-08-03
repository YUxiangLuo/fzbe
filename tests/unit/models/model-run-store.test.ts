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

  test("persists bounded model observability without storing prediction arrays", () => {
    const authoritative = buildAuthoritativeResult(
      "lstm",
      {
        ...baseBody,
        lstmNormalization: "zscore",
        lstmFeatures: JSON.stringify(["价格指数", "渠道"]),
        lstmTargetFeature: "销售数量",
      },
      {
        status: "success",
        results: {
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
            seed_runs: 3,
            feature_keys: ["价格指数", "渠道"],
            seed_stability: {
              scope: "internal_time_validation",
              run_count: 3,
              validation_rmse_mean: 1.5,
              validation_rmse_std: 0.2,
              seeds: [42, 314, 2718],
              runs: [{ seed: 42, validation_rmse: 1.4 }],
            },
            artifact_path: "/tmp/private-lstm.pkl",
            raw_losses: [1, 0.5],
          },
          uncertainty_summary: {
            method: "rolling_origin_quantiles",
            source: "empirical",
            calibration_source: "internal_time_validation",
            fallback: true,
            horizon_start: 4,
            fallback_horizons: [2, 2_000],
            horizons: Array.from({ length: 2_000 }, (_, index) => ({
              std_dev: index + 0.5,
              upper_error_p99: index + 1,
              calibration_mean_error: index / 10,
              calibration_count: index + 2,
              source: index === 1 || index === 1_999 ? "fallback" : "empirical",
              interval_kind: index === 1 || index === 1_999
                ? "fallback_normal_approximation"
                : "empirical_residual_quantile",
              coverage_claim: "none",
            })),
          },
          metrics: {
            rmse: 1.2,
            mae: 0.8,
            r2: 0.91,
            interval_evaluation: {
              scope: "outer_evaluation_descriptive",
              observed_inclusion_rate: 0.5,
            },
          },
          prediction_points: [{
            uncertainty_source: "empirical",
            interval_kind: "empirical_residual_quantile",
            coverage_claim: "none",
            calibration_source: "internal_time_validation",
            calibration_count: 6,
            upper_error_p99: 3,
          }],
        },
      },
    );

    expect(authoritative.metrics).toMatchObject({
      interval_evaluation: {
        scope: "outer_evaluation_descriptive",
        observed_inclusion_rate: 0.5,
      },
    });
    expect(authoritative.parameters.observability).toMatchObject({
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
        seed_runs: 3,
        feature_keys: ["价格指数", "渠道"],
        seed_stability: {
          scope: "internal_time_validation",
          run_count: 3,
          validation_rmse_mean: 1.5,
          validation_rmse_std: 0.2,
        },
      },
      uncertainty_summary: {
        method: "rolling_origin_quantiles",
        source: "empirical",
        calibration_source: "internal_time_validation",
        fallback: true,
        horizon_start: 4,
        horizons_total: 2_000,
        horizons_included: 0,
        horizons_truncated: true,
        fallback_horizons_total: 2,
        fallback_horizons_included: 0,
        fallback_horizons_truncated: true,
        fallback_horizon_start: 5,
        fallback_horizon_end: 2_003,
        sources: ["empirical", "fallback"],
        interval_kinds: [
          "empirical_residual_quantile",
          "fallback_normal_approximation",
        ],
        coverage_claims: ["none"],
        horizon_statistics: {
          std_dev: { min: 0.5, max: 1_999.5 },
          upper_error_p99: { min: 1, max: 2_000 },
          calibration_count: { min: 2, max: 2_001 },
        },
      },
      uncertainty_evidence: {
        sources: ["empirical"],
        coverage_claims: ["none"],
        calibration_count_min: 6,
        calibration_count_max: 6,
        upper_error_p99_points: 1,
      },
    });
    expect(authoritative.parameters.observability).not.toHaveProperty("prediction_points");
    expect(authoritative.parameters.observability).not.toHaveProperty(
      "uncertainty_summary.horizons",
    );
    expect(authoritative.parameters.observability).not.toHaveProperty(
      "uncertainty_summary.fallback_horizons",
    );
    expect(authoritative.parameters.observability).not.toHaveProperty(
      "training_metadata.artifact_path",
    );
    expect(authoritative.parameters.observability).not.toHaveProperty(
      "training_metadata.raw_losses",
    );
    expect(authoritative.parameters.observability).not.toHaveProperty(
      "training_metadata.seed_stability.runs",
    );
    expect(authoritative.parameters.observability).not.toHaveProperty(
      "training_metadata.seed_stability.seeds",
    );
    expect(JSON.stringify(authoritative.parameters.observability).length).toBeLessThan(10_000);
  });

  test("preserves the bounded maximum of 64 LSTM features plus the target type", () => {
    const features = Array.from({ length: 64 }, (_, index) => `特征${index + 1}`);
    const featureTypes = Object.fromEntries([
      ["销售数量", "numeric"],
      ...features.map((feature) => [feature, "numeric"]),
    ]);
    const authoritative = buildAuthoritativeResult(
      "lstm",
      {
        ...baseBody,
        lstmNormalization: "minmax",
        lstmFeatures: JSON.stringify(features),
        lstmTargetFeature: "销售数量",
      },
      {
        status: "success",
        results: {
          metrics: { rmse: 1.2, mae: 0.8, r2: 0.91 },
          target_key: "销售数量",
          feature_keys: features,
          feature_types: featureTypes,
          normalization: "minmax",
          forecast_horizon: 4,
          training_metadata: { feature_keys: features, look_back: 6 },
        },
      },
    );

    expect(authoritative.parameters.observability).toMatchObject({
      target_key: "销售数量",
      feature_keys: features,
      feature_types: featureTypes,
      normalization: "minmax",
      forecast_horizon: 4,
      training_metadata: { feature_keys: features, look_back: 6 },
    });
  });

  test("does not duplicate ARIMA evaluation predictions in persisted observability", () => {
    const authoritative = buildAuthoritativeResult(
      "arima",
      {
        ...baseBody,
        arimaP: 1,
        arimaD: 1,
        arimaQ: 1,
      },
      {
        status: "success",
        results: {
          best_order: { p: 1, d: 1, q: 1 },
          diagnostics: {
            ljung_box_pvalue: 0.3,
            y_pred_eval: [10, 11, 12],
          },
          metrics: { rmse: 1.2, mae: 0.8, r2: 0.91 },
          prediction_points: [{
            calibration_source: "arima_forecast_distribution",
            calibration_mean_error: 0.5,
            calibration_count: 6,
            bias_diagnostic_source: "outer_selection_assessment",
          }],
        },
      },
    );

    expect(authoritative.parameters.observability).toMatchObject({
      diagnostics: { ljung_box_pvalue: 0.3 },
      uncertainty_evidence: {
        bias_diagnostic_sources: ["outer_selection_assessment"],
        calibration_count_min: null,
        calibration_count_max: null,
        bias_diagnostic_count_min: 6,
        bias_diagnostic_count_max: 6,
      },
    });
    expect(authoritative.parameters.observability).not.toHaveProperty(
      "diagnostics.y_pred_eval",
    );
  });

  test("persists the explicit Boosting post-line-search shrinkage policy", () => {
    const policy = { enabled: false, factor: 1 };
    const authoritative = buildAuthoritativeResult(
      "boosting",
      { ...baseBody, models: "ma,es" },
      {
        status: "success",
        results: {
          metrics: { rmse: 1.2, mae: 0.8, r2: 0.91 },
          model_names: [],
          stage_coefficients: [],
          post_line_search_shrinkage: policy,
          search_summary: {
            algorithm: "residual_boosting",
            post_line_search_shrinkage: policy,
          },
        },
      },
    );

    expect(authoritative.parameters.observability).toMatchObject({
      model_names: [],
      stage_coefficients: [],
      post_line_search_shrinkage: policy,
      search_summary: {
        algorithm: "residual_boosting",
        post_line_search_shrinkage: policy,
      },
    });
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
      production_forecast_evidence: "signed-production-evidence",
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
      production_forecast_evidence: "signed-production-evidence",
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

  test("keeps both configurations when a base and an ensemble change together", () => {
    // A base-model iteration resets every ensemble, including each
    // `ensemble_*_base_models` selection. The exemption for newly selected
    // configuration must therefore not depend on MODEL_CONFIG_FIELDS listing
    // base models before ensembles.
    const invalidation = getServerInvalidationPatch(new Set([
      "moving_average_window",
      "ensemble_weighted_base_models",
    ]));

    expect(invalidation.patch.moving_average_window).toBeUndefined();
    expect(invalidation.patch.ensemble_weighted_base_models).toBeUndefined();
    // The ensembles the request did not reselect are still cleared.
    expect(invalidation.patch.ensemble_boosting_base_models).toBeNull();
    expect(invalidation.patch.ensemble_stacking_base_models).toBeNull();
    expect(invalidation.patch).toMatchObject({
      moving_average_completed: 0,
      ensemble_weighted_completed: 0,
      production_plan_completed: 0,
      production_forecast_evidence: null,
    });
  });

  test("still resets derived outputs a request tried to set alongside a config change", () => {
    // Only configuration is exempt. Outputs the client sent in the same request
    // must lose to server-side invalidation, otherwise a stale production plan
    // could survive the change that invalidated it.
    const invalidation = getServerInvalidationPatch(new Set([
      "moving_average_window",
      "production_plan_completed",
      "quiz_about_model_completed",
    ]));

    expect(invalidation.patch.moving_average_window).toBeUndefined();
    expect(invalidation.patch).toMatchObject({
      production_plan_completed: 0,
      quiz_about_model_completed: 0,
    });
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

  test("preserves new model configurations submitted with a context change", () => {
    const invalidation = getServerInvalidationPatch(new Set([
      "selected_product",
      "moving_average_window",
      "lstm_features",
      "ensemble_weighted_base_models",
    ]));

    expect(new Set(invalidation.modelsToDeactivate)).toEqual(
      new Set(["ma", "es", "arima", "lstm", "weighted_avg", "boosting", "stacking"]),
    );
    expect(invalidation.patch.moving_average_window).toBeUndefined();
    expect(invalidation.patch.lstm_features).toBeUndefined();
    expect(invalidation.patch.ensemble_weighted_base_models).toBeUndefined();
    expect(invalidation.patch).toMatchObject({
      moving_average_completed: 0,
      lstm_completed: 0,
      ensemble_weighted_completed: 0,
      production_plan_completed: 0,
    });
    // Configuration values not supplied by the request are still invalidated.
    expect(invalidation.patch.exponential_smoothing_alpha).toBeNull();
    expect(invalidation.patch.ensemble_boosting_base_models).toBeNull();
  });
});

describe("model training request integrity", () => {
  test("rejects duplicate ensemble members at the persistence boundary", () => {
    expect(() => buildExperimentTrainingRevision(
      "weighted_avg",
      {},
      { ...baseBody, models: "ma,ma" },
    )).toThrow("融合模型的基础模型不能重复");
  });

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
