import { describe, expect, test } from "bun:test";
import {
  buildExperimentFlowGradeValues,
  getReportSubmissionEligibilityError,
  type ReportSubmissionExperimentSnapshot,
} from "../../../src/modules/reports/report-submission-policy";
import {
  buildCanonicalProductionMps,
  buildProductionCapacityScenarios,
} from "../../../src/modules/experiments/experiments.schema";
import { issueProductionForecastEvidence } from "../../../src/modules/experiments/production-forecast-evidence";

const validProductionPredictionPoint = {
  prediction: 100,
  std_dev: 12,
  upper_error_p99: 28,
  uncertainty_source: "model",
  calibration_source: "arima_forecast_distribution",
  calibration_mean_error: 1.5,
  calibration_count: 8,
  bias_diagnostic_source: "outer_selection_assessment",
  interval_lower: 76.48,
  interval_upper: 123.52,
  interval_level: 0.95,
  interval_kind: "model_prediction_interval",
  coverage_claim: "model_nominal",
};

const legacyProductionPredictionPoint = {
  prediction: 100,
  std_dev: 12,
};

const buildValidProductionOutputs = () => {
  const forecasts = Array.from(
    { length: 6 },
    () => ({ ...validProductionPredictionPoint }),
  );
  const zScore = 1.65;
  const capacity = buildProductionCapacityScenarios(forecasts, zScore).normal;
  const rows = buildCanonicalProductionMps({
    forecasts,
    zScore,
    initialInventory: 0,
    capacity,
  });

  return { forecasts, rows, capacity };
};

const buildExperiment = (
  overrides: Partial<ReportSubmissionExperimentSnapshot> = {},
): ReportSubmissionExperimentSnapshot => {
  const { forecasts, rows, capacity } = buildValidProductionOutputs();
  return {
    experiment_id: 990001,
    student_id: 20240001,
    status: "In Progress",
    highest_completed_step: 7,
    selected_industry: "制造业",
    selected_company: "样本企业",
    selected_product: "样本产品",
    selected_best_model: "lstm",
    data_window_train_start_index: 0,
    data_window_train_end_index: 11,
    data_window_evaluate_start_index: 12,
    data_window_evaluate_end_index: 17,
    moving_average_completed: false,
    exponential_smoothing_completed: false,
    arima_completed: false,
    lstm_completed: true,
    ensemble_weighted_completed: false,
    ensemble_boosting_completed: false,
    ensemble_stacking_completed: false,
    moving_average_metrics_rmse: null,
    moving_average_metrics_mae: null,
    moving_average_metrics_r2: null,
    exponential_smoothing_metrics_rmse: null,
    exponential_smoothing_metrics_mae: null,
    exponential_smoothing_metrics_r2: null,
    arima_metrics_rmse: null,
    arima_metrics_mae: null,
    arima_metrics_r2: null,
    lstm_metrics_rmse: 1.2,
    lstm_metrics_mae: 0.8,
    lstm_metrics_r2: 0.91,
    ensemble_weighted_metrics_rmse: null,
    ensemble_weighted_metrics_mae: null,
    ensemble_weighted_metrics_r2: null,
    ensemble_boosting_metrics_rmse: null,
    ensemble_boosting_metrics_mae: null,
    ensemble_boosting_metrics_r2: null,
    ensemble_stacking_metrics_rmse: null,
    ensemble_stacking_metrics_mae: null,
    ensemble_stacking_metrics_r2: null,
    quiz_about_model_completed: true,
    quiz_about_plan_completed: true,
    production_plan_completed: true,
    production_forecast_periods: 6,
    production_initial_inventory: 0,
    production_target_service_level: 0.95,
    production_safety_stock_z_score: 1.65,
    production_forecast_results: forecasts,
    production_forecast_evidence: issueProductionForecastEvidence({
      experiment_id: 990001,
      student_id: 20240001,
      model_type: "lstm",
      run_id: "production-policy-run",
      production_revision: "production-policy-revision",
      forecast_steps: forecasts.length,
      forecasts,
    }),
    production_mps_table: rows,
    production_capacity_mode: "scenario",
    production_capacity_scenario: "normal",
    production_capacity: capacity,
    production_custom_capacity: null,
    ...overrides,
  };
};

describe("report submission policy", () => {
  test("allows submission only after the required checkpoints are complete", () => {
    expect(getReportSubmissionEligibilityError(buildExperiment())).toBeNull();
  });

  test("allows recovery of a completed run only after absence of a report is verified", () => {
    const completedExperiment = buildExperiment({ status: "Completed" });

    expect(getReportSubmissionEligibilityError(completedExperiment)).toContain("不能重复提交");
    expect(getReportSubmissionEligibilityError(completedExperiment, {
      allowCompletedWithoutReport: true,
    })).toBeNull();
  });

  test("rejects structurally valid client-fabricated forecasts without server evidence", () => {
    const zeroForecasts = Array.from(
      { length: 6 },
      () => ({ ...validProductionPredictionPoint, prediction: 0, std_dev: 0 }),
    );
    expect(getReportSubmissionEligibilityError(buildExperiment({
      production_forecast_results: zeroForecasts,
      production_forecast_evidence: null,
    }))).toContain("重新生成并保存生产计划");

    const grades = buildExperimentFlowGradeValues(buildExperiment({
      production_forecast_results: zeroForecasts,
      production_forecast_evidence: null,
    }));
    expect(grades.exp_flow_production_plan_creation).toBe(0);
    expect(grades.exp_flow_production_inventory_calc).toBe(0);
    expect(grades.exp_flow_production_service_level).toBe(0);
    expect(grades.exp_flow_production_variable_calc).toBe(0);
  });

  test("rejects submission when the production quiz is incomplete", () => {
    expect(
      getReportSubmissionEligibilityError(
        buildExperiment({ quiz_about_plan_completed: false }),
      ),
    ).toContain("生产计划测验");
  });

  test("rejects submission when no model results are available", () => {
    expect(
      getReportSubmissionEligibilityError(
        buildExperiment({
          lstm_completed: false,
          lstm_metrics_rmse: null,
          lstm_metrics_mae: null,
          lstm_metrics_r2: null,
        }),
      ),
    ).toContain("模型训练");
  });

  test("rejects submission when completed production-plan outputs are missing or malformed", () => {
    const invalidOutputs = [
      { production_forecast_results: [] },
      { production_forecast_results: [{ prediction: 100 }] },
      {
        production_forecast_results: Array.from(
          { length: 13 },
          () => validProductionPredictionPoint,
        ),
      },
      { production_mps_table: [] },
      { production_mps_table: [{ period: 1 }] },
      { production_forecast_periods: 5 },
      { production_safety_stock_z_score: 1.64 },
    ];

    for (const overrides of invalidOutputs) {
      expect(
        getReportSubmissionEligibilityError(buildExperiment(overrides)),
      ).toContain("重新生成并保存生产计划");
    }
  });

  test("derives flow grades from actual completed work instead of fixed full marks", () => {
    const grades = buildExperimentFlowGradeValues(
      buildExperiment({
        production_plan_completed: false,
        production_forecast_results: [],
        production_mps_table: [],
      }),
    );

    expect(grades.exp_flow_demand_data_preparation).toBe(100);
    expect(grades.exp_flow_demand_generate_results).toBe(100);
    expect(grades.exp_flow_production_plan_creation).toBe(0);
    expect(grades.exp_flow_report_submission).toBe(100);
  });

  test("requires the canonical service-level and Z-score pairing for plan credit", () => {
    const grades = buildExperimentFlowGradeValues(
      buildExperiment({ production_safety_stock_z_score: null }),
    );

    expect(grades.exp_flow_production_service_level).toBe(0);
  });

  test("accepts complete forecast points stored as DB JSON strings", () => {
    const { forecasts } = buildValidProductionOutputs();
    const grades = buildExperimentFlowGradeValues(
      buildExperiment({
        production_forecast_results: JSON.stringify(forecasts),
      }),
    );

    expect(grades.exp_flow_production_plan_creation).toBe(100);
  });

  test("does not award production-plan credit for malformed nonempty forecast arrays", () => {
    const invalidForecasts = [
      [{ prediction: 100, std_dev: 12 }],
      [{ ...validProductionPredictionPoint, interval_kind: "custom_interval" }],
      [{ ...validProductionPredictionPoint, coverage_claim: "none" }],
      Array.from({ length: 13 }, () => validProductionPredictionPoint),
    ];

    for (const productionForecastResults of invalidForecasts) {
      const grades = buildExperimentFlowGradeValues(
        buildExperiment({ production_forecast_results: productionForecastResults }),
      );
      expect(grades.exp_flow_production_plan_creation).toBe(0);
      expect(grades.exp_flow_production_inventory_calc).toBe(0);
      expect(grades.exp_flow_production_service_level).toBe(0);
      expect(grades.exp_flow_production_variable_calc).toBe(0);
    }
  });

  test("grandfathers v1 production plans only for completed historical experiments", () => {
    const legacyForecasts = Array.from(
      { length: 6 },
      () => ({ ...legacyProductionPredictionPoint }),
    );
    const completedGrades = buildExperimentFlowGradeValues(buildExperiment({
      status: "Completed",
      production_forecast_results: legacyForecasts,
      production_mps_table: Array.from({ length: 6 }, (_, index) => ({ period: index + 1 })),
    }));
    expect(completedGrades.exp_flow_production_plan_creation).toBe(100);

    const draftGrades = buildExperimentFlowGradeValues(buildExperiment({
      production_forecast_results: legacyForecasts,
      production_mps_table: Array.from({ length: 6 }, (_, index) => ({ period: index + 1 })),
    }));
    expect(draftGrades.exp_flow_production_plan_creation).toBe(0);
    expect(getReportSubmissionEligibilityError(buildExperiment({
      production_forecast_results: legacyForecasts,
      production_mps_table: Array.from({ length: 6 }, (_, index) => ({ period: index + 1 })),
    }))).toContain("重新生成并保存生产计划");
  });

  test("accepts an exact active v1 plan only when the rollout bridge is enabled", () => {
    const current = buildExperiment();
    const legacyForecasts = (current.production_forecast_results as Array<
      typeof validProductionPredictionPoint
    >).map((point) => {
      const {
        interval_lower: _lower,
        interval_upper: _upper,
        interval_level: _level,
        interval_kind: _kind,
        coverage_claim: _claim,
        ...legacyPoint
      } = point;
      return legacyPoint;
    });
    const legacyExperiment = buildExperiment({
      production_forecast_results: legacyForecasts,
      production_forecast_evidence: null,
    });

    expect(getReportSubmissionEligibilityError(legacyExperiment)).toContain(
      "重新生成并保存生产计划",
    );
    expect(getReportSubmissionEligibilityError(legacyExperiment, {
      allowLegacyV1Write: true,
    })).toBeNull();
    expect(buildExperimentFlowGradeValues(legacyExperiment, {
      allowLegacyV1Write: true,
    }).exp_flow_production_plan_creation).toBe(100);
  });
});
