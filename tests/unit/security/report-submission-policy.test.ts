import { describe, expect, test } from "bun:test";
import {
  buildExperimentFlowGradeValues,
  getReportSubmissionEligibilityError,
  type ReportSubmissionExperimentSnapshot,
} from "../../../src/modules/reports/report-submission-policy";

const buildExperiment = (
  overrides: Partial<ReportSubmissionExperimentSnapshot> = {},
): ReportSubmissionExperimentSnapshot => ({
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
  production_initial_inventory: 50,
  production_target_service_level: 0.95,
  production_safety_stock_z_score: 1.64,
  production_forecast_results: [{ prediction: 100, std_dev: 12 }],
  production_mps_table: [{ period: 1 }],
  production_capacity_mode: "auto",
  production_capacity_scenario: "normal",
  production_capacity: 120,
  production_custom_capacity: null,
  ...overrides,
});

describe("report submission policy", () => {
  test("allows submission only after the required checkpoints are complete", () => {
    expect(getReportSubmissionEligibilityError(buildExperiment())).toBeNull();
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
});