type NullableValue = string | number | boolean | null | undefined;

export interface ReportSubmissionExperimentSnapshot {
  status: string;
  highest_completed_step: number | null;
  selected_industry: string | null;
  selected_company: string | null;
  selected_product: string | null;
  selected_best_model: string | null;
  data_window_train_start_index: number | null;
  data_window_train_end_index: number | null;
  data_window_evaluate_start_index: number | null;
  data_window_evaluate_end_index: number | null;
  moving_average_completed: boolean | number | null;
  exponential_smoothing_completed: boolean | number | null;
  arima_completed: boolean | number | null;
  lstm_completed: boolean | number | null;
  ensemble_weighted_completed: boolean | number | null;
  ensemble_boosting_completed: boolean | number | null;
  ensemble_stacking_completed: boolean | number | null;
  moving_average_metrics_rmse: number | null;
  moving_average_metrics_mae: number | null;
  moving_average_metrics_r2: number | null;
  exponential_smoothing_metrics_rmse: number | null;
  exponential_smoothing_metrics_mae: number | null;
  exponential_smoothing_metrics_r2: number | null;
  arima_metrics_rmse: number | null;
  arima_metrics_mae: number | null;
  arima_metrics_r2: number | null;
  lstm_metrics_rmse: number | null;
  lstm_metrics_mae: number | null;
  lstm_metrics_r2: number | null;
  ensemble_weighted_metrics_rmse: number | null;
  ensemble_weighted_metrics_mae: number | null;
  ensemble_weighted_metrics_r2: number | null;
  ensemble_boosting_metrics_rmse: number | null;
  ensemble_boosting_metrics_mae: number | null;
  ensemble_boosting_metrics_r2: number | null;
  ensemble_stacking_metrics_rmse: number | null;
  ensemble_stacking_metrics_mae: number | null;
  ensemble_stacking_metrics_r2: number | null;
  quiz_about_model_completed: boolean | number | null;
  quiz_about_plan_completed: boolean | number | null;
  production_plan_completed: boolean | number | null;
  production_forecast_periods: number | null;
  production_initial_inventory: number | null;
  production_target_service_level: number | null;
  production_safety_stock_z_score: number | null;
  production_forecast_results: unknown;
  production_mps_table: unknown;
  production_capacity_mode: string | null;
  production_capacity_scenario: string | null;
  production_capacity: number | null;
  production_custom_capacity: number | null;
}

export interface ExperimentFlowGradeValues {
  exp_flow_demand_data_preparation: number;
  exp_flow_demand_descriptive_stats: number;
  exp_flow_demand_model_selection: number;
  exp_flow_demand_generate_results: number;
  exp_flow_production_inventory_calc: number;
  exp_flow_production_service_level: number;
  exp_flow_production_variable_calc: number;
  exp_flow_production_plan_creation: number;
  exp_flow_report_submission: number;
}

const toBoolean = (value: NullableValue): boolean => {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value === 1;
  if (typeof value === "string") {
    const lowered = value.trim().toLowerCase();
    return lowered === "1" || lowered === "true";
  }
  return false;
};

const hasValue = (value: NullableValue): boolean => value !== null && value !== undefined && value !== "";

const parseArray = (value: unknown): unknown[] => {
  if (Array.isArray(value)) {
    return value;
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return [];
    }

    try {
      const parsed = JSON.parse(trimmed);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  return [];
};

const hasCompletedAnyModel = (experiment: ReportSubmissionExperimentSnapshot): boolean => {
  return [
    experiment.moving_average_completed,
    experiment.exponential_smoothing_completed,
    experiment.arima_completed,
    experiment.lstm_completed,
    experiment.ensemble_weighted_completed,
    experiment.ensemble_boosting_completed,
    experiment.ensemble_stacking_completed,
  ].some(toBoolean);
};

const hasAnyModelMetrics = (experiment: ReportSubmissionExperimentSnapshot): boolean => {
  return [
    experiment.moving_average_metrics_rmse,
    experiment.moving_average_metrics_mae,
    experiment.moving_average_metrics_r2,
    experiment.exponential_smoothing_metrics_rmse,
    experiment.exponential_smoothing_metrics_mae,
    experiment.exponential_smoothing_metrics_r2,
    experiment.arima_metrics_rmse,
    experiment.arima_metrics_mae,
    experiment.arima_metrics_r2,
    experiment.lstm_metrics_rmse,
    experiment.lstm_metrics_mae,
    experiment.lstm_metrics_r2,
    experiment.ensemble_weighted_metrics_rmse,
    experiment.ensemble_weighted_metrics_mae,
    experiment.ensemble_weighted_metrics_r2,
    experiment.ensemble_boosting_metrics_rmse,
    experiment.ensemble_boosting_metrics_mae,
    experiment.ensemble_boosting_metrics_r2,
    experiment.ensemble_stacking_metrics_rmse,
    experiment.ensemble_stacking_metrics_mae,
    experiment.ensemble_stacking_metrics_r2,
  ].some(hasValue);
};

const hasSelectionContext = (experiment: ReportSubmissionExperimentSnapshot): boolean => {
  return hasValue(experiment.selected_industry)
    && hasValue(experiment.selected_company)
    && hasValue(experiment.selected_product);
};

const hasDataWindow = (experiment: ReportSubmissionExperimentSnapshot): boolean => {
  return [
    experiment.data_window_train_start_index,
    experiment.data_window_train_end_index,
    experiment.data_window_evaluate_start_index,
    experiment.data_window_evaluate_end_index,
  ].every(hasValue);
};

export const getReportSubmissionEligibilityError = (
  experiment: ReportSubmissionExperimentSnapshot,
): string | null => {
  if (experiment.status === "Completed") {
    return "该实验已提交报告，不能重复提交。";
  }

  if ((experiment.highest_completed_step ?? 0) < 7) {
    return "请先完成实验主流程后再提交报告。";
  }

  if (!toBoolean(experiment.production_plan_completed)) {
    return "请先完成生产计划后再提交报告。";
  }

  if (!toBoolean(experiment.quiz_about_model_completed)) {
    return "请先完成预测模型测验后再提交报告。";
  }

  if (!toBoolean(experiment.quiz_about_plan_completed)) {
    return "请先完成生产计划测验后再提交报告。";
  }

  if (!hasValue(experiment.selected_best_model)) {
    return "请先在结果评估中选择最优模型后再提交报告。";
  }

  if (!hasCompletedAnyModel(experiment) || !hasAnyModelMetrics(experiment)) {
    return "请至少完成一个模型训练并生成评价结果后再提交报告。";
  }

  return null;
};

export const buildExperimentFlowGradeValues = (
  experiment: ReportSubmissionExperimentSnapshot,
): ExperimentFlowGradeValues => {
  const preparedDemandData = hasSelectionContext(experiment) && hasDataWindow(experiment);
  const generatedDemandResults = hasCompletedAnyModel(experiment) && hasAnyModelMetrics(experiment);
  const selectedModel = generatedDemandResults && hasValue(experiment.selected_best_model);
  const hasForecastResults = parseArray(experiment.production_forecast_results).length > 0;
  const hasMpsTable = parseArray(experiment.production_mps_table).length > 0;
  const hasProductionPlan = toBoolean(experiment.production_plan_completed) && hasForecastResults && hasMpsTable;
  const hasInventoryInputs = hasValue(experiment.production_initial_inventory);
  const hasServiceLevelInputs = hasValue(experiment.production_target_service_level);
  const hasCapacityInputs = hasValue(experiment.production_capacity_mode)
    && (
      hasValue(experiment.production_capacity)
      || hasValue(experiment.production_custom_capacity)
      || hasValue(experiment.production_capacity_scenario)
    );

  return {
    exp_flow_demand_data_preparation: preparedDemandData ? 100 : 0,
    exp_flow_demand_descriptive_stats: preparedDemandData ? 100 : 0,
    exp_flow_demand_model_selection: selectedModel ? 100 : 0,
    exp_flow_demand_generate_results: generatedDemandResults ? 100 : 0,
    exp_flow_production_inventory_calc: hasProductionPlan && hasInventoryInputs ? 100 : 0,
    exp_flow_production_service_level: hasProductionPlan && hasServiceLevelInputs ? 100 : 0,
    exp_flow_production_variable_calc: hasProductionPlan && hasCapacityInputs ? 100 : 0,
    exp_flow_production_plan_creation: hasProductionPlan ? 100 : 0,
    exp_flow_report_submission: 100,
  };
};
