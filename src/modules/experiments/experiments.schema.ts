import { z } from "zod";

export const STATUS_VALUES = ["Not Started", "In Progress", "Completed"] as const;
export const BEST_MODEL_VALUES = ["ma", "exp", "arima", "lstm", "ensemble_weighted", "ensemble_boosting", "ensemble_stacking"] as const;
export const NORMALIZATION_VALUES = ["minmax", "zscore"] as const;
export const CAPACITY_MODE_VALUES = ["scenario", "auto", "custom"] as const;
export const CAPACITY_SCENARIO_VALUES = ["tight", "normal", "abundant"] as const;

const numberPreprocess = (value: unknown) => {
  if (value === undefined || value === null) return value;
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const parsed = Number(trimmed);
    return Number.isNaN(parsed) ? value : parsed;
  }
  return value;
};

const booleanPreprocess = (value: unknown) => {
  if (value === undefined || value === null) return value;
  if (typeof value === "string") {
    const trimmed = value.trim().toLowerCase();
    if (!trimmed) return null;
    if (trimmed === "true" || trimmed === "1") return true;
    if (trimmed === "false" || trimmed === "0") return false;
  }
  if (typeof value === "number") {
    if (value === 1) return true;
    if (value === 0) return false;
  }
  return value;
};

const stringPreprocess = (value: unknown) => {
  if (value === undefined || value === null) return null;
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length ? trimmed : null;
  }
  return value;
};

const jsonPreprocess = (value: unknown) => {
  if (value === undefined || value === null) return null;
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return null;
    try { return JSON.parse(trimmed); } catch { return trimmed; }
  }
  return value;
};

const buildIntSchema = (min?: number, max?: number) => {
  let schema = z.number().int();
  if (min !== undefined) schema = schema.min(min);
  if (max !== undefined) schema = schema.max(max);
  return schema;
};

const buildFloatSchema = (min?: number) => {
  let schema = z.number();
  if (min !== undefined) schema = schema.min(min);
  return schema;
};

const optionalInt = (min?: number, max?: number) => z.preprocess(numberPreprocess, buildIntSchema(min, max)).nullable().optional();
const optionalFloat = (min?: number) => z.preprocess(numberPreprocess, buildFloatSchema(min)).nullable().optional();
const optionalBoolean = z.preprocess(booleanPreprocess, z.boolean()).nullable().optional();
const optionalString = z.preprocess(stringPreprocess, z.string()).nullable().optional();
const optionalJson = z.preprocess(jsonPreprocess, z.union([z.array(z.any()), z.record(z.string(), z.any()), z.string()])).nullable().optional();

const optionalFields: Record<string, z.ZodTypeAny> = {
  highest_completed_step: optionalInt(0, 7),
  selected_industry: optionalString, selected_company: optionalString, selected_product: optionalString,
  selected_base_models: optionalJson, selected_ensemble_models: optionalJson,
  data_window_train_start_index: optionalInt(0), data_window_train_end_index: optionalInt(0),
  data_window_evaluate_start_index: optionalInt(0), data_window_evaluate_end_index: optionalInt(0),
  moving_average_completed: optionalBoolean, moving_average_window: optionalInt(1),
  moving_average_metrics_rmse: optionalFloat(0), moving_average_metrics_mae: optionalFloat(0), moving_average_metrics_r2: optionalFloat(),
  exponential_smoothing_completed: optionalBoolean, exponential_smoothing_alpha: optionalFloat(0),
  exponential_smoothing_metrics_rmse: optionalFloat(0), exponential_smoothing_metrics_mae: optionalFloat(0), exponential_smoothing_metrics_r2: optionalFloat(),
  arima_completed: optionalBoolean, arima_p: optionalInt(0), arima_d: optionalInt(0), arima_q: optionalInt(0),
  arima_metrics_rmse: optionalFloat(0), arima_metrics_mae: optionalFloat(0), arima_metrics_r2: optionalFloat(), arima_adf_stationarity: optionalJson,
  lstm_completed: optionalBoolean, lstm_normalization: z.preprocess(stringPreprocess, z.enum(NORMALIZATION_VALUES)).nullable().optional(),
  lstm_features: optionalJson, lstm_target_field: optionalString,
  lstm_metrics_rmse: optionalFloat(0), lstm_metrics_mae: optionalFloat(0), lstm_metrics_r2: optionalFloat(),
  ensemble_weighted_completed: optionalBoolean, ensemble_weighted_base_models: optionalJson,
  ensemble_weighted_metrics_rmse: optionalFloat(0), ensemble_weighted_metrics_mae: optionalFloat(0), ensemble_weighted_metrics_r2: optionalFloat(),
  ensemble_boosting_completed: optionalBoolean, ensemble_boosting_base_models: optionalJson,
  ensemble_boosting_metrics_rmse: optionalFloat(0), ensemble_boosting_metrics_mae: optionalFloat(0), ensemble_boosting_metrics_r2: optionalFloat(),
  ensemble_stacking_completed: optionalBoolean, ensemble_stacking_base_models: optionalJson,
  ensemble_stacking_metrics_rmse: optionalFloat(0), ensemble_stacking_metrics_mae: optionalFloat(0), ensemble_stacking_metrics_r2: optionalFloat(),
  selected_best_model: z.preprocess(stringPreprocess, z.enum(BEST_MODEL_VALUES)).nullable().optional(),
  quiz_about_model_completed: optionalBoolean, quiz_about_plan_completed: optionalBoolean,
  production_plan_completed: optionalBoolean, production_forecast_periods: optionalInt(1),
  production_initial_inventory: optionalFloat(0), production_target_service_level: optionalFloat(0), production_safety_stock_z_score: optionalFloat(),
  production_mps_table: optionalJson, production_forecast_results: optionalJson,
  production_capacity_mode: z.preprocess(stringPreprocess, z.enum(CAPACITY_MODE_VALUES)).nullable().optional(),
  production_capacity_scenario: z.preprocess(stringPreprocess, z.enum(CAPACITY_SCENARIO_VALUES)).nullable().optional(),
  production_capacity: optionalInt(0), production_custom_capacity: optionalInt(0),
  start_time: optionalString, last_activity_at: optionalString, completion_time: optionalString, total_active_duration_seconds: optionalInt(0),
};

export const experimentIdParamSchema = z.object({
  experimentId: z.string().regex(/^\d+$/, { message: "experimentId 必须是数字" }),
});

export const createExperimentSchema = z.object({
  status: z.enum(STATUS_VALUES).optional(),
  current_step: optionalInt(1, 7),
  ...optionalFields,
});

export const updateExperimentSchema = z.object({
  status: z.enum(STATUS_VALUES).optional(),
  current_step: optionalInt(1, 7),
  ...optionalFields,
}).refine((data) => Object.keys(data).length > 0, "至少需要提供一个字段");

export const experimentEventSchema = z.object({
  step_order: z.number().int().min(1).max(7),
  event_type: z.enum(["STARTED", "COMPLETED"]),
});

export type ExperimentIdParam = z.infer<typeof experimentIdParamSchema>;
export type CreateExperimentInput = z.infer<typeof createExperimentSchema>;
export type UpdateExperimentInput = z.infer<typeof updateExperimentSchema>;
export type ExperimentEventInput = z.infer<typeof experimentEventSchema>;
