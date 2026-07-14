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
const adfResultRowSchema = z.object({
  diff_order: z.number().int().min(0).max(2),
  statistic: z.number().finite(),
  p_value: z.number().finite().min(0).max(1),
  used_lags: z.number().int().min(0),
  n_obs: z.number().int().min(1),
  stationary: z.boolean(),
  critical_values: z.object({
    "1%": z.number().finite(),
    "5%": z.number().finite(),
    "10%": z.number().finite(),
  }).strict(),
}).strict();
const adfResultRowsSchema = z.array(adfResultRowSchema).max(3).superRefine((rows, context) => {
  const diffOrders = rows.map((row) => row.diff_order);
  if (new Set(diffOrders).size !== diffOrders.length) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "ADF 检验结果不能包含重复的差分阶数",
    });
  }
});
const optionalAdfResults = z.preprocess(jsonPreprocess, adfResultRowsSchema).nullable().optional();

const studentWritableFields: Record<string, z.ZodTypeAny> = {
  highest_completed_step: optionalInt(0, 7),
  selected_industry: optionalString, selected_company: optionalString, selected_product: optionalString,
  selected_base_models: optionalJson, selected_ensemble_models: optionalJson,
  data_window_train_start_index: optionalInt(0), data_window_train_end_index: optionalInt(0),
  data_window_evaluate_start_index: optionalInt(0), data_window_evaluate_end_index: optionalInt(0),
  moving_average_window: optionalInt(2),
  exponential_smoothing_alpha: optionalFloat(0),
  arima_d: optionalInt(0, 2), arima_adf_stationarity: optionalAdfResults,
  lstm_normalization: z.preprocess(stringPreprocess, z.enum(NORMALIZATION_VALUES)).nullable().optional(),
  lstm_features: optionalJson, lstm_target_field: optionalString,
  ensemble_weighted_base_models: optionalJson,
  ensemble_boosting_base_models: optionalJson,
  ensemble_stacking_base_models: optionalJson,
  selected_best_model: z.preprocess(stringPreprocess, z.enum(BEST_MODEL_VALUES)).nullable().optional(),
  production_plan_completed: optionalBoolean, production_forecast_periods: optionalInt(1),
  production_initial_inventory: optionalFloat(0), production_target_service_level: optionalFloat(0), production_safety_stock_z_score: optionalFloat(),
  production_mps_table: optionalJson, production_forecast_results: optionalJson,
  production_capacity_mode: z.preprocess(stringPreprocess, z.enum(CAPACITY_MODE_VALUES)).nullable().optional(),
  production_capacity_scenario: z.preprocess(stringPreprocess, z.enum(CAPACITY_SCENARIO_VALUES)).nullable().optional(),
  production_capacity: optionalInt(0), production_custom_capacity: optionalInt(0),
};

const studentInitialWritableFields: Record<string, z.ZodTypeAny> = {
  highest_completed_step: studentWritableFields.highest_completed_step!,
  selected_industry: studentWritableFields.selected_industry!,
  selected_company: studentWritableFields.selected_company!,
  selected_product: studentWritableFields.selected_product!,
  selected_base_models: studentWritableFields.selected_base_models!,
  selected_ensemble_models: studentWritableFields.selected_ensemble_models!,
  data_window_train_start_index: studentWritableFields.data_window_train_start_index!,
  data_window_train_end_index: studentWritableFields.data_window_train_end_index!,
  data_window_evaluate_start_index: studentWritableFields.data_window_evaluate_start_index!,
  data_window_evaluate_end_index: studentWritableFields.data_window_evaluate_end_index!,
  moving_average_window: studentWritableFields.moving_average_window!,
  exponential_smoothing_alpha: studentWritableFields.exponential_smoothing_alpha!,
  arima_d: studentWritableFields.arima_d!,
  arima_adf_stationarity: studentWritableFields.arima_adf_stationarity!,
  lstm_normalization: studentWritableFields.lstm_normalization!,
  lstm_features: studentWritableFields.lstm_features!,
  lstm_target_field: studentWritableFields.lstm_target_field!,
  ensemble_weighted_base_models: studentWritableFields.ensemble_weighted_base_models!,
  ensemble_boosting_base_models: studentWritableFields.ensemble_boosting_base_models!,
  ensemble_stacking_base_models: studentWritableFields.ensemble_stacking_base_models!,
};

export const experimentIdParamSchema = z.object({
  experimentId: z.string().regex(/^\d+$/, { message: "experimentId 必须是数字" }),
});

export const createExperimentSchema = z.object({
  current_step: optionalInt(1, 7),
  ...studentInitialWritableFields,
}).strict();

export const updateExperimentSchema = z.object({
  expected_version: z.coerce.number().int().min(0),
  current_step: optionalInt(1, 7),
  ...studentWritableFields,
}).strict().refine((data) => Object.keys(data).some((key) => key !== "expected_version"), "至少需要提供一个更新字段");

export const experimentEventSchema = z.object({
  step_order: z.number().int().min(1).max(7),
  event_type: z.enum(["STARTED", "COMPLETED"]),
});

export type ExperimentIdParam = z.infer<typeof experimentIdParamSchema>;
export type CreateExperimentInput = z.infer<typeof createExperimentSchema>;
export type UpdateExperimentInput = z.infer<typeof updateExperimentSchema>;
export type ExperimentEventInput = z.infer<typeof experimentEventSchema>;
