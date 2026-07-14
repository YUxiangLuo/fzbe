import crypto from "crypto";
import type { PoolConnection, RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import type { ModelType, TrainInput } from "./models.schema";
import type { BaseModelType, EnsembleModelType, GuidedModelType } from "./model-types";
import { ModelValidationError } from "./model-errors";

const BASE_MODELS: BaseModelType[] = ["ma", "es", "arima", "lstm"];
const ENSEMBLE_MODELS: EnsembleModelType[] = ["weighted_avg", "boosting", "stacking"];

const BASE_MODEL_FRONTEND_IDS: Record<BaseModelType, string> = {
  ma: "moving_average",
  es: "exponential_smoothing",
  arima: "arima",
  lstm: "lstm",
};

const TRAINING_CONTEXT_FIELDS = [
  "selected_industry",
  "selected_company",
  "selected_product",
  "data_window_train_start_index",
  "data_window_train_end_index",
  "data_window_evaluate_start_index",
  "data_window_evaluate_end_index",
] as const;

const BASE_MODEL_CONFIG_FIELDS: Record<BaseModelType, readonly string[]> = {
  ma: ["moving_average_window"],
  es: ["exponential_smoothing_alpha"],
  arima: ["arima_d"],
  lstm: ["lstm_normalization", "lstm_features", "lstm_target_field"],
};

const ENSEMBLE_CONFIG_FIELDS: Record<EnsembleModelType, string> = {
  weighted_avg: "ensemble_weighted_base_models",
  boosting: "ensemble_boosting_base_models",
  stacking: "ensemble_stacking_base_models",
};

const CLIENT_EXPERIMENT_STATE_FIELDS = [
  "experiment_id",
  "student_id",
  "status",
  "highest_completed_step",
  "current_step",
  "selected_industry",
  "selected_company",
  "selected_product",
  "selected_base_models",
  "selected_ensemble_models",
  "data_window_train_start_index",
  "data_window_train_end_index",
  "data_window_evaluate_start_index",
  "data_window_evaluate_end_index",
  "moving_average_completed",
  "moving_average_window",
  "moving_average_metrics_rmse",
  "moving_average_metrics_mae",
  "moving_average_metrics_r2",
  "exponential_smoothing_completed",
  "exponential_smoothing_alpha",
  "exponential_smoothing_metrics_rmse",
  "exponential_smoothing_metrics_mae",
  "exponential_smoothing_metrics_r2",
  "arima_completed",
  "arima_p",
  "arima_d",
  "arima_q",
  "arima_metrics_rmse",
  "arima_metrics_mae",
  "arima_metrics_r2",
  "arima_adf_stationarity",
  "lstm_completed",
  "lstm_normalization",
  "lstm_features",
  "lstm_target_field",
  "lstm_metrics_rmse",
  "lstm_metrics_mae",
  "lstm_metrics_r2",
  "ensemble_weighted_completed",
  "ensemble_weighted_base_models",
  "ensemble_weighted_metrics_rmse",
  "ensemble_weighted_metrics_mae",
  "ensemble_weighted_metrics_r2",
  "ensemble_boosting_completed",
  "ensemble_boosting_base_models",
  "ensemble_boosting_metrics_rmse",
  "ensemble_boosting_metrics_mae",
  "ensemble_boosting_metrics_r2",
  "ensemble_stacking_completed",
  "ensemble_stacking_base_models",
  "ensemble_stacking_metrics_rmse",
  "ensemble_stacking_metrics_mae",
  "ensemble_stacking_metrics_r2",
  "selected_best_model",
  "quiz_about_model_completed",
  "quiz_about_plan_completed",
  "production_plan_completed",
  "production_forecast_periods",
  "production_initial_inventory",
  "production_target_service_level",
  "production_safety_stock_z_score",
  "production_forecast_results",
  "production_mps_table",
  "production_capacity_mode",
  "production_capacity_scenario",
  "production_capacity",
  "production_custom_capacity",
  "start_time",
  "last_activity_at",
  "completion_time",
] as const;

const CLIENT_ARRAY_STATE_FIELDS = new Set([
  "selected_base_models",
  "selected_ensemble_models",
  "arima_adf_stationarity",
  "lstm_features",
  "ensemble_weighted_base_models",
  "ensemble_boosting_base_models",
  "ensemble_stacking_base_models",
  "production_mps_table",
]);

export interface ModelTrainingRunRow extends RowDataPacket {
  run_id: string;
  session_id: string | null;
  experiment_id: number;
  student_id: number;
  model_type: ModelType;
  request_hash: string;
  dataset_revision: string;
  runtime_revision: string;
  parameters: unknown;
  metrics: unknown;
  artifact_path: string;
  artifact_sha256: string | null;
  production_artifact_path: string | null;
  production_artifact_sha256: string | null;
  production_artifact_forecast_steps: number | null;
  production_revision: string | null;
  is_active: number | boolean;
}

export interface CompletedGuidedSessionUpdate {
  sessionId: string;
  activeAttemptId?: string;
  committedStatePath?: string;
  committedStateSha256?: string;
  currentStepId: string | null;
  completedStepIds: string[];
  stepOutputs: Record<string, unknown>;
  stepPlan: unknown[];
  result: {
    status: string;
    results: Record<string, unknown>;
    inferred_feature_types?: unknown;
    experiment_state_patch?: Record<string, unknown>;
  };
}

interface CommitModelTrainingRunInput {
  runId?: string;
  sessionId?: string | null;
  experimentId: number;
  studentId: number;
  modelType: GuidedModelType;
  requestHash: string;
  datasetRevision: string;
  runtimeRevision: string;
  body: TrainInput;
  result: {
    status: string;
    results: Record<string, unknown>;
    inferred_feature_types?: unknown;
    experiment_state_patch?: Record<string, unknown>;
  };
  artifactPath: string;
  artifactSha256?: string | null;
  guidedSession?: CompletedGuidedSessionUpdate;
}

export interface CommitModelTrainingRunResult {
  runId: string;
  stateVersion: number;
  reusedExistingRun: boolean;
}

const parseJson = <T>(value: unknown, fallback: T): T => {
  if (value === null || value === undefined) return fallback;
  if (typeof value !== "string") return value as T;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
};

const stableJsonStringify = (value: unknown): string => {
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableJsonStringify(item)).join(",")}]`;
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableJsonStringify(record[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
};

const requireFiniteMetric = (metrics: Record<string, unknown>, key: "rmse" | "mae" | "r2") => {
  const value = Number(metrics[key]);
  if (!Number.isFinite(value)) {
    throw new ModelValidationError(`模型训练结果缺少有效的 ${key.toUpperCase()} 指标`);
  }
  return value;
};

const parseRequestedBaseModels = (models: string | undefined): BaseModelType[] => {
  const parsed = (models ?? "")
    .split(",")
    .map((model) => model.trim())
    .filter((model) => model.length > 0);
  if (parsed.length < 2) {
    throw new ModelValidationError("融合模型至少需要两个基础模型");
  }
  if (new Set(parsed).size !== parsed.length) {
    throw new ModelValidationError("融合模型的基础模型不能重复");
  }
  if (parsed.some((model) => !BASE_MODELS.includes(model as BaseModelType))) {
    throw new ModelValidationError("融合模型包含不支持的基础模型");
  }
  return parsed as BaseModelType[];
};

const parseLstmFeatures = (value: unknown): string[] => {
  const normalize = (items: unknown[]) => items
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter((item) => item.length > 0);

  if (Array.isArray(value)) return normalize(value);
  if (typeof value !== "string") return [];

  const trimmed = value.trim();
  if (!trimmed) return [];
  const parsed = parseJson<unknown>(trimmed, null);
  if (Array.isArray(parsed)) return normalize(parsed);

  // The public training API historically accepted comma-separated feature
  // names. Keep that contract for legacy callers while JSON remains the
  // unambiguous representation for field names that themselves contain commas.
  return normalize(trimmed.split(","));
};

const normalizeRevisionValue = (field: string, value: unknown) => {
  if (field === "lstm_features") {
    return parseLstmFeatures(value);
  }
  if (field.endsWith("_base_models")) {
    return parseJson(value, value ?? null);
  }
  return value ?? null;
};

export const buildExperimentTrainingRevision = (
  modelType: GuidedModelType,
  experiment: Record<string, unknown>,
  body: TrainInput,
) => {
  const fields = new Set<string>(TRAINING_CONTEXT_FIELDS);
  if (BASE_MODELS.includes(modelType as BaseModelType)) {
    BASE_MODEL_CONFIG_FIELDS[modelType as BaseModelType].forEach((field) => fields.add(field));
  } else {
    fields.add(ENSEMBLE_CONFIG_FIELDS[modelType as EnsembleModelType]);
    for (const baseModel of parseRequestedBaseModels(body.models)) {
      BASE_MODEL_CONFIG_FIELDS[baseModel].forEach((field) => fields.add(field));
    }
  }

  const snapshot: Record<string, unknown> = {};
  for (const field of [...fields].sort()) {
    snapshot[field] = normalizeRevisionValue(field, experiment[field]);
  }
  return crypto.createHash("sha256").update(stableJsonStringify(snapshot)).digest("hex");
};

export const buildModelTrainingRequestHash = (input: {
  modelType: GuidedModelType;
  body: TrainInput;
  datasetRevision: string;
  runtimeRevision: string;
  experimentRevision: string;
  baseRunFingerprint: Record<string, string>;
}) => crypto
  .createHash("sha256")
  .update(stableJsonStringify(input))
  .digest("hex");

const productionResetPatch = (): Record<string, unknown> => ({
  selected_best_model: null,
  quiz_about_model_completed: 0,
  quiz_about_plan_completed: 0,
  production_plan_completed: 0,
  production_forecast_periods: null,
  production_initial_inventory: null,
  production_target_service_level: null,
  production_safety_stock_z_score: null,
  production_mps_table: null,
  production_forecast_results: null,
  production_capacity_mode: null,
  production_capacity_scenario: null,
  production_capacity: null,
  production_custom_capacity: null,
});

const ensembleResetPatch = (modelType?: EnsembleModelType): Record<string, unknown> => {
  const configs: Record<EnsembleModelType, string> = {
    weighted_avg: "ensemble_weighted",
    boosting: "ensemble_boosting",
    stacking: "ensemble_stacking",
  };
  const targetModels = modelType ? [modelType] : ENSEMBLE_MODELS;
  const patch: Record<string, unknown> = {};
  for (const targetModel of targetModels) {
    const prefix = configs[targetModel];
    patch[`${prefix}_completed`] = 0;
    patch[`${prefix}_base_models`] = null;
    patch[`${prefix}_metrics_rmse`] = null;
    patch[`${prefix}_metrics_mae`] = null;
    patch[`${prefix}_metrics_r2`] = null;
  }
  return patch;
};

const baseResetPatch = (modelType: BaseModelType, preserveConfig = true): Record<string, unknown> => {
  switch (modelType) {
    case "ma":
      return {
        moving_average_completed: 0,
        ...(preserveConfig ? {} : { moving_average_window: null }),
        moving_average_metrics_rmse: null,
        moving_average_metrics_mae: null,
        moving_average_metrics_r2: null,
      };
    case "es":
      return {
        exponential_smoothing_completed: 0,
        ...(preserveConfig ? {} : { exponential_smoothing_alpha: null }),
        exponential_smoothing_metrics_rmse: null,
        exponential_smoothing_metrics_mae: null,
        exponential_smoothing_metrics_r2: null,
      };
    case "arima":
      return {
        arima_completed: 0,
        arima_p: null,
        ...(preserveConfig ? {} : { arima_d: null, arima_adf_stationarity: null }),
        arima_q: null,
        arima_metrics_rmse: null,
        arima_metrics_mae: null,
        arima_metrics_r2: null,
      };
    case "lstm":
      return {
        lstm_completed: 0,
        ...(preserveConfig ? {} : {
          lstm_normalization: null,
          lstm_features: null,
          lstm_target_field: null,
        }),
        lstm_metrics_rmse: null,
        lstm_metrics_mae: null,
        lstm_metrics_r2: null,
      };
  }
};

const allModelResetPatch = () => ({
  ...baseResetPatch("ma", false),
  ...baseResetPatch("es", false),
  ...baseResetPatch("arima", false),
  ...baseResetPatch("lstm", false),
  ...ensembleResetPatch(),
  ...productionResetPatch(),
});

export const buildAuthoritativeResult = (
  modelType: GuidedModelType,
  body: TrainInput,
  result: CommitModelTrainingRunInput["result"],
) => {
  const resultMetrics = result.results.metrics;
  if (!resultMetrics || typeof resultMetrics !== "object" || Array.isArray(resultMetrics)) {
    throw new ModelValidationError("模型训练结果缺少评价指标");
  }
  const metricsRecord = resultMetrics as Record<string, unknown>;
  const metrics = {
    rmse: requireFiniteMetric(metricsRecord, "rmse"),
    mae: requireFiniteMetric(metricsRecord, "mae"),
    r2: requireFiniteMetric(metricsRecord, "r2"),
  };

  const parameters: Record<string, unknown> = {};
  const experimentPatch: Record<string, unknown> = {
    ...productionResetPatch(),
  };

  switch (modelType) {
    case "ma":
      parameters.moving_average_window = body.moving_average_window;
      Object.assign(experimentPatch, {
        moving_average_completed: 1,
        moving_average_window: body.moving_average_window,
        moving_average_metrics_rmse: metrics.rmse,
        moving_average_metrics_mae: metrics.mae,
        moving_average_metrics_r2: metrics.r2,
        ...ensembleResetPatch(),
      });
      break;
    case "es":
      parameters.exponential_smoothing_alpha = body.exponential_smoothing_alpha;
      Object.assign(experimentPatch, {
        exponential_smoothing_completed: 1,
        exponential_smoothing_alpha: body.exponential_smoothing_alpha,
        exponential_smoothing_metrics_rmse: metrics.rmse,
        exponential_smoothing_metrics_mae: metrics.mae,
        exponential_smoothing_metrics_r2: metrics.r2,
        ...ensembleResetPatch(),
      });
      break;
    case "arima": {
      const order = result.results.best_order;
      if (!order || typeof order !== "object" || Array.isArray(order)) {
        throw new ModelValidationError("ARIMA 训练结果缺少有效的最优阶数");
      }
      const orderRecord = order as Record<string, unknown>;
      const p = Number(orderRecord.p);
      const q = Number(orderRecord.q);
      const d = Number(body.arimaD);
      if (![p, d, q].every(Number.isInteger)) {
        throw new ModelValidationError("ARIMA 训练结果中的阶数无效");
      }
      Object.assign(parameters, { p, d, q });
      Object.assign(experimentPatch, {
        arima_completed: 1,
        arima_p: p,
        arima_d: d,
        arima_q: q,
        arima_metrics_rmse: metrics.rmse,
        arima_metrics_mae: metrics.mae,
        arima_metrics_r2: metrics.r2,
        ...ensembleResetPatch(),
      });
      break;
    }
    case "lstm": {
      const features = parseLstmFeatures(body.lstmFeatures);
      Object.assign(parameters, {
        normalization: body.lstmNormalization,
        target_field: body.lstmTargetFeature,
        features,
        look_back: result.results.look_back ?? null,
        epochs: result.results.epochs ?? null,
        forecast_horizon: result.results.forecast_horizon ?? null,
      });
      Object.assign(experimentPatch, {
        lstm_completed: 1,
        lstm_normalization: body.lstmNormalization,
        lstm_features: JSON.stringify(features),
        lstm_target_field: body.lstmTargetFeature,
        lstm_metrics_rmse: metrics.rmse,
        lstm_metrics_mae: metrics.mae,
        lstm_metrics_r2: metrics.r2,
        ...ensembleResetPatch(),
      });
      break;
    }
    case "weighted_avg":
    case "boosting":
    case "stacking": {
      const requestedModels = parseRequestedBaseModels(body.models);
      parameters.models = requestedModels;
      const prefix = modelType === "weighted_avg"
        ? "ensemble_weighted"
        : modelType === "boosting"
          ? "ensemble_boosting"
          : "ensemble_stacking";
      Object.assign(experimentPatch, {
        [`${prefix}_completed`]: 1,
        [`${prefix}_base_models`]: JSON.stringify(requestedModels.map((model) => BASE_MODEL_FRONTEND_IDS[model])),
        [`${prefix}_metrics_rmse`]: metrics.rmse,
        [`${prefix}_metrics_mae`]: metrics.mae,
        [`${prefix}_metrics_r2`]: metrics.r2,
      });
      break;
    }
  }

  return { parameters, metrics, experimentPatch };
};

const toClientExperimentPatch = (patch: Record<string, unknown>) => {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(patch)) {
    if (key.endsWith("_completed")) {
      result[key] = value === 1 || value === true;
      continue;
    }
    if (CLIENT_ARRAY_STATE_FIELDS.has(key)) {
      result[key] = parseJson(value, []);
      continue;
    }
    if (key === "production_forecast_results") {
      result[key] = parseJson(value, null);
      continue;
    }
    result[key] = value;
  }
  return result;
};

export const buildCurrentExperimentStatePatch = (
  experiment: Record<string, unknown>,
  stateVersion: number,
) => {
  const currentExperimentState = Object.fromEntries(
    CLIENT_EXPERIMENT_STATE_FIELDS.map((field) => [field, experiment[field] ?? null]),
  );
  return {
    ...toClientExperimentPatch(currentExperimentState),
    state_version: stateVersion,
  };
};

export const loadCurrentExperimentStatePatch = async (
  experimentId: number,
  studentId: number,
) => {
  const [rows] = await pool.query<RowDataPacket[]>(
    `SELECT *
       FROM experiment_status
      WHERE experiment_id = ?
        AND student_id = ?
      LIMIT 1`,
    [experimentId, studentId],
  );
  const experiment = rows[0];
  if (!experiment) {
    throw new ModelValidationError("实验不存在或不属于当前用户");
  }

  const stateVersion = Number(experiment.state_version ?? 0);
  if (!Number.isSafeInteger(stateVersion) || stateVersion < 0) {
    throw new ModelValidationError("实验状态版本无效");
  }
  return buildCurrentExperimentStatePatch(experiment, stateVersion);
};

const updateExperimentWithPatch = async (
  connection: PoolConnection,
  experimentId: number,
  studentId: number,
  patch: Record<string, unknown>,
) => {
  const columns = Object.keys(patch);
  const assignments = columns.map((column) => `${column} = ?`).join(", ");
  await connection.query(
    `UPDATE experiment_status
        SET ${assignments},
            state_version = state_version + 1,
            last_activity_at = NOW()
      WHERE experiment_id = ?
        AND student_id = ?`,
    [...columns.map((column) => patch[column]), experimentId, studentId],
  );
  const [rows] = await connection.query<RowDataPacket[]>(
    `SELECT state_version
       FROM experiment_status
      WHERE experiment_id = ?
        AND student_id = ?
      LIMIT 1`,
    [experimentId, studentId],
  );
  return Number(rows[0]?.state_version ?? 0);
};

const getExistingRunForSession = async (connection: PoolConnection, sessionId: string | null | undefined) => {
  if (!sessionId) return null;
  const [rows] = await connection.query<ModelTrainingRunRow[]>(
    `SELECT *
       FROM model_training_runs
      WHERE session_id = ?
      LIMIT 1
      FOR UPDATE`,
    [sessionId],
  );
  return rows[0] ?? null;
};

const deactivateRuns = async (
  connection: PoolConnection,
  experimentId: number,
  studentId: number,
  modelTypes: readonly ModelType[],
) => {
  if (modelTypes.length === 0) return;
  const placeholders = modelTypes.map(() => "?").join(", ");
  await connection.query(
    `UPDATE model_training_runs
        SET is_active = false
      WHERE experiment_id = ?
        AND student_id = ?
        AND model_type IN (${placeholders})
        AND is_active = true`,
    [experimentId, studentId, ...modelTypes],
  );
};

const replaceRunDependencies = async (
  connection: PoolConnection,
  runId: string,
  experimentId: number,
  studentId: number,
  requestedModels: BaseModelType[],
) => {
  await connection.query("DELETE FROM model_training_run_dependencies WHERE run_id = ?", [runId]);
  for (const baseModelType of requestedModels) {
    const [rows] = await connection.query<ModelTrainingRunRow[]>(
      `SELECT run_id
         FROM model_training_runs
        WHERE experiment_id = ?
          AND student_id = ?
          AND model_type = ?
          AND is_active = true
        ORDER BY activated_at DESC
        LIMIT 1`,
      [experimentId, studentId, baseModelType],
    );
    await connection.query(
      `INSERT INTO model_training_run_dependencies (run_id, base_model_type, base_run_id)
       VALUES (?, ?, ?)`,
      [runId, baseModelType, rows[0]?.run_id ?? null],
    );
  }
};

const assertExistingEnsembleDependenciesCurrent = async (
  connection: PoolConnection,
  runId: string,
  experimentId: number,
  studentId: number,
  requestedModels: BaseModelType[],
) => {
  const [dependencyRows] = await connection.query<RowDataPacket[]>(
    `SELECT base_model_type, base_run_id
       FROM model_training_run_dependencies
      WHERE run_id = ?`,
    [runId],
  );
  const dependencyMap = new Map(
    dependencyRows.map((row) => [String(row.base_model_type), row.base_run_id ? String(row.base_run_id) : null]),
  );
  for (const baseModelType of requestedModels) {
    const [activeRows] = await connection.query<RowDataPacket[]>(
      `SELECT run_id
         FROM model_training_runs
        WHERE experiment_id = ?
          AND student_id = ?
          AND model_type = ?
          AND is_active = true
        ORDER BY activated_at DESC
        LIMIT 1`,
      [experimentId, studentId, baseModelType],
    );
    const currentRunId = activeRows[0]?.run_id ? String(activeRows[0].run_id) : null;
    if (dependencyMap.get(baseModelType) !== currentRunId) {
      throw new ModelValidationError("基础模型版本已变化，请重新创建融合训练会话");
    }
  }
};

const buildCurrentBaseRunFingerprint = async (
  connection: PoolConnection,
  experimentId: number,
  studentId: number,
  body: TrainInput,
) => {
  const fingerprint: Record<string, string> = {};
  for (const baseModelType of parseRequestedBaseModels(body.models)) {
    const [activeRows] = await connection.query<RowDataPacket[]>(
      `SELECT run_id
         FROM model_training_runs
        WHERE experiment_id = ?
          AND student_id = ?
          AND model_type = ?
          AND is_active = true
        ORDER BY activated_at DESC
        LIMIT 1`,
      [experimentId, studentId, baseModelType],
    );
    fingerprint[baseModelType] = activeRows[0]?.run_id
      ? String(activeRows[0].run_id)
      : "legacy";
  }
  return fingerprint;
};

const assertTrainingRequestStillCurrent = async (
  connection: PoolConnection,
  input: CommitModelTrainingRunInput,
  experiment: RowDataPacket,
) => {
  const experimentRevision = buildExperimentTrainingRevision(
    input.modelType,
    experiment,
    input.body,
  );
  const baseRunFingerprint = ENSEMBLE_MODELS.includes(input.modelType as EnsembleModelType)
    ? await buildCurrentBaseRunFingerprint(
      connection,
      input.experimentId,
      input.studentId,
      input.body,
    )
    : {};
  const currentRequestHash = buildModelTrainingRequestHash({
    modelType: input.modelType,
    body: input.body,
    datasetRevision: input.datasetRevision,
    runtimeRevision: input.runtimeRevision,
    experimentRevision,
    baseRunFingerprint,
  });
  if (currentRequestHash !== input.requestHash) {
    throw new ModelValidationError("训练期间实验配置或基础模型版本已变化，请重新创建训练会话");
  }
};

export async function commitModelTrainingRun(
  input: CommitModelTrainingRunInput,
): Promise<CommitModelTrainingRunResult> {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();
    const [experimentRows] = await connection.query<RowDataPacket[]>(
      `SELECT *
         FROM experiment_status
        WHERE experiment_id = ?
        LIMIT 1
        FOR UPDATE`,
      [input.experimentId],
    );
    const experiment = experimentRows[0];
    if (!experiment || Number(experiment.student_id) !== input.studentId) {
      throw new ModelValidationError("实验不存在或不属于当前用户");
    }
    if (experiment.status === "Completed") {
      throw new ModelValidationError("已完成的实验不允许继续写入模型结果");
    }

    const { parameters, metrics, experimentPatch } = buildAuthoritativeResult(
      input.modelType,
      input.body,
      input.result,
    );
    const existingRun = await getExistingRunForSession(connection, input.sessionId);
    if (existingRun && Boolean(existingRun.is_active)) {
      if (
        input.artifactSha256
        && existingRun.artifact_sha256
        && existingRun.artifact_sha256.toLowerCase() !== input.artifactSha256.toLowerCase()
      ) {
        throw new ModelValidationError("已提交训练版本的模型产物校验和不一致，请重新创建训练会话");
      }
      const stateVersion = Number(experiment.state_version ?? 0);
      input.result.experiment_state_patch = buildCurrentExperimentStatePatch(
        experiment,
        stateVersion,
      );
      if (input.guidedSession) {
        await connection.query(
          `UPDATE model_guided_training_sessions
              SET experiment_state_version = ?,
                  result_payload = ?
            WHERE session_id = ?`,
          [stateVersion, JSON.stringify(input.result), input.guidedSession.sessionId],
        );
      }
      if (input.artifactSha256) {
        await connection.query(
          `UPDATE model_training_runs
              SET artifact_sha256 = COALESCE(artifact_sha256, ?)
            WHERE run_id = ?`,
          [input.artifactSha256, existingRun.run_id],
        );
      }
      await connection.commit();
      return { runId: existingRun.run_id, stateVersion, reusedExistingRun: true };
    }

    if (input.guidedSession?.activeAttemptId) {
      const [sessionRows] = await connection.query<RowDataPacket[]>(
        `SELECT active_attempt_id, status
           FROM model_guided_training_sessions
          WHERE session_id = ?
          LIMIT 1
          FOR UPDATE`,
        [input.guidedSession.sessionId],
      );
      const session = sessionRows[0];
      if (!session || session.active_attempt_id !== input.guidedSession.activeAttemptId) {
        throw new ModelValidationError("训练步骤 attempt 已被替换，已放弃本次提交");
      }
      if (session.status === "superseded") {
        throw new ModelValidationError("训练会话已失效，已放弃本次提交");
      }
    }

    await assertTrainingRequestStillCurrent(connection, input, experiment);

    if (existingRun) {
      const [newerActiveRows] = await connection.query<RowDataPacket[]>(
        `SELECT run_id
           FROM model_training_runs
          WHERE experiment_id = ?
            AND student_id = ?
            AND model_type = ?
            AND is_active = true
            AND run_id <> ?
          LIMIT 1`,
        [input.experimentId, input.studentId, input.modelType, existingRun.run_id],
      );
      if (newerActiveRows.length > 0) {
        throw new ModelValidationError("该训练会话已被更新的模型版本取代，请重新创建训练会话");
      }
    }

    const runId = existingRun?.run_id ?? input.runId ?? crypto.randomUUID();

    if (existingRun && ENSEMBLE_MODELS.includes(input.modelType as EnsembleModelType)) {
      await assertExistingEnsembleDependenciesCurrent(
        connection,
        existingRun.run_id,
        input.experimentId,
        input.studentId,
        parseRequestedBaseModels(input.body.models),
      );
    }

    await deactivateRuns(connection, input.experimentId, input.studentId, [input.modelType]);
    if (BASE_MODELS.includes(input.modelType as BaseModelType)) {
      await deactivateRuns(connection, input.experimentId, input.studentId, ENSEMBLE_MODELS);
    }

    if (existingRun) {
      await connection.query(
        `UPDATE model_training_runs
            SET artifact_path = ?,
                artifact_sha256 = ?,
                parameters = ?,
                metrics = ?,
                production_artifact_path = NULL,
                production_artifact_sha256 = NULL,
                production_artifact_forecast_steps = NULL,
                production_revision = NULL,
                is_active = true,
                activated_at = NOW()
          WHERE run_id = ?`,
        [input.artifactPath, input.artifactSha256 ?? null, JSON.stringify(parameters), JSON.stringify(metrics), runId],
      );
    } else {
      await connection.query(
        `INSERT INTO model_training_runs
           (run_id, session_id, experiment_id, student_id, model_type, request_hash,
            dataset_revision, runtime_revision, parameters, metrics, artifact_path, artifact_sha256, is_active)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, true)`,
        [
          runId,
          input.sessionId ?? null,
          input.experimentId,
          input.studentId,
          input.modelType,
          input.requestHash,
          input.datasetRevision,
          input.runtimeRevision,
          JSON.stringify(parameters),
          JSON.stringify(metrics),
          input.artifactPath,
          input.artifactSha256 ?? null,
        ],
      );
    }

    if (ENSEMBLE_MODELS.includes(input.modelType as EnsembleModelType)) {
      await replaceRunDependencies(
        connection,
        runId,
        input.experimentId,
        input.studentId,
        parseRequestedBaseModels(input.body.models),
      );
    }

    const stateVersion = await updateExperimentWithPatch(
      connection,
      input.experimentId,
      input.studentId,
      experimentPatch,
    );
    input.result.experiment_state_patch = {
      ...toClientExperimentPatch(experimentPatch),
      state_version: stateVersion,
    };

    if (input.guidedSession) {
      await connection.query(
        `UPDATE model_guided_training_sessions
            SET status = 'completed',
                current_step_id = ?,
                completed_step_ids = ?,
                step_outputs = ?,
                step_plan = ?,
                result_payload = ?,
                error_message = NULL,
                experiment_state_version = ?,
                committed_state_path = COALESCE(?, committed_state_path),
                committed_state_sha256 = COALESCE(?, committed_state_sha256),
                active_attempt_id = NULL
          WHERE session_id = ?`,
        [
          input.guidedSession.currentStepId,
          JSON.stringify(input.guidedSession.completedStepIds),
          JSON.stringify(input.guidedSession.stepOutputs),
          JSON.stringify(input.guidedSession.stepPlan),
          JSON.stringify(input.guidedSession.result),
          stateVersion,
          input.guidedSession.committedStatePath ?? null,
          input.guidedSession.committedStateSha256 ?? null,
          input.guidedSession.sessionId,
        ],
      );
      if (input.guidedSession.activeAttemptId) {
        await connection.query(
          `UPDATE model_guided_training_step_attempts
              SET status = 'committed',
                  candidate_state_sha256 = COALESCE(?, candidate_state_sha256),
                  output_payload = ?,
                  final_result_payload = ?,
                  error_message = NULL
            WHERE attempt_id = ?`,
          [
            input.guidedSession.committedStateSha256 ?? null,
            JSON.stringify(
              input.guidedSession.stepOutputs[
                input.guidedSession.completedStepIds.at(-1) ?? ""
              ] ?? {},
            ),
            JSON.stringify(input.guidedSession.result),
            input.guidedSession.activeAttemptId,
          ],
        );
      }
    }

    await connection.commit();
    return { runId, stateVersion, reusedExistingRun: Boolean(existingRun) };
  } catch (error) {
    await connection.rollback().catch(() => {});
    throw error;
  } finally {
    connection.release();
  }
}

export async function getActiveModelTrainingRun(
  experimentId: number,
  studentId: number,
  modelType: ModelType,
): Promise<ModelTrainingRunRow | null> {
  const [rows] = await pool.query<ModelTrainingRunRow[]>(
    `SELECT *
       FROM model_training_runs
      WHERE experiment_id = ?
        AND student_id = ?
        AND model_type = ?
        AND is_active = true
      ORDER BY activated_at DESC
      LIMIT 1`,
    [experimentId, studentId, modelType],
  );
  return rows[0] ?? null;
}

export async function setProductionArtifactForRun(
  runId: string,
  artifactPath: string,
  forecastSteps: number,
  productionRevision: string,
  productionArtifactSha256: string | null = null,
) {
  const [result] = await pool.query(
    `UPDATE model_training_runs
        SET production_artifact_path = ?,
            production_artifact_sha256 = ?,
            production_artifact_forecast_steps = ?,
            production_revision = ?
      WHERE run_id = ?
        AND is_active = true`,
    [artifactPath, productionArtifactSha256, forecastSteps, productionRevision, runId],
  );
  return Number((result as { affectedRows?: number }).affectedRows ?? 0) === 1;
}

export async function backfillArtifactSha256ForRun(runId: string, artifactSha256: string) {
  const [result] = await pool.query(
    `UPDATE model_training_runs
        SET artifact_sha256 = ?
      WHERE run_id = ?
        AND is_active = true
        AND artifact_sha256 IS NULL`,
    [artifactSha256, runId],
  );
  return Number((result as { affectedRows?: number }).affectedRows ?? 0) === 1;
}

export async function backfillProductionArtifactSha256ForRun(
  runId: string,
  productionRevision: string,
  productionArtifactSha256: string,
) {
  const [result] = await pool.query(
    `UPDATE model_training_runs
        SET production_artifact_sha256 = ?
      WHERE run_id = ?
        AND is_active = true
        AND production_revision = ?
        AND production_artifact_sha256 IS NULL`,
    [productionArtifactSha256, runId, productionRevision],
  );
  return Number((result as { affectedRows?: number }).affectedRows ?? 0) === 1;
}

export async function clearProductionArtifactForRun(
  runId: string,
  productionRevision: string,
) {
  const [result] = await pool.query(
    `UPDATE model_training_runs
        SET production_artifact_path = NULL,
            production_artifact_sha256 = NULL,
            production_artifact_forecast_steps = NULL,
            production_revision = NULL
      WHERE run_id = ?
        AND is_active = true
        AND production_revision = ?`,
    [runId, productionRevision],
  );
  return Number((result as { affectedRows?: number }).affectedRows ?? 0) === 1;
}

const CONTEXT_FIELDS = new Set([
  "selected_industry",
  "selected_company",
  "selected_product",
  "data_window_train_start_index",
  "data_window_train_end_index",
  "data_window_evaluate_start_index",
  "data_window_evaluate_end_index",
]);

const MODEL_CONFIG_FIELDS: Record<BaseModelType | EnsembleModelType, Set<string>> = {
  ma: new Set(["moving_average_window"]),
  es: new Set(["exponential_smoothing_alpha"]),
  arima: new Set(["arima_d"]),
  lstm: new Set(["lstm_normalization", "lstm_features", "lstm_target_field"]),
  weighted_avg: new Set(["ensemble_weighted_base_models"]),
  boosting: new Set(["ensemble_boosting_base_models"]),
  stacking: new Set(["ensemble_stacking_base_models"]),
};

export function getServerInvalidationPatch(changedFields: ReadonlySet<string>) {
  if ([...CONTEXT_FIELDS].some((field) => changedFields.has(field))) {
    return {
      modelsToDeactivate: [...BASE_MODELS, ...ENSEMBLE_MODELS] as ModelType[],
      patch: allModelResetPatch(),
    };
  }

  const modelsToDeactivate = new Set<ModelType>();
  const patch: Record<string, unknown> = {};
  for (const [modelType, fields] of Object.entries(MODEL_CONFIG_FIELDS) as Array<[
    BaseModelType | EnsembleModelType,
    Set<string>,
  ]>) {
    if (![...fields].some((field) => changedFields.has(field))) continue;
    modelsToDeactivate.add(modelType);
    if (BASE_MODELS.includes(modelType as BaseModelType)) {
      ENSEMBLE_MODELS.forEach((ensemble) => modelsToDeactivate.add(ensemble));
      Object.assign(patch, baseResetPatch(modelType as BaseModelType), ensembleResetPatch());
    } else {
      Object.assign(patch, ensembleResetPatch(modelType as EnsembleModelType));
    }
    Object.assign(patch, productionResetPatch());
  }

  if (changedFields.has("selected_best_model")) {
    Object.assign(patch, productionResetPatch());
    delete patch.selected_best_model;
  }

  return { modelsToDeactivate: [...modelsToDeactivate], patch };
}

export async function deactivateModelRunsForStudentUpdate(
  connection: PoolConnection,
  experimentId: number,
  studentId: number,
  modelTypes: readonly ModelType[],
) {
  await deactivateRuns(connection, experimentId, studentId, modelTypes);
}
