import path from "path";
import fs from "fs";
import crypto from "crypto";
import type { Stats } from "node:fs";
import type { PoolConnection, ResultSetHeader, RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import logger from "../../utils/logger";
import {
  MAX_FORECAST_STEPS,
  MIN_EVALUATION_POINTS,
  type ModelType,
  type TrainInput,
} from "./models.schema";
import { TIMEOUTS, CLEANUP_RETENTION } from "../../utils/constants";
import { PY_DIR, resolvePythonExecutable } from "../../utils/python";
import {
  GUIDED_TRAINING_ROOT,
  buildGuidedTrainingArtifactDir,
  buildGuidedTrainingStatePath,
  buildVersionedModelArtifactPath,
} from "./model-runtime";
import { loadDatasetFromDb } from "./model-datasets";
import {
  ModelAccessError,
  ModelCapacityError,
  ModelConflictError,
  ModelNotFoundError,
  ModelStaleAttemptError,
  ModelValidationError,
} from "./model-errors";
import {
  acquireModelArtifactMutationLocks,
  acquireModelSlot,
  releaseModelArtifactMutationLocks,
  releaseModelSlot,
} from "./model-locks";
import {
  allPossiblePythonParams,
  buildPythonModelProcessEnv,
  reportModelProgress,
  runPythonGuidedTrainingStep,
  runPythonModel,
  type ModelProgressReporter,
} from "./model-python-runner";
import { transformTrainingResult } from "./model-results";
import {
  BASE_MODEL_TYPES,
  ENSEMBLE_DERIVED_BODY_KEYS,
  ENSEMBLE_MODEL_TYPES,
  GUIDED_MODEL_TYPES,
  GUIDED_TRAINING_STEPS,
  type BaseModelType,
  type EnsembleModelType,
  type GuidedModelType,
  type GuidedTrainingSessionPayload,
  type GuidedTrainingStepDefinition,
  type GuidedTrainingStatus,
} from "./model-types";

export { loadDatasetFromDb } from "./model-datasets";
export {
  ModelAccessError,
  ModelCapacityError,
  ModelConflictError,
  ModelNotFoundError,
  ModelStaleAttemptError,
  ModelValidationError,
} from "./model-errors";
export {
  acquireModelArtifactMutationLocks,
  acquireModelExecutionLock,
  acquireModelSlot,
  releaseModelArtifactMutationLocks,
  releaseModelExecutionLock,
  releaseModelSlot,
} from "./model-locks";
export { runPythonGuidedTrainingStep, runPythonModel, type ModelProgressEvent, type ModelProgressReporter } from "./model-python-runner";
export { transformTrainingResult } from "./model-results";
export type { GuidedTrainingSessionPayload, GuidedTrainingStepDefinition } from "./model-types";

const TRAINING_TIMEOUT = TIMEOUTS.MODEL_TRAINING;
const GUIDED_TRAINING_CLEANUP_BATCH_SIZE = 500;
const GUIDED_STEP_LEASE_SECONDS = 90;
const GUIDED_STEP_HEARTBEAT_INTERVAL_MS = 30_000;

interface ExperimentOwnershipRow extends RowDataPacket {
  student_id: number;
  status: string;
  state_version: number | string;
  training_input_version: number | string;
}

interface ExperimentModelContextRow extends ExperimentOwnershipRow {
  selected_industry: string | null;
  selected_company: string | null;
  selected_product: string | null;
  data_window_train_start_index: number | null;
  data_window_train_end_index: number | null;
  data_window_evaluate_start_index: number | null;
  data_window_evaluate_end_index: number | null;
  moving_average_completed: number | boolean | null;
  moving_average_window: number | null;
  exponential_smoothing_completed: number | boolean | null;
  exponential_smoothing_alpha: number | null;
  arima_completed: number | boolean | null;
  arima_d: number | null;
  lstm_completed: number | boolean | null;
  lstm_normalization: "minmax" | "zscore" | null;
  lstm_features: unknown;
  lstm_target_field: string | null;
  ensemble_weighted_completed: number | boolean | null;
  ensemble_weighted_base_models: unknown;
  ensemble_boosting_completed: number | boolean | null;
  ensemble_boosting_base_models: unknown;
  ensemble_stacking_completed: number | boolean | null;
  ensemble_stacking_base_models: unknown;
}

interface ExperimentTrainingConfig {
  selected_industry: string;
  selected_company: string;
  selected_product: string;
  data_window_train_start_index: number;
  data_window_train_end_index: number;
  data_window_evaluate_start_index: number;
  data_window_evaluate_end_index: number;
}

interface GuidedTrainingSessionRow extends RowDataPacket {
  session_id: string;
  experiment_id: number;
  student_id: number;
  model_type: GuidedModelType;
  request_hash: string;
  request_payload: unknown;
  status: GuidedTrainingStatus;
  current_step_id: string | null;
  completed_step_ids: unknown;
  step_outputs: unknown;
  step_plan: unknown;
  result_payload: unknown;
  artifact_dir: string;
  error_message: string | null;
  training_input_version: number | string;
  run_token: string | null;
  run_version: number;
  lease_expires_at: Date | string | null;
  last_heartbeat_at: Date | string | null;
}

interface GuidedTrainingCleanupRow extends RowDataPacket {
  session_id: string;
  experiment_id: number;
  model_type: GuidedModelType;
  artifact_dir: string;
}

interface StaleModelExperimentEligibilityRow extends RowDataPacket {
  can_delete: 0 | 1 | boolean | string;
}

interface ActiveModelArtifactRow extends RowDataPacket {
  experiment_id: number;
  student_id: number;
  model_type: ModelType;
  revision_token: string;
  training_input_version: number | string;
  source_session_id: string | null;
  backtest_artifact_path: string;
  request_payload: unknown;
  result_payload: unknown;
  production_artifact_path: string | null;
  production_revision_token: string | null;
}

async function assertExperimentModelAccess(
  experimentId: number,
  studentId: number,
  connection?: PoolConnection,
  forUpdate = false,
) {
  const queryable = connection ?? pool;
  const [rows] = await queryable.query<ExperimentModelContextRow[]>(
    `SELECT student_id,
            status,
            state_version,
            training_input_version,
            selected_industry,
            selected_company,
            selected_product,
            data_window_train_start_index,
            data_window_train_end_index,
            data_window_evaluate_start_index,
            data_window_evaluate_end_index,
            moving_average_completed,
            moving_average_window,
            exponential_smoothing_completed,
            exponential_smoothing_alpha,
            arima_completed,
            arima_d,
            lstm_completed,
            lstm_normalization,
            lstm_features,
            lstm_target_field,
            ensemble_weighted_completed,
            ensemble_weighted_base_models,
            ensemble_boosting_completed,
            ensemble_boosting_base_models,
            ensemble_stacking_completed,
            ensemble_stacking_base_models
       FROM experiment_status
      WHERE experiment_id = ?
      LIMIT 1
      ${forUpdate ? "FOR UPDATE" : ""}`,
    [experimentId],
  );

  const experiment = rows[0];
  if (!experiment) {
    throw new ModelAccessError("实验不存在", 404);
  }

  if (experiment.student_id !== studentId) {
    throw new ModelAccessError("权限不足：该实验不属于当前用户", 403);
  }

  if (experiment.status === "Completed") {
    throw new ModelAccessError("已完成的实验不允许继续训练或预测模型", 400);
  }

  return experiment;
}

const hasDefinedValue = <T>(value: T | null | undefined): value is T => value !== null && value !== undefined;

const isDuplicateKeyError = (error: unknown) => {
  const dbError = error as { code?: unknown; errno?: unknown };
  return dbError?.code === "ER_DUP_ENTRY" || dbError?.errno === 1062;
};

const isBaseModelType = (modelType: ModelType): modelType is BaseModelType => {
  return BASE_MODEL_TYPES.has(modelType);
};

const isEnsembleModelType = (modelType: ModelType): modelType is Extract<ModelType, "weighted_avg" | "boosting" | "stacking"> => {
  return ENSEMBLE_MODEL_TYPES.has(modelType);
};

const getLstmSidecarPath = (modelPath: string) => modelPath.replace(/\.pkl$/, "_keras_model.h5");

const getGuidedSessionModelPath = (artifactDir: string, modelType: GuidedModelType) =>
  path.join(artifactDir, `${modelType}.pkl`);

const removeFileIfExists = (filePath: string) => {
  if (!fs.existsSync(filePath)) {
    return;
  }

  try {
    fs.unlinkSync(filePath);
  } catch (error) {
    logger.warn({ err: error, filePath }, "删除模型产物失败");
  }
};

const removeSiblingArtifactsByPrefix = (modelPath: string) => {
  const fileSystem = fs as typeof fs & { readdirSync?: (directoryPath: string) => string[] };
  if (typeof fileSystem.readdirSync !== "function") {
    return;
  }

  const directoryPath = path.dirname(modelPath);
  if (!fs.existsSync(directoryPath)) {
    return;
  }

  const baseName = path.basename(modelPath, ".pkl");
  for (const siblingFile of fileSystem.readdirSync(directoryPath)) {
    if (!siblingFile.startsWith(`${baseName}_`)) {
      continue;
    }

    removeFileIfExists(path.join(directoryPath, siblingFile));
  }
};

const invalidateArtifactAtPath = (modelType: ModelType, modelPath: string) => {
  removeFileIfExists(modelPath);
  if (modelType === "lstm") {
    removeFileIfExists(getLstmSidecarPath(modelPath));
    return;
  }

  if (isEnsembleModelType(modelType)) {
    removeSiblingArtifactsByPrefix(modelPath);
  }
};

const commonDownstreamStateResetValues: Record<string, unknown> = {
  selected_best_model: null,
  production_plan_completed: 0,
  production_forecast_periods: null,
  production_initial_inventory: null,
  production_target_service_level: null,
  production_safety_stock_z_score: null,
  production_forecast_results: null,
  production_mps_table: JSON.stringify([]),
  production_capacity_mode: null,
  production_capacity_scenario: null,
  production_capacity: null,
  production_custom_capacity: null,
  quiz_about_plan_completed: 0,
};

const modelStateResetValues: Record<ModelType, Record<string, unknown>> = {
  ma: {
    moving_average_completed: 0,
    moving_average_window: null,
    moving_average_metrics_rmse: null,
    moving_average_metrics_mae: null,
    moving_average_metrics_mape: null,
    moving_average_metrics_r2: null,
  },
  es: {
    exponential_smoothing_completed: 0,
    exponential_smoothing_alpha: null,
    exponential_smoothing_metrics_rmse: null,
    exponential_smoothing_metrics_mae: null,
    exponential_smoothing_metrics_mape: null,
    exponential_smoothing_metrics_r2: null,
  },
  arima: {
    arima_completed: 0,
    arima_p: null,
    arima_d: null,
    arima_q: null,
    arima_metrics_rmse: null,
    arima_metrics_mae: null,
    arima_metrics_mape: null,
    arima_metrics_r2: null,
  },
  lstm: {
    lstm_completed: 0,
    lstm_normalization: null,
    lstm_features: JSON.stringify([]),
    lstm_target_field: null,
    lstm_metrics_rmse: null,
    lstm_metrics_mae: null,
    lstm_metrics_mape: null,
    lstm_metrics_r2: null,
  },
  weighted_avg: {
    ensemble_weighted_completed: 0,
    ensemble_weighted_base_models: JSON.stringify([]),
    ensemble_weighted_metrics_rmse: null,
    ensemble_weighted_metrics_mae: null,
    ensemble_weighted_metrics_mape: null,
    ensemble_weighted_metrics_r2: null,
  },
  boosting: {
    ensemble_boosting_completed: 0,
    ensemble_boosting_base_models: JSON.stringify([]),
    ensemble_boosting_metrics_rmse: null,
    ensemble_boosting_metrics_mae: null,
    ensemble_boosting_metrics_mape: null,
    ensemble_boosting_metrics_r2: null,
  },
  stacking: {
    ensemble_stacking_completed: 0,
    ensemble_stacking_base_models: JSON.stringify([]),
    ensemble_stacking_metrics_rmse: null,
    ensemble_stacking_metrics_mae: null,
    ensemble_stacking_metrics_mape: null,
    ensemble_stacking_metrics_r2: null,
  },
};

const toFiniteMetric = (value: unknown) => (
  typeof value === "number" && Number.isFinite(value) ? value : null
);

const toFrontendBaseModelIds = (models: string | undefined) => {
  const aliases: Record<string, string> = {
    ma: "moving_average",
    es: "exponential_smoothing",
    arima: "arima",
    lstm: "lstm",
  };
  return (models ?? "")
    .split(",")
    .map((model) => model.trim())
    .filter((model) => model in aliases)
    .map((model) => aliases[model]!);
};

const buildActivatedModelStateValues = (
  modelType: ModelType,
  request: TrainInput,
  result: ReturnType<typeof transformTrainingResult>,
) => {
  const resultValues = result.results as Record<string, unknown>;
  const metrics = (resultValues.metrics ?? {}) as Record<string, unknown>;
  const metricValues = {
    rmse: toFiniteMetric(metrics.rmse),
    mae: toFiniteMetric(metrics.mae),
    mape: toFiniteMetric(metrics.mape),
    r2: toFiniteMetric(metrics.r2),
  };
  const resetValues: Record<string, unknown> = {
    ...modelStateResetValues[modelType],
    ...(isBaseModelType(modelType)
      ? Object.assign({}, ...[...ENSEMBLE_MODEL_TYPES].map((ensembleType) => modelStateResetValues[ensembleType]))
      : {}),
    ...commonDownstreamStateResetValues,
  };

  switch (modelType) {
    case "ma":
      return {
        ...resetValues,
        moving_average_window: request.moving_average_window ?? null,
        moving_average_metrics_rmse: metricValues.rmse,
        moving_average_metrics_mae: metricValues.mae,
        moving_average_metrics_mape: metricValues.mape,
        moving_average_metrics_r2: metricValues.r2,
      };
    case "es":
      return {
        ...resetValues,
        exponential_smoothing_alpha: request.exponential_smoothing_alpha ?? null,
        exponential_smoothing_metrics_rmse: metricValues.rmse,
        exponential_smoothing_metrics_mae: metricValues.mae,
        exponential_smoothing_metrics_mape: metricValues.mape,
        exponential_smoothing_metrics_r2: metricValues.r2,
      };
    case "arima": {
      const bestOrder = (resultValues.best_order ?? {}) as Record<string, unknown>;
      return {
        ...resetValues,
        arima_p: toFiniteMetric(bestOrder.p),
        arima_d: request.arimaD ?? null,
        arima_q: toFiniteMetric(bestOrder.q),
        arima_metrics_rmse: metricValues.rmse,
        arima_metrics_mae: metricValues.mae,
        arima_metrics_mape: metricValues.mape,
        arima_metrics_r2: metricValues.r2,
      };
    }
    case "lstm":
      return {
        ...resetValues,
        lstm_normalization: request.lstmNormalization ?? null,
        lstm_features: JSON.stringify(normalizePersistedStringArray(request.lstmFeatures)),
        lstm_target_field: request.lstmTargetFeature ?? null,
        lstm_metrics_rmse: metricValues.rmse,
        lstm_metrics_mae: metricValues.mae,
        lstm_metrics_mape: metricValues.mape,
        lstm_metrics_r2: metricValues.r2,
      };
    case "weighted_avg":
      return {
        ...resetValues,
        ensemble_weighted_base_models: JSON.stringify(toFrontendBaseModelIds(request.models)),
        ensemble_weighted_metrics_rmse: metricValues.rmse,
        ensemble_weighted_metrics_mae: metricValues.mae,
        ensemble_weighted_metrics_mape: metricValues.mape,
        ensemble_weighted_metrics_r2: metricValues.r2,
      };
    case "boosting":
      return {
        ...resetValues,
        ensemble_boosting_base_models: JSON.stringify(toFrontendBaseModelIds(request.models)),
        ensemble_boosting_metrics_rmse: metricValues.rmse,
        ensemble_boosting_metrics_mae: metricValues.mae,
        ensemble_boosting_metrics_mape: metricValues.mape,
        ensemble_boosting_metrics_r2: metricValues.r2,
      };
    case "stacking":
      return {
        ...resetValues,
        ensemble_stacking_base_models: JSON.stringify(toFrontendBaseModelIds(request.models)),
        ensemble_stacking_metrics_rmse: metricValues.rmse,
        ensemble_stacking_metrics_mae: metricValues.mae,
        ensemble_stacking_metrics_mape: metricValues.mape,
        ensemble_stacking_metrics_r2: metricValues.r2,
      };
  }
};

const fetchActiveModelArtifact = async (
  experimentId: number,
  studentId: number,
  modelType: ModelType,
  connection?: PoolConnection,
): Promise<ActiveModelArtifactRow | null> => {
  const sql = `SELECT experiment_id,
                      student_id,
                      model_type,
                      revision_token,
                      training_input_version,
                      source_session_id,
                      backtest_artifact_path,
                      request_payload,
                      result_payload,
                      production_artifact_path,
                      production_revision_token
                 FROM experiment_model_artifacts
                WHERE experiment_id = ?
                  AND student_id = ?
                  AND model_type = ?
                LIMIT 1`;
  const params = [experimentId, studentId, modelType];
  const [rows] = connection
    ? await connection.query<ActiveModelArtifactRow[]>(sql, params)
    : await pool.query<ActiveModelArtifactRow[]>(sql, params);
  return rows[0] ?? null;
};

const activateModelArtifactInTransaction = async (
  connection: PoolConnection,
  options: {
    experimentId: number;
    studentId: number;
    modelType: ModelType;
    revisionToken: string;
    sourceSessionId: string | null;
    artifactPath: string;
    request: TrainInput;
    result: ReturnType<typeof transformTrainingResult>;
    expectedTrainingInputVersion?: number;
  },
) => {
  const {
    experimentId,
    studentId,
    modelType,
    revisionToken,
    sourceSessionId,
    artifactPath,
    request,
    result,
    expectedTrainingInputVersion,
  } = options;
  const experiment = await assertExperimentModelAccess(
    experimentId,
    studentId,
    connection,
    true,
  );
  if (
    expectedTrainingInputVersion !== undefined
    && Number(experiment.training_input_version) !== expectedTrainingInputVersion
  ) {
    throw new ModelConflictError("训练期间实验输入已更新，本次结果未激活；请重新开始训练");
  }
  assertTrainingRequestMatchesExperiment(experiment, request);
  const [currentRows] = await connection.query<ActiveModelArtifactRow[]>(
    `SELECT experiment_id,
            student_id,
            model_type,
            revision_token,
            training_input_version,
            source_session_id,
            backtest_artifact_path,
            request_payload,
            result_payload,
            production_artifact_path,
            production_revision_token
       FROM experiment_model_artifacts
      WHERE experiment_id = ?
        AND model_type = ?
      LIMIT 1
      FOR UPDATE`,
    [experimentId, modelType],
  );
  const changed = currentRows[0]?.revision_token !== revisionToken;
  if (
    !changed
    && Number(currentRows[0]?.training_input_version) === Number(experiment.training_input_version)
  ) {
    return {
      changed: false,
      stateVersion: Number(experiment.state_version),
      trainingInputVersion: Number(experiment.training_input_version),
    };
  }
  const stateValues = buildActivatedModelStateValues(modelType, request, result);
  const stateEntries = Object.entries(stateValues);
  const stateAssignments = stateEntries.map(([field]) => `\`${field}\` = ?`).join(", ");
  const [stateResult] = await connection.query<ResultSetHeader>(
    `UPDATE experiment_status
        SET ${stateAssignments},
            last_activity_at = NOW(),
            state_version = state_version + 1
      WHERE experiment_id = ?
        AND student_id = ?`,
    [...stateEntries.map(([, value]) => value), experimentId, studentId],
  );
  if (stateResult.affectedRows !== 1) {
    throw new ModelAccessError("实验不存在或不属于当前用户", 404);
  }

  if (isBaseModelType(modelType)) {
    await connection.query(
      `DELETE FROM experiment_model_artifacts
        WHERE experiment_id = ?
          AND model_type IN ('weighted_avg', 'boosting', 'stacking')`,
      [experimentId],
    );
  }

  await connection.query(
    `INSERT INTO experiment_model_artifacts
       (experiment_id,
        student_id,
        model_type,
        revision_token,
        training_input_version,
        source_session_id,
        backtest_artifact_path,
        request_payload,
        result_payload,
        production_artifact_path,
        production_revision_token,
        activated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NOW(3))
     ON DUPLICATE KEY UPDATE
       student_id = VALUES(student_id),
       revision_token = VALUES(revision_token),
       training_input_version = VALUES(training_input_version),
       source_session_id = VALUES(source_session_id),
       backtest_artifact_path = VALUES(backtest_artifact_path),
       request_payload = VALUES(request_payload),
       result_payload = VALUES(result_payload),
       production_artifact_path = NULL,
       production_revision_token = NULL,
       activated_at = NOW(3)`,
    [
      experimentId,
      studentId,
      modelType,
      revisionToken,
      Number(experiment.training_input_version),
      sourceSessionId,
      artifactPath,
      JSON.stringify(request),
      JSON.stringify(result),
    ],
  );

  return {
    changed,
    stateVersion: Number(experiment.state_version) + 1,
    trainingInputVersion: Number(experiment.training_input_version),
  };
};

const assertActiveArtifactMatchesTrainingInput = (
  experiment: ExperimentModelContextRow,
  artifact: ActiveModelArtifactRow,
) => {
  if (Number(artifact.training_input_version) !== Number(experiment.training_input_version)) {
    throw new ModelConflictError("实验训练输入已更新，当前模型产物不再匹配；请重新训练模型");
  }
};

const hasSavedModelArtifact = (modelType: ModelType, modelPath: string) => {
  if (!fs.existsSync(modelPath)) {
    return false;
  }

  if (modelType === "lstm") {
    return fs.existsSync(getLstmSidecarPath(modelPath));
  }

  return true;
};

const ENSEMBLE_ARTIFACT_VALIDATION_SCRIPT = `
import json
import os
import pickle
import sys

def artifact_exists(model_type, model_file):
    if not isinstance(model_file, str) or not os.path.exists(model_file):
        return False
    if model_type == "lstm":
        return os.path.exists(os.path.splitext(model_file)[0] + "_keras_model.h5")
    return True

def iter_model_states(base_model_states):
    if not isinstance(base_model_states, dict):
        return []

    flattened = []
    for outer_name, model_state in base_model_states.items():
        if isinstance(model_state, dict) and isinstance(model_state.get("models"), list):
            candidate_states = model_state.get("models", [])
        else:
            candidate_states = [model_state]

        for candidate_state in candidate_states:
            if not isinstance(candidate_state, dict):
                return None
            flattened.append((candidate_state.get("model_type") or outer_name, candidate_state.get("model_file")))

    return flattened

try:
    with open(sys.argv[1], "rb") as handle:
        state = pickle.load(handle)

    flattened_states = iter_model_states(state.get("base_model_states"))
    is_valid = flattened_states is not None and len(flattened_states) > 0 and all(
        artifact_exists(model_type, model_file)
        for model_type, model_file in flattened_states
    )
    print(json.dumps({"valid": bool(is_valid)}))
except Exception:
    print(json.dumps({"valid": False}))
`;

const hasCompleteEnsembleArtifact = async (modelPath: string) => {
  try {
    const pythonExec = resolvePythonExecutable();
    const proc = Bun.spawn([
      pythonExec,
      "-c",
      ENSEMBLE_ARTIFACT_VALIDATION_SCRIPT,
      modelPath,
    ], {
      cwd: PY_DIR,
      stdout: "pipe",
      stderr: "pipe",
      env: {
        ...process.env,
        ...buildPythonModelProcessEnv(),
        PATH: `${path.dirname(pythonExec)}${path.delimiter}${process.env.PATH ?? ""}`,
      },
    });

    const [stdout, stderr] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);
    await proc.exited;

    if (proc.exitCode !== 0) {
      logger.warn({ modelPath, stderr: stderr.slice(0, 1000) }, "校验融合模型产物完整性失败");
      return false;
    }

    const payload = JSON.parse(stdout) as { valid?: unknown };
    return payload.valid === true;
  } catch (error) {
    logger.warn({ err: error, modelPath }, "解析融合模型产物校验结果失败");
    return false;
  }
};

const hasReusableSavedModelArtifact = async (modelType: ModelType, modelPath: string) => {
  if (!hasSavedModelArtifact(modelType, modelPath)) {
    return false;
  }

  if (!isEnsembleModelType(modelType)) {
    return true;
  }

  return await hasCompleteEnsembleArtifact(modelPath);
};

const isPathInsideDirectory = (candidatePath: string, rootDirectory: string) => {
  const resolvedCandidate = path.resolve(candidatePath);
  const resolvedRoot = path.resolve(rootDirectory);
  const relativePath = path.relative(resolvedRoot, resolvedCandidate);
  return relativePath.length > 0
    && !relativePath.startsWith("..")
    && !path.isAbsolute(relativePath);
};

const removeGuidedTrainingArtifactDir = (artifactDir: string) => {
  if (!isPathInsideDirectory(artifactDir, GUIDED_TRAINING_ROOT)) {
    logger.warn(
      { artifactDir, guidedTrainingRoot: GUIDED_TRAINING_ROOT },
      "跳过不安全的 guided training 工件目录清理",
    );
    return false;
  }

  fs.rmSync(artifactDir, { recursive: true, force: true });
  return true;
};

const deleteGuidedTrainingSession = async (
  row: GuidedTrainingSessionRow,
) => {
  let artifactLocks = null;
  try {
    artifactLocks = await acquireModelArtifactMutationLocks(row.experiment_id, row.model_type);
    if (!artifactLocks) {
      throw new ModelConflictError("该实验已有模型任务在执行，请稍后重试");
    }

    const refreshedRow = await fetchGuidedTrainingSessionRow(row.session_id);
    if (!refreshedRow) {
      return true;
    }
    const modelPath = getGuidedSessionModelPath(
      refreshedRow.artifact_dir,
      refreshedRow.model_type,
    );
    if (await hasReusableSavedModelArtifact(refreshedRow.model_type, modelPath)) {
      return false;
    }

    const connection = await pool.getConnection();
    try {
      await connection.beginTransaction();
      const [lockedRows] = await connection.query<RowDataPacket[]>(
        `SELECT session_id
           FROM model_guided_training_sessions
          WHERE session_id = ?
            AND status = 'completed'
          LIMIT 1
          FOR UPDATE`,
        [row.session_id],
      );
      if (lockedRows.length === 0) {
        await connection.rollback();
        return false;
      }
      // Keep an invalid registry row as an explicit stale revision, but detach
      // its source FK so the unusable session can be replaced safely.
      await connection.query(
        `UPDATE experiment_model_artifacts
            SET source_session_id = NULL
          WHERE source_session_id = ?`,
        [row.session_id],
      );
      const [deleteResult] = await connection.query<ResultSetHeader>(
        `DELETE FROM model_guided_training_sessions
          WHERE session_id = ?
            AND status = 'completed'`,
        [row.session_id],
      );
      if (deleteResult.affectedRows !== 1) {
        await connection.rollback();
        return false;
      }
      await connection.commit();
    } catch (error) {
      await connection.rollback().catch(() => {});
      throw error;
    } finally {
      connection.release();
    }

    try {
      removeGuidedTrainingArtifactDir(refreshedRow.artifact_dir);
    } catch (error) {
      logger.warn(
        {
          err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
          sessionId: row.session_id,
          artifactDir: refreshedRow.artifact_dir,
        },
        "删除不可用 guided training 工件目录失败，后续将按 orphan 目录重试清理",
      );
    }
    return true;
  } finally {
    await releaseModelArtifactMutationLocks(artifactLocks);
  }
};

const cleanupOrphanGuidedTrainingArtifactDirs = async (
  now: Date,
  maxAgeMs: number,
) => {
  let entries: string[];
  try {
    entries = fs.readdirSync(GUIDED_TRAINING_ROOT);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return 0;
    }
    throw error;
  }

  let deletedArtifactDirs = 0;
  for (const entry of entries) {
    const artifactDir = path.join(GUIDED_TRAINING_ROOT, entry);
    let stats: Stats;
    try {
      stats = fs.statSync(artifactDir);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        logger.warn({ err: error, artifactDir }, "读取 guided training 工件目录状态失败");
      }
      continue;
    }

    if (!stats.isDirectory() || now.getTime() - stats.mtimeMs < maxAgeMs) {
      continue;
    }

    const [rows] = await pool.query<RowDataPacket[]>(
      `SELECT session_id
         FROM model_guided_training_sessions
        WHERE artifact_dir = ?
        LIMIT 1`,
      [artifactDir],
    );
    if (rows.length > 0) {
      continue;
    }

    try {
      if (removeGuidedTrainingArtifactDir(artifactDir)) {
        deletedArtifactDirs += 1;
      }
    } catch (error) {
      logger.warn(
        {
          err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
          artifactDir,
        },
        "清理 orphan guided training 工件目录失败",
      );
    }
  }

  return deletedArtifactDirs;
};

export async function cleanupExpiredGuidedTrainingSessions(
  options: { now?: Date; limit?: number; orphanMaxAgeMs?: number } = {},
) {
  const now = options.now ?? new Date();
  const rawLimit = options.limit ?? GUIDED_TRAINING_CLEANUP_BATCH_SIZE;
  const limit = Number.isInteger(rawLimit) && rawLimit > 0
    ? Math.min(rawLimit, GUIDED_TRAINING_CLEANUP_BATCH_SIZE)
    : GUIDED_TRAINING_CLEANUP_BATCH_SIZE;

  const expirationCutoffExpression = options.now ? "?" : "NOW()";
  const [rows] = await pool.query<GuidedTrainingCleanupRow[]>(
    `SELECT session_id,
            experiment_id,
            model_type,
            artifact_dir
       FROM model_guided_training_sessions
      WHERE (
          status <> 'running'
          OR lease_expires_at IS NULL
          OR lease_expires_at <= ${expirationCutoffExpression}
        )
        AND expires_at IS NOT NULL
        AND expires_at <= ${expirationCutoffExpression}
        AND NOT EXISTS (
          SELECT 1
            FROM experiment_model_artifacts active_artifact
           WHERE active_artifact.source_session_id = model_guided_training_sessions.session_id
        )
      ORDER BY expires_at ASC
      LIMIT ${limit}`,
    options.now ? [options.now, options.now] : [],
  );

  let deletedArtifactDirs = 0;
  let deletedSessions = 0;
  for (const row of rows) {
    let artifactLocks = null;
    try {
      artifactLocks = await acquireModelArtifactMutationLocks(row.experiment_id, row.model_type);
      if (!artifactLocks) {
        continue;
      }
      const [deleteResult] = await pool.query<ResultSetHeader>(
        `DELETE FROM model_guided_training_sessions
          WHERE session_id = ?
            AND (
              status <> 'running'
              OR lease_expires_at IS NULL
              OR lease_expires_at <= ${expirationCutoffExpression}
            )
            AND expires_at IS NOT NULL
            AND expires_at <= ${expirationCutoffExpression}
            AND NOT EXISTS (
              SELECT 1
                FROM experiment_model_artifacts active_artifact
               WHERE active_artifact.source_session_id = model_guided_training_sessions.session_id
            )`,
        options.now ? [row.session_id, options.now, options.now] : [row.session_id],
      );
      if (deleteResult.affectedRows !== 1) {
        continue;
      }
      deletedSessions += 1;
      if (removeGuidedTrainingArtifactDir(row.artifact_dir)) {
        deletedArtifactDirs += 1;
      }
    } catch (error) {
      logger.warn(
        {
          err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
          sessionId: row.session_id,
          artifactDir: row.artifact_dir,
        },
        "清理过期 guided training 工件目录失败",
      );
    } finally {
      await releaseModelArtifactMutationLocks(artifactLocks);
    }
  }

  deletedArtifactDirs += await cleanupOrphanGuidedTrainingArtifactDirs(
    now,
    options.orphanMaxAgeMs ?? CLEANUP_RETENTION.GUIDED_TRAINING_ORPHAN_MS,
  );

  return {
    deletedSessions,
    deletedArtifactDirs,
  };
}

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

const hashGuidedTrainingRequest = (
  modelType: GuidedModelType,
  body: TrainInput,
  trainingInputVersion: number,
  dependencyRevisions?: unknown,
) =>
  crypto
    .createHash("sha256")
    .update(stableJsonStringify({ modelType, body, trainingInputVersion, dependencyRevisions }))
    .digest("hex");

const parseJsonColumn = <T>(value: unknown, fallback: T): T => {
  if (value === null || value === undefined) {
    return fallback;
  }

  if (typeof value === "string") {
    try {
      return JSON.parse(value) as T;
    } catch {
      return fallback;
    }
  }

  return value as T;
};

const isGuidedModelType = (modelType: ModelType): modelType is GuidedModelType => {
  return GUIDED_MODEL_TYPES.has(modelType);
};

const normalizeBooleanFlag = (value: number | boolean | null | undefined): boolean => {
  return value === true || value === 1;
};

const normalizeNumericArray = (value: unknown): number[] | undefined => {
  if (!Array.isArray(value)) {
    return undefined;
  }

  return value.filter((item): item is number => typeof item === "number" && Number.isFinite(item));
};

const normalizeStringArray = (value: unknown): string[] | undefined => {
  if (!Array.isArray(value)) {
    return undefined;
  }

  return value.filter((item): item is string => typeof item === "string");
};

const validateMovingAverageWindow = (
  trainingConfig: ExperimentTrainingConfig,
  movingAverageWindow: number | undefined,
) => {
  if (movingAverageWindow === undefined) {
    return;
  }

  const trainingLength =
    trainingConfig.data_window_train_end_index - trainingConfig.data_window_train_start_index + 1;

  if (movingAverageWindow > trainingLength) {
    throw new ModelValidationError("移动平均时间窗口不能超过训练数据长度", {
      moving_average_window: movingAverageWindow,
      training_length: trainingLength,
    });
  }
};

const resolveAutomaticLstmLookBack = (trainingSize: number) =>
  Math.max(3, Math.min(6, Math.floor(trainingSize / 4)));

const validateLstmProductionFeasibility = (trainingConfig: ExperimentTrainingConfig) => {
  const trainingSize =
    trainingConfig.data_window_train_end_index - trainingConfig.data_window_train_start_index + 1;
  const lookBack = resolveAutomaticLstmLookBack(trainingSize);
  const productionTrainingSize =
    trainingConfig.data_window_evaluate_end_index - trainingConfig.data_window_train_start_index + 1;
  const minimumProductionTrainingSize = lookBack + MAX_FORECAST_STEPS;

  if (productionTrainingSize < minimumProductionTrainingSize) {
    throw new ModelValidationError(
      `LSTM 生产预测固定为 ${MAX_FORECAST_STEPS} 期；当前数据在纳入评估区间后仅有 ${productionTrainingSize} 个训练点，至少需要 ${minimumProductionTrainingSize} 个`,
      {
        look_back: lookBack,
        production_forecast_steps: MAX_FORECAST_STEPS,
        production_training_size: productionTrainingSize,
        minimum_production_training_size: minimumProductionTrainingSize,
      },
    );
  }
};

const normalizePersistedStringArray = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value
      .filter((item): item is string => typeof item === "string")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
  }

  if (typeof value === "string") {
    try {
      return normalizePersistedStringArray(JSON.parse(value));
    } catch {
      return value
        .split(",")
        .map((item) => item.trim())
        .filter((item) => item.length > 0);
    }
  }

  return [];
};

const parseRequestedBaseModels = (value: string): BaseModelType[] => {
  const rawModels = value
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item.length > 0);

  const dedupedModels = [...new Set(rawModels)];
  const validModels = dedupedModels.filter(
    (model): model is BaseModelType => ["ma", "es", "arima", "lstm"].includes(model),
  );

  if (validModels.length !== dedupedModels.length || validModels.length < 2) {
    throw new ModelValidationError("融合模型至少需要选择两个有效且不重复的基础模型");
  }

  return validModels;
};

const assertRequestMatchesPersistedValue = (
  label: string,
  requestValue: unknown,
  persistedValue: unknown,
) => {
  if (requestValue === undefined) {
    return;
  }

  if (Array.isArray(persistedValue)) {
    const normalizedPersisted = normalizePersistedStringArray(persistedValue);
    const normalizedRequest = normalizePersistedStringArray(requestValue);
    if (JSON.stringify(normalizedRequest) !== JSON.stringify(normalizedPersisted)) {
      throw new ModelValidationError(`请求中的${label}与已完成基础模型记录不一致，请刷新页面后重试`);
    }
    return;
  }

  if (requestValue !== persistedValue) {
    throw new ModelValidationError(`请求中的${label}与已完成基础模型记录不一致，请刷新页面后重试`);
  }
};

const buildEnsemblePythonParams = (
  experiment: ExperimentModelContextRow,
  body: TrainInput,
): Record<string, unknown> => {
  if (!body.models) {
    throw new ModelValidationError("融合模型缺少基础模型列表");
  }

  const requestedModels = parseRequestedBaseModels(body.models);
  const pythonParams: Record<string, unknown> = {
    models: requestedModels.join(","),
  };

  for (const requestedModel of requestedModels) {
    switch (requestedModel) {
      case "ma": {
        if (!normalizeBooleanFlag(experiment.moving_average_completed)) {
          throw new ModelValidationError("融合模型依赖的移动平均基础模型尚未完成训练");
        }
        if (!hasDefinedValue(experiment.moving_average_window)) {
          throw new ModelValidationError("融合模型依赖的移动平均基础模型缺少时间窗口配置");
        }
        assertRequestMatchesPersistedValue(
          "移动平均时间窗口",
          body.moving_average_window,
          experiment.moving_average_window,
        );
        pythonParams.moving_average_window = experiment.moving_average_window;
        break;
      }
      case "es": {
        if (!normalizeBooleanFlag(experiment.exponential_smoothing_completed)) {
          throw new ModelValidationError("融合模型依赖的指数平滑基础模型尚未完成训练");
        }
        if (!hasDefinedValue(experiment.exponential_smoothing_alpha)) {
          throw new ModelValidationError("融合模型依赖的指数平滑基础模型缺少平滑系数配置");
        }
        assertRequestMatchesPersistedValue(
          "指数平滑系数",
          body.exponential_smoothing_alpha,
          experiment.exponential_smoothing_alpha,
        );
        pythonParams.exponential_smoothing_alpha = experiment.exponential_smoothing_alpha;
        break;
      }
      case "arima": {
        if (!normalizeBooleanFlag(experiment.arima_completed)) {
          throw new ModelValidationError("融合模型依赖的 ARIMA 基础模型尚未完成训练");
        }
        if (!hasDefinedValue(experiment.arima_d)) {
          throw new ModelValidationError("融合模型依赖的 ARIMA 基础模型缺少差分阶数配置");
        }
        assertRequestMatchesPersistedValue("ARIMA 差分阶数", body.arimaD, experiment.arima_d);
        pythonParams.arimaD = experiment.arima_d;
        break;
      }
      case "lstm": {
        if (!normalizeBooleanFlag(experiment.lstm_completed)) {
          throw new ModelValidationError("融合模型依赖的 LSTM 基础模型尚未完成训练");
        }

        const persistedFeatures = normalizePersistedStringArray(experiment.lstm_features);
        if (
          !hasDefinedValue(experiment.lstm_normalization)
          || !hasDefinedValue(experiment.lstm_target_field)
          || persistedFeatures.length === 0
        ) {
          throw new ModelValidationError("融合模型依赖的 LSTM 基础模型缺少特征或归一化配置");
        }

        assertRequestMatchesPersistedValue("LSTM 归一化方式", body.lstmNormalization, experiment.lstm_normalization);
        assertRequestMatchesPersistedValue("LSTM 目标特征", body.lstmTargetFeature, experiment.lstm_target_field);
        assertRequestMatchesPersistedValue("LSTM 输入特征", body.lstmFeatures, persistedFeatures);

        pythonParams.lstmNormalization = experiment.lstm_normalization;
        pythonParams.lstmTargetFeature = experiment.lstm_target_field;
        pythonParams.lstmFeatures = JSON.stringify(persistedFeatures);
        break;
      }
    }
  }

  return pythonParams;
};

const buildAvailableBaseModelArtifactMap = async (
  experimentId: number,
  studentId: number,
  modelsValue: string,
  expectedTrainingInputVersion: number,
): Promise<Partial<Record<BaseModelType, string>>> => {
  const requestedModels = parseRequestedBaseModels(modelsValue);
  const availableModelFiles: Partial<Record<BaseModelType, string>> = {};

  for (const modelName of requestedModels) {
    const activeArtifact = await fetchActiveModelArtifact(experimentId, studentId, modelName);
    if (
      activeArtifact
      && Number(activeArtifact.training_input_version) === expectedTrainingInputVersion
      && await hasReusableSavedModelArtifact(modelName, activeArtifact.backtest_artifact_path)
    ) {
      availableModelFiles[modelName] = activeArtifact.backtest_artifact_path;
    }
  }

  return availableModelFiles;
};

const buildGuidedDependencyRevisions = async (
  modelType: GuidedModelType,
  experimentId: number,
  studentId: number,
  modelsValue: string | undefined,
  options: {
    connection?: PoolConnection;
    targetRevisionOverride?: string;
  } = {},
) => {
  const activeTargetArtifact = options.targetRevisionOverride
    ? null
    : await fetchActiveModelArtifact(
        experimentId,
        studentId,
        modelType,
        options.connection,
      );
  const baseRevisions = [];
  if (isEnsembleModelType(modelType) && modelsValue) {
    for (const baseModelType of parseRequestedBaseModels(modelsValue)) {
      const activeArtifact = await fetchActiveModelArtifact(
        experimentId,
        studentId,
        baseModelType,
        options.connection,
      );
      baseRevisions.push({
        model_type: baseModelType,
        revision_token: activeArtifact?.revision_token ?? null,
      });
    }
  }
  return {
    target_revision: options.targetRevisionOverride
      ?? activeTargetArtifact?.revision_token
      ?? null,
    base_revisions: baseRevisions,
  };
};

async function assertGuidedTrainingSessionDependenciesCurrent(row: GuidedTrainingSessionRow) {
  const body = parseJsonColumn<TrainInput>(row.request_payload, {} as TrainInput);
  const experiment = await assertExperimentModelAccess(row.experiment_id, row.student_id);
  const currentTrainingInputVersion = Number(experiment.training_input_version);
  const currentRequestHash = hashGuidedTrainingRequest(
    row.model_type,
    body,
    currentTrainingInputVersion,
    await buildGuidedDependencyRevisions(
      row.model_type,
      row.experiment_id,
      row.student_id,
      body.models,
    ),
  );

  if (currentRequestHash !== row.request_hash) {
    throw new ModelConflictError("实验训练输入或基础模型版本已更新，请重新开始分阶段训练");
  }
}

const assertTrainingRequestMatchesExperiment = (
  experiment: ExperimentModelContextRow,
  body: TrainInput,
) => {
  const comparisons = [
    ["selected_industry", "行业", body.selected_industry, experiment.selected_industry],
    ["selected_company", "企业", body.selected_company, experiment.selected_company],
    ["selected_product", "产品", body.selected_product, experiment.selected_product],
    [
      "data_window_train_start_index",
      "训练起始索引",
      body.data_window_train_start_index,
      experiment.data_window_train_start_index,
    ],
    [
      "data_window_train_end_index",
      "训练结束索引",
      body.data_window_train_end_index,
      experiment.data_window_train_end_index,
    ],
    [
      "data_window_evaluate_start_index",
      "评估起始索引",
      body.data_window_evaluate_start_index,
      experiment.data_window_evaluate_start_index,
    ],
    [
      "data_window_evaluate_end_index",
      "评估结束索引",
      body.data_window_evaluate_end_index,
      experiment.data_window_evaluate_end_index,
    ],
  ] as const;

  for (const [field, label, requestValue, persistedValue] of comparisons) {
    if (requestValue !== undefined && requestValue !== persistedValue) {
      throw new ModelValidationError(`请求中的${label}与实验记录不一致，请刷新页面后重试`, {
        field,
        request_value: requestValue,
        persisted_value: persistedValue,
      });
    }
  }
};

const getExperimentTrainingConfig = (
  experiment: ExperimentModelContextRow,
): ExperimentTrainingConfig => {
  if (
    !hasDefinedValue(experiment.selected_industry)
    || !hasDefinedValue(experiment.selected_company)
    || !hasDefinedValue(experiment.selected_product)
  ) {
    throw new ModelValidationError("实验尚未完成数据集选择，无法训练模型");
  }

  if (
    !hasDefinedValue(experiment.data_window_train_start_index)
    || !hasDefinedValue(experiment.data_window_train_end_index)
    || !hasDefinedValue(experiment.data_window_evaluate_start_index)
    || !hasDefinedValue(experiment.data_window_evaluate_end_index)
  ) {
    throw new ModelValidationError("实验尚未完成数据窗口配置，无法训练模型");
  }

  if (experiment.data_window_train_start_index > experiment.data_window_train_end_index) {
    throw new ModelValidationError("实验记录中的训练数据窗口无效，请重新选择数据窗口后重试");
  }

  if (experiment.data_window_evaluate_start_index > experiment.data_window_evaluate_end_index) {
    throw new ModelValidationError("实验记录中的评估数据窗口无效，请重新选择数据窗口后重试");
  }

  if (
    experiment.data_window_evaluate_end_index
      - experiment.data_window_evaluate_start_index
      + 1
    < MIN_EVALUATION_POINTS
  ) {
    throw new ModelValidationError(
      `实验记录中的评估区间至少需要 ${MIN_EVALUATION_POINTS} 个数据点，请重新选择数据窗口后重试`,
    );
  }

  if (experiment.data_window_evaluate_start_index <= experiment.data_window_train_end_index) {
    throw new ModelValidationError("实验记录中的评估起始索引必须大于训练结束索引，请重新选择数据窗口后重试");
  }

  return {
    selected_industry: experiment.selected_industry,
    selected_company: experiment.selected_company,
    selected_product: experiment.selected_product,
    data_window_train_start_index: experiment.data_window_train_start_index,
    data_window_train_end_index: experiment.data_window_train_end_index,
    data_window_evaluate_start_index: experiment.data_window_evaluate_start_index,
    data_window_evaluate_end_index: experiment.data_window_evaluate_end_index,
  };
}

const assertBaseModelReadyForProduction = (
  experiment: ExperimentModelContextRow,
  modelType: BaseModelType,
) => {
  switch (modelType) {
    case "ma":
      if (!normalizeBooleanFlag(experiment.moving_average_completed) || !hasDefinedValue(experiment.moving_average_window)) {
        throw new ModelValidationError("移动平均基础模型尚未完成训练，无法准备生产预测模型");
      }
      return;
    case "es":
      if (!normalizeBooleanFlag(experiment.exponential_smoothing_completed) || !hasDefinedValue(experiment.exponential_smoothing_alpha)) {
        throw new ModelValidationError("指数平滑基础模型尚未完成训练，无法准备生产预测模型");
      }
      return;
    case "arima":
      if (!normalizeBooleanFlag(experiment.arima_completed) || !hasDefinedValue(experiment.arima_d)) {
        throw new ModelValidationError("ARIMA 基础模型尚未完成训练，无法准备生产预测模型");
      }
      return;
    case "lstm": {
      const lstmFeatures = normalizePersistedStringArray(experiment.lstm_features);
      if (
        !normalizeBooleanFlag(experiment.lstm_completed)
        || !hasDefinedValue(experiment.lstm_normalization)
        || !hasDefinedValue(experiment.lstm_target_field)
        || lstmFeatures.length === 0
      ) {
        throw new ModelValidationError("LSTM 基础模型尚未完成训练，无法准备生产预测模型");
      }
      return;
    }
  }
};

const assertEnsembleModelReadyForProduction = (
  experiment: ExperimentModelContextRow,
  modelType: EnsembleModelType,
) => {
  const configs = {
    weighted_avg: {
      completed: normalizeBooleanFlag(experiment.ensemble_weighted_completed),
      baseModels: normalizePersistedStringArray(experiment.ensemble_weighted_base_models),
      errorMessage: "加权平均融合模型尚未完成训练，无法准备生产预测模型",
    },
    boosting: {
      completed: normalizeBooleanFlag(experiment.ensemble_boosting_completed),
      baseModels: normalizePersistedStringArray(experiment.ensemble_boosting_base_models),
      errorMessage: "Boosting 融合模型尚未完成训练，无法准备生产预测模型",
    },
    stacking: {
      completed: normalizeBooleanFlag(experiment.ensemble_stacking_completed),
      baseModels: normalizePersistedStringArray(experiment.ensemble_stacking_base_models),
      errorMessage: "Stacking 融合模型尚未完成训练，无法准备生产预测模型",
    },
  } satisfies Record<EnsembleModelType, { completed: boolean; baseModels: string[]; errorMessage: string }>;

  const config = configs[modelType];
  if (!config.completed || config.baseModels.length < 2) {
    throw new ModelValidationError(config.errorMessage);
  }
};

const assertModelReadyForProduction = (
  experiment: ExperimentModelContextRow,
  modelType: ModelType,
) => {
  if (isBaseModelType(modelType)) {
    assertBaseModelReadyForProduction(experiment, modelType);
    return;
  }

  if (isEnsembleModelType(modelType)) {
    assertEnsembleModelReadyForProduction(experiment, modelType);
  }
};

const buildBoostingRoundStepDefinitions = (maxRounds: number): GuidedTrainingStepDefinition[] => {
  const steps: GuidedTrainingStepDefinition[] = [];
  for (let round = 1; round <= maxRounds; round += 1) {
    steps.push(
      {
        id: `round_${round}_evaluate_candidates`,
        label: `第 ${round} 轮评估候选模型`,
        description: "用当前残差训练每个候选基础模型，并比较它们降低验证残差 RMSE 的能力。",
        actionLabel: "评估候选",
      },
      {
        id: `round_${round}_select_winner`,
        label: `第 ${round} 轮选择胜出模型`,
        description: "选择本轮最能降低验证残差的基础模型加入模型链；若没有改善，则提前停止迭代。",
        actionLabel: "选择模型",
      },
    );
  }
  return steps;
};

const buildGuidedTrainingStepPlan = (
  modelType: GuidedModelType,
  body: TrainInput,
): GuidedTrainingStepDefinition[] => {
  if (modelType !== "boosting") {
    return GUIDED_TRAINING_STEPS[modelType];
  }

  const requestedModels = body.models ? parseRequestedBaseModels(body.models) : [];
  const maxRounds = Math.max(1, requestedModels.length);
  const [prepareStep, splitStep, retrainStep, combineStep, saveStep] = GUIDED_TRAINING_STEPS.boosting;

  return [
    prepareStep!,
    splitStep!,
    ...buildBoostingRoundStepDefinitions(maxRounds),
    retrainStep!,
    combineStep!,
    saveStep!,
  ];
};

const parseGuidedStepPlan = (
  modelType: GuidedModelType,
  stepPlan: unknown,
): GuidedTrainingStepDefinition[] => {
  if (!Array.isArray(stepPlan)) {
    return GUIDED_TRAINING_STEPS[modelType];
  }

  const parsedSteps = stepPlan.filter((item): item is GuidedTrainingStepDefinition => {
    if (!item || typeof item !== "object") {
      return false;
    }
    const candidate = item as Record<string, unknown>;
    return typeof candidate.id === "string"
      && typeof candidate.label === "string"
      && typeof candidate.description === "string"
      && typeof candidate.actionLabel === "string";
  });

  return parsedSteps.length > 0 ? parsedSteps : GUIDED_TRAINING_STEPS[modelType];
};

const getGuidedTrainingSteps = (row: Pick<GuidedTrainingSessionRow, "model_type" | "step_plan">) =>
  parseGuidedStepPlan(row.model_type, parseJsonColumn<unknown>(row.step_plan, null));

const getGuidedNextStepId = (steps: GuidedTrainingStepDefinition[], completedStepIds: string[]) => {
  const completedSet = new Set(completedStepIds);
  return steps.find((step) => !completedSet.has(step.id))?.id ?? null;
};

const isRecordValue = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const shouldTrimBoostingRoundSteps = (
  modelType: GuidedModelType,
  stepId: string,
  output: unknown,
) => {
  if (modelType !== "boosting" || !/^round_\d+_select_winner$/.test(stepId) || !isRecordValue(output)) {
    return false;
  }

  return output.action === "early_stop";
};

const trimBoostingStepPlanAfterEarlyStop = (
  stepPlan: GuidedTrainingStepDefinition[],
  stepId: string,
) => {
  const currentIndex = stepPlan.findIndex((step) => step.id === stepId);
  if (currentIndex === -1) {
    return stepPlan;
  }

  const finalStepIds = new Set(["retrain_chain", "combine_evaluate", "save_artifact"]);
  return [
    ...stepPlan.slice(0, currentIndex + 1),
    ...stepPlan.filter((step, index) => index > currentIndex && finalStepIds.has(step.id)),
  ];
};

const buildGuidedTrainingSessionPayload = (
  row: GuidedTrainingSessionRow,
  options: {
    backtestArtifactChanged?: boolean;
    artifactRevision?: string | null;
    experimentStateVersion?: number | null;
  } = {},
): GuidedTrainingSessionPayload => {
  const completedStepIds = parseJsonColumn<string[]>(row.completed_step_ids, []);
  const stepOutputs = parseJsonColumn<Record<string, unknown>>(row.step_outputs, {});
  const result = parseJsonColumn<GuidedTrainingSessionPayload["result"]>(row.result_payload, null);
  const stepPlan = getGuidedTrainingSteps(row);
  const nextStepId = getGuidedNextStepId(stepPlan, completedStepIds);
  const activeStepId = row.current_step_id ?? nextStepId;
  const completedSet = new Set(completedStepIds);
  const isCompletedSession = row.status === "completed";

  return {
    session_id: row.session_id,
    experiment_id: row.experiment_id,
    model_type: row.model_type,
    status: row.status,
    current_step_id: activeStepId,
    next_step_id: row.status === "completed" ? null : nextStepId,
    steps: stepPlan.map((step) => ({
      ...step,
      status: isCompletedSession || completedSet.has(step.id)
        ? "completed"
        : row.status === "failed" && activeStepId === step.id
          ? "failed"
          : activeStepId === step.id
            ? "active"
            : "pending",
      output: stepOutputs[step.id],
    })),
    step_outputs: stepOutputs,
    result,
    error_message: row.error_message,
    backtest_artifact_changed: options.backtestArtifactChanged === true,
    artifact_revision: options.artifactRevision ?? null,
    experiment_state_version: options.experimentStateVersion ?? null,
  };
};

const fetchGuidedTrainingSessionRow = async (
  sessionId: string,
): Promise<GuidedTrainingSessionRow | null> => {
  const [rows] = await pool.query<GuidedTrainingSessionRow[]>(
    `SELECT session_id,
            experiment_id,
            student_id,
            model_type,
            request_hash,
            request_payload,
            status,
            current_step_id,
            completed_step_ids,
            step_outputs,
            step_plan,
            result_payload,
            artifact_dir,
            error_message,
            training_input_version,
            run_token,
            run_version,
            lease_expires_at,
            last_heartbeat_at
       FROM model_guided_training_sessions
      WHERE session_id = ?
      LIMIT 1`,
    [sessionId],
  );

  return rows[0] ?? null;
};

interface GuidedTrainingAttempt {
  token: string;
  version: number;
}

const claimGuidedTrainingAttempt = async (
  row: GuidedTrainingSessionRow,
  stepId: string,
): Promise<GuidedTrainingAttempt> => {
  const token = crypto.randomUUID();
  const currentVersion = Number(row.run_version ?? 0);
  const [result] = await pool.query<ResultSetHeader>(
    `UPDATE model_guided_training_sessions
        SET status = 'running',
            current_step_id = ?,
            error_message = NULL,
            run_token = ?,
            run_version = run_version + 1,
            last_heartbeat_at = NOW(3),
            lease_expires_at = TIMESTAMPADD(SECOND, ?, NOW(3)),
            expires_at = DATE_ADD(NOW(), INTERVAL 7 DAY)
      WHERE session_id = ?
        AND run_version = ?
        AND (
          status IN ('ready', 'failed')
          OR (
            status = 'running'
            AND (lease_expires_at IS NULL OR lease_expires_at <= NOW(3))
          )
        )`,
    [stepId, token, GUIDED_STEP_LEASE_SECONDS, row.session_id, currentVersion],
  );

  if (result.affectedRows !== 1) {
    throw new ModelConflictError("该分阶段训练会话已有执行中的步骤，请稍后重试");
  }

  return { token, version: currentVersion + 1 };
};

const startGuidedTrainingHeartbeat = (
  sessionId: string,
  attempt: GuidedTrainingAttempt,
) => {
  let stopped = false;
  let pendingHeartbeat = Promise.resolve();

  const heartbeat = async () => {
    if (stopped) return;
    try {
      const [result] = await pool.query<ResultSetHeader>(
        `UPDATE model_guided_training_sessions
            SET last_heartbeat_at = NOW(3),
                lease_expires_at = TIMESTAMPADD(SECOND, ?, NOW(3))
          WHERE session_id = ?
            AND status = 'running'
            AND run_token = ?
            AND run_version = ?`,
        [GUIDED_STEP_LEASE_SECONDS, sessionId, attempt.token, attempt.version],
      );
      if (result.affectedRows !== 1) {
        logger.warn(
          { sessionId, runVersion: attempt.version },
          "guided training 租约已丢失，当前执行将不允许提交",
        );
      }
    } catch (error) {
      logger.warn(
        { err: error, sessionId, runVersion: attempt.version },
        "续期 guided training 租约失败",
      );
    }
  };

  const timer = setInterval(() => {
    pendingHeartbeat = pendingHeartbeat.then(heartbeat, heartbeat);
  }, GUIDED_STEP_HEARTBEAT_INTERVAL_MS);

  return async () => {
    if (stopped) return;
    stopped = true;
    clearInterval(timer);
    await pendingHeartbeat;
  };
};

const assertGuidedTrainingAttemptOwned = async (
  sessionId: string,
  attempt: GuidedTrainingAttempt,
) => {
  const [rows] = await pool.query<RowDataPacket[]>(
    `SELECT session_id
       FROM model_guided_training_sessions
      WHERE session_id = ?
        AND status = 'running'
        AND run_token = ?
        AND run_version = ?
        AND lease_expires_at > NOW(3)
      LIMIT 1`,
    [sessionId, attempt.token, attempt.version],
  );
  if (rows.length !== 1) {
    throw new ModelStaleAttemptError("分阶段训练执行权已转移，本次结果已丢弃");
  }
};

const getGuidedPublicErrorMessage = (error: unknown) => {
  if (
    error instanceof ModelAccessError
    || error instanceof ModelCapacityError
    || error instanceof ModelConflictError
    || error instanceof ModelNotFoundError
    || error instanceof ModelStaleAttemptError
    || error instanceof ModelValidationError
  ) {
    return error.message;
  }

  const message = error instanceof Error ? error.message : "";
  if (message.startsWith("[PYTHON_VALIDATION]") || message.startsWith("[PYTHON_DATA]")) {
    return message;
  }
  if (message.includes("超时") || message.includes("timed out")) {
    return "分阶段训练执行超时，可以重试当前步骤";
  }
  return "分阶段训练执行失败，可以重试当前步骤";
};

const recordGuidedTrainingAttemptFailure = async (
  sessionId: string,
  stepId: string,
  attempt: GuidedTrainingAttempt,
  error: unknown,
) => {
  try {
    await pool.query(
      `UPDATE model_guided_training_sessions
          SET status = 'failed',
              current_step_id = ?,
              error_message = ?,
              run_token = NULL,
              lease_expires_at = NULL,
              last_heartbeat_at = NULL
        WHERE session_id = ?
          AND status = 'running'
          AND run_token = ?
          AND run_version = ?`,
      [stepId, getGuidedPublicErrorMessage(error), sessionId, attempt.token, attempt.version],
    );
  } catch (recordError) {
    logger.error(
      { err: recordError, sessionId, stepId, runVersion: attempt.version },
      "记录 guided training 失败状态时发生二次错误",
    );
  }
};

const buildGuidedTrainingPythonParams = async (
  modelType: GuidedModelType,
  studentId: number,
  body: TrainInput,
  modelSaveLoc: string,
): Promise<Record<string, unknown>> => {
  const experiment = await assertExperimentModelAccess(body.experiment_id, studentId);
  assertTrainingRequestMatchesExperiment(experiment, body);
  const trainingConfig = getExperimentTrainingConfig(experiment);
  if (modelType === "ma") {
    validateMovingAverageWindow(trainingConfig, body.moving_average_window);
  } else if (modelType === "lstm") {
    validateLstmProductionFeasibility(trainingConfig);
  }
  if (isEnsembleModelType(modelType)) {
    buildEnsemblePythonParams(experiment, body);
  }
  const derivedModelParams: Record<string, unknown> = isEnsembleModelType(modelType)
    ? buildEnsemblePythonParams(experiment, body)
    : {};

  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
  );

  if (
    trainingConfig.data_window_train_end_index >= allData.length
    || trainingConfig.data_window_evaluate_end_index >= allData.length
  ) {
    throw new ModelValidationError("数据窗口索引超出范围", { dataset_size: allData.length });
  }

  const pythonParams: Record<string, unknown> = {
    data: JSON.stringify(allData),
    trainStart: trainingConfig.data_window_train_start_index,
    trainEnd: trainingConfig.data_window_train_end_index,
    evalStart: trainingConfig.data_window_evaluate_start_index,
    evalEnd: trainingConfig.data_window_evaluate_end_index,
    saveModel: modelSaveLoc,
    pretty: true,
  };

  for (const bodyKey of Object.keys(allPossiblePythonParams)) {
    const value = body[bodyKey as keyof TrainInput];
    if (value !== undefined) {
      pythonParams[bodyKey] = value;
    }
  }

  Object.assign(pythonParams, derivedModelParams);
  if (isEnsembleModelType(modelType) && body.models) {
    const baseModelArtifactMap = await buildAvailableBaseModelArtifactMap(
      body.experiment_id,
      studentId,
      body.models,
      Number(experiment.training_input_version),
    );
    if (Object.keys(baseModelArtifactMap).length > 0) {
      pythonParams.baseModelFiles = JSON.stringify(baseModelArtifactMap);
    }
  }

  return pythonParams;
};

export async function createOrResumeGuidedTrainingSession(
  modelType: ModelType,
  studentId: number,
  body: TrainInput,
): Promise<GuidedTrainingSessionPayload> {
  if (!isGuidedModelType(modelType)) {
    throw new ModelValidationError("当前模型暂不支持分阶段训练");
  }

  const experiment = await assertExperimentModelAccess(body.experiment_id, studentId);
  assertTrainingRequestMatchesExperiment(experiment, body);
  const trainingConfig = getExperimentTrainingConfig(experiment);
  if (modelType === "ma") {
    validateMovingAverageWindow(trainingConfig, body.moving_average_window);
  } else if (modelType === "lstm") {
    validateLstmProductionFeasibility(trainingConfig);
  }

  const requestHash = hashGuidedTrainingRequest(
    modelType,
    body,
    Number(experiment.training_input_version),
    await buildGuidedDependencyRevisions(
      modelType,
      body.experiment_id,
      studentId,
      body.models,
    ),
  );
  const [existingRows] = await pool.query<GuidedTrainingSessionRow[]>(
    `SELECT session_id,
            experiment_id,
            student_id,
            model_type,
            request_hash,
            request_payload,
            status,
            current_step_id,
            completed_step_ids,
            step_outputs,
            step_plan,
            result_payload,
            artifact_dir,
            error_message,
            training_input_version,
            run_token,
            run_version,
            lease_expires_at,
            last_heartbeat_at
       FROM model_guided_training_sessions
      WHERE experiment_id = ?
        AND student_id = ?
        AND model_type = ?
        AND request_hash = ?
      LIMIT 1`,
    [body.experiment_id, studentId, modelType, requestHash],
  );

  const existingSession = existingRows[0];
  if (existingSession) {
    if (existingSession.status === "completed") {
      const sourceModelPath = getGuidedSessionModelPath(existingSession.artifact_dir, modelType);
      if (await hasReusableSavedModelArtifact(modelType, sourceModelPath)) {
        return buildGuidedTrainingSessionPayload(existingSession);
      }

      logger.warn(
        {
          sessionId: existingSession.session_id,
          artifactDir: existingSession.artifact_dir,
          modelType,
        },
        "已完成 guided training 会话缺少可复用工件，将删除会话并重新创建",
      );
      const deletedSession = await deleteGuidedTrainingSession(existingSession);
      if (!deletedSession) {
        const refreshedSession = await fetchGuidedTrainingSessionRow(existingSession.session_id);
        if (refreshedSession) {
          return buildGuidedTrainingSessionPayload(refreshedSession);
        }
        throw new ModelConflictError("旧的分阶段训练会话状态已更新，请重试");
      }
    } else {
      return buildGuidedTrainingSessionPayload(existingSession);
    }
  }

  const sessionId = crypto.randomUUID();
  const artifactDir = buildGuidedTrainingArtifactDir(body.experiment_id, studentId, modelType, sessionId);

  const completedStepIds: string[] = [];
  const stepPlan = buildGuidedTrainingStepPlan(modelType, body);
  const currentStepId = getGuidedNextStepId(stepPlan, completedStepIds);
  try {
    await pool.query(
      `INSERT INTO model_guided_training_sessions
       (session_id,
        experiment_id,
        student_id,
        model_type,
        request_hash,
        request_payload,
        status,
        current_step_id,
        completed_step_ids,
        step_outputs,
        step_plan,
        result_payload,
        artifact_dir,
        training_input_version,
        expires_at)
       VALUES (?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, ?, NULL, ?, ?, DATE_ADD(NOW(), INTERVAL 7 DAY))`,
      [
        sessionId,
        body.experiment_id,
        studentId,
        modelType,
        requestHash,
        JSON.stringify(body),
        currentStepId,
        JSON.stringify(completedStepIds),
        JSON.stringify({}),
        JSON.stringify(stepPlan),
        artifactDir,
        Number(experiment.training_input_version),
      ],
    );
  } catch (error) {
    if (!isDuplicateKeyError(error)) throw error;

    const [winnerRows] = await pool.query<GuidedTrainingSessionRow[]>(
      `SELECT session_id,
              experiment_id,
              student_id,
              model_type,
              request_hash,
              request_payload,
              status,
              current_step_id,
              completed_step_ids,
              step_outputs,
              step_plan,
              result_payload,
              artifact_dir,
              error_message,
              training_input_version,
              run_token,
              run_version,
              lease_expires_at,
              last_heartbeat_at
         FROM model_guided_training_sessions
        WHERE experiment_id = ?
          AND student_id = ?
          AND model_type = ?
          AND request_hash = ?
        LIMIT 1`,
      [body.experiment_id, studentId, modelType, requestHash],
    );
    if (winnerRows[0]) {
      return buildGuidedTrainingSessionPayload(winnerRows[0]);
    }
    throw error;
  }

  const row = await fetchGuidedTrainingSessionRow(sessionId);
  if (!row) {
    throw new Error("创建分阶段训练会话失败");
  }

  return buildGuidedTrainingSessionPayload(row);
}

export async function getGuidedTrainingSession(
  modelType: ModelType,
  studentId: number,
  sessionId: string,
): Promise<GuidedTrainingSessionPayload> {
  if (!isGuidedModelType(modelType)) {
    throw new ModelValidationError("当前模型暂不支持分阶段训练");
  }

  const row = await fetchGuidedTrainingSessionRow(sessionId);
  if (!row || row.model_type !== modelType) {
    throw new ModelNotFoundError("未找到分阶段训练会话");
  }

  if (row.student_id !== studentId) {
    throw new ModelAccessError("权限不足：该训练会话不属于当前用户", 403);
  }

  await assertExperimentModelAccess(row.experiment_id, studentId);
  await assertGuidedTrainingSessionDependenciesCurrent(row);
  return buildGuidedTrainingSessionPayload(row);
}

export async function activateGuidedTrainingSession(
  modelType: ModelType,
  studentId: number,
  sessionId: string,
): Promise<GuidedTrainingSessionPayload> {
  if (!isGuidedModelType(modelType)) {
    throw new ModelValidationError("当前模型暂不支持分阶段训练");
  }

  const initialRow = await fetchGuidedTrainingSessionRow(sessionId);
  if (!initialRow || initialRow.model_type !== modelType) {
    throw new ModelNotFoundError("未找到分阶段训练会话");
  }
  if (initialRow.student_id !== studentId) {
    throw new ModelAccessError("权限不足：该训练会话不属于当前用户", 403);
  }

  let artifactLocks = null;
  try {
    artifactLocks = await acquireModelArtifactMutationLocks(initialRow.experiment_id, modelType);
    if (!artifactLocks) {
      throw new ModelConflictError("该实验已有模型任务在执行，请稍后重试");
    }

    await assertExperimentModelAccess(initialRow.experiment_id, studentId);
    const row = await fetchGuidedTrainingSessionRow(sessionId);
    if (!row || row.model_type !== modelType || row.student_id !== studentId) {
      throw new ModelNotFoundError("未找到分阶段训练会话");
    }
    if (row.status !== "completed" || !row.result_payload) {
      throw new ModelValidationError("只能激活已完成的分阶段训练会话");
    }

    await assertGuidedTrainingSessionDependenciesCurrent(row);
    const artifactPath = getGuidedSessionModelPath(row.artifact_dir, modelType);
    if (!(await hasReusableSavedModelArtifact(modelType, artifactPath))) {
      throw new ModelNotFoundError("分阶段训练产物已过期或已被清理，请重新开始分阶段训练");
    }
    const request = parseJsonColumn<TrainInput>(row.request_payload, {} as TrainInput);
    const result = parseJsonColumn<ReturnType<typeof transformTrainingResult> | null>(
      row.result_payload,
      null,
    );
    if (!result) {
      throw new ModelValidationError("分阶段训练结果不完整，无法激活");
    }

    const connection = await pool.getConnection();
    let changed = false;
    let experimentStateVersion: number | null = null;
    try {
      await connection.beginTransaction();
      const [sessionRows] = await connection.query<RowDataPacket[]>(
        `SELECT session_id
           FROM model_guided_training_sessions
          WHERE session_id = ?
            AND student_id = ?
            AND model_type = ?
            AND status = 'completed'
          LIMIT 1
          FOR UPDATE`,
        [sessionId, studentId, modelType],
      );
      if (sessionRows.length !== 1) {
        throw new ModelConflictError("分阶段训练会话状态已变更，请刷新后重试");
      }
      const activation = await activateModelArtifactInTransaction(connection, {
        experimentId: row.experiment_id,
        studentId,
        modelType,
        revisionToken: sessionId,
        sourceSessionId: sessionId,
        artifactPath,
        request,
        result,
        expectedTrainingInputVersion: Number(row.training_input_version),
      });
      changed = activation.changed;
      experimentStateVersion = activation.stateVersion;
      const activatedRequestHash = hashGuidedTrainingRequest(
        modelType,
        request,
        activation.trainingInputVersion,
        await buildGuidedDependencyRevisions(
          modelType,
          row.experiment_id,
          studentId,
          request.models,
          { connection, targetRevisionOverride: sessionId },
        ),
      );
      await connection.query(
        `UPDATE model_guided_training_sessions
            SET request_hash = ?,
                training_input_version = ?,
                expires_at = DATE_ADD(NOW(), INTERVAL 7 DAY)
          WHERE session_id = ?`,
        [activatedRequestHash, activation.trainingInputVersion, sessionId],
      );
      await connection.commit();
    } catch (error) {
      await connection.rollback().catch(() => {});
      throw error;
    } finally {
      connection.release();
    }

    return buildGuidedTrainingSessionPayload(row, {
      backtestArtifactChanged: changed,
      artifactRevision: sessionId,
      experimentStateVersion,
    });
  } finally {
    await releaseModelArtifactMutationLocks(artifactLocks);
  }
}

export async function runGuidedTrainingStep(
  modelType: ModelType,
  studentId: number,
  sessionId: string,
  stepId: string,
): Promise<GuidedTrainingSessionPayload> {
  if (!isGuidedModelType(modelType)) {
    throw new ModelValidationError("当前模型暂不支持分阶段训练");
  }

  const initialRow = await fetchGuidedTrainingSessionRow(sessionId);
  if (!initialRow || initialRow.model_type !== modelType) {
    throw new ModelNotFoundError("未找到分阶段训练会话");
  }

  if (initialRow.student_id !== studentId) {
    throw new ModelAccessError("权限不足：该训练会话不属于当前用户", 403);
  }

  let artifactLocks = null;
  let slotLock = null;
  let attempt: GuidedTrainingAttempt | null = null;
  let stopHeartbeat: (() => Promise<void>) | null = null;
  let experimentStateVersion: number | null = null;

  try {
    artifactLocks = await acquireModelArtifactMutationLocks(initialRow.experiment_id, modelType);
    if (!artifactLocks) {
      throw new ModelConflictError("该实验已有模型任务在执行，请稍后重试");
    }

    const row = await fetchGuidedTrainingSessionRow(sessionId);
    if (!row || row.model_type !== modelType) {
      throw new ModelNotFoundError("未找到分阶段训练会话");
    }
    if (row.student_id !== studentId) {
      throw new ModelAccessError("权限不足：该训练会话不属于当前用户", 403);
    }
    await assertGuidedTrainingSessionDependenciesCurrent(row);

    if (row.status === "completed") {
      return buildGuidedTrainingSessionPayload(row);
    }

    const body = parseJsonColumn<TrainInput>(row.request_payload, {} as TrainInput);
    const completedStepIds = parseJsonColumn<string[]>(row.completed_step_ids, []);
    if (completedStepIds.includes(stepId)) {
      return buildGuidedTrainingSessionPayload(row);
    }

    const stepPlan = getGuidedTrainingSteps(row);
    const nextStepId = getGuidedNextStepId(stepPlan, completedStepIds);
    if (stepId !== nextStepId) {
      throw new ModelValidationError("只能按训练流程顺序执行下一个阶段");
    }

    slotLock = await acquireModelSlot();
    if (!slotLock) {
      throw new ModelCapacityError("模型服务繁忙，请稍后重试");
    }

    attempt = await claimGuidedTrainingAttempt(row, stepId);
    stopHeartbeat = startGuidedTrainingHeartbeat(sessionId, attempt);

    const sessionModelPath = getGuidedSessionModelPath(row.artifact_dir, modelType);
    const pythonParams = await buildGuidedTrainingPythonParams(
      modelType,
      studentId,
      body,
      sessionModelPath,
    );
    const stateFilePath = buildGuidedTrainingStatePath(row.experiment_id, studentId, modelType, sessionId);
    const pythonStepResult = await runPythonGuidedTrainingStep(
      modelType,
      stepId,
      pythonParams,
      stateFilePath,
      row.artifact_dir,
    );
    await assertGuidedTrainingAttemptOwned(sessionId, attempt);

    const nextCompletedStepIds = [...completedStepIds, stepId];
    const stepOutputs = parseJsonColumn<Record<string, unknown>>(row.step_outputs, {});
    const nextStepOutputs = {
      ...stepOutputs,
      [stepId]: pythonStepResult.output ?? {},
    };
    const nextStepPlan = shouldTrimBoostingRoundSteps(modelType, stepId, pythonStepResult.output)
      ? trimBoostingStepPlanAfterEarlyStop(stepPlan, stepId)
      : stepPlan;

    let transformedResult: ReturnType<typeof transformTrainingResult> | null = null;
    let backtestArtifactChanged = false;
    if (pythonStepResult.final_result) {
      if (!(await hasReusableSavedModelArtifact(modelType, sessionModelPath))) {
        throw new ModelNotFoundError("分阶段训练未生成完整的模型产物");
      }
      pythonStepResult.final_result.saved_model = sessionModelPath;
      transformedResult = transformTrainingResult(modelType, pythonStepResult.final_result);
    }

    const status: GuidedTrainingStatus = transformedResult ? "completed" : "ready";
    const currentStepId = status === "completed"
      ? null
      : getGuidedNextStepId(nextStepPlan, nextCompletedStepIds);

    await stopHeartbeat();
    stopHeartbeat = null;
    const buildCommitParams = (
      requestHash: string,
      trainingInputVersion: number,
    ) => [
        status,
        currentStepId,
        JSON.stringify(nextCompletedStepIds),
        JSON.stringify(nextStepOutputs),
        JSON.stringify(nextStepPlan),
        transformedResult ? JSON.stringify(transformedResult) : null,
        requestHash,
        trainingInputVersion,
        sessionId,
        attempt!.token,
        attempt!.version,
      ];
    const commitSql = `UPDATE model_guided_training_sessions
                          SET status = ?,
                              current_step_id = ?,
                              completed_step_ids = ?,
                              step_outputs = ?,
                              step_plan = ?,
                              result_payload = ?,
                              request_hash = ?,
                              training_input_version = ?,
                              error_message = NULL,
                              run_token = NULL,
                              lease_expires_at = NULL,
                              last_heartbeat_at = NULL,
                              expires_at = DATE_ADD(NOW(), INTERVAL 7 DAY)
                        WHERE session_id = ?
                          AND status = 'running'
                          AND run_token = ?
                          AND run_version = ?
                          AND lease_expires_at > NOW(3)`;

    if (transformedResult) {
      const connection = await pool.getConnection();
      try {
        await connection.beginTransaction();
        const [ownershipRows] = await connection.query<RowDataPacket[]>(
          `SELECT session_id
             FROM model_guided_training_sessions
            WHERE session_id = ?
              AND status = 'running'
              AND run_token = ?
              AND run_version = ?
              AND lease_expires_at > NOW(3)
            LIMIT 1
            FOR UPDATE`,
          [sessionId, attempt.token, attempt.version],
        );
        if (ownershipRows.length !== 1) {
          throw new ModelStaleAttemptError("分阶段训练执行权已转移，本次结果已丢弃");
        }

        const activation = await activateModelArtifactInTransaction(connection, {
          experimentId: row.experiment_id,
          studentId,
          modelType,
          revisionToken: sessionId,
          sourceSessionId: sessionId,
          artifactPath: sessionModelPath,
          request: body,
          result: transformedResult,
          expectedTrainingInputVersion: Number(row.training_input_version),
        });
        backtestArtifactChanged = activation.changed;
        experimentStateVersion = activation.stateVersion;
        const activatedRequestHash = hashGuidedTrainingRequest(
          modelType,
          body,
          activation.trainingInputVersion,
          await buildGuidedDependencyRevisions(
            modelType,
            row.experiment_id,
            studentId,
            body.models,
            { connection, targetRevisionOverride: sessionId },
          ),
        );
        const [commitResult] = await connection.query<ResultSetHeader>(
          commitSql,
          buildCommitParams(activatedRequestHash, activation.trainingInputVersion),
        );
        if (commitResult.affectedRows !== 1) {
          throw new ModelStaleAttemptError("分阶段训练执行权已转移，本次结果已丢弃");
        }
        await connection.commit();
      } catch (error) {
        await connection.rollback().catch(() => {});
        throw error;
      } finally {
        connection.release();
      }
    } else {
      const [commitResult] = await pool.query<ResultSetHeader>(
        commitSql,
        buildCommitParams(row.request_hash, Number(row.training_input_version)),
      );
      if (commitResult.affectedRows !== 1) {
        throw new ModelStaleAttemptError("分阶段训练执行权已转移，本次结果已丢弃");
      }
    }

    const updatedRow = await fetchGuidedTrainingSessionRow(sessionId);
    if (!updatedRow) {
      throw new Error("分阶段训练会话已丢失");
    }

    return buildGuidedTrainingSessionPayload(updatedRow, {
      backtestArtifactChanged,
      artifactRevision: transformedResult ? sessionId : null,
      experimentStateVersion,
    });
  } catch (error) {
    if (stopHeartbeat) {
      await stopHeartbeat();
      stopHeartbeat = null;
    }
    if (attempt) {
      await recordGuidedTrainingAttemptFailure(sessionId, stepId, attempt, error);
    }
    throw error;
  } finally {
    if (stopHeartbeat) {
      await stopHeartbeat();
    }
    await releaseModelSlot(slotLock);
    await releaseModelArtifactMutationLocks(artifactLocks);
  }
}

export async function trainModel(
  modelType: ModelType,
  studentId: number,
  body: TrainInput,
  progressReporter?: ModelProgressReporter,
) {
  reportModelProgress(progressReporter, {
    type: "phase",
    phase: "validate",
    percent: 4,
    source: "backend",
    message: "正在校验实验状态、模型参数和数据窗口",
  });
  const experiment = await assertExperimentModelAccess(body.experiment_id, studentId);
  assertTrainingRequestMatchesExperiment(experiment, body);
  const trainingConfig = getExperimentTrainingConfig(experiment);
  if (modelType === "ma") {
    validateMovingAverageWindow(trainingConfig, body.moving_average_window);
  } else if (modelType === "lstm") {
    validateLstmProductionFeasibility(trainingConfig);
  }
  const derivedModelParams: Record<string, unknown> = isEnsembleModelType(modelType)
    ? buildEnsemblePythonParams(experiment, body)
    : {};
  if (isEnsembleModelType(modelType) && body.models) {
    const baseModelArtifactMap = await buildAvailableBaseModelArtifactMap(
      body.experiment_id,
      studentId,
      body.models,
      Number(experiment.training_input_version),
    );
    if (Object.keys(baseModelArtifactMap).length > 0) {
      derivedModelParams.baseModelFiles = JSON.stringify(baseModelArtifactMap);
      reportModelProgress(progressReporter, {
        type: "phase",
        phase: "reuse_artifacts",
        percent: 9,
        source: "backend",
        message: "已找到可复用的基础模型产物，融合训练将优先复用这些结果",
      });
    }
  }

  reportModelProgress(progressReporter, {
    type: "phase",
    phase: "load_data",
    percent: 12,
    source: "backend",
    message: `正在加载 ${trainingConfig.selected_industry} / ${trainingConfig.selected_company} / ${trainingConfig.selected_product} 的实验数据`,
  });
  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
  );

  if (
    trainingConfig.data_window_train_end_index >= allData.length
    || trainingConfig.data_window_evaluate_end_index >= allData.length
  ) {
    throw new ModelValidationError("数据窗口索引超出范围", { dataset_size: allData.length });
  }

  reportModelProgress(progressReporter, {
    type: "phase",
    phase: "prepare_python_params",
    percent: 18,
    source: "backend",
    message: "实验数据已加载，正在准备 Python 训练参数",
  });

  const revisionToken = crypto.randomUUID();
  const modelSaveLoc = buildVersionedModelArtifactPath(
    body.experiment_id,
    studentId,
    modelType,
    revisionToken,
    "backtest",
  );

  const pythonParams: Record<string, unknown> = {
    data: JSON.stringify(allData),
    trainStart: trainingConfig.data_window_train_start_index,
    trainEnd: trainingConfig.data_window_train_end_index,
    evalStart: trainingConfig.data_window_evaluate_start_index,
    evalEnd: trainingConfig.data_window_evaluate_end_index,
    saveModel: modelSaveLoc,
    pretty: true,
    ...derivedModelParams,
  };

  for (const bodyKey of Object.keys(allPossiblePythonParams)) {
    if (isEnsembleModelType(modelType) && ENSEMBLE_DERIVED_BODY_KEYS.has(bodyKey)) {
      continue;
    }

    const value = body[bodyKey as keyof TrainInput];
    if (value !== undefined) {
      pythonParams[bodyKey] = value;
    }
  }

  const timeout = TRAINING_TIMEOUT;
  const result = (await runPythonModel(modelType, pythonParams, timeout, progressReporter)) as Record<string, unknown>;
  reportModelProgress(progressReporter, {
    type: "phase",
    phase: "invalidate_production_artifact",
    percent: 92,
    source: "backend",
    message: "训练结果已生成，正在更新模型产物状态",
  });
  reportModelProgress(progressReporter, {
    type: "phase",
    phase: "transform_result",
    percent: 96,
    source: "backend",
    message: "正在整理评估指标、预测值和模型元数据",
  });
  if (!(await hasReusableSavedModelArtifact(modelType, modelSaveLoc))) {
    throw new ModelNotFoundError("训练完成但未生成完整的模型产物");
  }
  const transformedResult = transformTrainingResult(modelType, result);
  const connection = await pool.getConnection();
  let experimentStateVersion: number | null = null;
  try {
    await connection.beginTransaction();
    const activation = await activateModelArtifactInTransaction(connection, {
      experimentId: body.experiment_id,
      studentId,
      modelType,
      revisionToken,
      sourceSessionId: null,
      artifactPath: modelSaveLoc,
      request: body,
      result: transformedResult,
      expectedTrainingInputVersion: Number(experiment.training_input_version),
    });
    experimentStateVersion = activation.stateVersion;
    await connection.commit();
  } catch (error) {
    await connection.rollback().catch(() => {});
    throw error;
  } finally {
    connection.release();
  }
  transformedResult.results.artifact_revision = revisionToken;
  transformedResult.results.experiment_state_version = experimentStateVersion;
  return transformedResult;
}

export async function getReusableProductionModelArtifactResult(
  modelType: ModelType,
  studentId: number,
  experimentId: number,
) {
  const experiment = await assertExperimentModelAccess(experimentId, studentId);
  assertModelReadyForProduction(experiment, modelType);
  if (modelType === "lstm") {
    validateLstmProductionFeasibility(getExperimentTrainingConfig(experiment));
  }

  const activeArtifact = await fetchActiveModelArtifact(experimentId, studentId, modelType);
  if (activeArtifact) {
    assertActiveArtifactMatchesTrainingInput(experiment, activeArtifact);
  }
  if (
    !activeArtifact
    || activeArtifact.production_revision_token !== activeArtifact.revision_token
    || !activeArtifact.production_artifact_path
    || !await hasReusableSavedModelArtifact(modelType, activeArtifact.production_artifact_path)
  ) {
    return null;
  }

  return {
    status: "success",
    results: {
      saved_model: activeArtifact.production_artifact_path,
      artifact_phase: "production",
      reused_existing: true,
      artifact_revision: activeArtifact.revision_token,
    },
  };
}

export async function completeExperimentModel(
  modelType: ModelType,
  studentId: number,
  experimentId: number,
  artifactRevision: string,
) {
  const completedFields = {
    ma: "moving_average_completed",
    es: "exponential_smoothing_completed",
    arima: "arima_completed",
    lstm: "lstm_completed",
    weighted_avg: "ensemble_weighted_completed",
    boosting: "ensemble_boosting_completed",
    stacking: "ensemble_stacking_completed",
  } as const satisfies Record<ModelType, keyof ExperimentModelContextRow>;
  const completedField = completedFields[modelType];
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();
    const experiment = await assertExperimentModelAccess(
      experimentId,
      studentId,
      connection,
      true,
    );
    const activeArtifact = await fetchActiveModelArtifact(
      experimentId,
      studentId,
      modelType,
      connection,
    );
    if (!activeArtifact || activeArtifact.revision_token !== artifactRevision) {
      throw new ModelConflictError("模型结果版本已更新，请刷新页面后重试");
    }
    assertActiveArtifactMatchesTrainingInput(experiment, activeArtifact);

    if (normalizeBooleanFlag(experiment[completedField])) {
      await connection.commit();
      return {
        status: "success",
        artifact_revision: artifactRevision,
        experiment_state_version: Number(experiment.state_version),
      };
    }

    const [result] = await connection.query<ResultSetHeader>(
      `UPDATE experiment_status
          SET \`${completedField}\` = 1,
              last_activity_at = NOW(),
              state_version = state_version + 1
        WHERE experiment_id = ?
          AND student_id = ?`,
      [experimentId, studentId],
    );
    if (result.affectedRows !== 1) {
      throw new ModelConflictError("模型结果版本已更新，请刷新页面后重试");
    }
    await connection.commit();
    return {
      status: "success",
      artifact_revision: artifactRevision,
      experiment_state_version: Number(experiment.state_version) + 1,
    };
  } catch (error) {
    await connection.rollback().catch(() => {});
    throw error;
  } finally {
    connection.release();
  }
}

export async function prepareExperimentModelForProduction(
  modelType: ModelType,
  studentId: number,
  experimentId: number,
) {
  const reusableArtifact = await getReusableProductionModelArtifactResult(modelType, studentId, experimentId);
  if (reusableArtifact) {
    return reusableArtifact;
  }

  const experiment = await assertExperimentModelAccess(experimentId, studentId);
  assertModelReadyForProduction(experiment, modelType);
  if (modelType === "lstm") {
    validateLstmProductionFeasibility(getExperimentTrainingConfig(experiment));
  }
  const trainingConfig = getExperimentTrainingConfig(experiment);

  const activeArtifact = await fetchActiveModelArtifact(experimentId, studentId, modelType);
  if (activeArtifact) {
    assertActiveArtifactMatchesTrainingInput(experiment, activeArtifact);
  }
  if (
    !activeArtifact
    || !await hasReusableSavedModelArtifact(modelType, activeArtifact.backtest_artifact_path)
  ) {
    throw new ModelNotFoundError(`未找到该实验的已训练 ${modelType.toUpperCase()} 回测模型`);
  }
  const backtestModelPath = activeArtifact.backtest_artifact_path;
  const productionAttemptToken = crypto.randomUUID();
  const productionModelPath = buildVersionedModelArtifactPath(
    experimentId,
    studentId,
    modelType,
    `${activeArtifact.revision_token}-${productionAttemptToken}`,
    "production",
  );

  const [claimResult] = await pool.query<ResultSetHeader>(
    `UPDATE experiment_model_artifacts
        SET production_artifact_path = NULL,
            production_revision_token = ?
      WHERE experiment_id = ?
        AND student_id = ?
        AND model_type = ?
        AND revision_token = ?
        AND training_input_version = ?
        AND EXISTS (
          SELECT 1
            FROM experiment_status current_experiment
           WHERE current_experiment.experiment_id = experiment_model_artifacts.experiment_id
             AND current_experiment.student_id = experiment_model_artifacts.student_id
             AND current_experiment.training_input_version = experiment_model_artifacts.training_input_version
        )`,
    [
      productionAttemptToken,
      experimentId,
      studentId,
      modelType,
      activeArtifact.revision_token,
      Number(activeArtifact.training_input_version),
    ],
  );
  if (claimResult.affectedRows !== 1) {
    throw new ModelStaleAttemptError("回测模型版本已更新，本次生产模型任务未启动");
  }

  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
  );

  if (trainingConfig.data_window_evaluate_end_index >= allData.length) {
    throw new ModelValidationError("数据窗口索引超出范围", { dataset_size: allData.length });
  }

  const pythonParams: Record<string, unknown> = {
    data: JSON.stringify(allData),
    trainStart: trainingConfig.data_window_train_start_index,
    trainEnd: trainingConfig.data_window_evaluate_end_index,
    saveModel: productionModelPath,
    loadModel: backtestModelPath,
    fitOnly: true,
    pretty: true,
  };

  await runPythonModel(modelType, pythonParams, TRAINING_TIMEOUT);
  if (!(await hasReusableSavedModelArtifact(modelType, productionModelPath))) {
    throw new ModelNotFoundError("生产预测模型准备完成但产物不完整");
  }
  const [updateResult] = await pool.query<ResultSetHeader>(
    `UPDATE experiment_model_artifacts
        SET production_artifact_path = ?,
            production_revision_token = ?
      WHERE experiment_id = ?
        AND student_id = ?
        AND model_type = ?
        AND revision_token = ?
        AND training_input_version = ?
        AND production_revision_token = ?
        AND EXISTS (
          SELECT 1
            FROM experiment_status current_experiment
           WHERE current_experiment.experiment_id = experiment_model_artifacts.experiment_id
             AND current_experiment.student_id = experiment_model_artifacts.student_id
             AND current_experiment.training_input_version = experiment_model_artifacts.training_input_version
        )`,
    [
      productionModelPath,
      activeArtifact.revision_token,
      experimentId,
      studentId,
      modelType,
      activeArtifact.revision_token,
      Number(activeArtifact.training_input_version),
      productionAttemptToken,
    ],
  );
  if (updateResult.affectedRows !== 1) {
    invalidateArtifactAtPath(modelType, productionModelPath);
    throw new ModelStaleAttemptError("回测模型版本已更新，本次生产模型结果已丢弃");
  }

  return {
    status: "success",
    results: {
      saved_model: productionModelPath,
      artifact_phase: "production",
      reused_existing: false,
      artifact_revision: activeArtifact.revision_token,
    },
  };
}

export function cleanupExperimentModelArtifacts(experimentId: number, studentId: number) {
  const trainedModelsDir = path.join(PY_DIR, "trained_models");
  let entries: string[];
  try {
    entries = fs.readdirSync(trainedModelsDir);
  } catch {
    return;
  }

  const prefix = `exp${experimentId}-student${studentId}-`;
  let deleted = 0;
  for (const entry of entries) {
    if (!entry.startsWith(prefix)) continue;
    try {
      fs.unlinkSync(path.join(trainedModelsDir, entry));
      deleted++;
    } catch (error: unknown) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        logger.warn({ err: error, file: entry }, "删除模型产物失败");
      }
    }
  }

  if (deleted > 0) {
    logger.info({ experimentId, studentId, deleted }, "已清理实验模型产物");
  }
}

interface ModelArtifactGroup {
  experimentId: number;
  studentId: number;
  prefix: string;
  files: string[];
  newestMtimeMs: number;
}

const MODEL_ARTIFACT_PREFIX_PATTERN = /^exp(\d+)-student(\d+)-/;

const toCleanupAgeSeconds = (maxAgeMs: number) => Math.max(1, Math.ceil(maxAgeMs / 1000));

const isDbTruthy = (value: unknown) =>
  value === true
  || value === 1
  || value === "1"
  || value === BigInt(1);

const shouldDeleteStaleModelArtifactGroup = async (
  group: ModelArtifactGroup,
  maxAgeSeconds: number,
) => {
  const [rows] = await pool.query<StaleModelExperimentEligibilityRow[]>(
    `SELECT CASE
              WHEN status = 'Completed' THEN 1
              WHEN TIMESTAMPDIFF(
                SECOND,
                GREATEST(
                  COALESCE(last_activity_at, CAST('1970-01-01 00:00:00' AS DATETIME)),
                  COALESCE(completion_time, CAST('1970-01-01 00:00:00' AS DATETIME)),
                  COALESCE(start_time, CAST('1970-01-01 00:00:00' AS DATETIME))
                ),
                NOW()
              ) >= ? THEN 1
              ELSE 0
            END AS can_delete
       FROM experiment_status
      WHERE experiment_id = ?
        AND student_id = ?
      LIMIT 1`,
    [maxAgeSeconds, group.experimentId, group.studentId],
  );

  if (!rows[0]) return true;
  return isDbTruthy(rows[0].can_delete);
};

export async function cleanupStaleModelArtifacts(
  options: { now?: Date; maxAgeMs?: number } = {},
) {
  const trainedModelsDir = path.join(PY_DIR, "trained_models");
  const now = options.now ?? new Date();
  const maxAgeMs = options.maxAgeMs ?? CLEANUP_RETENTION.MODEL_ARTIFACTS_MS;
  const cutoffMs = now.getTime() - maxAgeMs;
  const maxAgeSeconds = toCleanupAgeSeconds(maxAgeMs);
  const result = {
    scannedGroups: 0,
    deletedGroups: 0,
    deletedFiles: 0,
    failedFiles: 0,
  };

  let entries: string[];
  try {
    entries = fs.readdirSync(trainedModelsDir);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return result;
    }
    throw error;
  }

  const groups = new Map<string, ModelArtifactGroup>();
  for (const entry of entries) {
    const match = MODEL_ARTIFACT_PREFIX_PATTERN.exec(entry);
    if (!match) continue;

    const filePath = path.join(trainedModelsDir, entry);
    let stats: Stats;
    try {
      stats = fs.statSync(filePath);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        logger.warn({ err: error, file: entry }, "读取模型产物状态失败");
      }
      continue;
    }
    if (!stats.isFile()) continue;

    const prefix = match[0]!;
    const existing = groups.get(prefix);
    if (existing) {
      existing.files.push(entry);
      existing.newestMtimeMs = Math.max(existing.newestMtimeMs, stats.mtimeMs);
    } else {
      groups.set(prefix, {
        experimentId: Number(match[1]),
        studentId: Number(match[2]),
        prefix,
        files: [entry],
        newestMtimeMs: stats.mtimeMs,
      });
    }
  }

  for (const group of groups.values()) {
    if (group.newestMtimeMs > cutoffMs) {
      continue;
    }

    result.scannedGroups += 1;
    if (!await shouldDeleteStaleModelArtifactGroup(group, maxAgeSeconds)) {
      continue;
    }

    const [activeArtifactRows] = await pool.query<RowDataPacket[]>(
      `SELECT backtest_artifact_path,
              production_artifact_path
         FROM experiment_model_artifacts
        WHERE experiment_id = ?
          AND student_id = ?`,
      [group.experimentId, group.studentId],
    );
    const referencedPaths = new Set<string>();
    const referencedFamilyPrefixes: string[] = [];
    const registerArtifactFamily = (artifactPath: string) => {
      const resolvedArtifactPath = path.resolve(artifactPath);
      const artifactDirectory = path.dirname(resolvedArtifactPath);
      const artifactBaseName = path.basename(resolvedArtifactPath, ".pkl");
      referencedPaths.add(resolvedArtifactPath);
      referencedPaths.add(path.join(artifactDirectory, `${artifactBaseName}_keras_model.h5`));
      referencedFamilyPrefixes.push(path.join(artifactDirectory, `${artifactBaseName}_base_`));
    };
    for (const activeArtifact of activeArtifactRows) {
      if (typeof activeArtifact.backtest_artifact_path === "string") {
        registerArtifactFamily(activeArtifact.backtest_artifact_path);
      }
      if (typeof activeArtifact.production_artifact_path === "string") {
        registerArtifactFamily(activeArtifact.production_artifact_path);
      }
    }

    let deletedInGroup = 0;
    for (const file of group.files) {
      const filePath = path.join(trainedModelsDir, file);
      const resolvedFilePath = path.resolve(filePath);
      if (
        referencedPaths.has(resolvedFilePath)
        || referencedFamilyPrefixes.some((prefix) => resolvedFilePath.startsWith(prefix))
      ) {
        continue;
      }
      try {
        fs.unlinkSync(filePath);
        deletedInGroup += 1;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
          result.failedFiles += 1;
          logger.warn({ err: error, file }, "删除过期模型产物失败");
        }
      }
    }

    if (deletedInGroup > 0) {
      result.deletedGroups += 1;
      result.deletedFiles += deletedInGroup;
      logger.info(
        {
          experimentId: group.experimentId,
          studentId: group.studentId,
          prefix: group.prefix,
          deletedFiles: deletedInGroup,
        },
        "已清理过期模型产物组",
      );
    }
  }

  return result;
}

export async function predictExperimentModel(
  modelType: ModelType,
  studentId: number,
  experimentId: number,
  forecastSteps: number,
) {
  const experiment = await assertExperimentModelAccess(experimentId, studentId);
  assertModelReadyForProduction(experiment, modelType);
  if (modelType === "lstm") {
    validateLstmProductionFeasibility(getExperimentTrainingConfig(experiment));
  }

  let activeArtifact = await fetchActiveModelArtifact(experimentId, studentId, modelType);
  if (activeArtifact) {
    assertActiveArtifactMatchesTrainingInput(experiment, activeArtifact);
  }
  if (
    !activeArtifact
    || activeArtifact.production_revision_token !== activeArtifact.revision_token
    || !activeArtifact.production_artifact_path
    || !await hasReusableSavedModelArtifact(modelType, activeArtifact.production_artifact_path)
  ) {
    await prepareExperimentModelForProduction(modelType, studentId, experimentId);
    activeArtifact = await fetchActiveModelArtifact(experimentId, studentId, modelType);
    if (activeArtifact) {
      assertActiveArtifactMatchesTrainingInput(experiment, activeArtifact);
    }
  }

  if (
    !activeArtifact
    || activeArtifact.production_revision_token !== activeArtifact.revision_token
    || !activeArtifact.production_artifact_path
    || !await hasReusableSavedModelArtifact(modelType, activeArtifact.production_artifact_path)
  ) {
    throw new ModelNotFoundError(`未找到该实验的 ${modelType.toUpperCase()} 生产预测模型`);
  }
  const productionModelPath = activeArtifact.production_artifact_path;

  const pythonParams = {
    loadModel: productionModelPath,
    forecastSteps,
    pretty: true,
  };

  const result = (await runPythonModel(modelType, pythonParams, TIMEOUTS.MODEL_PREDICTION)) as Record<string, unknown>;
  const currentExperiment = await assertExperimentModelAccess(experimentId, studentId);
  const currentArtifact = await fetchActiveModelArtifact(experimentId, studentId, modelType);
  if (
    !currentArtifact
    || currentArtifact.revision_token !== activeArtifact.revision_token
    || currentArtifact.production_revision_token !== activeArtifact.production_revision_token
  ) {
    throw new ModelStaleAttemptError("预测期间模型版本已更新，本次预测结果已丢弃");
  }
  assertActiveArtifactMatchesTrainingInput(currentExperiment, currentArtifact);

  return {
    status: "success",
    results: {
      predictions: result.predictions || [],
      method_name: result.method_name,
      forecast_strategy: result.forecast_strategy,
      implementation_notes: result.implementation_notes,
    },
  };
}
