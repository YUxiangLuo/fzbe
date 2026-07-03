import path from "path";
import fs from "fs";
import crypto from "crypto";
import type { PoolConnection, RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import logger from "../../utils/logger";
import type { ModelType, TrainInput } from "./models.schema";
import { TIMEOUTS, CONCURRENCY_LIMITS, PYTHON_RUNTIME_CONFIG } from "../../utils/constants";
import { PY_DIR, resolvePythonExecutable } from "../../utils/python";
import { CsvEncodingError, readCsvRows } from "../../utils/csv-processor";
import {
  buildGuidedTrainingArtifactDir,
  buildGuidedTrainingStatePath,
  buildExperimentScopedModelPath,
  buildModelExecutionLockName,
  buildModelSlotLockName,
} from "./model-runtime";

const TRAINING_TIMEOUT = TIMEOUTS.MODEL_TRAINING;
const MAX_CONCURRENT_MODEL_JOBS = CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS;

function buildPythonModelProcessEnv(): Record<string, string> {
  const modelThreads = String(PYTHON_RUNTIME_CONFIG.MODEL_PROCESS_THREADS);
  const tfInterOpThreads = String(PYTHON_RUNTIME_CONFIG.TF_INTER_OP_THREADS);

  return {
    OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? modelThreads,
    OPENBLAS_NUM_THREADS: process.env.OPENBLAS_NUM_THREADS ?? modelThreads,
    MKL_NUM_THREADS: process.env.MKL_NUM_THREADS ?? modelThreads,
    VECLIB_MAXIMUM_THREADS: process.env.VECLIB_MAXIMUM_THREADS ?? modelThreads,
    NUMEXPR_NUM_THREADS: process.env.NUMEXPR_NUM_THREADS ?? modelThreads,
    TF_NUM_INTRAOP_THREADS: process.env.TF_NUM_INTRAOP_THREADS ?? modelThreads,
    TF_NUM_INTEROP_THREADS: process.env.TF_NUM_INTEROP_THREADS ?? tfInterOpThreads,
    TF_CPP_MIN_LOG_LEVEL: process.env.TF_CPP_MIN_LOG_LEVEL ?? "3",
  };
}

interface AdvisoryLockRow extends RowDataPacket {
  acquired: 0 | 1 | null;
}

interface ExperimentOwnershipRow extends RowDataPacket {
  student_id: number;
  status: string;
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

interface TrainingMetricsPayload extends Record<string, unknown> {
  evaluate_indices?: unknown;
  evaluate_months?: unknown;
}

type BaseModelType = "ma" | "es" | "arima" | "lstm";
type EnsembleModelType = Extract<ModelType, "weighted_avg" | "boosting" | "stacking">;
type GuidedModelType = BaseModelType;
type GuidedTrainingStatus = "ready" | "running" | "failed" | "completed";

const BASE_MODEL_TYPES = new Set<ModelType>(["ma", "es", "arima", "lstm"]);
const ENSEMBLE_MODEL_TYPES = new Set<ModelType>(["weighted_avg", "boosting", "stacking"]);
const GUIDED_MODEL_TYPES = new Set<ModelType>(["ma", "es", "arima", "lstm"]);
const ENSEMBLE_DERIVED_BODY_KEYS = new Set([
  "models",
  "moving_average_window",
  "exponential_smoothing_alpha",
  "arimaD",
  "lstmFeatures",
  "lstmTargetFeature",
  "lstmNormalization",
]);

export interface GuidedTrainingStepDefinition {
  id: string;
  label: string;
  description: string;
  actionLabel: string;
}

export interface GuidedTrainingSessionPayload {
  session_id: string;
  experiment_id: number;
  model_type: GuidedModelType;
  status: GuidedTrainingStatus;
  current_step_id: string | null;
  next_step_id: string | null;
  steps: Array<GuidedTrainingStepDefinition & {
    status: "pending" | "active" | "completed" | "failed";
    output?: unknown;
  }>;
  step_outputs: Record<string, unknown>;
  result: { status: string; results: Record<string, unknown>; inferred_feature_types?: unknown } | null;
  error_message: string | null;
}

const GUIDED_TRAINING_STEPS: Record<GuidedModelType, GuidedTrainingStepDefinition[]> = {
  ma: [
    {
      id: "prepare_data",
      label: "校验训练窗口",
      description: "读取实验数据，确认训练区间、评估区间和移动平均窗口可以用于建模。",
      actionLabel: "校验数据",
    },
    {
      id: "build_windows",
      label: "构造滚动窗口",
      description: "取训练集末尾的历史销量作为初始窗口，准备按窗口均值递推预测。",
      actionLabel: "构造窗口",
    },
    {
      id: "forecast",
      label: "逐期生成预测",
      description: "每预测一期，就把预测值放回窗口，为下一期预测提供历史。",
      actionLabel: "生成预测",
    },
    {
      id: "evaluate",
      label: "计算指标并保存",
      description: "比较预测值和评估集真实销量，计算 RMSE、MAE、R² 并保存模型产物。",
      actionLabel: "完成评估",
    },
  ],
  es: [
    {
      id: "prepare_data",
      label: "校验平滑参数",
      description: "读取实验数据，确认训练/评估窗口和 alpha 平滑系数有效。",
      actionLabel: "校验数据",
    },
    {
      id: "fit_model",
      label: "拟合指数平滑",
      description: "用训练集销量和 alpha 参数拟合一次指数平滑水平项。",
      actionLabel: "拟合模型",
    },
    {
      id: "forecast",
      label: "生成评估预测",
      description: "使用拟合后的水平项，对评估区间进行多期外推预测。",
      actionLabel: "生成预测",
    },
    {
      id: "evaluate",
      label: "计算指标并保存",
      description: "计算误差指标，保存可复用的指数平滑模型产物。",
      actionLabel: "完成评估",
    },
  ],
  arima: [
    {
      id: "prepare_data",
      label: "校验差分和搜索范围",
      description: "读取训练序列，确认 d 阶差分后仍有足够样本，并确定 p/q 搜索上限。",
      actionLabel: "准备搜索",
    },
    {
      id: "aic_search",
      label: "执行 AIC 搜索",
      description: "按 AIC 信息准则搜索候选 ARIMA(p,d,q) 模型。",
      actionLabel: "运行 AIC",
    },
    {
      id: "bic_search",
      label: "执行 BIC 搜索",
      description: "按 BIC 信息准则搜索候选模型，增强小样本下的复杂度惩罚。",
      actionLabel: "运行 BIC",
    },
    {
      id: "select_model",
      label: "选择最优阶数",
      description: "比较 AIC/BIC 候选模型，并根据样本量选择最终 p、d、q。",
      actionLabel: "选择模型",
    },
    {
      id: "evaluate",
      label: "预测评估并保存",
      description: "用最终 ARIMA 模型预测评估区间，计算指标并保存模型产物。",
      actionLabel: "完成评估",
    },
  ],
  lstm: [
    {
      id: "prepare_data",
      label: "解析字段并校验窗口",
      description: "读取多变量数据，确认目标字段、特征字段、归一化方式和训练窗口有效。",
      actionLabel: "解析数据",
    },
    {
      id: "preprocess",
      label: "预处理并构造样本",
      description: "仅用训练集拟合归一化/编码器，再构造 LSTM 输入序列和多步预测标签。",
      actionLabel: "构造样本",
    },
    {
      id: "build_model",
      label: "确定网络结构",
      description: "根据特征维度和样本量确定 LSTM 层、Dense 层和 batch size。",
      actionLabel: "构建网络",
    },
    {
      id: "train_evaluate",
      label: "训练、预测并保存",
      description: "执行真实 Keras 训练，生成评估预测，反归一化后计算指标并保存模型产物。",
      actionLabel: "训练并评估",
    },
  ],
};

export interface AdvisoryLockHandle {
  connection: PoolConnection;
  name: string;
}

function safeRelease(conn: PoolConnection) {
  try { conn.release(); } catch { /* connection may already be destroyed */ }
}

async function acquireAdvisoryLock(name: string): Promise<AdvisoryLockHandle | null> {
  const connection = await pool.getConnection();

  try {
    const [rows] = await connection.query<AdvisoryLockRow[]>(
      "SELECT GET_LOCK(?, 0) AS acquired",
      [name],
    );

    if (rows[0]?.acquired === 1) {
      return { connection, name };
    }

    safeRelease(connection);
    return null;
  } catch (error) {
    safeRelease(connection);
    throw error;
  }
}

async function releaseAdvisoryLock(handle: AdvisoryLockHandle | null | undefined) {
  if (!handle) {
    return;
  }

  try {
    await handle.connection.query("SELECT RELEASE_LOCK(?)", [handle.name]);
  } catch (error) {
    logger.warn({ err: error, lock: handle.name }, "释放数据库 advisory lock 失败");
  } finally {
    safeRelease(handle.connection);
  }
}

export const acquireModelSlot = async (): Promise<AdvisoryLockHandle | null> => {
  const connection = await pool.getConnection();

  try {
    for (let slot = 1; slot <= MAX_CONCURRENT_MODEL_JOBS; slot += 1) {
      const name = buildModelSlotLockName(slot);
      const [rows] = await connection.query<AdvisoryLockRow[]>(
        "SELECT GET_LOCK(?, 0) AS acquired",
        [name],
      );

      if (rows[0]?.acquired === 1) {
        return { connection, name };
      }
    }

    safeRelease(connection);
    return null;
  } catch (error) {
    safeRelease(connection);
    throw error;
  }
};

export const releaseModelSlot = async (handle: AdvisoryLockHandle | null | undefined) => {
  await releaseAdvisoryLock(handle);
};

export const acquireModelExecutionLock = async (
  experimentId: number,
  modelType: ModelType,
): Promise<AdvisoryLockHandle | null> => {
  return await acquireAdvisoryLock(buildModelExecutionLockName(experimentId, modelType));
};

export const releaseModelExecutionLock = async (handle: AdvisoryLockHandle | null | undefined) => {
  await releaseAdvisoryLock(handle);
};

interface SegmentRow extends RowDataPacket {
  file_path: string;
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
  result_payload: unknown;
  artifact_dir: string;
  error_message: string | null;
}

export class ModelValidationError extends Error {
  constructor(message: string, public details?: Record<string, unknown>) {
    super(message);
  }
}

export class ModelNotFoundError extends Error {}
export class ModelAccessError extends Error {
  constructor(message: string, public status: 400 | 403 | 404) {
    super(message);
  }
}


/** TS 参数名 → Python CLI 参数名 的映射 */
const allPossiblePythonParams: Record<string, string> = {
  // 通用参数（由 service 层手动设置）
  data: "data",
  trainStart: "train-start",
  trainEnd: "train-end",
  evalStart: "eval-start",
  evalEnd: "eval-end",
  saveModel: "save-model",
  loadModel: "load-model",
  forecastSteps: "forecast-steps",
  fitOnly: "fit-only",
  pretty: "pretty",
  baseModelFiles: "base-model-files",
  // 模型参数（从前端 body 透传）
  moving_average_window: "window",
  exponential_smoothing_alpha: "alpha",
  arimaD: "d",
  lstmFeatures: "lstm-features",
  lstmTargetFeature: "lstm-target-feature",
  lstmNormalization: "normalization",
  models: "models",
};

const SETSID_PATH = process.platform === "win32" ? undefined : Bun.which("setsid");

const extractPythonTaggedError = (stderr: string): Error | null => {
  const markers = [
    { marker: "[VALIDATION]", prefix: "[PYTHON_VALIDATION]" },
    { marker: "[DATA]", prefix: "[PYTHON_DATA]" },
  ] as const;

  const lines = stderr
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index]!;
    for (const { marker, prefix } of markers) {
      const markerIndex = line.indexOf(marker);
      if (markerIndex === -1) {
        continue;
      }

      const detail = line.slice(markerIndex + marker.length).trim();
      return new Error(`${prefix} ${detail}`.trim());
    }
  }

  return null;
};

export async function loadDatasetFromDb(
  industry: string,
  company: string,
  product: string,
): Promise<Record<string, unknown>[]> {
  const [segmentRows] = await pool.query<SegmentRow[]>(
    "SELECT file_path FROM dataset_segments WHERE industry_name = ? AND company_name = ? AND product_name = ? LIMIT 1",
    [industry, company, product],
  );

  if (segmentRows.length === 0) {
    throw new ModelValidationError("未找到数据集分段文件");
  }

  try {
    return await readCsvRows(segmentRows[0]!.file_path);
  } catch (error) {
    if (error instanceof CsvEncodingError) {
      throw new ModelValidationError(error.message);
    }
    throw error;
  }
}

async function assertExperimentModelAccess(experimentId: number, studentId: number) {
  const [rows] = await pool.query<ExperimentModelContextRow[]>(
    `SELECT student_id,
            status,
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
      LIMIT 1`,
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

const isBaseModelType = (modelType: ModelType): modelType is BaseModelType => {
  return BASE_MODEL_TYPES.has(modelType);
};

const isEnsembleModelType = (modelType: ModelType): modelType is Extract<ModelType, "weighted_avg" | "boosting" | "stacking"> => {
  return ENSEMBLE_MODEL_TYPES.has(modelType);
};

const getBacktestModelPath = (experimentId: number, studentId: number, modelType: ModelType) =>
  buildExperimentScopedModelPath(experimentId, studentId, modelType, "backtest");

const getProductionModelPath = (experimentId: number, studentId: number, modelType: ModelType) =>
  buildExperimentScopedModelPath(experimentId, studentId, modelType, "production");

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

const invalidateProductionArtifact = (experimentId: number, studentId: number, modelType: ModelType) => {
  const productionModelPath = getProductionModelPath(experimentId, studentId, modelType);
  removeFileIfExists(productionModelPath);

  if (modelType === "lstm") {
    removeFileIfExists(getLstmSidecarPath(productionModelPath));
    return;
  }

  if (isEnsembleModelType(modelType)) {
    removeSiblingArtifactsByPrefix(productionModelPath);
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

const LSTM_ARTIFACT_COPY_SCRIPT = `
import os
import pickle
import shutil
import sys
import tempfile

source_model_path = sys.argv[1]
destination_model_path = sys.argv[2]

if not os.path.exists(source_model_path):
    raise FileNotFoundError(f"source model not found: {source_model_path}")

os.makedirs(os.path.dirname(destination_model_path), exist_ok=True)
shutil.copy2(source_model_path, destination_model_path)

with open(destination_model_path, "rb") as handle:
    state = pickle.load(handle)

source_keras_path = state.get("keras_model_path")
if not source_keras_path:
    source_keras_path = os.path.splitext(source_model_path)[0] + "_keras_model.h5"

if not os.path.exists(source_keras_path):
    raise FileNotFoundError(f"source keras model not found: {source_keras_path}")

destination_keras_path = os.path.splitext(destination_model_path)[0] + "_keras_model.h5"
shutil.copy2(source_keras_path, destination_keras_path)
state["keras_model_path"] = destination_keras_path

fd, temp_name = tempfile.mkstemp(
    prefix=f".{os.path.basename(destination_model_path)}.",
    suffix=".tmp",
    dir=os.path.dirname(destination_model_path),
)
try:
    with os.fdopen(fd, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_name, destination_model_path)
finally:
    if os.path.exists(temp_name):
        try:
            os.remove(temp_name)
        except OSError:
            pass
`;

const copyLstmModelArtifact = async (sourceModelPath: string, destinationModelPath: string) => {
  const pythonExec = resolvePythonExecutable();
  const proc = Bun.spawn([
    pythonExec,
    "-c",
    LSTM_ARTIFACT_COPY_SCRIPT,
    sourceModelPath,
    destinationModelPath,
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

  const [, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  await proc.exited;

  if (proc.exitCode !== 0) {
    logger.error(
      { stderr: stderr.slice(0, 2000), sourceModelPath, destinationModelPath },
      "恢复 guided LSTM 模型产物失败",
    );
    throw new Error("恢复 LSTM 模型产物失败");
  }
};

const copyModelArtifact = async (
  modelType: GuidedModelType,
  sourceModelPath: string,
  destinationModelPath: string,
) => {
  if (!fs.existsSync(sourceModelPath)) {
    throw new ModelNotFoundError("分阶段训练模型产物已丢失，请重新训练");
  }

  fs.mkdirSync(path.dirname(destinationModelPath), { recursive: true });

  if (modelType === "lstm") {
    await copyLstmModelArtifact(sourceModelPath, destinationModelPath);
    return;
  }

  fs.copyFileSync(sourceModelPath, destinationModelPath);
};

const restoreGuidedSessionBacktestArtifact = async (row: GuidedTrainingSessionRow) => {
  const sourceModelPath = getGuidedSessionModelPath(row.artifact_dir, row.model_type);
  const backtestModelPath = getBacktestModelPath(row.experiment_id, row.student_id, row.model_type);

  await copyModelArtifact(row.model_type, sourceModelPath, backtestModelPath);
  invalidateProductionArtifact(row.experiment_id, row.student_id, row.model_type);
  return backtestModelPath;
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

const hashGuidedTrainingRequest = (modelType: GuidedModelType, body: TrainInput) =>
  crypto
    .createHash("sha256")
    .update(stableJsonStringify({ modelType, body }))
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

const buildAvailableBaseModelArtifactMap = (
  experimentId: number,
  studentId: number,
  modelsValue: string,
): Partial<Record<BaseModelType, string>> => {
  const requestedModels = parseRequestedBaseModels(modelsValue);
  const availableModelFiles: Partial<Record<BaseModelType, string>> = {};

  for (const modelName of requestedModels) {
    const modelPath = getBacktestModelPath(experimentId, studentId, modelName);
    if (hasSavedModelArtifact(modelName, modelPath)) {
      availableModelFiles[modelName] = modelPath;
    }
  }

  return availableModelFiles;
};

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

export interface ModelProgressEvent {
  type: "phase" | "log" | "progress";
  message: string;
  phase?: string;
  percent?: number;
  source?: "backend" | "python";
  timestamp?: string;
}

export type ModelProgressReporter = (event: ModelProgressEvent) => void;

const reportModelProgress = (
  reporter: ModelProgressReporter | undefined,
  event: ModelProgressEvent,
) => {
  reporter?.({
    ...event,
    timestamp: event.timestamp ?? new Date().toISOString(),
  });
};

const SUPPRESSED_PYTHON_PROGRESS_PREFIXES = [
  "Skipping non-numeric feature for subprocess model:",
];

const PYTHON_PROGRESS_LOG_LINE_PATTERN = /^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+-\s+(DEBUG|INFO|WARNING|ERROR)\s+-\s+(.*)$/;

const UNSAFE_PYTHON_PROGRESS_PATTERNS = [
  /^Traceback \(most recent call last\):/,
  /^File \".*\", line \d+/,
  /^\^+$/,
  /^(?:[A-Za-z_][\w.]*)(?:Error|Exception|Warning):/,
  /\b(?:args|vars\(args\))\s*[:=]/i,
  /\b--data\b/i,
  /\bJSONDecodeError\b/i,
  /\bsite-packages\b/i,
  /\bvenv\b/i,
  /\/(?:opt|root|home|usr|tmp)\//i,
  /[A-Za-z]:\\/,
];

const SAFE_PYTHON_PROGRESS_PATTERNS = [
  /^\[[^\]]+\]/,
  /^✓\s/,
  /^(?:数据划分|权重训练集|权重验证集|评估集|误差计算|权重归一化)[:：]/,
  /^(?:训练|重新训练|复用|生成|计算|评估|预测|扩展horizon后生成|稳定性回退)\b/,
  /^Round \d+ candidate /,
  /^Final RMSE:/,
  /^Model chain:/,
  /^Entering training mode\.$/,
  /^Starting (?:MA|ES|ARIMA|LSTM|model training|prediction|.*train_and_evaluate)/,
  /^Data loaded successfully\./,
  /^Validation passed:/,
  /^Auto-selected /,
  /^Using /,
  /^Running auto_arima /,
  /^Selected /,
  /^Final selected order:/,
  /^Model complexity:/,
  /^Model converged successfully$/,
  /^Model did not converge properly$/,
  /^Forecasting \d+ steps/,
  /^Fitting /,
  /^Generated \d+ /,
  /^Calculated residuals:/,
  /^Evaluation metrics computed:/,
  /^Training finished\./,
  /^Prediction finished successfully$/,
  /^Run finished successfully\.$/,
  /^Training dataset created:/,
  /^Creating autoregressive /,
  /^Training (?:direct|autoregressive) LSTM model /,
  /^Initial learning rate:/,
  /^Model training completed$/,
  /^Fitting preprocessor /,
  /^No feature_types provided/,
  /^Using provided feature_types/,
  /^Resolved feature_types:/,
  /^After d=/,
  /^(?:Very small|Small) effective sample size /,
  /^High parameter-to-sample ratio /,
  /^Empty evaluation range/,
];

const sanitizePythonProgressLine = (line: string): string | null => {
  const trimmed = line.trim();
  if (!trimmed) return null;

  const logMatch = trimmed.match(PYTHON_PROGRESS_LOG_LINE_PATTERN);
  const level = logMatch?.[1];
  const message = logMatch?.[2]?.trim() ?? trimmed;
  if (!message) return null;

  if (level === "DEBUG" || level === "ERROR") {
    return null;
  }

  if (SUPPRESSED_PYTHON_PROGRESS_PREFIXES.some((prefix) => message.startsWith(prefix))) {
    return null;
  }

  if (UNSAFE_PYTHON_PROGRESS_PATTERNS.some((pattern) => pattern.test(message))) {
    return null;
  }

  if (!SAFE_PYTHON_PROGRESS_PATTERNS.some((pattern) => pattern.test(message))) {
    return null;
  }

  return message.length > 320
    ? `${message.slice(0, 320)}...`
    : message;
};

const readStreamText = async (
  stream: ReadableStream<Uint8Array> | null | undefined,
  onLine?: (line: string) => void,
): Promise<string> => {
  if (!stream) return "";

  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let text = "";
  let lineBuffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    text += chunk;

    if (onLine) {
      lineBuffer += chunk;
      const lines = lineBuffer.split(/\r?\n/);
      lineBuffer = lines.pop() ?? "";
      for (const line of lines) onLine(line);
    }
  }

  const finalChunk = decoder.decode();
  if (finalChunk) {
    text += finalChunk;
    if (onLine) lineBuffer += finalChunk;
  }

  if (onLine && lineBuffer.trim()) {
    onLine(lineBuffer);
  }

  return text;
};

export async function runPythonModel(
  modelType: string,
  params: Record<string, unknown>,
  timeoutMs?: number,
  progressReporter?: ModelProgressReporter,
): Promise<unknown> {
  const pythonExec = resolvePythonExecutable();

  const scriptPath = path.join(PY_DIR, "run_model.py");
  if (!fs.existsSync(scriptPath)) {
    throw new Error(`未找到 Python 入口脚本: ${scriptPath}`);
  }

  const args = Object.entries(params).flatMap(([key, value]) => {
    const scriptKey = allPossiblePythonParams[key] || key;
    const cliArg = `--${scriptKey}`;
    if (typeof value === "boolean") return value ? [cliArg] : [];
    if (Array.isArray(value)) return [cliArg, JSON.stringify(value)];
    return [cliArg, String(value)];
  });

  const fullArgs = [modelType, ...args];
  const command = process.platform !== "win32" && SETSID_PATH
    ? [SETSID_PATH, pythonExec, scriptPath, ...fullArgs]
    : [pythonExec, scriptPath, ...fullArgs];

  reportModelProgress(progressReporter, {
    type: "phase",
    phase: "python_start",
    percent: 24,
    source: "backend",
    message: `已启动 ${modelType.toUpperCase()} Python 训练进程`,
  });

  const proc = Bun.spawn(command, {
    cwd: PY_DIR,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      ...buildPythonModelProcessEnv(),
      PATH: `${path.dirname(pythonExec)}${path.delimiter}${process.env.PATH ?? ""}`,
    },
  });

  const killPythonProcessTree = () => {
    if (proc.exitCode !== null || proc.killed) return;

    if (process.platform !== "win32" && SETSID_PATH) {
      try {
        process.kill(-proc.pid, "SIGKILL");
        return;
      } catch (error) {
        logger.warn(
          { error, pid: proc.pid, modelType },
          "终止 Python 进程组失败，回退为终止主进程",
        );
      }
    }

    try {
      proc.kill("SIGKILL");
    } catch (error) {
      logger.warn({ err: error, pid: proc.pid, modelType }, "终止 Python 主进程失败");
    }
  };

  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  const executionPromise = new Promise<unknown>(async (resolve, reject) => {
    try {
      let pythonLogProgress = 26;
      const [stdout, stderr] = await Promise.all([
        readStreamText(proc.stdout),
        readStreamText(proc.stderr, (line) => {
          const message = sanitizePythonProgressLine(line);
          if (!message) return;

          pythonLogProgress = Math.min(86, pythonLogProgress + 2);
          reportModelProgress(progressReporter, {
            type: "log",
            phase: "python_training",
            percent: pythonLogProgress,
            source: "python",
            message,
          });
        }),
      ]);

      await proc.exited;

      reportModelProgress(progressReporter, {
        type: "phase",
        phase: "python_finished",
        percent: 88,
        source: "backend",
        message: `${modelType.toUpperCase()} Python 训练进程已结束，正在解析结果`,
      });

      if (proc.exitCode !== 0) {
        // 截断日志，避免泄露完整数据
        const truncatedArgs = fullArgs.map((arg) =>
          typeof arg === "string" && arg.length > 200 ? `${arg.slice(0, 200)}...[truncated]` : arg,
        );
        logger.error({ stderr: stderr.slice(0, 2000), modelType, args: truncatedArgs }, "Python 执行失败");
        const taggedError = extractPythonTaggedError(stderr);
        if (taggedError) {
          reject(taggedError);
        } else {
          reject(new Error("Python 执行失败"));
        }
      } else {
        try {
          resolve(JSON.parse(stdout));
        } catch {
          const truncatedArgs = fullArgs.map((arg) =>
            typeof arg === "string" && arg.length > 200 ? `${arg.slice(0, 200)}...[truncated]` : arg,
          );
          logger.error({ stdout: stdout.slice(0, 2000), modelType, args: truncatedArgs }, "解析 Python 输出失败");
          reject(new Error("Python 输出无效"));
        }
      }
    } catch (err) {
      reject(err);
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
    }
  });

  if (!timeoutMs) return executionPromise;

  const timeoutPromise = new Promise<unknown>((_, reject) => {
    timeoutId = setTimeout(() => {
      killPythonProcessTree();
      reject(new Error(`Python 执行超时 (${timeoutMs}ms)`));
    }, timeoutMs);
  });

  // 超时后 executionPromise 仍会 reject（进程被杀后 exitCode !== 0），
  // 若不捕获会触发 unhandledRejection。用 .catch 吞掉滞后的 rejection。
  executionPromise.catch((err) => {
    logger.debug({ err, modelType }, "超时后 executionPromise 滞后 rejection（已忽略）");
  });

  return Promise.race([executionPromise, timeoutPromise]);
}

interface GuidedPythonStepResult {
  step_id: string;
  output?: unknown;
  final_result?: Record<string, unknown>;
}

async function runPythonGuidedTrainingStep(
  modelType: GuidedModelType,
  stepId: string,
  params: Record<string, unknown>,
  stateFilePath: string,
  artifactDir: string,
  timeoutMs = TRAINING_TIMEOUT,
): Promise<GuidedPythonStepResult> {
  const pythonExec = resolvePythonExecutable();
  const scriptPath = path.join(PY_DIR, "guided_model.py");
  if (!fs.existsSync(scriptPath)) {
    throw new Error(`未找到 Python guided 训练入口脚本: ${scriptPath}`);
  }

  fs.mkdirSync(artifactDir, { recursive: true });
  const args = Object.entries(params).flatMap(([key, value]) => {
    const scriptKey = allPossiblePythonParams[key] || key;
    const cliArg = `--${scriptKey}`;
    if (typeof value === "boolean") return value ? [cliArg] : [];
    if (Array.isArray(value)) return [cliArg, JSON.stringify(value)];
    return [cliArg, String(value)];
  });

  const fullArgs = [
    modelType,
    stepId,
    "--state-file",
    stateFilePath,
    "--artifact-dir",
    artifactDir,
    ...args,
  ];
  const command = process.platform !== "win32" && SETSID_PATH
    ? [SETSID_PATH, pythonExec, scriptPath, ...fullArgs]
    : [pythonExec, scriptPath, ...fullArgs];

  const proc = Bun.spawn(command, {
    cwd: PY_DIR,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      ...buildPythonModelProcessEnv(),
      PATH: `${path.dirname(pythonExec)}${path.delimiter}${process.env.PATH ?? ""}`,
    },
  });

  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  const killPythonProcessTree = () => {
    if (proc.exitCode !== null || proc.killed) return;

    if (process.platform !== "win32" && SETSID_PATH) {
      try {
        process.kill(-proc.pid, "SIGKILL");
        return;
      } catch (error) {
        logger.warn({ error, pid: proc.pid, modelType, stepId }, "终止 guided Python 进程组失败");
      }
    }

    try {
      proc.kill("SIGKILL");
    } catch (error) {
      logger.warn({ err: error, pid: proc.pid, modelType, stepId }, "终止 guided Python 主进程失败");
    }
  };

  const executionPromise = new Promise<GuidedPythonStepResult>(async (resolve, reject) => {
    try {
      const [stdout, stderr] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
      ]);
      await proc.exited;

      if (proc.exitCode !== 0) {
        logger.error(
          { stderr: stderr.slice(0, 2000), modelType, stepId },
          "guided Python 阶段执行失败",
        );
        const taggedError = extractPythonTaggedError(stderr);
        reject(taggedError ?? new Error("Python 执行失败"));
        return;
      }

      try {
        resolve(JSON.parse(stdout) as GuidedPythonStepResult);
      } catch {
        logger.error(
          { stdout: stdout.slice(0, 2000), modelType, stepId },
          "解析 guided Python 输出失败",
        );
        reject(new Error("Python 输出无效"));
      }
    } catch (err) {
      reject(err);
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
    }
  });

  const timeoutPromise = new Promise<GuidedPythonStepResult>((_, reject) => {
    timeoutId = setTimeout(() => {
      killPythonProcessTree();
      reject(new Error(`Python 执行超时 (${timeoutMs}ms)`));
    }, timeoutMs);
  });

  executionPromise.catch((err) => {
    logger.debug({ err, modelType, stepId }, "guided Python 超时后滞后 rejection（已忽略）");
  });

  return Promise.race([executionPromise, timeoutPromise]);
}

const resultTransformers: Record<ModelType, (result: Record<string, unknown>) => Record<string, unknown>> = {
  arima: (result) => ({
    best_order: result.best_order,
    search_summary: result.search_summary,
    diagnostics: result.diagnostics
      ? { ...(result.diagnostics as object), y_pred_eval: (result.predictions as { prediction: number }[] | undefined)?.map((p) => p.prediction) || [] }
      : undefined,
  }),
  es: () => ({}),
  ma: () => ({}),
  lstm: (result) => ({
    feature_types: result.feature_types,
    diagnostics: result.diagnostics
      ? { ...(result.diagnostics as object), y_pred_eval: (result.predictions as { prediction: number }[] | undefined)?.map((p) => p.prediction) || [] }
      : undefined,
  }),
  stacking: (result) => ({ model_names: result.models, meta_model: result.meta_model }),
  weighted_avg: (result) => ({
    model_names: result.models,
    weights: (result.models as string[] | undefined)?.map((name) => (result.weights as Record<string, number>)?.[name]),
  }),
  boosting: (result) => ({
    model_names: result.model_chain,
    search_summary: result.search_summary,
  }),
};

export function transformTrainingResult(
  modelType: ModelType,
  result: Record<string, unknown>,
): { status: string; results: Record<string, unknown>; inferred_feature_types?: unknown } {
  const metrics =
    result.metrics && typeof result.metrics === "object"
      ? (result.metrics as TrainingMetricsPayload)
      : undefined;

  const baseResults = {
    metrics: result.metrics,
    eval_y_true: result.eval_y_true,
    eval_predictions: (result.predictions as { prediction: number }[] | undefined)?.map((p) => p.prediction) || [],
    evaluate_indices: normalizeNumericArray(metrics?.evaluate_indices),
    evaluate_months: normalizeStringArray(metrics?.evaluate_months),
    method_name: result.method_name,
    forecast_strategy: result.forecast_strategy,
    implementation_notes: result.implementation_notes,
    notes: result.notes || [],
    saved_model: result.saved_model,
    logs: result.logs,
  };

  const modelSpecificTransformedResult = resultTransformers[modelType]?.(result) || {};
  const finalTransformedResult = { ...baseResults, ...modelSpecificTransformedResult };

  const finalResponse: { status: string; results: Record<string, unknown>; inferred_feature_types?: unknown } = {
    status: "success",
    results: finalTransformedResult,
  };

  if (modelType === "lstm" && result.feature_types) {
    finalResponse.inferred_feature_types = result.feature_types;
  }

  return finalResponse;
}

const getGuidedTrainingSteps = (modelType: GuidedModelType) => GUIDED_TRAINING_STEPS[modelType];

const getGuidedNextStepId = (modelType: GuidedModelType, completedStepIds: string[]) => {
  const completedSet = new Set(completedStepIds);
  return getGuidedTrainingSteps(modelType).find((step) => !completedSet.has(step.id))?.id ?? null;
};

const buildGuidedTrainingSessionPayload = (
  row: GuidedTrainingSessionRow,
): GuidedTrainingSessionPayload => {
  const completedStepIds = parseJsonColumn<string[]>(row.completed_step_ids, []);
  const stepOutputs = parseJsonColumn<Record<string, unknown>>(row.step_outputs, {});
  const result = parseJsonColumn<GuidedTrainingSessionPayload["result"]>(row.result_payload, null);
  const nextStepId = getGuidedNextStepId(row.model_type, completedStepIds);
  const activeStepId = row.current_step_id ?? nextStepId;
  const completedSet = new Set(completedStepIds);

  return {
    session_id: row.session_id,
    experiment_id: row.experiment_id,
    model_type: row.model_type,
    status: row.status,
    current_step_id: activeStepId,
    next_step_id: row.status === "completed" ? null : nextStepId,
    steps: getGuidedTrainingSteps(row.model_type).map((step) => ({
      ...step,
      status: completedSet.has(step.id)
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
            result_payload,
            artifact_dir,
            error_message
       FROM model_guided_training_sessions
      WHERE session_id = ?
      LIMIT 1`,
    [sessionId],
  );

  return rows[0] ?? null;
};

const buildGuidedTrainingPythonParams = async (
  modelType: GuidedModelType,
  studentId: number,
  body: TrainInput,
): Promise<Record<string, unknown>> => {
  const experiment = await assertExperimentModelAccess(body.experiment_id, studentId);
  assertTrainingRequestMatchesExperiment(experiment, body);
  const trainingConfig = getExperimentTrainingConfig(experiment);
  if (modelType === "ma") {
    validateMovingAverageWindow(trainingConfig, body.moving_average_window);
  }

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

  const modelSaveLoc = getBacktestModelPath(body.experiment_id, studentId, modelType);
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
  }

  const requestHash = hashGuidedTrainingRequest(modelType, body);
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
            result_payload,
            artifact_dir,
            error_message
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
    if (existingSession.status === "running") {
      await pool.query(
        `UPDATE model_guided_training_sessions
            SET status = 'ready'
          WHERE session_id = ?`,
        [existingSession.session_id],
      );
      existingSession.status = "ready";
    }
    if (existingSession.status === "completed") {
      await restoreGuidedSessionBacktestArtifact(existingSession);
    }
    return buildGuidedTrainingSessionPayload(existingSession);
  }

  const sessionId = crypto.randomUUID();
  const artifactDir = buildGuidedTrainingArtifactDir(body.experiment_id, studentId, modelType, sessionId);
  fs.mkdirSync(artifactDir, { recursive: true });

  const completedStepIds: string[] = [];
  const currentStepId = getGuidedNextStepId(modelType, completedStepIds);
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
        result_payload,
        artifact_dir,
        expires_at)
     VALUES (?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, NULL, ?, DATE_ADD(NOW(), INTERVAL 7 DAY))`,
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
      artifactDir,
    ],
  );

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
  if (row.status === "completed") {
    await restoreGuidedSessionBacktestArtifact(row);
  }
  return buildGuidedTrainingSessionPayload(row);
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

  const row = await fetchGuidedTrainingSessionRow(sessionId);
  if (!row || row.model_type !== modelType) {
    throw new ModelNotFoundError("未找到分阶段训练会话");
  }

  if (row.student_id !== studentId) {
    throw new ModelAccessError("权限不足：该训练会话不属于当前用户", 403);
  }

  if (row.status === "completed") {
    await restoreGuidedSessionBacktestArtifact(row);
    return buildGuidedTrainingSessionPayload(row);
  }

  const body = parseJsonColumn<TrainInput>(row.request_payload, {} as TrainInput);
  const completedStepIds = parseJsonColumn<string[]>(row.completed_step_ids, []);
  if (completedStepIds.includes(stepId)) {
    return buildGuidedTrainingSessionPayload(row);
  }

  const nextStepId = getGuidedNextStepId(modelType, completedStepIds);
  if (stepId !== nextStepId) {
    throw new ModelValidationError("只能按训练流程顺序执行下一个阶段");
  }

  let executionLock = null;
  let slotLock = null;
  let sessionMarkedRunning = false;

  try {
    executionLock = await acquireModelExecutionLock(row.experiment_id, modelType);
    if (!executionLock) {
      throw new ModelValidationError("同一模型正在训练或预测，请稍后再试");
    }

    slotLock = await acquireModelSlot();
    if (!slotLock) {
      throw new ModelValidationError("模型服务繁忙，请稍后再试");
    }

    await pool.query(
      `UPDATE model_guided_training_sessions
          SET status = 'running',
              current_step_id = ?,
              error_message = NULL
        WHERE session_id = ?`,
      [stepId, sessionId],
    );
    sessionMarkedRunning = true;

    const sessionModelPath = getGuidedSessionModelPath(row.artifact_dir, modelType);
    const pythonParams = await buildGuidedTrainingPythonParams(modelType, studentId, body);
    pythonParams.saveModel = sessionModelPath;
    const stateFilePath = buildGuidedTrainingStatePath(row.experiment_id, studentId, modelType, sessionId);
    const pythonStepResult = await runPythonGuidedTrainingStep(
      modelType,
      stepId,
      pythonParams,
      stateFilePath,
      row.artifact_dir,
    );

    const nextCompletedStepIds = [...completedStepIds, stepId];
    const stepOutputs = parseJsonColumn<Record<string, unknown>>(row.step_outputs, {});
    const nextStepOutputs = {
      ...stepOutputs,
      [stepId]: pythonStepResult.output ?? {},
    };

    let transformedResult: ReturnType<typeof transformTrainingResult> | null = null;
    if (pythonStepResult.final_result) {
      const backtestModelPath = await restoreGuidedSessionBacktestArtifact(row);
      pythonStepResult.final_result.saved_model = backtestModelPath;
      transformedResult = transformTrainingResult(modelType, pythonStepResult.final_result);
    }

    const status: GuidedTrainingStatus = transformedResult ? "completed" : "ready";
    const currentStepId = status === "completed"
      ? null
      : getGuidedNextStepId(modelType, nextCompletedStepIds);

    await pool.query(
      `UPDATE model_guided_training_sessions
          SET status = ?,
              current_step_id = ?,
              completed_step_ids = ?,
              step_outputs = ?,
              result_payload = ?,
              error_message = NULL
        WHERE session_id = ?`,
      [
        status,
        currentStepId,
        JSON.stringify(nextCompletedStepIds),
        JSON.stringify(nextStepOutputs),
        transformedResult ? JSON.stringify(transformedResult) : null,
        sessionId,
      ],
    );

    const updatedRow = await fetchGuidedTrainingSessionRow(sessionId);
    if (!updatedRow) {
      throw new Error("分阶段训练会话已丢失");
    }

    return buildGuidedTrainingSessionPayload(updatedRow);
  } catch (error) {
    if (sessionMarkedRunning) {
      const detail = error instanceof Error ? error.message : "分阶段训练执行失败";
      await pool.query(
        `UPDATE model_guided_training_sessions
            SET status = 'failed',
                current_step_id = ?,
                error_message = ?
          WHERE session_id = ?`,
        [stepId, detail, sessionId],
      );
    }
    throw error;
  } finally {
    await releaseModelSlot(slotLock);
    await releaseModelExecutionLock(executionLock);
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
  }
  const derivedModelParams: Record<string, unknown> = isEnsembleModelType(modelType)
    ? buildEnsemblePythonParams(experiment, body)
    : {};
  if (isEnsembleModelType(modelType) && body.models) {
    const baseModelArtifactMap = buildAvailableBaseModelArtifactMap(body.experiment_id, studentId, body.models);
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

  const modelSaveLoc = getBacktestModelPath(body.experiment_id, studentId, modelType);

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
  invalidateProductionArtifact(body.experiment_id, studentId, modelType);

  reportModelProgress(progressReporter, {
    type: "phase",
    phase: "transform_result",
    percent: 96,
    source: "backend",
    message: "正在整理评估指标、预测值和模型元数据",
  });
  return transformTrainingResult(modelType, result);
}

export async function getReusableProductionModelArtifactResult(
  modelType: ModelType,
  studentId: number,
  experimentId: number,
) {
  const experiment = await assertExperimentModelAccess(experimentId, studentId);
  assertModelReadyForProduction(experiment, modelType);

  const productionModelPath = getProductionModelPath(experimentId, studentId, modelType);
  if (!await hasReusableSavedModelArtifact(modelType, productionModelPath)) {
    return null;
  }

  return {
    status: "success",
    results: {
      saved_model: productionModelPath,
      artifact_phase: "production",
      reused_existing: true,
    },
  };
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
  const trainingConfig = getExperimentTrainingConfig(experiment);

  const productionModelPath = getProductionModelPath(experimentId, studentId, modelType);
  const backtestModelPath = getBacktestModelPath(experimentId, studentId, modelType);
  if (!hasSavedModelArtifact(modelType, backtestModelPath)) {
    throw new ModelNotFoundError(`未找到该实验的已训练 ${modelType.toUpperCase()} 回测模型`);
  }

  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
  );

  if (trainingConfig.data_window_evaluate_end_index >= allData.length) {
    throw new ModelValidationError("数据窗口索引超出范围", { dataset_size: allData.length });
  }

  invalidateProductionArtifact(experimentId, studentId, modelType);

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

  return {
    status: "success",
    results: {
      saved_model: productionModelPath,
      artifact_phase: "production",
      reused_existing: false,
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

export async function predictExperimentModel(
  modelType: ModelType,
  studentId: number,
  experimentId: number,
  forecastSteps: number,
) {
  const experiment = await assertExperimentModelAccess(experimentId, studentId);
  assertModelReadyForProduction(experiment, modelType);

  const productionModelPath = getProductionModelPath(experimentId, studentId, modelType);
  if (!await hasReusableSavedModelArtifact(modelType, productionModelPath)) {
    await prepareExperimentModelForProduction(modelType, studentId, experimentId);
  }

  if (!await hasReusableSavedModelArtifact(modelType, productionModelPath)) {
    throw new ModelNotFoundError(`未找到该实验的 ${modelType.toUpperCase()} 生产预测模型`);
  }

  const pythonParams = {
    loadModel: productionModelPath,
    forecastSteps,
    pretty: true,
  };

  const result = (await runPythonModel(modelType, pythonParams, TIMEOUTS.MODEL_PREDICTION)) as Record<string, unknown>;

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
