import path from "path";
import fs from "fs";
import crypto from "crypto";
import os from "node:os";
import type { Stats } from "node:fs";
import type { RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import logger from "../../utils/logger";
import type { ModelType, TrainInput } from "./models.schema";
import { TIMEOUTS, CLEANUP_RETENTION } from "../../utils/constants";
import { PY_DIR, resolvePythonExecutable } from "../../utils/python";
import {
  GUIDED_TRAINING_ROOT,
  buildGuidedTrainingArtifactDir,
  buildGuidedTrainingStatePath,
  buildExperimentScopedModelPath,
  buildVersionedModelPath,
} from "./model-runtime";
import { loadDatasetFromDb } from "./model-datasets";
import { ModelAccessError, ModelNotFoundError, ModelValidationError } from "./model-errors";
import {
  acquireModelExecutionLock,
  acquireModelSlot,
  releaseModelExecutionLock,
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
  buildExperimentTrainingRevision,
  buildModelTrainingRequestHash,
  commitModelTrainingRun,
  getActiveModelTrainingRun,
  setProductionArtifactForRun,
  type ModelTrainingRunRow,
} from "./model-run-store";
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
export { ModelAccessError, ModelNotFoundError, ModelValidationError } from "./model-errors";
export {
  acquireModelExecutionLock,
  acquireModelSlot,
  releaseModelExecutionLock,
  releaseModelSlot,
} from "./model-locks";
export { runPythonGuidedTrainingStep, runPythonModel, type ModelProgressEvent, type ModelProgressReporter } from "./model-python-runner";
export { transformTrainingResult } from "./model-results";
export type { GuidedTrainingSessionPayload, GuidedTrainingStepDefinition } from "./model-types";

const TRAINING_TIMEOUT = TIMEOUTS.MODEL_TRAINING;
const GUIDED_TRAINING_CLEANUP_BATCH_SIZE = 500;
const GUIDED_DATASET_FILE_NAME = "dataset.json";
const MODEL_PIPELINE_REVISION = "model-run-v3";
let cachedModelRuntimeRevision: string | null = null;

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

interface GuidedTrainingSessionRow extends RowDataPacket {
  session_id: string;
  experiment_id: number;
  student_id: number;
  model_type: GuidedModelType;
  request_hash: string;
  dataset_revision: string;
  runtime_revision: string;
  request_payload: unknown;
  status: GuidedTrainingStatus;
  current_step_id: string | null;
  completed_step_ids: unknown;
  step_outputs: unknown;
  step_plan: unknown;
  result_payload: unknown;
  artifact_dir: string;
  error_message: string | null;
  experiment_state_version: number | null;
  updated_at: string | Date;
}

interface GuidedTrainingCleanupRow extends RowDataPacket {
  session_id: string;
  artifact_dir: string;
}

interface StaleModelExperimentEligibilityRow extends RowDataPacket {
  can_delete: 0 | 1 | boolean | string;
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
with open(source_model_path, "rb") as handle:
    state = pickle.load(handle)

source_keras_path = state.get("keras_model_path")
if not source_keras_path:
    source_keras_path = os.path.splitext(source_model_path)[0] + "_keras_model.h5"

if not os.path.exists(source_keras_path):
    raise FileNotFoundError(f"source keras model not found: {source_keras_path}")

destination_keras_path = os.path.splitext(destination_model_path)[0] + "_keras_model.h5"
state["keras_model_path"] = destination_keras_path

keras_fd, temp_keras_name = tempfile.mkstemp(
    prefix=f".{os.path.basename(destination_keras_path)}.",
    suffix=".tmp",
    dir=os.path.dirname(destination_model_path),
)
os.close(keras_fd)
temp_model_name = None
try:
    shutil.copy2(source_keras_path, temp_keras_name)
    model_fd, temp_model_name = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination_model_path)}.",
        suffix=".tmp",
        dir=os.path.dirname(destination_model_path),
    )
    with os.fdopen(model_fd, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_keras_name, destination_keras_path)
    os.replace(temp_model_name, destination_model_path)
finally:
    if os.path.exists(temp_keras_name):
        try:
            os.remove(temp_keras_name)
        except OSError:
            pass
    if temp_model_name and os.path.exists(temp_model_name):
        try:
            os.remove(temp_model_name)
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

const ENSEMBLE_ARTIFACT_COPY_SCRIPT = `
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from ensemble_utils import get_member_model_path, persist_member_model_artifact
from serialization import dump_model, load_model

source_model_path = sys.argv[1]
destination_model_path = sys.argv[2]

if not os.path.exists(source_model_path):
    raise FileNotFoundError(f"source model not found: {source_model_path}")

state = load_model(source_model_path)
if not isinstance(state, dict):
    raise ValueError("invalid ensemble state")

base_model_states = state.get("base_model_states")
if not isinstance(base_model_states, dict):
    raise ValueError("ensemble state is missing base_model_states")

os.makedirs(os.path.dirname(destination_model_path), exist_ok=True)

def persist_state(model_name, model_state, round_idx=None):
    if not isinstance(model_state, dict):
        return
    source_member_path = model_state.get("model_file")
    if not isinstance(source_member_path, str) or not source_member_path:
        raise ValueError(f"missing member artifact for {model_name}")
    destination_member_path = get_member_model_path(destination_model_path, model_name, round_idx)
    persist_member_model_artifact(
        model_name,
        source_member_path,
        destination_member_path,
        move_source=False,
    )
    model_state["model_file"] = destination_member_path
    model_state.pop("artifact_origin", None)

for outer_name, model_info in base_model_states.items():
    if isinstance(model_info, dict) and isinstance(model_info.get("models"), list):
        for index, model_state in enumerate(model_info["models"]):
            if not isinstance(model_state, dict):
                continue
            model_name = str(model_state.get("model_type") or outer_name)
            round_value = model_state.get("round", index)
            try:
                round_idx = int(round_value)
            except (TypeError, ValueError):
                round_idx = index
            persist_state(model_name, model_state, round_idx)
    elif isinstance(model_info, dict):
        model_name = str(model_info.get("model_type") or outer_name)
        persist_state(model_name, model_info)

dump_model(state, destination_model_path)
`;

const copyEnsembleModelArtifact = async (sourceModelPath: string, destinationModelPath: string) => {
  const pythonExec = resolvePythonExecutable();
  const proc = Bun.spawn([
    pythonExec,
    "-c",
    ENSEMBLE_ARTIFACT_COPY_SCRIPT,
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
      "恢复 guided 融合模型产物失败",
    );
    throw new Error("恢复融合模型产物失败");
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

  if (isEnsembleModelType(modelType)) {
    removeSiblingArtifactsByPrefix(destinationModelPath);
    await copyEnsembleModelArtifact(sourceModelPath, destinationModelPath);
    return;
  }

  const temporaryModelPath = `${destinationModelPath}.tmp-${crypto.randomUUID()}`;
  try {
    fs.copyFileSync(sourceModelPath, temporaryModelPath);
    fs.renameSync(temporaryModelPath, destinationModelPath);
  } finally {
    removeFileIfExists(temporaryModelPath);
  }
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
  row: Pick<GuidedTrainingSessionRow, "session_id" | "artifact_dir">,
) => {
  let canDeleteSessionRow = false;
  try {
    removeGuidedTrainingArtifactDir(row.artifact_dir);
    canDeleteSessionRow = true;
  } catch (error) {
    logger.warn(
      {
        err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
        sessionId: row.session_id,
        artifactDir: row.artifact_dir,
      },
      "删除 guided training 工件目录失败，将保留会话记录以便后续重试",
    );
  }

  if (!canDeleteSessionRow) {
    return false;
  }

  await pool.query(
    `DELETE FROM model_guided_training_sessions
      WHERE session_id = ?`,
    [row.session_id],
  );
  return true;
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
  const expirationCutoffParams = options.now ? [options.now] : [];
  const [rows] = await pool.query<GuidedTrainingCleanupRow[]>(
    `SELECT session_id,
            artifact_dir
       FROM model_guided_training_sessions
      WHERE expires_at IS NOT NULL
        AND expires_at <= ${expirationCutoffExpression}
      ORDER BY expires_at ASC
      LIMIT ${limit}`,
    expirationCutoffParams,
  );

  let deletedArtifactDirs = 0;
  const sessionIdsToDelete: string[] = [];
  for (const row of rows) {
    try {
      if (removeGuidedTrainingArtifactDir(row.artifact_dir)) {
        deletedArtifactDirs += 1;
      }
      sessionIdsToDelete.push(row.session_id);
    } catch (error) {
      logger.warn(
        {
          err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
          sessionId: row.session_id,
          artifactDir: row.artifact_dir,
        },
        "清理过期 guided training 工件目录失败",
      );
    }
  }

  if (sessionIdsToDelete.length > 0) {
    const placeholders = sessionIdsToDelete.map(() => "?").join(", ");
    await pool.query(
      `DELETE FROM model_guided_training_sessions
        WHERE session_id IN (${placeholders})`,
      sessionIdsToDelete,
    );
  }

  deletedArtifactDirs += await cleanupOrphanGuidedTrainingArtifactDirs(
    now,
    options.orphanMaxAgeMs ?? CLEANUP_RETENTION.GUIDED_TRAINING_ORPHAN_MS,
  );

  return {
    deletedSessions: sessionIdsToDelete.length,
    deletedArtifactDirs,
  };
}

const collectPythonRuntimeFiles = (directoryPath: string): string[] => {
  if (!fs.existsSync(directoryPath)) return [];
  const files: string[] = [];
  for (const entry of fs.readdirSync(directoryPath, { withFileTypes: true })) {
    const entryPath = path.join(directoryPath, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "__pycache__") continue;
      files.push(...collectPythonRuntimeFiles(entryPath));
    } else if (entry.isFile() && entry.name.endsWith(".py")) {
      files.push(entryPath);
    }
  }
  return files;
};

const getModelRuntimeRevision = () => {
  if (cachedModelRuntimeRevision) return cachedModelRuntimeRevision;
  const runtimeFiles = [
    path.join(PY_DIR, "guided_model.py"),
    path.join(PY_DIR, "guided_ensemble_model.py"),
    path.join(PY_DIR, "run_model.py"),
    ...collectPythonRuntimeFiles(path.join(PY_DIR, "src")),
  ].filter((filePath) => fs.existsSync(filePath)).sort();
  const hash = crypto.createHash("sha256").update(MODEL_PIPELINE_REVISION);
  for (const filePath of runtimeFiles) {
    try {
      hash.update(path.relative(PY_DIR, filePath));
      hash.update(fs.readFileSync(filePath));
    } catch {
      // A deployment may atomically replace files while the revision is first
      // calculated. Missing files are excluded and the cache is rebuilt on the
      // next process start.
    }
  }
  cachedModelRuntimeRevision = hash.digest("hex");
  return cachedModelRuntimeRevision;
};

const hashDatasetSnapshot = (datasetJson: string) =>
  crypto.createHash("sha256").update(datasetJson).digest("hex");

const writeGuidedDatasetSnapshot = async (artifactDir: string, datasetJson: string) => {
  await fs.promises.mkdir(artifactDir, { recursive: true });
  const targetPath = path.join(artifactDir, GUIDED_DATASET_FILE_NAME);
  const temporaryPath = `${targetPath}.tmp-${crypto.randomUUID()}`;
  await Bun.write(temporaryPath, datasetJson);
  fs.renameSync(temporaryPath, targetPath);
  return targetPath;
};

const buildBaseRunFingerprint = async (
  experimentId: number,
  studentId: number,
  body: TrainInput,
) => {
  const fingerprint: Record<string, string> = {};
  if (!body.models) return fingerprint;
  for (const baseModel of parseRequestedBaseModels(body.models)) {
    const activeRun = await getActiveModelTrainingRun(experimentId, studentId, baseModel);
    fingerprint[baseModel] = activeRun?.run_id ?? "legacy";
  }
  return fingerprint;
};

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

const buildAvailableBaseModelArtifactMap = async (
  experimentId: number,
  studentId: number,
  modelsValue: string,
): Promise<Partial<Record<BaseModelType, string>>> => {
  const requestedModels = parseRequestedBaseModels(modelsValue);
  const availableModelFiles: Partial<Record<BaseModelType, string>> = {};

  for (const modelName of requestedModels) {
    const activeRun = await getActiveModelTrainingRun(experimentId, studentId, modelName);
    const modelPath = activeRun?.artifact_path
      ?? getBacktestModelPath(experimentId, studentId, modelName);
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
    experiment_state_version: row.experiment_state_version === null
      ? null
      : Number(row.experiment_state_version),
  };
};

const buildCanonicalTrainingBody = (
  body: TrainInput,
  trainingConfig: ExperimentTrainingConfig,
): TrainInput => ({
  ...body,
  selected_industry: trainingConfig.selected_industry,
  selected_company: trainingConfig.selected_company,
  selected_product: trainingConfig.selected_product,
  data_window_train_start_index: trainingConfig.data_window_train_start_index,
  data_window_train_end_index: trainingConfig.data_window_train_end_index,
  data_window_evaluate_start_index: trainingConfig.data_window_evaluate_start_index,
  data_window_evaluate_end_index: trainingConfig.data_window_evaluate_end_index,
});

const fetchGuidedTrainingSessionRow = async (
  sessionId: string,
): Promise<GuidedTrainingSessionRow | null> => {
  const [rows] = await pool.query<GuidedTrainingSessionRow[]>(
    `SELECT session_id,
            experiment_id,
            student_id,
            model_type,
            request_hash,
            dataset_revision,
            runtime_revision,
            request_payload,
            status,
            current_step_id,
            completed_step_ids,
            step_outputs,
            step_plan,
            result_payload,
            artifact_dir,
            error_message,
            experiment_state_version,
            updated_at
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
  artifactDir: string,
): Promise<Record<string, unknown>> => {
  const experiment = await assertExperimentModelAccess(body.experiment_id, studentId);
  assertTrainingRequestMatchesExperiment(experiment, body);
  const trainingConfig = getExperimentTrainingConfig(experiment);
  const canonicalBody = buildCanonicalTrainingBody(body, trainingConfig);
  if (modelType === "ma") {
    validateMovingAverageWindow(trainingConfig, canonicalBody.moving_average_window);
  }
  const derivedModelParams: Record<string, unknown> = isEnsembleModelType(modelType)
    ? buildEnsemblePythonParams(experiment, body)
    : {};
  const datasetFile = path.join(artifactDir, GUIDED_DATASET_FILE_NAME);
  if (!fs.existsSync(datasetFile)) {
    throw new ModelNotFoundError("训练数据快照已丢失，请重新创建训练会话");
  }
  const modelSaveLoc = getGuidedSessionModelPath(artifactDir, modelType);
  const pythonParams: Record<string, unknown> = {
    dataFile: datasetFile,
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
    const baseModelArtifactMap = await buildAvailableBaseModelArtifactMap(body.experiment_id, studentId, body.models);
    if (Object.keys(baseModelArtifactMap).length > 0) {
      pythonParams.baseModelFiles = JSON.stringify(baseModelArtifactMap);
    }
  }

  return pythonParams;
};

const materializeCompletedGuidedTrainingRun = async (row: GuidedTrainingSessionRow) => {
  const body = parseJsonColumn<TrainInput>(row.request_payload, {} as TrainInput);
  const result = parseJsonColumn<GuidedTrainingSessionPayload["result"]>(row.result_payload, null);
  if (!result) {
    throw new ModelValidationError("已完成训练会话缺少结果，无法恢复");
  }

  const sourceModelPath = getGuidedSessionModelPath(row.artifact_dir, row.model_type);
  if (!await hasReusableSavedModelArtifact(row.model_type, sourceModelPath)) {
    throw new ModelNotFoundError("分阶段训练产物已过期或已被清理，请重新开始分阶段训练");
  }
  const versionedArtifactPath = buildVersionedModelPath(
    row.experiment_id,
    row.student_id,
    row.model_type,
    row.session_id,
    "backtest",
  );
  if (!await hasReusableSavedModelArtifact(row.model_type, versionedArtifactPath)) {
    await copyModelArtifact(row.model_type, sourceModelPath, versionedArtifactPath);
  }
  result.results.saved_model = versionedArtifactPath;

  const completedStepIds = parseJsonColumn<string[]>(row.completed_step_ids, []);
  const stepOutputs = parseJsonColumn<Record<string, unknown>>(row.step_outputs, {});
  const stepPlan = getGuidedTrainingSteps(row);
  const committed = await commitModelTrainingRun({
    runId: row.session_id,
    sessionId: row.session_id,
    experimentId: row.experiment_id,
    studentId: row.student_id,
    modelType: row.model_type,
    requestHash: row.request_hash,
    datasetRevision: row.dataset_revision,
    runtimeRevision: row.runtime_revision,
    body,
    result,
    artifactPath: versionedArtifactPath,
    guidedSession: {
      sessionId: row.session_id,
      currentStepId: null,
      completedStepIds,
      stepOutputs,
      stepPlan,
      result,
    },
  });
  row.result_payload = JSON.stringify(result);
  row.experiment_state_version = committed.stateVersion;
  return row;
};

const isDuplicateGuidedTrainingRequestError = (error: unknown) => {
  if (!error || typeof error !== "object") return false;
  const candidate = error as { code?: string; errno?: number };
  return candidate.code === "ER_DUP_ENTRY" || candidate.errno === 1062;
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
  const canonicalBody = buildCanonicalTrainingBody(body, trainingConfig);
  if (modelType === "ma") {
    validateMovingAverageWindow(trainingConfig, canonicalBody.moving_average_window);
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
  const datasetJson = JSON.stringify(allData);
  const datasetRevision = hashDatasetSnapshot(datasetJson);
  const runtimeRevision = getModelRuntimeRevision();
  const baseRunFingerprint = isEnsembleModelType(modelType)
    ? await buildBaseRunFingerprint(canonicalBody.experiment_id, studentId, canonicalBody)
    : {};
  const experimentRevision = buildExperimentTrainingRevision(modelType, experiment, canonicalBody);
  const requestHash = buildModelTrainingRequestHash({
    modelType,
    body: canonicalBody,
    datasetRevision,
    runtimeRevision,
    experimentRevision,
    baseRunFingerprint,
  });
  const [existingRows] = await pool.query<GuidedTrainingSessionRow[]>(
    `SELECT session_id,
            experiment_id,
            student_id,
            model_type,
            request_hash,
            dataset_revision,
            runtime_revision,
            request_payload,
            status,
            current_step_id,
            completed_step_ids,
            step_outputs,
            step_plan,
            result_payload,
            artifact_dir,
            error_message,
            experiment_state_version,
            updated_at
       FROM model_guided_training_sessions
      WHERE experiment_id = ?
        AND student_id = ?
        AND model_type = ?
        AND request_hash = ?
      LIMIT 1`,
    [canonicalBody.experiment_id, studentId, modelType, requestHash],
  );

  const existingSession = existingRows[0];
  if (existingSession) {
    if (existingSession.status === "running") {
      const staleAfterSeconds = Math.ceil(TIMEOUTS.MODEL_TRAINING / 1000) + 30;
      const [resetResult] = await pool.query(
        `UPDATE model_guided_training_sessions
            SET status = 'ready',
                error_message = NULL
          WHERE session_id = ?
            AND status = 'running'
            AND updated_at < DATE_SUB(NOW(), INTERVAL ? SECOND)`,
        [existingSession.session_id, staleAfterSeconds],
      );
      if (Number((resetResult as { affectedRows?: number }).affectedRows ?? 0) === 1) {
        existingSession.status = "ready";
      }
    }
    if (existingSession.status === "completed") {
      try {
        const restoredSession = await materializeCompletedGuidedTrainingRun(existingSession);
        return buildGuidedTrainingSessionPayload(restoredSession);
      } catch (error) {
        if (!(error instanceof ModelNotFoundError)) {
          throw error;
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
          throw new ModelValidationError("旧的分阶段训练会话工件清理失败，请稍后重试");
        }
      }
    } else {
      const datasetFile = path.join(existingSession.artifact_dir, GUIDED_DATASET_FILE_NAME);
      if (!fs.existsSync(datasetFile)) {
        await writeGuidedDatasetSnapshot(existingSession.artifact_dir, datasetJson);
      }
      return buildGuidedTrainingSessionPayload(existingSession);
    }
  }

  const sessionId = crypto.randomUUID();
  const artifactDir = buildGuidedTrainingArtifactDir(canonicalBody.experiment_id, studentId, modelType, sessionId);
  fs.mkdirSync(artifactDir, { recursive: true });
  await writeGuidedDatasetSnapshot(artifactDir, datasetJson);

  const completedStepIds: string[] = [];
  const stepPlan = buildGuidedTrainingStepPlan(modelType, canonicalBody);
  const currentStepId = getGuidedNextStepId(stepPlan, completedStepIds);
  try {
    await pool.query(
      `INSERT INTO model_guided_training_sessions
       (session_id,
        experiment_id,
        student_id,
        model_type,
        request_hash,
        dataset_revision,
        runtime_revision,
        request_payload,
        status,
        current_step_id,
        completed_step_ids,
        step_outputs,
        step_plan,
        result_payload,
        artifact_dir,
        expires_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, ?, NULL, ?, DATE_ADD(NOW(), INTERVAL 7 DAY))`,
      [
        sessionId,
        canonicalBody.experiment_id,
        studentId,
        modelType,
        requestHash,
        datasetRevision,
        runtimeRevision,
        JSON.stringify(canonicalBody),
        currentStepId,
        JSON.stringify(completedStepIds),
        JSON.stringify({}),
        JSON.stringify(stepPlan),
        artifactDir,
      ],
    );
  } catch (error) {
    if (!isDuplicateGuidedTrainingRequestError(error)) throw error;
    fs.rmSync(artifactDir, { recursive: true, force: true });
    const [racedRows] = await pool.query<GuidedTrainingSessionRow[]>(
      `SELECT *
         FROM model_guided_training_sessions
        WHERE experiment_id = ?
          AND student_id = ?
          AND model_type = ?
          AND request_hash = ?
        LIMIT 1`,
      [canonicalBody.experiment_id, studentId, modelType, requestHash],
    );
    if (racedRows[0]) return buildGuidedTrainingSessionPayload(racedRows[0]);
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
  if (row.status === "completed") {
    const sourceModelPath = getGuidedSessionModelPath(row.artifact_dir, row.model_type);
    if (!await hasReusableSavedModelArtifact(row.model_type, sourceModelPath)) {
      throw new ModelNotFoundError("分阶段训练产物已过期或已被清理");
    }
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
    const restoredRow = await materializeCompletedGuidedTrainingRun(row);
    return buildGuidedTrainingSessionPayload(restoredRow);
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
    const pythonParams = await buildGuidedTrainingPythonParams(modelType, studentId, body, row.artifact_dir);
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
    const nextStepPlan = shouldTrimBoostingRoundSteps(modelType, stepId, pythonStepResult.output)
      ? trimBoostingStepPlanAfterEarlyStop(stepPlan, stepId)
      : stepPlan;

    let transformedResult: ReturnType<typeof transformTrainingResult> | null = null;
    let versionedArtifactPath: string | null = null;
    if (pythonStepResult.final_result) {
      const sourceModelPath = getGuidedSessionModelPath(row.artifact_dir, row.model_type);
      versionedArtifactPath = buildVersionedModelPath(
        row.experiment_id,
        row.student_id,
        row.model_type,
        row.session_id,
        "backtest",
      );
      await copyModelArtifact(row.model_type, sourceModelPath, versionedArtifactPath);
      pythonStepResult.final_result.saved_model = versionedArtifactPath;
      transformedResult = transformTrainingResult(modelType, pythonStepResult.final_result);
    }

    const status: GuidedTrainingStatus = transformedResult ? "completed" : "ready";
    const currentStepId = status === "completed"
      ? null
      : getGuidedNextStepId(nextStepPlan, nextCompletedStepIds);

    if (transformedResult && versionedArtifactPath) {
      await commitModelTrainingRun({
        runId: row.session_id,
        sessionId: row.session_id,
        experimentId: row.experiment_id,
        studentId: row.student_id,
        modelType: row.model_type,
        requestHash: row.request_hash,
        datasetRevision: row.dataset_revision,
        runtimeRevision: row.runtime_revision,
        body,
        result: transformedResult,
        artifactPath: versionedArtifactPath,
        guidedSession: {
          sessionId,
          currentStepId,
          completedStepIds: nextCompletedStepIds,
          stepOutputs: nextStepOutputs,
          stepPlan: nextStepPlan,
          result: transformedResult,
        },
      });
    } else {
      await pool.query(
        `UPDATE model_guided_training_sessions
          SET status = ?,
              current_step_id = ?,
              completed_step_ids = ?,
              step_outputs = ?,
              step_plan = ?,
              result_payload = ?,
              error_message = NULL
        WHERE session_id = ?`,
      [
        status,
        currentStepId,
        JSON.stringify(nextCompletedStepIds),
        JSON.stringify(nextStepOutputs),
        JSON.stringify(nextStepPlan),
        transformedResult ? JSON.stringify(transformedResult) : null,
        sessionId,
        ],
      );
    }
    sessionMarkedRunning = false;

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
  const canonicalBody = buildCanonicalTrainingBody(body, trainingConfig);
  if (modelType === "ma") {
    validateMovingAverageWindow(trainingConfig, canonicalBody.moving_average_window);
  }
  const derivedModelParams: Record<string, unknown> = isEnsembleModelType(modelType)
    ? buildEnsemblePythonParams(experiment, body)
    : {};
  if (isEnsembleModelType(modelType) && body.models) {
    const baseModelArtifactMap = await buildAvailableBaseModelArtifactMap(body.experiment_id, studentId, body.models);
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

  const datasetJson = JSON.stringify(allData);
  const datasetRevision = hashDatasetSnapshot(datasetJson);
  const runtimeRevision = getModelRuntimeRevision();
  const baseRunFingerprint = isEnsembleModelType(modelType)
    ? await buildBaseRunFingerprint(canonicalBody.experiment_id, studentId, canonicalBody)
    : {};
  const experimentRevision = buildExperimentTrainingRevision(modelType, experiment, canonicalBody);
  const requestHash = buildModelTrainingRequestHash({
    modelType,
    body: canonicalBody,
    datasetRevision,
    runtimeRevision,
    experimentRevision,
    baseRunFingerprint,
  });
  const runId = crypto.randomUUID();
  const modelSaveLoc = buildVersionedModelPath(
    body.experiment_id,
    studentId,
    modelType,
    runId,
    "backtest",
  );
  const snapshotDir = fs.mkdtempSync(path.join(os.tmpdir(), "fz-training-snapshot-"));
  const dataFile = path.join(snapshotDir, GUIDED_DATASET_FILE_NAME);
  await Bun.write(dataFile, datasetJson);

  const pythonParams: Record<string, unknown> = {
    dataFile,
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
  let result: Record<string, unknown>;
  try {
    result = (await runPythonModel(modelType, pythonParams, timeout, progressReporter)) as Record<string, unknown>;
  } finally {
    fs.rmSync(snapshotDir, { recursive: true, force: true });
  }
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
  const transformedResult = transformTrainingResult(modelType, result);
  await commitModelTrainingRun({
    runId,
    sessionId: null,
    experimentId: canonicalBody.experiment_id,
    studentId,
    modelType,
    requestHash,
    datasetRevision,
    runtimeRevision,
    body: canonicalBody,
    result: transformedResult,
    artifactPath: modelSaveLoc,
  });
  return transformedResult;
}

export async function getReusableProductionModelArtifactResult(
  modelType: ModelType,
  studentId: number,
  experimentId: number,
) {
  const experiment = await assertExperimentModelAccess(experimentId, studentId);
  assertModelReadyForProduction(experiment, modelType);

  const activeRun = await getActiveModelTrainingRun(experimentId, studentId, modelType);
  const productionModelPath = activeRun
    ? activeRun.production_artifact_path
    : getProductionModelPath(experimentId, studentId, modelType);
  if (!productionModelPath) {
    return null;
  }
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

  const activeRun = await getActiveModelTrainingRun(experimentId, studentId, modelType);
  const productionModelPath = activeRun
    ? buildVersionedModelPath(experimentId, studentId, modelType, activeRun.run_id, "production")
    : getProductionModelPath(experimentId, studentId, modelType);
  const backtestModelPath = activeRun?.artifact_path
    ?? getBacktestModelPath(experimentId, studentId, modelType);
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

  const snapshotDir = fs.mkdtempSync(path.join(os.tmpdir(), "fz-production-snapshot-"));
  const dataFile = path.join(snapshotDir, GUIDED_DATASET_FILE_NAME);
  await Bun.write(dataFile, JSON.stringify(allData));
  const pythonParams: Record<string, unknown> = {
    dataFile,
    trainStart: trainingConfig.data_window_train_start_index,
    trainEnd: trainingConfig.data_window_evaluate_end_index,
    saveModel: productionModelPath,
    loadModel: backtestModelPath,
    fitOnly: true,
    pretty: true,
  };

  try {
    await runPythonModel(modelType, pythonParams, TRAINING_TIMEOUT);
  } finally {
    fs.rmSync(snapshotDir, { recursive: true, force: true });
  }
  if (activeRun) {
    const recorded = await setProductionArtifactForRun(activeRun.run_id, productionModelPath);
    if (!recorded) {
      throw new ModelValidationError("模型版本已变化，已放弃本次生产模型，请重试");
    }
  }

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

    let deletedInGroup = 0;
    for (const file of group.files) {
      try {
        fs.unlinkSync(path.join(trainedModelsDir, file));
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

  const reusableArtifact = await prepareExperimentModelForProduction(modelType, studentId, experimentId);
  const productionModelPath = reusableArtifact?.results.saved_model;
  if (!productionModelPath || !await hasReusableSavedModelArtifact(modelType, productionModelPath)) {
    throw new ModelNotFoundError(`未找到该实验的 ${modelType.toUpperCase()} 生产预测模型`);
  }
  const predictionRun = await getActiveModelTrainingRun(experimentId, studentId, modelType);

  const pythonParams = {
    loadModel: productionModelPath,
    forecastSteps,
    pretty: true,
  };

  const result = (await runPythonModel(modelType, pythonParams, TIMEOUTS.MODEL_PREDICTION)) as Record<string, unknown>;
  if (predictionRun) {
    const currentRun = await getActiveModelTrainingRun(experimentId, studentId, modelType);
    if (currentRun?.run_id !== predictionRun.run_id) {
      throw new ModelValidationError("预测期间模型版本已变化，请重新预测");
    }
  }

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
