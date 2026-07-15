import path from "path";
import fs from "fs";
import crypto from "crypto";
import os from "node:os";
import type { Stats } from "node:fs";
import type { PoolConnection, RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import { acquireConnectionWithTimeout, queryWithTimeout } from "./model-db-guard";
import logger from "../../utils/logger";
import type { ModelType, TrainInput } from "./models.schema";
import { TIMEOUTS, CLEANUP_RETENTION } from "../../utils/constants";
import { withBoundedPoolTransaction } from "../../utils/bounded-pool-transaction";
import { PY_DIR, resolvePythonExecutable } from "../../utils/python";
import { terminateProcessTree } from "../../utils/process-tree";
import {
  GUIDED_TRAINING_ROOT,
  buildGuidedTrainingArtifactDir,
  buildGuidedTrainingAttemptPaths,
  buildVersionedModelPath,
} from "./model-runtime";
import { loadDatasetFromDb } from "./model-datasets";
import {
  ModelAccessError,
  ModelArtifactError,
  ModelConflictError,
  ModelNotFoundError,
  ModelTimeoutError,
  ModelValidationError,
} from "./model-errors";
import {
  withModelOperationContext,
  withModelOperationLease,
  type ModelOperationContext,
  type ModelOperationLease,
} from "./model-operation-lease";
import {
  allPossiblePythonParams,
  buildPythonModelProcessEnv,
  runPythonGuidedTrainingStep,
  runPythonModel,
} from "./model-python-runner";
import { normalizeDemandPredictionPoints, transformTrainingResult } from "./model-results";
import {
  backfillArtifactSha256ForRun,
  backfillProductionArtifactSha256ForRun,
  buildExperimentTrainingRevision,
  buildModelTrainingRequestHash,
  clearProductionArtifactForRun,
  commitModelTrainingRun,
  getActiveModelTrainingRun,
  loadCurrentExperimentStatePatch,
  setProductionArtifactForRun,
  type ModelTrainingRunRow,
} from "./model-run-store";
import {
  BASE_MODEL_TYPES,
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
  ModelArtifactError,
  ModelCapacityBusyError,
  ModelConflictError,
  ModelNotFoundError,
  ModelOperationDeadlineError,
  ModelServiceUnavailableError,
  ModelTimeoutError,
  ModelValidationError,
} from "./model-errors";
export { runPythonGuidedTrainingStep, runPythonModel, type ModelProgressEvent, type ModelProgressReporter } from "./model-python-runner";
export { transformTrainingResult } from "./model-results";
export type { GuidedTrainingSessionPayload, GuidedTrainingStepDefinition } from "./model-types";

const TRAINING_TIMEOUT = TIMEOUTS.MODEL_TRAINING;
const MODEL_EXECUTION_HARD_CAP = TIMEOUTS.MODEL_EXECUTION_HARD_CAP;
const DEFAULT_PRODUCTION_FORECAST_STEPS = 6;
const MIN_TRAINING_POINTS = 8;
const MIN_EVALUATION_POINTS = 2;
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
  protocol_version: number;
  request_payload: unknown;
  status: GuidedTrainingStatus;
  current_step_id: string | null;
  completed_step_ids: unknown;
  step_outputs: unknown;
  step_plan: unknown;
  result_payload: unknown;
  artifact_dir: string;
  committed_state_path: string | null;
  committed_state_sha256: string | null;
  active_attempt_id: string | null;
  error_message: string | null;
  superseded_reason: string | null;
  experiment_state_version: number | null;
  updated_at: string | Date;
}

interface GuidedTrainingStepAttemptRow extends RowDataPacket {
  attempt_id: string;
  session_id: string;
  step_id: string;
  status: "running" | "python_completed" | "committed" | "failed";
  input_state_path: string | null;
  input_state_sha256: string | null;
  candidate_state_path: string;
  candidate_state_sha256: string | null;
  receipt_path: string;
  output_payload: unknown;
  final_result_payload: unknown;
  error_message: string | null;
  run_count: number;
  lease_expires_at: Date | string | null;
}

interface GuidedTrainingCleanupRow extends RowDataPacket {
  session_id: string;
  artifact_dir: string;
}

interface ModelTrainingDependencyRow extends RowDataPacket {
  base_model_type: BaseModelType;
  base_run_id: string | null;
}

interface StaleModelExperimentEligibilityRow extends RowDataPacket {
  can_delete: 0 | 1 | boolean | string;
}

async function assertExperimentModelAccess(experimentId: number, studentId: number) {
  const [rows] = await queryWithTimeout<ExperimentModelContextRow[]>(pool, 
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

const requireActiveModelTrainingRun = async (
  experimentId: number,
  studentId: number,
  modelType: ModelType,
) => {
  const activeRun = await getActiveModelTrainingRun(experimentId, studentId, modelType);
  if (!activeRun) {
    throw new ModelValidationError(
      `${modelType.toUpperCase()} 模型缺少可验证的训练版本，请重新训练该模型`,
    );
  }
  return activeRun;
};

const buildProductionCandidateModelPath = (
  experimentId: number,
  studentId: number,
  modelType: ModelType,
  runId: string,
) => {
  const stablePath = buildVersionedModelPath(
    experimentId,
    studentId,
    modelType,
    runId,
    "production",
  );
  const extension = path.extname(stablePath);
  const basename = path.basename(stablePath, extension);
  return path.join(
    path.dirname(stablePath),
    `${basename}-${crypto.randomUUID()}${extension}`,
  );
};

const getLstmSidecarPath = (modelPath: string) => modelPath.replace(/\.pkl$/, "_keras_model.h5");
const AUXILIARY_PYTHON_TIMEOUT_MS = 30_000;
const AUXILIARY_SETSID_PATH = process.platform === "win32" ? undefined : Bun.which("setsid");

interface AuxiliaryPythonProcess {
  stdout: ReadableStream<Uint8Array>;
  stderr: ReadableStream<Uint8Array>;
  exited: Promise<number>;
  exitCode: number | null;
  pid: number;
  kill: (signal?: number | NodeJS.Signals) => void;
}

interface AuxiliaryPythonWaitOptions {
  timeoutMs?: number;
  terminationSignal?: AbortSignal;
  processGroup?: boolean;
  terminationGraceMs?: number;
}

const buildAuxiliaryPythonCommand = (...args: string[]) => ({
  command: AUXILIARY_SETSID_PATH ? [AUXILIARY_SETSID_PATH, ...args] : args,
  processGroup: Boolean(AUXILIARY_SETSID_PATH),
});

const getAuxiliaryTerminationError = (signal: AbortSignal) => signal.reason instanceof Error
  ? signal.reason
  : new Error("辅助 Python 任务已终止");

export const waitForAuxiliaryPythonProcess = async (
  proc: AuxiliaryPythonProcess,
  label: string,
  options: AuxiliaryPythonWaitOptions = {},
) => {
  const timeoutMs = options.timeoutMs ?? AUXILIARY_PYTHON_TIMEOUT_MS;
  const terminationGraceMs = options.terminationGraceMs
    ?? TIMEOUTS.MODEL_PROCESS_TERMINATION_GRACE;
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  let abortHandler: (() => void) | undefined;
  let forcedTerminationError: Error | null = null;
  let terminationPromise: Promise<void> | null = null;

  const terminateAndWait = (error: Error): Promise<Error> => {
    forcedTerminationError ??= error;
    if (!terminationPromise) {
      terminationPromise = terminateProcessTree(proc, {
        graceMs: terminationGraceMs,
        processGroup: options.processGroup === true,
        label,
      });
    }
    return terminationPromise.then(() => forcedTerminationError!);
  };

  const completion = (async () => {
    let stdout = "";
    let stderr = "";
    let streamError: unknown;
    try {
      [stdout, stderr] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
      ]);
    } catch (error) {
      streamError = error;
    }
    await proc.exited;
    if (forcedTerminationError) throw forcedTerminationError;
    if (streamError !== undefined) throw streamError;
    return { stdout, stderr };
  })();

  const competingPromises: Array<Promise<{ stdout: string; stderr: string }>> = [completion];
  competingPromises.push(new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      const error = new ModelTimeoutError(`${label}超时`, timeoutMs, "python_process");
      void terminateAndWait(error).then(reject, reject);
    }, timeoutMs);
    timeoutId.unref?.();
  }));
  if (options.terminationSignal) {
    competingPromises.push(new Promise((_, reject) => {
      abortHandler = () => {
        void terminateAndWait(getAuxiliaryTerminationError(options.terminationSignal!))
          .then(reject, reject);
      };
      options.terminationSignal!.addEventListener("abort", abortHandler, { once: true });
      if (options.terminationSignal!.aborted) abortHandler();
    }));
  }

  try {
    return await Promise.race(competingPromises);
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
    if (abortHandler && options.terminationSignal) {
      options.terminationSignal.removeEventListener("abort", abortHandler);
    }
  }
};

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

const removeModelArtifactGroup = (modelPath: string) => {
  const trainedModelsRoot = path.join(PY_DIR, "trained_models");
  if (!isPathInsideDirectory(modelPath, trainedModelsRoot)) {
    logger.warn({ modelPath }, "拒绝清理训练模型目录之外的产物组");
    return;
  }

  removeFileIfExists(modelPath);
  removeSiblingArtifactsByPrefix(modelPath);
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

const hasCompleteEnsembleArtifact = async (
  modelPath: string,
  terminationSignal?: AbortSignal,
) => {
  try {
    const pythonExec = resolvePythonExecutable();
    const auxiliaryCommand = buildAuxiliaryPythonCommand(
      pythonExec,
      "-c",
      ENSEMBLE_ARTIFACT_VALIDATION_SCRIPT,
      modelPath,
    );
    const proc = Bun.spawn(auxiliaryCommand.command, {
      cwd: PY_DIR,
      stdout: "pipe",
      stderr: "pipe",
      env: {
        ...process.env,
        ...buildPythonModelProcessEnv(),
        PATH: `${path.dirname(pythonExec)}${path.delimiter}${process.env.PATH ?? ""}`,
      },
    });

    const { stdout, stderr } = await waitForAuxiliaryPythonProcess(proc, "校验融合模型产物", {
      terminationSignal,
      processGroup: auxiliaryCommand.processGroup,
    });

    if (proc.exitCode !== 0) {
      logger.warn({ modelPath, stderr: stderr.slice(0, 1000) }, "校验融合模型产物完整性失败");
      return false;
    }

    const payload = JSON.parse(stdout) as { valid?: unknown };
    return payload.valid === true;
  } catch (error) {
    if (terminationSignal?.aborted) {
      throw getAuxiliaryTerminationError(terminationSignal);
    }
    if (error instanceof ModelTimeoutError) {
      throw error;
    }
    logger.warn({ err: error, modelPath }, "解析融合模型产物校验结果失败");
    return false;
  }
};

const BASE_ARTIFACT_DIGEST_PROTOCOL = "fz-base-model-artifact-v1";

const hashFileContents = async (
  filePath: string,
  terminationSignal?: AbortSignal,
) => await new Promise<string>((resolve, reject) => {
  if (terminationSignal?.aborted) {
    reject(getAuxiliaryTerminationError(terminationSignal));
    return;
  }
  const digest = crypto.createHash("sha256");
  const stream = fs.createReadStream(filePath);
  const onAbort = () => stream.destroy(getAuxiliaryTerminationError(terminationSignal!));
  terminationSignal?.addEventListener("abort", onAbort, { once: true });
  const cleanup = () => terminationSignal?.removeEventListener("abort", onAbort);
  stream.on("data", (chunk) => digest.update(chunk));
  stream.once("error", (error) => {
    cleanup();
    reject(error);
  });
  stream.once("end", () => {
    cleanup();
    resolve(digest.digest("hex"));
  });
});

const getBaseArtifactFiles = (modelType: BaseModelType, modelPath: string) => {
  const files = [{ role: "state", filePath: modelPath }];
  if (modelType === "lstm") {
    files.push({ role: "keras", filePath: getLstmSidecarPath(modelPath) });
  }
  return files;
};

const computeBaseModelArtifactSha256 = async (
  modelType: BaseModelType,
  modelPath: string,
  terminationSignal?: AbortSignal,
) => {
  if (!hasSavedModelArtifact(modelType, modelPath)) {
    throw new ModelArtifactError(`${modelType.toUpperCase()} 模型产物缺失或不完整`);
  }

  try {
    const manifest = crypto.createHash("sha256");
    manifest.update(BASE_ARTIFACT_DIGEST_PROTOCOL);
    manifest.update("\0");
    manifest.update(modelType);
    for (const artifact of getBaseArtifactFiles(modelType, modelPath)) {
      if (terminationSignal?.aborted) throw getAuxiliaryTerminationError(terminationSignal);
      manifest.update("\0");
      manifest.update(artifact.role);
      manifest.update("\0");
      manifest.update(await hashFileContents(artifact.filePath, terminationSignal));
    }
    return manifest.digest("hex");
  } catch (error) {
    if (terminationSignal?.aborted) throw getAuxiliaryTerminationError(terminationSignal);
    if (error instanceof ModelArtifactError) throw error;
    throw new ModelArtifactError(
      `${modelType.toUpperCase()} 模型产物无法读取：${error instanceof Error ? error.message : "未知错误"}`,
    );
  }
};

const artifactDigestMatches = (actual: string, expected: string) => {
  if (!/^[a-f\d]{64}$/i.test(expected)) return false;
  return crypto.timingSafeEqual(
    Buffer.from(actual.toLowerCase(), "hex"),
    Buffer.from(expected.toLowerCase(), "hex"),
  );
};

// Enumerate an ensemble's real constituent files (root pickle + every member
// state + LSTM member sidecars) using pickle-truth paths, then hash them into a
// stable manifest. Member/sidecar paths only exist inside the root pickle, so
// this must run in Python; every path is resolved and constrained to the
// trained-models root to reject traversal or redirected artifacts.
const ENSEMBLE_ARTIFACT_DIGEST_SCRIPT = `
import hashlib
import json
import os
import pickle
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_within(root, path):
    if not isinstance(path, str) or not path:
        return None
    real = os.path.realpath(path)
    if os.path.commonpath([real, root]) != root:
        return None
    if not os.path.isfile(real):
        return None
    return real


def iter_model_states(base_model_states):
    if not isinstance(base_model_states, dict):
        return None
    flattened = []
    for outer_name, model_state in base_model_states.items():
        if isinstance(model_state, dict) and isinstance(model_state.get("models"), list):
            candidate_states = model_state.get("models", [])
        else:
            candidate_states = [model_state]
        for candidate_state in candidate_states:
            if not isinstance(candidate_state, dict):
                return None
            flattened.append((
                str(candidate_state.get("model_type") or outer_name),
                candidate_state.get("model_file"),
                candidate_state.get("keras_model_path"),
            ))
    return flattened


try:
    model_type = sys.argv[1]
    model_path = sys.argv[2]
    root = os.path.realpath(sys.argv[3])
    root_file = resolve_within(root, model_path)
    if root_file is None:
        raise ValueError("ensemble root artifact missing or outside root")
    with open(root_file, "rb") as handle:
        state = pickle.load(handle)
    members = iter_model_states(state.get("base_model_states"))
    if members is None or len(members) == 0:
        raise ValueError("ensemble has no base members")
    ordered = []
    for member_type, member_file, member_keras in sorted(members, key=lambda item: (item[0], str(item[1]))):
        member_real = resolve_within(root, member_file)
        if member_real is None:
            raise ValueError("ensemble member artifact missing or outside root")
        ordered.append(("member:" + member_type, member_real))
        if member_type == "lstm":
            sidecar = member_keras if isinstance(member_keras, str) and member_keras else os.path.splitext(member_file)[0] + "_keras_model.h5"
            sidecar_real = resolve_within(root, sidecar)
            if sidecar_real is None:
                raise ValueError("ensemble member keras sidecar missing or outside root")
            ordered.append(("member_keras:" + member_type, sidecar_real))
    manifest = hashlib.sha256()
    manifest.update(b"fz-ensemble-model-artifact-v1")
    manifest.update(b"\\0")
    manifest.update(model_type.encode("utf-8"))
    manifest.update(b"\\0root\\0")
    manifest.update(sha256_file(root_file).encode("ascii"))
    for role, filepath in ordered:
        manifest.update(b"\\0")
        manifest.update(role.encode("utf-8"))
        manifest.update(b"\\0")
        manifest.update(sha256_file(filepath).encode("ascii"))
    print(json.dumps({"ok": True, "digest": manifest.hexdigest()}))
except Exception as exc:
    print(json.dumps({"ok": False, "reason": str(exc)}))
`;

// Confirm a legacy LSTM artifact's pickle points keras_model_path at exactly the
// canonical sidecar (and inside the artifact root) before we bless a checksum.
// Otherwise the digest could cover a decoy while Python loads a different H5.
const LSTM_SIDECAR_CANONICAL_SCRIPT = `
import json
import os
import pickle
import sys

try:
    model_path = sys.argv[1]
    expected_sidecar = sys.argv[2]
    root = os.path.realpath(sys.argv[3])
    with open(model_path, "rb") as handle:
        state = pickle.load(handle)
    keras_path = state.get("keras_model_path")
    if not isinstance(keras_path, str) or not keras_path:
        raise ValueError("missing keras_model_path")
    real = os.path.realpath(keras_path)
    expected_real = os.path.realpath(expected_sidecar)
    if os.path.commonpath([real, root]) != root:
        raise ValueError("keras path outside artifact root")
    if real != expected_real:
        raise ValueError("keras path is not the canonical sidecar")
    print(json.dumps({"ok": True}))
except Exception as exc:
    print(json.dumps({"ok": False, "reason": str(exc)}))
`;

const runAuxiliaryModelJsonScript = async <T,>(
  label: string,
  script: string,
  scriptArgs: string[],
  terminationSignal?: AbortSignal,
): Promise<T> => {
  try {
    const pythonExec = resolvePythonExecutable();
    const auxiliaryCommand = buildAuxiliaryPythonCommand(pythonExec, "-c", script, ...scriptArgs);
    const proc = Bun.spawn(auxiliaryCommand.command, {
      cwd: PY_DIR,
      stdout: "pipe",
      stderr: "pipe",
      env: {
        ...process.env,
        ...buildPythonModelProcessEnv(),
        PATH: `${path.dirname(pythonExec)}${path.delimiter}${process.env.PATH ?? ""}`,
      },
    });
    const { stdout, stderr } = await waitForAuxiliaryPythonProcess(proc, label, {
      terminationSignal,
      processGroup: auxiliaryCommand.processGroup,
    });
    if (proc.exitCode !== 0) {
      throw new ModelArtifactError(`${label}失败：${stderr.slice(0, 500)}`);
    }
    return JSON.parse(stdout) as T;
  } catch (error) {
    if (terminationSignal?.aborted) throw getAuxiliaryTerminationError(terminationSignal);
    if (error instanceof ModelTimeoutError || error instanceof ModelArtifactError) throw error;
    throw new ModelArtifactError(`${label}失败：${error instanceof Error ? error.message : "未知错误"}`);
  }
};

const computeEnsembleArtifactSha256 = async (
  modelType: ModelType,
  modelPath: string,
  terminationSignal?: AbortSignal,
): Promise<string> => {
  const trainedModelsRoot = path.join(PY_DIR, "trained_models");
  const payload = await runAuxiliaryModelJsonScript<{ ok?: unknown; digest?: unknown; reason?: unknown }>(
    `${modelType.toUpperCase()} 融合模型产物摘要`,
    ENSEMBLE_ARTIFACT_DIGEST_SCRIPT,
    [modelType, modelPath, trainedModelsRoot],
    terminationSignal,
  );
  if (payload.ok !== true || typeof payload.digest !== "string" || !/^[a-f\d]{64}$/i.test(payload.digest)) {
    throw new ModelArtifactError(
      `${modelType.toUpperCase()} 融合模型产物无法生成摘要：${typeof payload.reason === "string" ? payload.reason : "结构无效"}`,
    );
  }
  return payload.digest.toLowerCase();
};

const assertLstmSidecarCanonical = async (
  modelPath: string,
  terminationSignal?: AbortSignal,
) => {
  const trainedModelsRoot = path.join(PY_DIR, "trained_models");
  const payload = await runAuxiliaryModelJsonScript<{ ok?: unknown; reason?: unknown }>(
    "LSTM 模型 sidecar 校验",
    LSTM_SIDECAR_CANONICAL_SCRIPT,
    [modelPath, getLstmSidecarPath(modelPath), trainedModelsRoot],
    terminationSignal,
  );
  if (payload.ok !== true) {
    throw new ModelArtifactError(
      `LSTM 模型 sidecar 路径不可信：${typeof payload.reason === "string" ? payload.reason : "校验失败"}`,
    );
  }
};

const computeModelArtifactSha256 = async (
  modelType: ModelType,
  modelPath: string,
  terminationSignal?: AbortSignal,
): Promise<string> => {
  if (isEnsembleModelType(modelType)) {
    return await computeEnsembleArtifactSha256(modelType, modelPath, terminationSignal);
  }
  return await computeBaseModelArtifactSha256(modelType, modelPath, terminationSignal);
};

const validateBaseModelArtifactSemantically = async (
  modelType: ModelType,
  modelPath: string,
  terminationSignal?: AbortSignal,
) => {
  if (isEnsembleModelType(modelType)) {
    if (!await hasCompleteEnsembleArtifact(modelPath, terminationSignal)) {
      throw new ModelArtifactError(`${modelType.toUpperCase()} 融合模型产物无法加载或成员缺失`);
    }
    return;
  }
  try {
    const result = await runPythonModel(
      modelType,
      { loadModel: modelPath, forecastSteps: 1, pretty: true },
      TIMEOUTS.MODEL_PREDICTION,
      undefined,
      terminationSignal,
    ) as Record<string, unknown>;
    normalizeDemandPredictionPoints(result.predictions, 1, { requireStdDev: true });
  } catch (error) {
    if (terminationSignal?.aborted || error instanceof ModelTimeoutError) throw error;
    if (error instanceof ModelArtifactError) throw error;
    if (error instanceof Error && /^\[PYTHON_(?:VALIDATION|DATA)\]/.test(error.message)) {
      throw new ModelArtifactError(`${modelType.toUpperCase()} 模型产物无法加载：${error.message}`);
    }
    throw error;
  }
};

interface BaseArtifactIntegrityResult {
  sha256: string;
  legacyChecksum: boolean;
}

const verifyBaseModelArtifactIntegrity = async (
  modelType: ModelType,
  modelPath: string,
  expectedSha256: string | null | undefined,
  terminationSignal?: AbortSignal,
): Promise<BaseArtifactIntegrityResult> => {
  const beforeValidationSha256 = await computeModelArtifactSha256(modelType, modelPath, terminationSignal);
  if (expectedSha256) {
    if (!artifactDigestMatches(beforeValidationSha256, expectedSha256)) {
      throw new ModelArtifactError(`${modelType.toUpperCase()} 模型产物校验和不匹配`);
    }
    return { sha256: beforeValidationSha256, legacyChecksum: false };
  }

  await validateBaseModelArtifactSemantically(modelType, modelPath, terminationSignal);
  // Legacy LSTM artifacts store their sidecar path inside the pickle; require it
  // to be the canonical sidecar before blessing so the digest covers the file
  // Python actually loads, not a decoy left at the derived path.
  if (modelType === "lstm") {
    await assertLstmSidecarCanonical(modelPath, terminationSignal);
  }
  const afterValidationSha256 = await computeModelArtifactSha256(modelType, modelPath, terminationSignal);
  if (!artifactDigestMatches(afterValidationSha256, beforeValidationSha256)) {
    throw new ModelArtifactError(`${modelType.toUpperCase()} 模型产物在加载校验期间发生变化`);
  }
  return { sha256: afterValidationSha256, legacyChecksum: true };
};

const verifyBacktestBaseArtifactForRun = async (
  modelType: ModelType,
  activeRun: ModelTrainingRunRow,
  terminationSignal?: AbortSignal,
) => {
  const integrity = await verifyBaseModelArtifactIntegrity(
    modelType,
    activeRun.artifact_path,
    activeRun.artifact_sha256,
    terminationSignal,
  );
  if (integrity.legacyChecksum) {
    await backfillArtifactSha256ForRun(activeRun.run_id, integrity.sha256);
  }
  return integrity.sha256;
};

const verifyEnsembleBaseArtifactForRun = async (
  modelType: BaseModelType,
  activeRun: ModelTrainingRunRow,
  terminationSignal?: AbortSignal,
) => {
  try {
    return await verifyBacktestBaseArtifactForRun(modelType, activeRun, terminationSignal);
  } catch (error) {
    if (!(error instanceof ModelArtifactError)) throw error;
    throw new ModelValidationError(
      `融合模型依赖的 ${modelType.toUpperCase()} 基础模型产物已损坏，请重新训练该基础模型`,
      {
        code: error.code,
        required_action: "retrain",
        base_model: modelType,
      },
    );
  }
};

const verifyProductionBaseArtifactForRun = async (
  modelType: ModelType,
  activeRun: ModelTrainingRunRow,
  terminationSignal?: AbortSignal,
) => {
  if (!activeRun.production_artifact_path || !activeRun.production_revision) {
    throw new ModelArtifactError(`${modelType.toUpperCase()} 生产模型产物记录不完整`);
  }
  const integrity = await verifyBaseModelArtifactIntegrity(
    modelType,
    activeRun.production_artifact_path,
    activeRun.production_artifact_sha256,
    terminationSignal,
  );
  if (integrity.legacyChecksum) {
    await backfillProductionArtifactSha256ForRun(
      activeRun.run_id,
      activeRun.production_revision,
      integrity.sha256,
    );
  }
  return integrity.sha256;
};

const invalidateProductionArtifactForRun = async (activeRun: ModelTrainingRunRow) => {
  const artifactPath = activeRun.production_artifact_path;
  const productionRevision = activeRun.production_revision;
  if (!artifactPath || !productionRevision) return false;

  const cleared = await clearProductionArtifactForRun(activeRun.run_id, productionRevision);
  if (cleared) {
    removeModelArtifactGroup(artifactPath);
  }
  return cleared;
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

const copyLstmModelArtifact = async (
  sourceModelPath: string,
  destinationModelPath: string,
  terminationSignal?: AbortSignal,
) => {
  const pythonExec = resolvePythonExecutable();
  const auxiliaryCommand = buildAuxiliaryPythonCommand(
    pythonExec,
    "-c",
    LSTM_ARTIFACT_COPY_SCRIPT,
    sourceModelPath,
    destinationModelPath,
  );
  const proc = Bun.spawn(auxiliaryCommand.command, {
    cwd: PY_DIR,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      ...buildPythonModelProcessEnv(),
      PATH: `${path.dirname(pythonExec)}${path.delimiter}${process.env.PATH ?? ""}`,
    },
  });

  const { stderr } = await waitForAuxiliaryPythonProcess(proc, "复制 LSTM 模型产物", {
    terminationSignal,
    processGroup: auxiliaryCommand.processGroup,
  });

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

const copyEnsembleModelArtifact = async (
  sourceModelPath: string,
  destinationModelPath: string,
  terminationSignal?: AbortSignal,
) => {
  const pythonExec = resolvePythonExecutable();
  const auxiliaryCommand = buildAuxiliaryPythonCommand(
    pythonExec,
    "-c",
    ENSEMBLE_ARTIFACT_COPY_SCRIPT,
    sourceModelPath,
    destinationModelPath,
  );
  const proc = Bun.spawn(auxiliaryCommand.command, {
    cwd: PY_DIR,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      ...buildPythonModelProcessEnv(),
      PATH: `${path.dirname(pythonExec)}${path.delimiter}${process.env.PATH ?? ""}`,
    },
  });

  const { stderr } = await waitForAuxiliaryPythonProcess(proc, "复制融合模型产物", {
    terminationSignal,
    processGroup: auxiliaryCommand.processGroup,
  });

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
  terminationSignal?: AbortSignal,
) => {
  if (!fs.existsSync(sourceModelPath)) {
    throw new ModelNotFoundError("分阶段训练模型产物已丢失，请重新训练");
  }

  fs.mkdirSync(path.dirname(destinationModelPath), { recursive: true });

  if (modelType === "lstm") {
    await copyLstmModelArtifact(sourceModelPath, destinationModelPath, terminationSignal);
    return;
  }

  if (isEnsembleModelType(modelType)) {
    removeSiblingArtifactsByPrefix(destinationModelPath);
    await copyEnsembleModelArtifact(sourceModelPath, destinationModelPath, terminationSignal);
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

  await queryWithTimeout(pool, 
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

    const [rows] = await queryWithTimeout<RowDataPacket[]>(pool, 
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
  const [rows] = await queryWithTimeout<GuidedTrainingCleanupRow[]>(pool, 
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
    await queryWithTimeout(pool, 
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

export const getModelRuntimeRevision = (fresh = false) => {
  const configuredRevision = process.env.MODEL_RUNTIME_REVISION?.trim();
  if (configuredRevision) {
    return crypto.createHash("sha256").update(configuredRevision).digest("hex");
  }
  if (!fresh && cachedModelRuntimeRevision) return cachedModelRuntimeRevision;
  const runtimeFiles = [
    path.join(PY_DIR, "guided_model.py"),
    path.join(PY_DIR, "guided_ensemble_model.py"),
    path.join(PY_DIR, "run_model.py"),
    path.join(PY_DIR, "requirements.txt"),
    ...collectPythonRuntimeFiles(path.join(PY_DIR, "src")),
  ].filter((filePath) => fs.existsSync(filePath)).sort();
  const hash = crypto.createHash("sha256").update(MODEL_PIPELINE_REVISION);
  for (const filePath of runtimeFiles) {
    try {
      const contents = fs.readFileSync(filePath);
      hash.update(path.relative(PY_DIR, filePath));
      hash.update(contents);
    } catch {
      // A deployment may atomically replace files while the revision is first
      // calculated. Missing files are excluded and the cache is rebuilt on the
      // next process start.
    }
  }
  const revision = hash.digest("hex");
  if (!fresh) cachedModelRuntimeRevision = revision;
  return revision;
};

export const hashDatasetSnapshot = (datasetJson: string) =>
  crypto.createHash("sha256").update(datasetJson).digest("hex");

const assertRunRevisionMatches = (
  run: ModelTrainingRunRow,
  datasetRevision: string,
  runtimeRevision: string,
  label = run.model_type.toUpperCase(),
) => {
  if (typeof run.runtime_revision === "string" && run.runtime_revision !== runtimeRevision) {
    throw new ModelValidationError(`${label} 模型运行环境已变化，请重新训练该模型`);
  }
  if (typeof run.dataset_revision === "string" && run.dataset_revision !== datasetRevision) {
    throw new ModelValidationError(`${label} 模型对应的数据集版本已变化，请重新训练该模型`);
  }
};

const validateActiveRunAgainstSnapshot = (
  run: ModelTrainingRunRow | null,
  datasetJson: string,
  runtimeRevision = getModelRuntimeRevision(true),
) => {
  if (!run) return;
  assertRunRevisionMatches(run, hashDatasetSnapshot(datasetJson), runtimeRevision);
};

const writeGuidedDatasetSnapshot = async (artifactDir: string, datasetJson: string) => {
  await fs.promises.mkdir(artifactDir, { recursive: true });
  const targetPath = path.join(artifactDir, GUIDED_DATASET_FILE_NAME);
  const temporaryPath = `${targetPath}.tmp-${crypto.randomUUID()}`;
  await Bun.write(temporaryPath, datasetJson);
  fs.renameSync(temporaryPath, targetPath);
  return targetPath;
};

interface ArtifactValidationOptions {
  verifyArtifacts?: boolean;
  terminationSignal?: AbortSignal;
}

const buildBaseRunContext = async (
  experimentId: number,
  studentId: number,
  body: TrainInput,
  datasetRevision: string,
  runtimeRevision: string,
  options: ArtifactValidationOptions = {},
) => {
  const fingerprint: Record<string, string> = {};
  const artifacts: Partial<Record<BaseModelType, string>> = {};
  if (!body.models) return { fingerprint, artifacts };
  for (const baseModel of parseRequestedBaseModels(body.models)) {
    const activeRun = await requireActiveModelTrainingRun(experimentId, studentId, baseModel);
    if (!hasSavedModelArtifact(baseModel, activeRun.artifact_path)) {
      throw new ModelNotFoundError(
        `${baseModel.toUpperCase()} 模型的当前训练产物已丢失，请重新训练该模型`,
      );
    }
    assertRunRevisionMatches(activeRun, datasetRevision, runtimeRevision, baseModel.toUpperCase());
    if (options.verifyArtifacts !== false) {
      await verifyEnsembleBaseArtifactForRun(baseModel, activeRun, options.terminationSignal);
    }
    fingerprint[baseModel] = activeRun.run_id;
    artifacts[baseModel] = activeRun.artifact_path;
  }
  return { fingerprint, artifacts };
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
  if (dedupedModels.length !== rawModels.length) {
    throw new ModelValidationError("融合模型的基础模型不能重复");
  }
  const validModels = dedupedModels.filter(
    (model): model is BaseModelType => ["ma", "es", "arima", "lstm"].includes(model),
  );

  if (validModels.length !== dedupedModels.length || validModels.length < 2) {
    throw new ModelValidationError("融合模型至少需要选择两个有效且不重复的基础模型");
  }

  return validModels;
};

const BASE_MODEL_TIMEOUT_SECONDS: Record<BaseModelType, number> = {
  ma: 120,
  es: 120,
  arima: 120,
  lstm: 300,
};

export const resolveGuidedStepTimeoutMs = (
  modelType: GuidedModelType,
  stepId: string,
  body: TrainInput,
) => {
  if (!isEnsembleModelType(modelType) || !body.models) return TRAINING_TIMEOUT;
  const models = parseRequestedBaseModels(body.models);
  const sumBudgetSeconds = models.reduce(
    (sum, model) => sum + BASE_MODEL_TIMEOUT_SECONDS[model],
    30,
  );
  if (
    (modelType === "weighted_avg" && ["validation_predictions", "eval_predictions"].includes(stepId))
    || (modelType === "stacking" && ["level1_predictions", "eval_predictions"].includes(stepId))
    || (modelType === "boosting" && /^round_\d+_evaluate_candidates$/.test(stepId))
  ) {
    return Math.min(
      MODEL_EXECUTION_HARD_CAP,
      Math.max(TRAINING_TIMEOUT, sumBudgetSeconds * 1000),
    );
  }
  if (modelType === "boosting" && stepId === "retrain_chain") {
    const maxCandidateSeconds = Math.max(...models.map((model) => BASE_MODEL_TIMEOUT_SECONDS[model]));
    return Math.min(
      MODEL_EXECUTION_HARD_CAP,
      Math.max(TRAINING_TIMEOUT, (models.length * maxCandidateSeconds + 30) * 1000),
    );
  }
  return TRAINING_TIMEOUT;
};

export const resolveProductionPreparationTimeoutMs = (
  modelType: ModelType,
  activeRun: ModelTrainingRunRow,
) => {
  if (!isEnsembleModelType(modelType)) return TRAINING_TIMEOUT;
  const parameters = parseJsonColumn<Record<string, unknown>>(activeRun.parameters, {});
  const rawModels = Array.isArray(parameters.models)
    ? parameters.models.filter((model): model is BaseModelType => BASE_MODEL_TYPES.has(model as ModelType))
    : [];
  const models = rawModels.length > 0 ? rawModels : [...BASE_MODEL_TYPES] as BaseModelType[];
  if (modelType === "boosting") {
    const maxCandidateSeconds = Math.max(...models.map((model) => BASE_MODEL_TIMEOUT_SECONDS[model]));
    return Math.min(
      MODEL_EXECUTION_HARD_CAP,
      Math.max(TRAINING_TIMEOUT, (models.length * maxCandidateSeconds + 60) * 1000),
    );
  }
  const sumBudgetSeconds = models.reduce(
    (sum, model) => sum + BASE_MODEL_TIMEOUT_SECONDS[model],
    60,
  );
  return Math.min(
    MODEL_EXECUTION_HARD_CAP,
    Math.max(TRAINING_TIMEOUT, sumBudgetSeconds * 1000),
  );
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

export const computeEnsemblePhase1TrainSize = (trainSize: number) => {
  if (trainSize <= 15) {
    return Math.max(trainSize - 3, Math.trunc(trainSize * 0.85));
  }
  return Math.max(3, Math.trunc(trainSize * 0.8));
};

const buildEnsemblePythonParams = (
  experiment: ExperimentModelContextRow,
  body: TrainInput,
  trainingConfig: ExperimentTrainingConfig,
): Record<string, unknown> => {
  if (!body.models) {
    throw new ModelValidationError("融合模型缺少基础模型列表");
  }

  const requestedModels = parseRequestedBaseModels(body.models);
  const trainSize = trainingConfig.data_window_train_end_index
    - trainingConfig.data_window_train_start_index
    + 1;
  const internalTrainingSize = computeEnsemblePhase1TrainSize(trainSize);
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
        if (Number(experiment.moving_average_window) > internalTrainingSize) {
          throw new ModelValidationError(
            "移动平均时间窗口超过融合模型内部训练区间长度",
            {
              moving_average_window: Number(experiment.moving_average_window),
              internal_training_length: internalTrainingSize,
            },
          );
        }
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
        if (internalTrainingSize < Number(experiment.arima_d) + 2) {
          throw new ModelValidationError(
            "融合模型内部训练区间不足以拟合当前 ARIMA 差分阶数",
            {
              arima_d: Number(experiment.arima_d),
              internal_training_length: internalTrainingSize,
              minimum_training_length: Number(experiment.arima_d) + 2,
            },
          );
        }
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
          || !hasDefinedValue(experiment.lstm_features)
        ) {
          throw new ModelValidationError("融合模型依赖的 LSTM 基础模型缺少特征配置、目标字段或归一化配置");
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
  terminationSignal?: AbortSignal,
): Promise<Partial<Record<BaseModelType, string>>> => {
  const requestedModels = parseRequestedBaseModels(modelsValue);
  const availableModelFiles: Partial<Record<BaseModelType, string>> = {};

  for (const modelName of requestedModels) {
    const activeRun = await requireActiveModelTrainingRun(experimentId, studentId, modelName);
    const modelPath = activeRun.artifact_path;
    if (!hasSavedModelArtifact(modelName, modelPath)) {
      throw new ModelNotFoundError(
        `${modelName.toUpperCase()} 模型的当前训练产物已丢失，请重新训练该模型`,
      );
    }
    await verifyEnsembleBaseArtifactForRun(modelName, activeRun, terminationSignal);
    availableModelFiles[modelName] = modelPath;
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

  const trainingPoints = experiment.data_window_train_end_index
    - experiment.data_window_train_start_index
    + 1;
  if (trainingPoints < MIN_TRAINING_POINTS) {
    throw new ModelValidationError(
      `训练数据窗口至少需要 ${MIN_TRAINING_POINTS} 个数据点，请重新选择数据窗口后重试`,
      { training_points: trainingPoints, minimum_training_points: MIN_TRAINING_POINTS },
    );
  }

  const evaluationPoints = experiment.data_window_evaluate_end_index
    - experiment.data_window_evaluate_start_index
    + 1;
  if (evaluationPoints < MIN_EVALUATION_POINTS) {
    throw new ModelValidationError(
      `评估数据窗口至少需要 ${MIN_EVALUATION_POINTS} 个数据点，请重新选择数据窗口后重试`,
      { evaluation_points: evaluationPoints, minimum_evaluation_points: MIN_EVALUATION_POINTS },
    );
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
      if (
        !normalizeBooleanFlag(experiment.lstm_completed)
        || !hasDefinedValue(experiment.lstm_normalization)
        || !hasDefinedValue(experiment.lstm_target_field)
        || !hasDefinedValue(experiment.lstm_features)
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
        description: round === 1
          ? "用原始销量训练全部候选，将候选阶段输出截断为非负后比较验证RMSE；胜出者作为第一阶段，系数为1。"
          : `用当前有正有负的残差训练全部候选，保留其有符号输出，再比较乘以默认学习率0.3后降低验证残差RMSE的能力；候选类型可重复入选（第${round}轮）。`,
        actionLabel: "评估候选",
      },
      {
        id: `round_${round}_select_winner`,
        label: `第 ${round} 轮选择胜出模型`,
        description: "选择本轮使同一时间留出验证段 RMSE 最小且确有改善的模型加入链；若所有候选都没有改善，则提前停止。",
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
  const completedSet = new Set(completedStepIds);
  const isCompletedSession = row.status === "completed";
  const isTerminalSession = isCompletedSession || row.status === "superseded";
  const activeStepId = isTerminalSession ? null : row.current_step_id ?? nextStepId;

  return {
    session_id: row.session_id,
    experiment_id: row.experiment_id,
    model_type: row.model_type,
    status: row.status,
    current_step_id: activeStepId,
    next_step_id: isTerminalSession ? null : nextStepId,
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
    error_message: row.superseded_reason ?? row.error_message,
    error_code: row.status === "superseded" ? "GUIDED_SESSION_SUPERSEDED" : null,
    experiment_state_version: row.experiment_state_version === null
      ? null
      : Number(row.experiment_state_version),
  };
};

const buildRefreshedCompletedGuidedTrainingSessionPayload = async (
  row: GuidedTrainingSessionRow,
  studentId: number,
): Promise<GuidedTrainingSessionPayload> => {
  const payload = buildGuidedTrainingSessionPayload(row);
  if (row.status !== "completed" || !payload.result) {
    throw new ModelValidationError("已完成训练会话缺少有效结果，无法恢复");
  }

  const experimentStatePatch = await loadCurrentExperimentStatePatch(
    row.experiment_id,
    studentId,
  );
  const stateVersion = Number(experimentStatePatch.state_version);
  payload.result = {
    ...payload.result,
    experiment_state_patch: experimentStatePatch,
  };
  payload.experiment_state_version = stateVersion;
  return payload;
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

interface GuidedTrainingIdentity {
  requestHash: string;
  datasetRevision: string;
  runtimeRevision: string;
  baseRunFingerprint: Record<string, string>;
}

const computeCurrentGuidedTrainingIdentity = async (
  row: Pick<GuidedTrainingSessionRow, "experiment_id" | "model_type" | "request_payload">,
  studentId: number,
  options: ArtifactValidationOptions = {},
): Promise<GuidedTrainingIdentity> => {
  const body = parseJsonColumn<TrainInput>(row.request_payload, {} as TrainInput);
  const experiment = await assertExperimentModelAccess(row.experiment_id, studentId);
  assertTrainingRequestMatchesExperiment(experiment, body);
  const trainingConfig = getExperimentTrainingConfig(experiment);
  const canonicalBody = buildCanonicalTrainingBody(body, trainingConfig);
  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
  );
  const datasetRevision = hashDatasetSnapshot(JSON.stringify(allData));
  const runtimeRevision = getModelRuntimeRevision(true);
  const baseRunFingerprint = isEnsembleModelType(row.model_type)
    ? (await buildBaseRunContext(
      row.experiment_id,
      studentId,
      canonicalBody,
      datasetRevision,
      runtimeRevision,
      options,
    )).fingerprint
    : {};
  const experimentRevision = buildExperimentTrainingRevision(
    row.model_type,
    experiment,
    canonicalBody,
  );
  return {
    datasetRevision,
    runtimeRevision,
    baseRunFingerprint,
    requestHash: buildModelTrainingRequestHash({
      modelType: row.model_type,
      body: canonicalBody,
      datasetRevision,
      runtimeRevision,
      experimentRevision,
      baseRunFingerprint,
    }),
  };
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

const guidedRequestPayloadMatches = (
  row: Pick<GuidedTrainingSessionRow, "request_payload">,
  canonicalBody: TrainInput,
) => stableJsonStringify(parseJsonColumn<TrainInput>(row.request_payload, {} as TrainInput))
  === stableJsonStringify(canonicalBody);

const markGuidedTrainingSessionSuperseded = async (
  row: Pick<GuidedTrainingSessionRow, "session_id">,
  reason: string,
) => {
  await queryWithTimeout(pool, 
    `UPDATE model_guided_training_sessions
        SET status = 'superseded',
            current_step_id = NULL,
            active_attempt_id = NULL,
            superseded_reason = ?,
            error_message = NULL
      WHERE session_id = ?
        AND status <> 'superseded'`,
    [reason, row.session_id],
  );
};

const assertGuidedTrainingIdentityCurrent = async (
  row: Pick<
    GuidedTrainingSessionRow,
    "session_id" | "experiment_id" | "model_type" | "request_payload" | "request_hash" | "dataset_revision" | "runtime_revision"
  >,
  studentId: number,
  options: ArtifactValidationOptions = {},
) => {
  let current: GuidedTrainingIdentity;
  try {
    current = await computeCurrentGuidedTrainingIdentity(row, studentId, options);
  } catch (error) {
    if (
      !(error instanceof ModelAccessError)
      && !(error instanceof ModelValidationError)
      && !(error instanceof ModelNotFoundError)
      && !(error instanceof ModelConflictError)
    ) {
      throw error;
    }
    const reason = error instanceof Error
      ? `训练会话依赖已变化：${error.message}`
      : "训练会话依赖已变化";
    await markGuidedTrainingSessionSuperseded(row, reason);
    throw new ModelConflictError(reason, "GUIDED_SESSION_SUPERSEDED");
  }

  if (
    current.requestHash !== row.request_hash
    || current.datasetRevision !== row.dataset_revision
    || current.runtimeRevision !== row.runtime_revision
  ) {
    const reason = "训练数据、实验配置、基础模型或运行环境已变化，请重新创建训练会话";
    await markGuidedTrainingSessionSuperseded(row, reason);
    throw new ModelConflictError(reason, "GUIDED_SESSION_SUPERSEDED");
  }
  return current;
};

const assertCompletedGuidedTrainingRunCurrent = async (
  row: Pick<
    GuidedTrainingSessionRow,
    "session_id" | "experiment_id" | "student_id" | "model_type" | "request_payload" | "request_hash" | "dataset_revision" | "runtime_revision"
  >,
  studentId: number,
  options: ArtifactValidationOptions = {},
) => {
  try {
    const current = await computeCurrentGuidedTrainingIdentity(row, studentId, options);
    if (
      current.datasetRevision !== row.dataset_revision
      || current.runtimeRevision !== row.runtime_revision
    ) {
      throw new ModelValidationError("训练数据或运行环境已变化");
    }

    const activeRun = await getActiveModelTrainingRun(
      row.experiment_id,
      studentId,
      row.model_type,
    );
    if (
      !activeRun
      || activeRun.session_id !== row.session_id
      || activeRun.request_hash !== row.request_hash
      || activeRun.dataset_revision !== row.dataset_revision
      || activeRun.runtime_revision !== row.runtime_revision
    ) {
      throw new ModelValidationError("已完成会话不再对应当前生效的训练版本");
    }

    if (options.verifyArtifacts === false) {
      if (!hasSavedModelArtifact(row.model_type, activeRun.artifact_path)) {
        throw new ModelNotFoundError("已完成训练版本的模型产物已丢失");
      }
    } else {
      await verifyBacktestBaseArtifactForRun(
        row.model_type,
        activeRun,
        options.terminationSignal,
      );
    }

    if (isEnsembleModelType(row.model_type)) {
      const body = parseJsonColumn<TrainInput>(row.request_payload, {} as TrainInput);
      const requestedModels = parseRequestedBaseModels(body.models ?? "");
      const [dependencyRows] = await queryWithTimeout<ModelTrainingDependencyRow[]>(pool, 
        `SELECT base_model_type, base_run_id
           FROM model_training_run_dependencies
          WHERE run_id = ?`,
        [activeRun.run_id],
      );
      const dependencyMap = new Map(
        dependencyRows.map((dependency) => [dependency.base_model_type, dependency.base_run_id]),
      );
      if (
        dependencyRows.length !== requestedModels.length
        || requestedModels.some((modelName) => (
          !dependencyMap.has(modelName)
          || dependencyMap.get(modelName) !== current.baseRunFingerprint[modelName]
        ))
      ) {
        throw new ModelValidationError("融合模型依赖的基础训练版本已变化");
      }
    }

    return activeRun;
  } catch (error) {
    if (
      !(error instanceof ModelAccessError)
      && !(error instanceof ModelValidationError)
      && !(error instanceof ModelNotFoundError)
      && !(error instanceof ModelArtifactError)
      && !(error instanceof ModelConflictError)
    ) {
      throw error;
    }
    const reason = error instanceof Error
      ? `已完成训练会话依赖已变化：${error.message}`
      : "已完成训练会话依赖已变化";
    await markGuidedTrainingSessionSuperseded(row, reason);
    throw new ModelConflictError(reason, "GUIDED_SESSION_SUPERSEDED");
  }
};

const fetchGuidedTrainingSessionRow = async (
  sessionId: string,
): Promise<GuidedTrainingSessionRow | null> => {
  const [rows] = await queryWithTimeout<GuidedTrainingSessionRow[]>(pool, 
    `SELECT session_id,
            experiment_id,
            student_id,
            model_type,
            request_hash,
            dataset_revision,
            runtime_revision,
            protocol_version,
            request_payload,
            status,
            current_step_id,
            completed_step_ids,
            step_outputs,
            step_plan,
            result_payload,
            artifact_dir,
            committed_state_path,
            committed_state_sha256,
            active_attempt_id,
            error_message,
            superseded_reason,
            experiment_state_version,
            updated_at
       FROM model_guided_training_sessions
      WHERE session_id = ?
      LIMIT 1`,
    [sessionId],
  );

  return rows[0] ?? null;
};

const sha256File = (filePath: string) => {
  const digest = crypto.createHash("sha256");
  const file = fs.openSync(filePath, "r");
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  try {
    while (true) {
      const bytesRead = fs.readSync(file, buffer, 0, buffer.length, null);
      if (bytesRead === 0) break;
      digest.update(buffer.subarray(0, bytesRead));
    }
  } finally {
    fs.closeSync(file);
  }
  return digest.digest("hex");
};

const parseGuidedAttemptReceipt = (
  attempt: GuidedTrainingStepAttemptRow,
): Record<string, unknown> | null => {
  if (!fs.existsSync(attempt.receipt_path) || !fs.existsSync(attempt.candidate_state_path)) {
    return null;
  }
  let receipt: Record<string, unknown>;
  try {
    receipt = JSON.parse(fs.readFileSync(attempt.receipt_path, "utf8")) as Record<string, unknown>;
  } catch {
    return null;
  }
  if (
    receipt.protocol_version !== 2
    || receipt.attempt_id !== attempt.attempt_id
    || receipt.step_id !== attempt.step_id
    || receipt.candidate_state_path !== attempt.candidate_state_path
  ) {
    return null;
  }
  const candidateSha256 = sha256File(attempt.candidate_state_path);
  if (receipt.candidate_state_sha256 !== candidateSha256) {
    return null;
  }
  if (attempt.input_state_path) {
    if (!fs.existsSync(attempt.input_state_path)) return null;
    const inputSha256 = sha256File(attempt.input_state_path);
    if (receipt.input_state_sha256 !== inputSha256 || attempt.input_state_sha256 !== inputSha256) {
      return null;
    }
  }
  return receipt;
};

const cleanupIncompleteGuidedAttemptArtifacts = (
  attempt: GuidedTrainingStepAttemptRow,
) => {
  try {
    if (parseGuidedAttemptReceipt(attempt)) return;
  } catch (error) {
    logger.warn(
      { err: error, attemptId: attempt.attempt_id },
      "校验超时 guided attempt 工件失败，将按未完成工件清理",
    );
  }

  for (const filePath of [attempt.candidate_state_path, attempt.receipt_path]) {
    try {
      fs.rmSync(filePath, { force: true });
    } catch (error) {
      logger.warn(
        { err: error, attemptId: attempt.attempt_id, filePath },
        "清理超时 guided attempt 工件失败",
      );
    }
  }
};

const fetchGuidedAttempt = async (
  connection: PoolConnection,
  attemptId: string,
) => {
  const [rows] = await connection.query<GuidedTrainingStepAttemptRow[]>(
    `SELECT *
       FROM model_guided_training_step_attempts
      WHERE attempt_id = ?
      LIMIT 1`,
    [attemptId],
  );
  return rows[0] ?? null;
};

const beginGuidedStepAttempt = async (
  row: GuidedTrainingSessionRow,
  stepId: string,
  timeoutMs: number,
) => {
  const connection = await acquireConnectionWithTimeout(pool);
  try {
    await connection.beginTransaction();
    const [sessionRows] = await connection.query<GuidedTrainingSessionRow[]>(
      `SELECT *
         FROM model_guided_training_sessions
        WHERE session_id = ?
        LIMIT 1
        FOR UPDATE`,
      [row.session_id],
    );
    const session = sessionRows[0];
    if (!session) throw new ModelNotFoundError("分阶段训练会话已丢失");
    if (session.status === "superseded") {
      throw new ModelConflictError(
        session.superseded_reason ?? "训练会话已失效，请重新创建",
        "GUIDED_SESSION_SUPERSEDED",
      );
    }

    if (session.active_attempt_id) {
      const activeAttempt = await fetchGuidedAttempt(connection, session.active_attempt_id);
      if (activeAttempt?.step_id === stepId) {
        const receipt = parseGuidedAttemptReceipt(activeAttempt);
        if (receipt) {
          await connection.query(
            `UPDATE model_guided_training_sessions
                SET status = 'running', error_message = NULL
              WHERE session_id = ?`,
            [row.session_id],
          );
          await connection.commit();
          return { attempt: activeAttempt, receipt, recovered: true };
        }
        const leaseExpiresAt = activeAttempt.lease_expires_at
          ? new Date(activeAttempt.lease_expires_at).getTime()
          : 0;
        if (activeAttempt.status === "running" && leaseExpiresAt > Date.now()) {
          throw new ModelConflictError(
            "当前训练步骤仍在执行",
            "GUIDED_STEP_IN_PROGRESS",
          );
        }
        await connection.query(
          `UPDATE model_guided_training_step_attempts
              SET status = 'failed', error_message = COALESCE(error_message, 'Attempt lease expired without a valid receipt')
            WHERE attempt_id = ?`,
          [activeAttempt.attempt_id],
        );
      }
    }

    const attemptId = crypto.randomUUID();
    const { candidateStatePath, receiptPath } = buildGuidedTrainingAttemptPaths(
      row.artifact_dir,
      stepId,
      attemptId,
    );
    const inputStatePath = session.committed_state_path;
    const inputStateSha256 = inputStatePath && fs.existsSync(inputStatePath)
      ? sha256File(inputStatePath)
      : null;
    const leaseSeconds = Math.ceil((timeoutMs + 60_000) / 1000);
    await connection.query(
      `INSERT INTO model_guided_training_step_attempts
       (attempt_id, session_id, step_id, status, input_state_path, input_state_sha256,
        candidate_state_path, receipt_path, lease_expires_at)
       VALUES (?, ?, ?, 'running', ?, ?, ?, ?, DATE_ADD(NOW(), INTERVAL ? SECOND))`,
      [
        attemptId,
        row.session_id,
        stepId,
        inputStatePath,
        inputStateSha256,
        candidateStatePath,
        receiptPath,
        leaseSeconds,
      ],
    );
    const [updateResult] = await connection.query(
      `UPDATE model_guided_training_sessions
          SET status = 'running',
              current_step_id = ?,
              active_attempt_id = ?,
              error_message = NULL,
              superseded_reason = NULL
        WHERE session_id = ?
          AND status IN ('ready', 'failed', 'running')`,
      [stepId, attemptId, row.session_id],
    );
    if (Number((updateResult as { affectedRows?: number }).affectedRows ?? 0) !== 1) {
      throw new ModelConflictError("训练会话状态已变化", "GUIDED_STEP_IN_PROGRESS");
    }
    await connection.commit();
    const attempt = await fetchGuidedAttempt(connection, attemptId);
    if (!attempt) throw new Error("创建训练步骤 attempt 失败");
    return { attempt, receipt: null, recovered: false };
  } catch (error) {
    await connection.rollback().catch(() => {});
    throw error;
  } finally {
    connection.release();
  }
};

const commitGuidedIntermediateStep = async (input: {
  sessionId: string;
  attempt: GuidedTrainingStepAttemptRow;
  status: GuidedTrainingStatus;
  currentStepId: string | null;
  completedStepIds: string[];
  stepOutputs: Record<string, unknown>;
  stepPlan: GuidedTrainingStepDefinition[];
  receipt: Record<string, unknown>;
}) => {
  const connection = await acquireConnectionWithTimeout(pool);
  try {
    await connection.beginTransaction();
    const [sessionRows] = await connection.query<RowDataPacket[]>(
      `SELECT active_attempt_id, status
         FROM model_guided_training_sessions
        WHERE session_id = ?
        LIMIT 1
        FOR UPDATE`,
      [input.sessionId],
    );
    const session = sessionRows[0];
    if (!session || session.active_attempt_id !== input.attempt.attempt_id) {
      throw new ModelConflictError("训练步骤 attempt 已被替换", "GUIDED_STEP_IN_PROGRESS");
    }
    if (session.status === "superseded") {
      throw new ModelConflictError("训练会话已失效", "GUIDED_SESSION_SUPERSEDED");
    }

    await connection.query(
      `UPDATE model_guided_training_step_attempts
          SET status = 'committed',
              candidate_state_sha256 = ?,
              output_payload = ?,
              final_result_payload = ?,
              error_message = NULL
        WHERE attempt_id = ?`,
      [
        input.receipt.candidate_state_sha256,
        JSON.stringify(input.receipt.output ?? {}),
        input.receipt.final_result === undefined
          ? null
          : JSON.stringify(input.receipt.final_result),
        input.attempt.attempt_id,
      ],
    );
    await connection.query(
      `UPDATE model_guided_training_sessions
          SET status = ?,
              current_step_id = ?,
              completed_step_ids = ?,
              step_outputs = ?,
              step_plan = ?,
              result_payload = NULL,
              committed_state_path = ?,
              committed_state_sha256 = ?,
              active_attempt_id = NULL,
              error_message = NULL
        WHERE session_id = ?`,
      [
        input.status,
        input.currentStepId,
        JSON.stringify(input.completedStepIds),
        JSON.stringify(input.stepOutputs),
        JSON.stringify(input.stepPlan),
        input.attempt.candidate_state_path,
        input.receipt.candidate_state_sha256,
        input.sessionId,
      ],
    );
    await connection.commit();
  } catch (error) {
    await connection.rollback().catch(() => {});
    throw error;
  } finally {
    connection.release();
  }
};

const recordGuidedAttemptPythonCompleted = async (
  attempt: GuidedTrainingStepAttemptRow,
  receipt: Record<string, unknown>,
) => {
  await queryWithTimeout(pool, 
    `UPDATE model_guided_training_step_attempts
        SET status = 'python_completed',
            candidate_state_sha256 = ?,
            output_payload = ?,
            final_result_payload = ?,
            error_message = NULL
      WHERE attempt_id = ?
        AND status IN ('running', 'python_completed')`,
    [
      receipt.candidate_state_sha256,
      JSON.stringify(receipt.output ?? {}),
      receipt.final_result === undefined ? null : JSON.stringify(receipt.final_result),
      attempt.attempt_id,
    ],
  );
};

const recordGuidedAttemptFailure = async (
  sessionId: string,
  attemptId: string,
  stepId: string,
  error: unknown,
) => {
  const detail = error instanceof Error ? error.message : "分阶段训练执行失败";
  try {
    await withBoundedPoolTransaction(
      pool,
      TIMEOUTS.MODEL_FAILURE_RECORD,
      async (connection) => {
        await connection.query(
          `UPDATE model_guided_training_step_attempts
              SET status = IF(status = 'python_completed', status, 'failed'),
                  error_message = ?
            WHERE attempt_id = ?
              AND status IN ('running', 'python_completed')`,
          [detail, attemptId],
        );
        await connection.query(
          `UPDATE model_guided_training_sessions
              SET status = 'failed',
                  current_step_id = ?,
                  error_message = ?
            WHERE session_id = ?
              AND active_attempt_id = ?
              AND status = 'running'`,
          [stepId, detail, sessionId, attemptId],
        );
      },
    );
  } catch (recordError) {
    logger.warn(
      {
        err: recordError,
        sessionId,
        attemptId,
        timeoutMs: TIMEOUTS.MODEL_FAILURE_RECORD,
      },
      "记录 guided attempt 失败状态失败",
    );
  }
};

const buildGuidedTrainingPythonParams = async (
  modelType: GuidedModelType,
  studentId: number,
  body: TrainInput,
  artifactDir: string,
  terminationSignal?: AbortSignal,
): Promise<Record<string, unknown>> => {
  const experiment = await assertExperimentModelAccess(body.experiment_id, studentId);
  assertTrainingRequestMatchesExperiment(experiment, body);
  const trainingConfig = getExperimentTrainingConfig(experiment);
  const canonicalBody = buildCanonicalTrainingBody(body, trainingConfig);
  if (modelType === "ma") {
    validateMovingAverageWindow(trainingConfig, canonicalBody.moving_average_window);
  }
  const derivedModelParams: Record<string, unknown> = isEnsembleModelType(modelType)
    ? buildEnsemblePythonParams(experiment, body, trainingConfig)
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
    const baseModelArtifactMap = await buildAvailableBaseModelArtifactMap(
      body.experiment_id,
      studentId,
      body.models,
      terminationSignal,
    );
    if (Object.keys(baseModelArtifactMap).length > 0) {
      pythonParams.baseModelFiles = JSON.stringify(baseModelArtifactMap);
    }
  }

  return pythonParams;
};

const isDuplicateGuidedTrainingRequestError = (error: unknown) => {
  if (!error || typeof error !== "object") return false;
  const candidate = error as { code?: string; errno?: number };
  return candidate.code === "ER_DUP_ENTRY" || candidate.errno === 1062;
};

const createOrResumeGuidedTrainingSessionInternal = async (
  modelType: ModelType,
  studentId: number,
  body: TrainInput,
  operationContext: ModelOperationContext,
  terminationSignal: AbortSignal,
): Promise<GuidedTrainingSessionPayload> => {
  const experiment = await assertExperimentModelAccess(body.experiment_id, studentId);
  operationContext.throwIfTerminated();
  assertTrainingRequestMatchesExperiment(experiment, body);
  const trainingConfig = getExperimentTrainingConfig(experiment);
  const canonicalBody = buildCanonicalTrainingBody(body, trainingConfig);
  if (modelType === "ma") {
    validateMovingAverageWindow(trainingConfig, canonicalBody.moving_average_window);
  }
  if (isEnsembleModelType(modelType)) {
    buildEnsemblePythonParams(experiment, canonicalBody, trainingConfig);
  }

  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
  );
  operationContext.throwIfTerminated();
  if (
    trainingConfig.data_window_train_end_index >= allData.length
    || trainingConfig.data_window_evaluate_end_index >= allData.length
  ) {
    throw new ModelValidationError("数据窗口索引超出范围", { dataset_size: allData.length });
  }
  const datasetJson = JSON.stringify(allData);
  const datasetRevision = hashDatasetSnapshot(datasetJson);
  const runtimeRevision = getModelRuntimeRevision(true);
  const baseRunFingerprint = isEnsembleModelType(modelType)
    ? (await buildBaseRunContext(
      canonicalBody.experiment_id,
      studentId,
      canonicalBody,
      datasetRevision,
      runtimeRevision,
      { terminationSignal },
    )).fingerprint
    : {};
  operationContext.throwIfTerminated();
  const experimentRevision = buildExperimentTrainingRevision(modelType, experiment, canonicalBody);
  const requestHash = buildModelTrainingRequestHash({
    modelType,
    body: canonicalBody,
    datasetRevision,
    runtimeRevision,
    experimentRevision,
    baseRunFingerprint,
  });
  const [existingRows] = await queryWithTimeout<GuidedTrainingSessionRow[]>(pool, 
    `SELECT session_id,
            experiment_id,
            student_id,
            model_type,
            request_hash,
            dataset_revision,
            runtime_revision,
            protocol_version,
            request_payload,
            status,
            current_step_id,
            completed_step_ids,
            step_outputs,
            step_plan,
            result_payload,
            artifact_dir,
            committed_state_path,
            committed_state_sha256,
            active_attempt_id,
            error_message,
            superseded_reason,
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
  operationContext.throwIfTerminated();

  const existingSession = existingRows[0];
  if (existingSession) {
    if (existingSession.status === "superseded" || existingSession.protocol_version !== 2) {
      operationContext.throwIfTerminated();
      const deletedSession = await deleteGuidedTrainingSession(existingSession);
      operationContext.throwIfTerminated();
      if (!deletedSession) {
        throw new ModelValidationError("旧的分阶段训练会话清理失败，请稍后重试");
      }
    } else if (existingSession.status === "completed") {
      try {
        await assertCompletedGuidedTrainingRunCurrent(existingSession, studentId, { terminationSignal });
        return await buildRefreshedCompletedGuidedTrainingSessionPayload(existingSession, studentId);
      } catch (error) {
        if (!(error instanceof ModelConflictError) || error.code !== "GUIDED_SESSION_SUPERSEDED") {
          throw error;
        }

        logger.warn(
          {
            sessionId: existingSession.session_id,
            artifactDir: existingSession.artifact_dir,
            modelType,
          },
          "已完成 guided training 会话不再对应当前训练版本，将删除会话并重新创建",
        );
        operationContext.throwIfTerminated();
        const deletedSession = await deleteGuidedTrainingSession(existingSession);
        operationContext.throwIfTerminated();
        if (!deletedSession) {
          throw new ModelValidationError("旧的分阶段训练会话工件清理失败，请稍后重试");
        }
      }
    } else {
      const datasetFile = path.join(existingSession.artifact_dir, GUIDED_DATASET_FILE_NAME);
      if (!fs.existsSync(datasetFile)) {
        operationContext.throwIfTerminated();
        await writeGuidedDatasetSnapshot(existingSession.artifact_dir, datasetJson);
        operationContext.throwIfTerminated();
      }
      operationContext.throwIfTerminated();
      return buildGuidedTrainingSessionPayload(existingSession);
    }
  }

  if (!existingSession) {
    const activeRun = await getActiveModelTrainingRun(
      canonicalBody.experiment_id,
      studentId,
      modelType,
    );
    operationContext.throwIfTerminated();
    if (activeRun?.session_id) {
      const completedSession = await fetchGuidedTrainingSessionRow(activeRun.session_id);
      operationContext.throwIfTerminated();
      if (
        completedSession
        && completedSession.status === "completed"
        && completedSession.protocol_version === 2
        && completedSession.model_type === modelType
        && completedSession.student_id === studentId
        && guidedRequestPayloadMatches(completedSession, canonicalBody)
      ) {
        try {
          await assertCompletedGuidedTrainingRunCurrent(completedSession, studentId, { terminationSignal });
          return await buildRefreshedCompletedGuidedTrainingSessionPayload(completedSession, studentId);
        } catch (error) {
          if (!(error instanceof ModelConflictError) || error.code !== "GUIDED_SESSION_SUPERSEDED") {
            throw error;
          }
          operationContext.throwIfTerminated();
          const deletedSession = await deleteGuidedTrainingSession(completedSession);
          operationContext.throwIfTerminated();
          if (!deletedSession) {
            throw new ModelValidationError("失效的分阶段训练会话清理失败，请稍后重试");
          }
        }
      }
    }
  }

  const sessionId = crypto.randomUUID();
  const artifactDir = buildGuidedTrainingArtifactDir(canonicalBody.experiment_id, studentId, modelType, sessionId);
  const completedStepIds: string[] = [];
  const stepPlan = buildGuidedTrainingStepPlan(modelType, canonicalBody);
  const currentStepId = getGuidedNextStepId(stepPlan, completedStepIds);
  let insertAttempted = false;
  try {
    operationContext.throwIfTerminated();
    fs.mkdirSync(artifactDir, { recursive: true });
    await writeGuidedDatasetSnapshot(artifactDir, datasetJson);
    operationContext.throwIfTerminated();
    insertAttempted = true;
    await queryWithTimeout(pool, 
      `INSERT INTO model_guided_training_sessions
       (session_id,
        experiment_id,
        student_id,
        model_type,
        request_hash,
        dataset_revision,
        runtime_revision,
        protocol_version,
        request_payload,
        status,
        current_step_id,
        completed_step_ids,
        step_outputs,
        step_plan,
        result_payload,
        artifact_dir,
        expires_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, 2, ?, 'ready', ?, ?, ?, ?, NULL, ?, DATE_ADD(NOW(), INTERVAL 7 DAY))`,
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
    operationContext.throwIfTerminated();
  } catch (error) {
    if (!isDuplicateGuidedTrainingRequestError(error)) {
      // If no INSERT was sent, this directory cannot be referenced by a DB
      // row. Once INSERT starts its outcome can be ambiguous on connection
      // errors, so leave cleanup to the stale-artifact sweeper.
      if (!insertAttempted) {
        fs.rmSync(artifactDir, { recursive: true, force: true });
      }
      throw error;
    }
    fs.rmSync(artifactDir, { recursive: true, force: true });
    operationContext.throwIfTerminated();
    const [racedRows] = await queryWithTimeout<GuidedTrainingSessionRow[]>(pool, 
      `SELECT *
         FROM model_guided_training_sessions
        WHERE experiment_id = ?
          AND student_id = ?
          AND model_type = ?
          AND request_hash = ?
        LIMIT 1`,
      [canonicalBody.experiment_id, studentId, modelType, requestHash],
    );
    operationContext.throwIfTerminated();
    if (racedRows[0]) {
      if (racedRows[0].status === "completed") {
        await assertCompletedGuidedTrainingRunCurrent(racedRows[0], studentId, { terminationSignal });
        return await buildRefreshedCompletedGuidedTrainingSessionPayload(racedRows[0], studentId);
      }
      return buildGuidedTrainingSessionPayload(racedRows[0]);
    }
    throw error;
  }

  operationContext.throwIfTerminated();
  const row = await fetchGuidedTrainingSessionRow(sessionId);
  operationContext.throwIfTerminated();
  if (!row) {
    throw new Error("创建分阶段训练会话失败");
  }

  return buildGuidedTrainingSessionPayload(row);
};

export async function createOrResumeGuidedTrainingSession(
  modelType: ModelType,
  studentId: number,
  body: TrainInput,
): Promise<GuidedTrainingSessionPayload> {
  if (!isGuidedModelType(modelType)) {
    throw new ModelValidationError("当前模型暂不支持分阶段训练");
  }

  return await withModelOperationContext("guided_training_session", async (operationContext) => {
    return await withModelOperationLease(
      operationContext,
      { mode: "slot_and_execution", experimentId: body.experiment_id, modelType },
      async (lease) => await createOrResumeGuidedTrainingSessionInternal(
        modelType,
        studentId,
        body,
        operationContext,
        lease.terminationSignal,
      ),
    );
  });
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
  if (row.status !== "superseded") {
    try {
      // Keep GET metadata reads free of artifact hashing and Python validation.
      // The leased create/resume and step endpoints perform those checks.
      if (row.status === "completed") {
        await assertCompletedGuidedTrainingRunCurrent(row, studentId, { verifyArtifacts: false });
        return await buildRefreshedCompletedGuidedTrainingSessionPayload(row, studentId);
      } else {
        await assertGuidedTrainingIdentityCurrent(row, studentId, { verifyArtifacts: false });
      }
    } catch (error) {
      if (!(error instanceof ModelConflictError) || error.code !== "GUIDED_SESSION_SUPERSEDED") {
        throw error;
      }
      const supersededRow = await fetchGuidedTrainingSessionRow(sessionId);
      if (!supersededRow) throw new ModelNotFoundError("未找到分阶段训练会话");
      return buildGuidedTrainingSessionPayload(supersededRow);
    }
  }
  return buildGuidedTrainingSessionPayload(row);
}

export async function discardGuidedTrainingSession(
  modelType: ModelType,
  studentId: number,
  sessionId: string,
) {
  if (!isGuidedModelType(modelType)) {
    throw new ModelValidationError("当前模型不支持分阶段训练");
  }
  return await withModelOperationContext("guided_training_discard", async (operationContext) => {
    const row = await fetchGuidedTrainingSessionRow(sessionId);
    operationContext.throwIfTerminated();
    if (!row || row.model_type !== modelType) {
      throw new ModelNotFoundError("未找到分阶段训练会话");
    }
    if (row.student_id !== studentId) {
      throw new ModelAccessError("权限不足：该训练会话不属于当前用户", 403);
    }
    if (row.status === "running" || row.status === "completed") {
      throw new ModelConflictError(
        row.status === "running" ? "运行中的训练会话不能删除" : "已完成的训练会话不能删除",
        "GUIDED_SESSION_NOT_DISCARDABLE",
      );
    }

    return await withModelOperationLease(
      operationContext,
      { mode: "execution_only", experimentId: row.experiment_id, modelType },
      async (lease) => {
        await lease.verifyOwned();
        const [result] = await queryWithTimeout(pool, 
          `DELETE FROM model_guided_training_sessions
            WHERE session_id = ?
              AND student_id = ?
              AND status IN ('ready', 'failed', 'superseded')`,
          [sessionId, studentId],
        );
        if (Number((result as { affectedRows?: number }).affectedRows ?? 0) !== 1) {
          throw new ModelConflictError("训练会话状态已变化，无法删除", "GUIDED_SESSION_NOT_DISCARDABLE");
        }
        try {
          removeGuidedTrainingArtifactDir(row.artifact_dir);
        } catch (error) {
          logger.warn({ err: error, sessionId, artifactDir: row.artifact_dir }, "删除会话后清理 guided 工件失败");
        }
        return { status: "success" as const };
      },
    );
  });
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

  const activeAttemptRef: { current: GuidedTrainingStepAttemptRow | null } = { current: null };
  let stepCommitted = false;

  try {
    return await withModelOperationContext("guided_training_step", async (operationContext) => {
      const row = await fetchGuidedTrainingSessionRow(sessionId);
      operationContext.throwIfTerminated();
      if (!row || row.model_type !== modelType) {
        throw new ModelNotFoundError("未找到分阶段训练会话");
      }
      if (row.student_id !== studentId) {
        throw new ModelAccessError("权限不足：该训练会话不属于当前用户", 403);
      }

      if (row.status === "completed") {
        return await withModelOperationLease(
          operationContext,
          { mode: "slot_and_execution", experimentId: row.experiment_id, modelType },
          async (lease) => {
            await assertCompletedGuidedTrainingRunCurrent(row, studentId, {
              terminationSignal: lease.terminationSignal,
            });
            return await buildRefreshedCompletedGuidedTrainingSessionPayload(row, studentId);
          },
        );
      }
      if (row.status === "superseded" || row.protocol_version !== 2) {
        if (row.status !== "superseded") {
          await markGuidedTrainingSessionSuperseded(row, "训练会话协议已升级，请重新创建训练会话");
        }
        throw new ModelConflictError(
          row.superseded_reason ?? "训练会话已失效，请重新创建训练会话",
          "GUIDED_SESSION_SUPERSEDED",
        );
      }

      const body = parseJsonColumn<TrainInput>(row.request_payload, {} as TrainInput);
      const evaluationStart = Number(body.data_window_evaluate_start_index);
      const evaluationEnd = Number(body.data_window_evaluate_end_index);
      const expectedEvaluationLength = evaluationEnd - evaluationStart + 1;
      if (
        !Number.isSafeInteger(evaluationStart)
        || !Number.isSafeInteger(evaluationEnd)
        || !Number.isSafeInteger(expectedEvaluationLength)
        || expectedEvaluationLength <= 0
      ) {
        throw new ModelValidationError("训练会话缺少有效的评估区间");
      }
      const completedStepIds = parseJsonColumn<string[]>(row.completed_step_ids, []);
      if (completedStepIds.includes(stepId)) return buildGuidedTrainingSessionPayload(row);

      const stepPlan = getGuidedTrainingSteps(row);
      const nextStepId = getGuidedNextStepId(stepPlan, completedStepIds);
      if (stepId !== nextStepId) {
        throw new ModelValidationError("只能按训练流程顺序执行下一个阶段");
      }

      return await withModelOperationLease(
        operationContext,
        { mode: "slot_and_execution", experimentId: row.experiment_id, modelType },
        async (lease) => {
          await assertGuidedTrainingIdentityCurrent(row, studentId, {
            terminationSignal: lease.terminationSignal,
          });
          const stepTimeoutMs = resolveGuidedStepTimeoutMs(modelType, stepId, body);
          const attemptContext = await beginGuidedStepAttempt(row, stepId, stepTimeoutMs);
          const activeAttempt = attemptContext.attempt;
          activeAttemptRef.current = activeAttempt;
          let receipt = attemptContext.receipt;

          if (!receipt) {
            const sessionModelPath = getGuidedSessionModelPath(row.artifact_dir, modelType);
            const pythonParams = await buildGuidedTrainingPythonParams(
              modelType,
              studentId,
              body,
              row.artifact_dir,
              lease.terminationSignal,
            );
            pythonParams.saveModel = sessionModelPath;
            await runPythonGuidedTrainingStep(
              modelType,
              stepId,
              pythonParams,
              activeAttempt.attempt_id,
              activeAttempt.input_state_path,
              activeAttempt.candidate_state_path,
              activeAttempt.receipt_path,
              row.artifact_dir,
              stepTimeoutMs,
              lease.terminationSignal,
            );
            receipt = parseGuidedAttemptReceipt(activeAttempt);
            if (!receipt) throw new Error("Python 步骤已结束但未生成有效的 checkpoint receipt");
          }

          await lease.verifyOwned();
          await recordGuidedAttemptPythonCompleted(activeAttempt, receipt);
            await assertGuidedTrainingIdentityCurrent(row, studentId, {
              terminationSignal: lease.terminationSignal,
            });

          const nextCompletedStepIds = [...completedStepIds, stepId];
          const stepOutputs = parseJsonColumn<Record<string, unknown>>(row.step_outputs, {});
          const nextStepOutputs = { ...stepOutputs, [stepId]: receipt.output ?? {} };
          const nextStepPlan = shouldTrimBoostingRoundSteps(modelType, stepId, receipt.output)
            ? trimBoostingStepPlanAfterEarlyStop(stepPlan, stepId)
            : stepPlan;

          let cleanupUncommittedArtifactPath: string | null = null;
          try {
            let transformedResult: ReturnType<typeof transformTrainingResult> | null = null;
            let versionedArtifactPath: string | null = null;
            let artifactSha256: string | null = null;
            const receiptFinalResult = isRecordValue(receipt.final_result) ? receipt.final_result : null;
            if (receiptFinalResult) {
              const sourceModelPath = getGuidedSessionModelPath(row.artifact_dir, row.model_type);
              versionedArtifactPath = buildVersionedModelPath(
                row.experiment_id,
                row.student_id,
                row.model_type,
                row.session_id,
                "backtest",
              );
              let versionedArtifactWasReusable: boolean;
              if (hasSavedModelArtifact(row.model_type, versionedArtifactPath)) {
                try {
                  const integrity = await verifyBaseModelArtifactIntegrity(
                    row.model_type,
                    versionedArtifactPath,
                    null,
                    lease.terminationSignal,
                  );
                  artifactSha256 = integrity.sha256;
                  versionedArtifactWasReusable = true;
                } catch (error) {
                  if (!(error instanceof ModelArtifactError)) throw error;
                  removeModelArtifactGroup(versionedArtifactPath);
                  versionedArtifactWasReusable = false;
                }
              } else {
                versionedArtifactWasReusable = false;
              }
              if (!versionedArtifactWasReusable) {
                await lease.verifyOwned();
                cleanupUncommittedArtifactPath = versionedArtifactPath;
                await copyModelArtifact(
                  row.model_type,
                  sourceModelPath,
                  versionedArtifactPath,
                  lease.terminationSignal,
                );
                await lease.verifyOwned();
              }
              if (!artifactSha256) {
                artifactSha256 = (
                  await verifyBaseModelArtifactIntegrity(
                    row.model_type,
                    versionedArtifactPath,
                    null,
                    lease.terminationSignal,
                  )
                ).sha256;
                await lease.verifyOwned();
              }
              receiptFinalResult.saved_model = versionedArtifactPath;
              transformedResult = transformTrainingResult(
                modelType,
                receiptFinalResult,
                expectedEvaluationLength,
              );
            }

            const status: GuidedTrainingStatus = transformedResult ? "completed" : "ready";
            const currentStepId = status === "completed"
              ? null
              : getGuidedNextStepId(nextStepPlan, nextCompletedStepIds);

            await lease.verifyOwned();
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
                artifactSha256,
                guidedSession: {
                  sessionId,
                  currentStepId,
                  completedStepIds: nextCompletedStepIds,
                  stepOutputs: nextStepOutputs,
                  stepPlan: nextStepPlan,
                  result: transformedResult,
                  activeAttemptId: activeAttempt.attempt_id,
                  committedStatePath: activeAttempt.candidate_state_path,
                  committedStateSha256: String(receipt.candidate_state_sha256),
                },
              });
            } else {
              await commitGuidedIntermediateStep({
                sessionId,
                attempt: activeAttempt,
                status,
                currentStepId,
                completedStepIds: nextCompletedStepIds,
                stepOutputs: nextStepOutputs,
                stepPlan: nextStepPlan,
                receipt,
              });
            }
            stepCommitted = true;
            cleanupUncommittedArtifactPath = null;

            const updatedRow = await fetchGuidedTrainingSessionRow(sessionId);
            if (!updatedRow) throw new Error("分阶段训练会话已丢失");
            return buildGuidedTrainingSessionPayload(updatedRow);
          } catch (error) {
            if (cleanupUncommittedArtifactPath) {
              removeModelArtifactGroup(cleanupUncommittedArtifactPath);
            }
            throw error;
          }
        },
      );
    });
  } catch (error) {
    const failedAttempt = activeAttemptRef.current;
    if (!stepCommitted && failedAttempt) {
      // Failure recording is best-effort and must never replace the original
      // model/deadline/lock error returned to the caller.
      await recordGuidedAttemptFailure(
        sessionId,
        failedAttempt.attempt_id,
        stepId,
        error,
      );
      // Invalid/partial candidates are removed for every failure mode. A fully
      // validated receipt is intentionally retained for idempotent recovery.
      cleanupIncompleteGuidedAttemptArtifacts(failedAttempt);
    }
    throw error;
  }
}

const getReusableProductionModelArtifactResult = async (
  modelType: ModelType,
  forecastSteps: number,
  experiment: ExperimentModelContextRow,
  activeRun: ModelTrainingRunRow,
  terminationSignal?: AbortSignal,
) => {
  const productionModelPath = activeRun.production_artifact_path;
  if (!productionModelPath) {
    return null;
  }
  if (
    !activeRun.production_revision
    || activeRun.production_artifact_forecast_steps === null
    || Number(activeRun.production_artifact_forecast_steps) < forecastSteps
  ) {
    return null;
  }
  const trainingConfig = getExperimentTrainingConfig(experiment);
  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
  );
  validateActiveRunAgainstSnapshot(activeRun, JSON.stringify(allData));

  try {
    await verifyProductionBaseArtifactForRun(modelType, activeRun, terminationSignal);
  } catch (error) {
    if (!(error instanceof ModelArtifactError)) throw error;
    logger.warn(
      { err: error, runId: activeRun.run_id, modelType, productionModelPath },
      "检测到无效生产模型产物，将在当前请求中重新拟合",
    );
    await invalidateProductionArtifactForRun(activeRun);
    return null;
  }

  return {
    status: "success",
    results: {
      preparation_token: activeRun.production_revision,
      prepared_forecast_steps: Number(activeRun.production_artifact_forecast_steps),
      reused_existing: true,
    },
  };
}

const prepareExperimentModelForProductionWithLease = async (
  modelType: ModelType,
  studentId: number,
  experimentId: number,
  forecastSteps: number,
  forceRefit: boolean,
  experiment: ExperimentModelContextRow,
  lease: ModelOperationLease,
) => {
  const activeRun = await requireActiveModelTrainingRun(experimentId, studentId, modelType);
  const reusableArtifact = forceRefit
    ? null
    : await getReusableProductionModelArtifactResult(
      modelType,
      forecastSteps,
      experiment,
      activeRun,
      lease.terminationSignal,
    );
  if (reusableArtifact) {
    await lease.verifyOwned();
    return reusableArtifact;
  }

  const trainingConfig = getExperimentTrainingConfig(experiment);
  const previousProductionModelPath = activeRun.production_artifact_path;
  const productionModelPath = buildProductionCandidateModelPath(
    experimentId,
    studentId,
    modelType,
    activeRun.run_id,
  );
  const backtestModelPath = activeRun.artifact_path;
  let verifiedBacktestSha256: string | null = null;
  try {
    verifiedBacktestSha256 = await verifyBacktestBaseArtifactForRun(
      modelType,
      activeRun,
      lease.terminationSignal,
    );
  } catch (error) {
    if (!(error instanceof ModelArtifactError)) throw error;
    throw new ModelConflictError(
      `${modelType.toUpperCase()} 回测模型产物已损坏或缺失，请重新训练该模型`,
      "MODEL_ARTIFACT_INVALID",
      { required_action: "retrain" },
    );
  }

  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
  );

  validateActiveRunAgainstSnapshot(activeRun, JSON.stringify(allData));

  if (trainingConfig.data_window_evaluate_end_index >= allData.length) {
    throw new ModelValidationError("数据窗口索引超出范围", { dataset_size: allData.length });
  }

  const snapshotDir = fs.mkdtempSync(path.join(os.tmpdir(), "fz-production-snapshot-"));
  let productionArtifactPromoted = false;
  let productionRevision: string | null = null;
  let productionArtifactSha256: string | null = null;
  try {
    const dataFile = path.join(snapshotDir, GUIDED_DATASET_FILE_NAME);
    await Bun.write(dataFile, JSON.stringify(allData));
    const pythonParams: Record<string, unknown> = {
      dataFile,
      trainStart: trainingConfig.data_window_train_start_index,
      trainEnd: trainingConfig.data_window_evaluate_end_index,
      saveModel: productionModelPath,
      loadModel: backtestModelPath,
      fitOnly: true,
      fitForecastSteps: forecastSteps,
      pretty: true,
    };

    try {
      await runPythonModel(
        modelType,
        pythonParams,
        resolveProductionPreparationTimeoutMs(modelType, activeRun),
        undefined,
        lease.terminationSignal,
      );
    } catch (error) {
      if (!(error instanceof ModelArtifactError)) throw error;
      throw new ModelConflictError(
        `${modelType.toUpperCase()} 回测模型产物无法加载，请重新训练该模型`,
        "MODEL_ARTIFACT_INVALID",
        { required_action: "retrain" },
      );
    }
    if (verifiedBacktestSha256) {
      try {
        await verifyBaseModelArtifactIntegrity(
          modelType,
          backtestModelPath,
          verifiedBacktestSha256,
          lease.terminationSignal,
        );
      } catch (error) {
        if (!(error instanceof ModelArtifactError)) throw error;
        throw new ModelConflictError(
          `${modelType.toUpperCase()} 回测模型产物在生产拟合期间发生变化，请重新训练该模型`,
          "MODEL_ARTIFACT_INVALID",
          { required_action: "retrain" },
        );
      }
    }
    productionArtifactSha256 = (
      await verifyBaseModelArtifactIntegrity(
        modelType,
        productionModelPath,
        null,
        lease.terminationSignal,
      )
    ).sha256;

    await lease.verifyOwned();
    productionRevision = crypto.randomUUID();
    const recorded = await setProductionArtifactForRun(
      activeRun.run_id,
      productionModelPath,
      forecastSteps,
      productionRevision,
      productionArtifactSha256,
    );
    if (!recorded) {
      throw new ModelValidationError("模型版本已变化，已放弃本次生产模型，请重试");
    }
    productionArtifactPromoted = true;
  } catch (error) {
    removeModelArtifactGroup(productionModelPath);
    throw error;
  } finally {
    fs.rmSync(snapshotDir, { recursive: true, force: true });
  }

  if (
    productionArtifactPromoted
    && previousProductionModelPath
    && previousProductionModelPath !== productionModelPath
    && previousProductionModelPath !== backtestModelPath
  ) {
    removeModelArtifactGroup(previousProductionModelPath);
  }
  if (!productionRevision) {
    throw new ModelValidationError("生产模型准备完成但版本标识缺失，请重试");
  }

  return {
    status: "success",
    results: {
      preparation_token: productionRevision,
      prepared_forecast_steps: forecastSteps,
      reused_existing: false,
    },
  };
};

export async function prepareExperimentModelForProduction(
  modelType: ModelType,
  studentId: number,
  experimentId: number,
  forecastSteps = DEFAULT_PRODUCTION_FORECAST_STEPS,
  forceRefit = false,
) {
  return await withModelOperationContext("production_preparation", async (operationContext) => {
    const experiment = await assertExperimentModelAccess(experimentId, studentId);
    assertModelReadyForProduction(experiment, modelType);
    operationContext.throwIfTerminated();

    return await withModelOperationLease(
      operationContext,
      { mode: "slot_and_execution", experimentId, modelType },
      async (lease) => await prepareExperimentModelForProductionWithLease(
        modelType,
        studentId,
        experimentId,
        forecastSteps,
        forceRefit,
        experiment,
        lease,
      ),
    );
  });
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
  const [rows] = await queryWithTimeout<StaleModelExperimentEligibilityRow[]>(pool, 
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

const predictExperimentModelWithLease = async (
  modelType: ModelType,
  studentId: number,
  experimentId: number,
  forecastSteps: number,
  preparationToken: string,
  experiment: ExperimentModelContextRow,
  lease: ModelOperationLease,
) => {
  const activeRun = await requireActiveModelTrainingRun(experimentId, studentId, modelType);
  if (
    activeRun.production_revision !== preparationToken
    || activeRun.production_artifact_forecast_steps === null
    || Number(activeRun.production_artifact_forecast_steps) < forecastSteps
  ) {
    throw new ModelConflictError(
      "生产预测准备已失效或预测期数不足，请重新准备生产模型",
      "PRODUCTION_PREPARATION_STALE",
    );
  }
  const productionModelPath = activeRun.production_artifact_path;
  if (!productionModelPath) {
    throw new ModelConflictError(
      `未找到该实验已准备的 ${modelType.toUpperCase()} 生产预测模型`,
      "PRODUCTION_PREPARATION_REQUIRED",
    );
  }
  let verifiedProductionSha256: string | null = null;
  try {
    verifiedProductionSha256 = await verifyProductionBaseArtifactForRun(
      modelType,
      activeRun,
      lease.terminationSignal,
    );
  } catch (error) {
    if (!(error instanceof ModelArtifactError)) throw error;
    await invalidateProductionArtifactForRun(activeRun);
    throw new ModelConflictError(
      `${modelType.toUpperCase()} 生产模型产物已失效，请重新准备生产模型`,
      "PRODUCTION_PREPARATION_REQUIRED",
      { required_action: "prepare" },
    );
  }

  const trainingConfig = getExperimentTrainingConfig(experiment);
  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
  );
  validateActiveRunAgainstSnapshot(activeRun, JSON.stringify(allData));

  const pythonParams = {
    loadModel: productionModelPath,
    forecastSteps,
    pretty: true,
  };

  let result: Record<string, unknown>;
  try {
    result = (await runPythonModel(
      modelType,
      pythonParams,
      TIMEOUTS.MODEL_PREDICTION,
      undefined,
      lease.terminationSignal,
    )) as Record<string, unknown>;
  } catch (error) {
    if (!(error instanceof ModelArtifactError)) throw error;
    const currentRun = await getActiveModelTrainingRun(experimentId, studentId, modelType);
    if (
      currentRun?.run_id !== activeRun.run_id
      || currentRun.production_revision !== preparationToken
    ) {
      throw new ModelConflictError(
        "预测期间模型版本已变化，请重新准备生产模型",
        "PRODUCTION_PREPARATION_STALE",
      );
    }
    await invalidateProductionArtifactForRun(activeRun);
    throw new ModelConflictError(
      `${modelType.toUpperCase()} 生产模型产物无法加载，请重新准备生产模型`,
      "PRODUCTION_PREPARATION_REQUIRED",
      { required_action: "prepare" },
    );
  }
  lease.throwIfTerminated();
  const currentRun = await getActiveModelTrainingRun(experimentId, studentId, modelType);
  if (
    currentRun?.run_id !== activeRun.run_id
    || currentRun.production_revision !== preparationToken
  ) {
    throw new ModelConflictError(
      "预测期间模型版本已变化，请重新准备生产模型",
      "PRODUCTION_PREPARATION_STALE",
    );
  }
  if (verifiedProductionSha256) {
    try {
      await verifyBaseModelArtifactIntegrity(
        modelType,
        productionModelPath,
        verifiedProductionSha256,
        lease.terminationSignal,
      );
    } catch (error) {
      if (!(error instanceof ModelArtifactError)) throw error;
      await invalidateProductionArtifactForRun(activeRun);
      throw new ModelConflictError(
        `${modelType.toUpperCase()} 生产模型产物在预测期间发生变化，请重新准备生产模型`,
        "PRODUCTION_PREPARATION_REQUIRED",
        { required_action: "prepare" },
      );
    }
  }
  await lease.verifyOwned();

  let predictions: ReturnType<typeof normalizeDemandPredictionPoints>;
  try {
    predictions = normalizeDemandPredictionPoints(
      result.predictions,
      forecastSteps,
      { requireStdDev: true },
    );
  } catch (error) {
    if (!(error instanceof Error) || !error.message.startsWith("[PYTHON_DATA]")) {
      throw error;
    }
    await invalidateProductionArtifactForRun(activeRun);
    throw new ModelConflictError(
      `${modelType.toUpperCase()} 生产模型返回了无效预测结果，请重新准备生产模型`,
      "PRODUCTION_PREPARATION_REQUIRED",
      { required_action: "prepare" },
    );
  }

  const metadata = Object.fromEntries(
    ["method_name", "forecast_strategy", "implementation_notes"]
      .filter((key) => result[key] !== undefined)
      .map((key) => [key, result[key]]),
  );

  return {
    status: "success",
    results: {
      predictions,
      ...metadata,
    },
  };
};

export async function predictExperimentModel(
  modelType: ModelType,
  studentId: number,
  experimentId: number,
  forecastSteps: number,
  preparationToken: string,
) {
  return await withModelOperationContext("prediction", async (operationContext) => {
    const experiment = await assertExperimentModelAccess(experimentId, studentId);
    assertModelReadyForProduction(experiment, modelType);
    operationContext.throwIfTerminated();

    return await withModelOperationLease(
      operationContext,
      { mode: "slot_and_execution", experimentId, modelType },
      async (lease) => await predictExperimentModelWithLease(
        modelType,
        studentId,
        experimentId,
        forecastSteps,
        preparationToken,
        experiment,
        lease,
      ),
    );
  });
}
