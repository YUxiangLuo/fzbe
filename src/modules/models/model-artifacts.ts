import path from "path";
import fs from "fs";
import crypto from "crypto";
import { TIMEOUTS } from "../../utils/constants";
import logger from "../../utils/logger";
import { PY_DIR, resolvePythonExecutable } from "../../utils/python";
import { terminateProcessTree } from "../../utils/process-tree";
import type { ModelType } from "./models.schema";
import {
  ModelArtifactError,
  ModelNotFoundError,
  ModelTimeoutError,
  ModelValidationError,
} from "./model-errors";
import { buildVersionedModelPath } from "./model-runtime";
import { buildPythonModelProcessEnv, runPythonModel } from "./model-python-runner";
import { normalizeDemandPredictionPoints } from "./model-results";
import {
  backfillArtifactSha256ForRun,
  backfillProductionArtifactSha256ForRun,
  clearProductionArtifactForRun,
  type ModelTrainingRunRow,
} from "./model-run-store";
import {
  ENSEMBLE_MODEL_TYPES,
  type BaseModelType,
  type EnsembleModelType,
  type GuidedModelType,
} from "./model-types";
import {
  computeBaseModelArtifactManifestSha256,
  getLstmSidecarPath,
  hashFileContents,
} from "./base-model-artifact-digest";

export { hashFileContents };

const isPathInsideDirectory = (candidatePath: string, rootDirectory: string) => {
  const resolvedCandidate = path.resolve(candidatePath);
  const resolvedRoot = path.resolve(rootDirectory);
  const relativePath = path.relative(resolvedRoot, resolvedCandidate);
  return relativePath.length > 0
    && !relativePath.startsWith("..")
    && !path.isAbsolute(relativePath);
};

const isEnsembleModelType = (modelType: ModelType): modelType is EnsembleModelType =>
  ENSEMBLE_MODEL_TYPES.has(modelType);

export const buildProductionCandidateModelPath = (
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
  } catch (error) {
    await terminateAndWait(error instanceof Error ? error : new Error(String(error)));
    throw error;
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
    if (abortHandler && options.terminationSignal) {
      options.terminationSignal.removeEventListener("abort", abortHandler);
    }
    if (terminationPromise) await terminationPromise;
  }
};

export const getGuidedSessionModelPath = (artifactDir: string, modelType: GuidedModelType) =>
  path.join(artifactDir, `${modelType}.pkl`);

export const removeFileIfExists = (filePath: string) => {
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

export const removeModelArtifactGroup = (modelPath: string) => {
  const trainedModelsRoot = path.join(PY_DIR, "trained_models");
  if (!isPathInsideDirectory(modelPath, trainedModelsRoot)) {
    logger.warn({ modelPath }, "拒绝清理训练模型目录之外的产物组");
    return;
  }

  removeFileIfExists(modelPath);
  removeSiblingArtifactsByPrefix(modelPath);
};

export const hasSavedModelArtifact = (modelType: ModelType, modelPath: string) => {
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
import math
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

def is_finite_number(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return not isinstance(value, bool) and math.isfinite(numeric)

def has_valid_persisted_baseline(state):
    baseline = state.get("baseline")
    if not isinstance(baseline, dict):
        return False
    kind = baseline.get("kind")
    if kind == "mean":
        return is_finite_number(baseline.get("training_mean"))
    if kind == "naive":
        return is_finite_number(baseline.get("last_value"))
    if kind == "seasonal_naive":
        try:
            period = int(baseline.get("seasonal_period"))
        except (TypeError, ValueError, OverflowError):
            return False
        tail = baseline.get("seasonal_tail")
        return (
            period >= 2
            and isinstance(tail, (list, tuple))
            and len(tail) == period
            and all(is_finite_number(value) for value in tail)
        )
    # A zero baseline is supported only for replaying legacy artifacts that
    # still contain member stages. By itself it is not a meaningful model.
    return False

def is_baseline_only_boosting(state):
    return (
        isinstance(state, dict)
        and state.get("model_type") == "boosting"
        and state.get("boosting_variant") == "residual"
        and state.get("model_chain") == []
        and state.get("stage_coefficients") == []
        and state.get("base_model_states") == {}
        and has_valid_persisted_baseline(state)
    )

try:
    with open(sys.argv[1], "rb") as handle:
        state = pickle.load(handle)

    flattened_states = iter_model_states(state.get("base_model_states"))
    is_valid = flattened_states is not None and (
        (
            len(flattened_states) > 0
            and all(
                artifact_exists(model_type, model_file)
                for model_type, model_file in flattened_states
            )
        )
        or (
            len(flattened_states) == 0
            and is_baseline_only_boosting(state)
        )
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

export const computeBaseModelArtifactSha256 = async (
  modelType: BaseModelType,
  modelPath: string,
  terminationSignal?: AbortSignal,
) => {
  if (!hasSavedModelArtifact(modelType, modelPath)) {
    throw new ModelArtifactError(`${modelType.toUpperCase()} 模型产物缺失或不完整`);
  }

  try {
    return await computeBaseModelArtifactManifestSha256(
      modelType,
      modelPath,
      terminationSignal,
    );
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
import math
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

def is_finite_number(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return not isinstance(value, bool) and math.isfinite(numeric)

def has_valid_persisted_baseline(state):
    baseline = state.get("baseline")
    if not isinstance(baseline, dict):
        return False
    kind = baseline.get("kind")
    if kind == "mean":
        return is_finite_number(baseline.get("training_mean"))
    if kind == "naive":
        return is_finite_number(baseline.get("last_value"))
    if kind == "seasonal_naive":
        try:
            period = int(baseline.get("seasonal_period"))
        except (TypeError, ValueError, OverflowError):
            return False
        tail = baseline.get("seasonal_tail")
        return (
            period >= 2
            and isinstance(tail, (list, tuple))
            and len(tail) == period
            and all(is_finite_number(value) for value in tail)
        )
    # A zero baseline is supported only for replaying legacy artifacts that
    # still contain member stages. By itself it is not a meaningful model.
    return False

def is_baseline_only_boosting(state, requested_model_type):
    return (
        requested_model_type == "boosting"
        and isinstance(state, dict)
        and state.get("model_type") == "boosting"
        and state.get("boosting_variant") == "residual"
        and state.get("model_chain") == []
        and state.get("stage_coefficients") == []
        and state.get("base_model_states") == {}
        and has_valid_persisted_baseline(state)
    )


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
    if members is None:
        raise ValueError("ensemble has invalid base members")
    if len(members) == 0 and not is_baseline_only_boosting(state, model_type):
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

export const computeEnsembleArtifactSha256 = async (
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

export const computeModelArtifactSha256 = async (
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
    normalizeDemandPredictionPoints(result.predictions, 1, {
      requireStdDev: true,
      requireUpperErrorP99: true,
      requireCalibrationDiagnostics: true,
    });
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

export const verifyBaseModelArtifactIntegrity = async (
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

export const verifyBacktestBaseArtifactForRun = async (
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

export const verifyEnsembleBaseArtifactForRun = async (
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

export const verifyProductionBaseArtifactForRun = async (
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

export const invalidateProductionArtifactForRun = async (activeRun: ModelTrainingRunRow) => {
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

export const copyModelArtifact = async (
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

export const sha256File = (filePath: string) => {
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
