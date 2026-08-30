import crypto from "crypto";
import fs from "fs";
import path from "path";
import { PY_DIR } from "../../utils/python";
import {
  buildModelRetrainingRequiredDetails,
  ModelValidationError,
} from "./model-errors";
import type { ModelTrainingRunRow } from "./model-run-store";

const MODEL_PIPELINE_REVISION = "model-run-v3";
let cachedModelRuntimeRevision: string | null = null;

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

export const calculateModelRuntimeRevision = (pythonDirectory: string) => {
  const runtimeFiles = [
    path.join(pythonDirectory, ".python-version"),
    path.join(pythonDirectory, "production-dependencies.lock"),
    path.join(pythonDirectory, "guided_model.py"),
    path.join(pythonDirectory, "guided_ensemble_model.py"),
    path.join(pythonDirectory, "production_prepare_model.py"),
    path.join(pythonDirectory, "run_model.py"),
    ...collectPythonRuntimeFiles(path.join(pythonDirectory, "src")),
  ].filter((filePath) => fs.existsSync(filePath)).sort();
  const hash = crypto.createHash("sha256").update(MODEL_PIPELINE_REVISION);
  for (const filePath of runtimeFiles) {
    try {
      const contents = fs.readFileSync(filePath);
      hash.update(path.relative(pythonDirectory, filePath));
      hash.update(contents);
    } catch {
      // A deployment may atomically replace files while the revision is first
      // calculated. Missing files are excluded and the cache is rebuilt on the
      // next process start.
    }
  }
  return hash.digest("hex");
};

export const getModelRuntimeRevision = (fresh = false) => {
  const configuredRevision = process.env.MODEL_RUNTIME_REVISION?.trim();
  if (configuredRevision) {
    return crypto.createHash("sha256").update(configuredRevision).digest("hex");
  }
  if (!fresh && cachedModelRuntimeRevision) return cachedModelRuntimeRevision;
  const revision = calculateModelRuntimeRevision(PY_DIR);
  if (!fresh) cachedModelRuntimeRevision = revision;
  return revision;
};

export const hashDatasetSnapshot = (datasetJson: string) =>
  crypto.createHash("sha256").update(datasetJson).digest("hex");

export const assertRunRevisionMatches = (
  run: ModelTrainingRunRow,
  datasetRevision: string,
  runtimeRevision: string,
  label = run.model_type.toUpperCase(),
) => {
  if (typeof run.runtime_revision === "string" && run.runtime_revision !== runtimeRevision) {
    throw new ModelValidationError(
      `${label} 模型运行环境已变化，请重新训练该模型`,
      buildModelRetrainingRequiredDetails(run.model_type),
    );
  }
  if (typeof run.dataset_revision === "string" && run.dataset_revision !== datasetRevision) {
    throw new ModelValidationError(
      `${label} 模型对应的数据集版本已变化，请重新训练该模型`,
      buildModelRetrainingRequiredDetails(run.model_type),
    );
  }
};

export const validateActiveRunAgainstSnapshot = (
  run: ModelTrainingRunRow | null,
  datasetJson: string,
  runtimeRevision = getModelRuntimeRevision(true),
) => {
  if (!run) return;
  assertRunRevisionMatches(run, hashDatasetSnapshot(datasetJson), runtimeRevision);
};
