import path from "path";
import type { ModelType } from "./models.schema";
import { PY_DIR } from "../../utils/python";

const LOCK_NAMESPACE = "fangzhen-model";
export type ModelArtifactPhase = "backtest" | "production";
export const GUIDED_TRAINING_ROOT = path.join(PY_DIR, "guided_training");

export const buildExperimentScopedModelPath = (
  experimentId: number,
  studentId: number,
  modelType: ModelType,
  phase: ModelArtifactPhase = "backtest",
): string => path.join(
  PY_DIR,
  "trained_models",
  `exp${experimentId}-student${studentId}-${modelType}${phase === "production" ? "-production" : ""}.pkl`,
);

export const buildVersionedModelPath = (
  experimentId: number,
  studentId: number,
  modelType: ModelType,
  runId: string,
  phase: ModelArtifactPhase = "backtest",
): string => path.join(
  PY_DIR,
  "trained_models",
  `exp${experimentId}-student${studentId}-${modelType}-${runId}-${phase}.pkl`,
);

export const buildGuidedTrainingArtifactDir = (
  experimentId: number,
  studentId: number,
  modelType: ModelType,
  sessionId: string,
): string => path.join(
  GUIDED_TRAINING_ROOT,
  `exp${experimentId}-student${studentId}-${modelType}-${sessionId}`,
);

export const buildGuidedTrainingStatePath = (
  experimentId: number,
  studentId: number,
  modelType: ModelType,
  sessionId: string,
): string => path.join(
  buildGuidedTrainingArtifactDir(experimentId, studentId, modelType, sessionId),
  "state.pkl",
);

export const buildGuidedTrainingAttemptPaths = (
  artifactDir: string,
  stepId: string,
  attemptId: string,
) => {
  const safeStepId = stepId.replace(/[^A-Za-z0-9_-]/g, "_");
  const checkpointDir = path.join(artifactDir, "checkpoints");
  const basename = `${safeStepId}-${attemptId}`;
  return {
    candidateStatePath: path.join(checkpointDir, `${basename}.pkl`),
    receiptPath: path.join(checkpointDir, `${basename}.receipt.json`),
  };
};

export const buildModelExecutionLockName = (
  experimentId: number,
  modelType: ModelType,
): string => `${LOCK_NAMESPACE}:exp:${experimentId}:${modelType}`;

export const buildModelSlotLockName = (slot: number): string => `${LOCK_NAMESPACE}:slot:${slot}`;
