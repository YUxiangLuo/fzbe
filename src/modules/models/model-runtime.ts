import path from "path";
import type { ModelType } from "./models.schema";
import { PY_DIR } from "../../utils/python";

const LOCK_NAMESPACE = "fangzhen-model";
export type ModelArtifactPhase = "backtest" | "production";
export const GUIDED_TRAINING_ROOT = path.join(PY_DIR, "guided_training");

export const buildVersionedModelArtifactPath = (
  experimentId: number,
  studentId: number,
  modelType: ModelType,
  revisionToken: string,
  phase: ModelArtifactPhase = "backtest",
): string => path.join(
  PY_DIR,
  "trained_models",
  `exp${experimentId}-student${studentId}-${modelType}-${phase}-${revisionToken}.pkl`,
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

export const buildModelExecutionLockName = (
  experimentId: number,
  _modelType?: ModelType,
): string => `${LOCK_NAMESPACE}:exp:${experimentId}`;

export const buildModelSlotLockName = (slot: number): string => `${LOCK_NAMESPACE}:slot:${slot}`;
