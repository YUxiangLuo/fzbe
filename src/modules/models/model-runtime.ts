import path from "path";
import type { ModelType } from "./models.schema";
import { PY_DIR } from "../../utils/python";

const LOCK_NAMESPACE = "fangzhen-model";
export type ModelArtifactPhase = "backtest" | "production";

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

export const buildModelExecutionLockName = (
  experimentId: number,
  modelType: ModelType,
): string => `${LOCK_NAMESPACE}:exp:${experimentId}:${modelType}`;

export const buildModelSlotLockName = (slot: number): string => `${LOCK_NAMESPACE}:slot:${slot}`;