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

/**
 * 单实验单模型执行去重的按键锁 key（进程内 KeyedLock 使用）。
 * 沿用旧 advisory lock 的命名格式，便于日志排查。
 */
export const buildModelExecutionLockName = (
  experimentId: number,
  modelType: ModelType,
): string => `${LOCK_NAMESPACE}:exp:${experimentId}:${modelType}`;
