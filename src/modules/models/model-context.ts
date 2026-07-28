import crypto from "crypto";
import fs from "fs";
import path from "path";
import type { RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import { queryWithTimeout } from "./model-db-guard";
import type { ModelType, TrainInput } from "./models.schema";
import {
  ModelAccessError,
  ModelNotFoundError,
  ModelValidationError,
} from "./model-errors";
import {
  getActiveModelTrainingRun,
} from "./model-run-store";
import {
  BASE_MODEL_TYPES,
  ENSEMBLE_MODEL_TYPES,
  GUIDED_MODEL_TYPES,
  type BaseModelType,
  type EnsembleModelType,
  type GuidedModelType,
} from "./model-types";
import {
  hasSavedModelArtifact,
  verifyEnsembleBaseArtifactForRun,
} from "./model-artifacts";
import {
  assertRunRevisionMatches,
} from "./model-runtime-identity";
import {
  GUIDED_DATASET_FILE_NAME,
  MIN_EVALUATION_POINTS,
  MIN_TRAINING_POINTS,
  MODEL_ORCHESTRATION_HARD_CAP,
  TRAINING_TIMEOUT,
} from "./model-service-constants";

interface ExperimentOwnershipRow extends RowDataPacket {
  student_id: number;
  status: string;
}

export interface ExperimentModelContextRow extends ExperimentOwnershipRow {
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

export interface ExperimentTrainingConfig {
  selected_industry: string;
  selected_company: string;
  selected_product: string;
  data_window_train_start_index: number;
  data_window_train_end_index: number;
  data_window_evaluate_start_index: number;
  data_window_evaluate_end_index: number;
}
export async function assertExperimentModelAccess(experimentId: number, studentId: number) {
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

export const isBaseModelType = (modelType: ModelType): modelType is BaseModelType => {
  return BASE_MODEL_TYPES.has(modelType);
};

export const isEnsembleModelType = (modelType: ModelType): modelType is Extract<ModelType, "weighted_avg" | "boosting" | "stacking"> => {
  return ENSEMBLE_MODEL_TYPES.has(modelType);
};

export const requireActiveModelTrainingRun = async (
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

export const writeGuidedDatasetSnapshot = async (artifactDir: string, datasetJson: string) => {
  await fs.promises.mkdir(artifactDir, { recursive: true });
  const targetPath = path.join(artifactDir, GUIDED_DATASET_FILE_NAME);
  const temporaryPath = `${targetPath}.tmp-${crypto.randomUUID()}`;
  await Bun.write(temporaryPath, datasetJson);
  fs.renameSync(temporaryPath, targetPath);
  return targetPath;
};

export interface ArtifactValidationOptions {
  verifyArtifacts?: boolean;
  terminationSignal?: AbortSignal;
}

export const buildBaseRunContext = async (
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

export const parseJsonColumn = <T>(value: unknown, fallback: T): T => {
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

export const isRecordValue = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

export const isGuidedModelType = (modelType: ModelType): modelType is GuidedModelType => {
  return GUIDED_MODEL_TYPES.has(modelType);
};

const normalizeBooleanFlag = (value: number | boolean | null | undefined): boolean => {
  return value === true || value === 1;
};

export const validateMovingAverageWindow = (
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

export const parseRequestedBaseModels = (value: string): BaseModelType[] => {
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

export const BASE_MODEL_TIMEOUT_SECONDS: Record<BaseModelType, number> = {
  ma: 120,
  es: 120,
  arima: 120,
  lstm: 600,
};

export const resolveGuidedStepTimeoutMs = (
  modelType: GuidedModelType,
  stepId: string,
  body: TrainInput,
) => {
  if (!isEnsembleModelType(modelType) || !body.models) return TRAINING_TIMEOUT;
  parseRequestedBaseModels(body.models);
  const memberMatch = /_(ma|es|arima|lstm)$/.exec(stepId);
  if (memberMatch) {
    const member = memberMatch[1] as BaseModelType;
    return Math.min(
      MODEL_ORCHESTRATION_HARD_CAP,
      Math.max(TRAINING_TIMEOUT, BASE_MODEL_TIMEOUT_SECONDS[member] * 1000 + 30_000),
    );
  }
  // Protocol-v2 sessions are superseded before execution; this only keeps the
  // old aggregate id safely bounded during a rolling deployment.
  if (stepId === "retrain_chain") return MODEL_ORCHESTRATION_HARD_CAP;
  return TRAINING_TIMEOUT;
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

export const buildEnsemblePythonParams = (
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

export const buildAvailableBaseModelArtifactMap = async (
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

export const assertTrainingRequestMatchesExperiment = (
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

export const getExperimentTrainingConfig = (
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

export const assertModelReadyForProduction = (
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

export const BASE_MODEL_LABELS: Record<BaseModelType, string> = {
  ma: "移动平均",
  es: "指数平滑",
  arima: "ARIMA",
  lstm: "LSTM",
};

export const isDuplicateModelSessionRequestError = (error: unknown) => {
  if (!error || typeof error !== "object") return false;
  const candidate = error as { code?: string; errno?: number };
  return candidate.code === "ER_DUP_ENTRY" || candidate.errno === 1062;
};
