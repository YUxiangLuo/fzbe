import crypto from "crypto";
import fs from "fs";
import path from "path";
import type { PoolConnection, RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import { TIMEOUTS } from "../../utils/constants";
import { withBoundedPoolTransaction } from "../../utils/bounded-pool-transaction";
import logger from "../../utils/logger";
import {
  acquireConnectionWithTimeout,
  commitConnectionWithOutcomeGuard,
  queryWithTimeout,
} from "./model-db-guard";
import type { ModelType, TrainInput } from "./models.schema";
import {
  ModelAccessError,
  ModelArtifactError,
  ModelConflictError,
  ModelNotFoundError,
  ModelValidationError,
} from "./model-errors";
import {
  buildGuidedTrainingAttemptPaths,
} from "./model-runtime";
import { loadDatasetFromDb } from "./model-datasets";
import {
  allPossiblePythonParams,
} from "./model-python-runner";
import {
  buildExperimentTrainingRevision,
  buildModelTrainingRequestHash,
  getActiveModelTrainingRun,
  loadCurrentExperimentStatePatch,
} from "./model-run-store";
import {
  BASE_MODEL_TYPES,
  GUIDED_TRAINING_STEPS,
  type BaseModelType,
  type GuidedModelType,
  type GuidedTrainingSessionPayload,
  type GuidedTrainingStepDefinition,
  type GuidedTrainingStatus,
} from "./model-types";
import {
  getGuidedSessionModelPath,
  hasSavedModelArtifact,
  sha256File,
  verifyBacktestBaseArtifactForRun,
} from "./model-artifacts";
import {
  type ArtifactValidationOptions,
  BASE_MODEL_LABELS,
  type ExperimentTrainingConfig,
  assertExperimentModelAccess,
  assertTrainingRequestMatchesExperiment,
  buildAvailableBaseModelArtifactMap,
  buildBaseRunContext,
  buildEnsemblePythonParams,
  getExperimentTrainingConfig,
  isEnsembleModelType,
  parseJsonColumn,
  parseRequestedBaseModels,
  validateMovingAverageWindow,
} from "./model-context";
import {
  getModelRuntimeRevision,
  hashDatasetSnapshot,
} from "./model-runtime-identity";
import {
  GUIDED_DATASET_FILE_NAME,
  GUIDED_TRAINING_PROTOCOL_VERSION,
} from "./model-service-constants";

export interface ModelTrainingDependencyRow extends RowDataPacket {
  base_model_type: BaseModelType;
  base_run_id: string | null;
}

export interface GuidedTrainingSessionRow extends RowDataPacket {
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

export interface GuidedTrainingStepAttemptRow extends RowDataPacket {
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
  /** Computed by MySQL so DB_DATE_STRINGS and application timezone cannot skew lease ownership. */
  lease_is_active?: number | boolean | string | null;
}

export const buildMemberStepDefinitions = (
  models: BaseModelType[],
  idPrefix: string,
  labelPrefix: string,
  description: (model: BaseModelType) => string,
): GuidedTrainingStepDefinition[] => models.map((model) => ({
  id: `${idPrefix}_${model}`,
  label: `${labelPrefix}：${BASE_MODEL_LABELS[model]}`,
  description: description(model),
  actionLabel: `执行 ${BASE_MODEL_LABELS[model]}`,
}));

export const buildBoostingRoundStepDefinitions = (
  maxRounds: number,
  models: BaseModelType[],
): GuidedTrainingStepDefinition[] => {
  const steps: GuidedTrainingStepDefinition[] = [];
  for (let round = 1; round <= maxRounds; round += 1) {
    steps.push(
      ...buildMemberStepDefinitions(
        models,
        `round_${round}_evaluate`,
        `第 ${round} 轮评估候选`,
        (model) => round === 1
          ? `用第 0 阶段基线留下的有符号残差训练 ${BASE_MODEL_LABELS[model]}，通过非负平方损失线搜索求阶段系数并记录选择段残差 RMSE。`
          : `用当前有正有负的残差训练 ${BASE_MODEL_LABELS[model]}，动态求非负阶段系数并记录第 ${round} 轮选择残差 RMSE。`,
      ),
      {
        id: `round_${round}_select_winner`,
        label: `第 ${round} 轮选择胜出模型`,
        description: "选择本轮使内部时间选择段 RMSE 最小且达到相对改善阈值的模型；短选择段要求更强改善，无候选达标则提前停止。",
        actionLabel: "选择模型",
      },
    );
  }
  return steps;
};

export const buildGuidedTrainingStepPlan = (
  modelType: GuidedModelType,
  body: TrainInput,
): GuidedTrainingStepDefinition[] => {
  if (!isEnsembleModelType(modelType)) {
    return GUIDED_TRAINING_STEPS[modelType];
  }

  const requestedModels = body.models ? parseRequestedBaseModels(body.models) : [];
  if (modelType === "weighted_avg") {
    return GUIDED_TRAINING_STEPS.weighted_avg.flatMap((step) => {
      if (step.id === "validation_predictions") {
        return buildMemberStepDefinitions(
          requestedModels,
          "validation_prediction",
          "生成权重选择预测",
          (model) => `只在权重训练前缀拟合 ${BASE_MODEL_LABELS[model]}，并为紧随其后的权重选择段生成预测。`,
        );
      }
      if (step.id === "eval_predictions") {
        return buildMemberStepDefinitions(
          requestedModels,
          "eval_prediction",
          "生成评估预测",
          (model) => `在完整训练区间拟合或复用 ${BASE_MODEL_LABELS[model]}，生成统一评估区间预测。`,
        );
      }
      return [step];
    });
  }
  if (modelType === "stacking") {
    return GUIDED_TRAINING_STEPS.stacking.flatMap((step) => {
      if (step.id === "level1_predictions") {
        return buildMemberStepDefinitions(
          requestedModels,
          "level1_prediction",
          "生成元模型训练特征",
          (model) => `在多个扩展窗口起点拟合 ${BASE_MODEL_LABELS[model]}，生成严格按时间顺序排列的折外预测。`,
        );
      }
      if (step.id === "eval_predictions") {
        return buildMemberStepDefinitions(
          requestedModels,
          "eval_prediction",
          "生成评估预测",
          (model) => `在完整训练区间拟合或复用 ${BASE_MODEL_LABELS[model]}，生成统一评估区间预测。`,
        );
      }
      return [step];
    });
  }
  const maxRounds = Math.max(1, requestedModels.length * 2);
  const [prepareStep, splitStep, retrainStep, combineStep, saveStep] = GUIDED_TRAINING_STEPS.boosting;

  return [
    prepareStep!,
    splitStep!,
    ...buildBoostingRoundStepDefinitions(maxRounds, requestedModels),
    retrainStep!,
    combineStep!,
    saveStep!,
  ];
};

export const parseGuidedStepPlan = (
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

export const getGuidedTrainingSteps = (row: Pick<GuidedTrainingSessionRow, "model_type" | "step_plan">) =>
  parseGuidedStepPlan(row.model_type, parseJsonColumn<unknown>(row.step_plan, null));

export const getGuidedNextStepId = (steps: GuidedTrainingStepDefinition[], completedStepIds: string[]) => {
  const completedSet = new Set(completedStepIds);
  return steps.find((step) => !completedSet.has(step.id))?.id ?? null;
};

export const isRecordValue = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

export const shouldTrimBoostingRoundSteps = (
  modelType: GuidedModelType,
  stepId: string,
  output: unknown,
) => {
  if (modelType !== "boosting" || !/^round_\d+_select_winner$/.test(stepId) || !isRecordValue(output)) {
    return false;
  }

  return output.action === "early_stop" || output.stop_reason === "round_budget";
};

export const trimBoostingStepPlanAfterEarlyStop = (
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

export const expandBoostingRefitSteps = (
  stepPlan: GuidedTrainingStepDefinition[],
  output: Record<string, unknown>,
) => {
  const rawChain = output.model_chain;
  const modelChain = Array.isArray(rawChain)
    ? rawChain.filter((model): model is BaseModelType => (
      typeof model === "string" && BASE_MODEL_TYPES.has(model as ModelType)
    ))
    : [];
  const refitSteps: GuidedTrainingStepDefinition[] = [
    {
      id: "initialize_refit",
      label: "准备留出校准与完整重拟合",
      description: "先冻结内部选择链，并用更晚时间留出段之前的历史生成候选残差；随后初始化会重新纳入该留出段的完整训练重拟合。",
      actionLabel: "初始化重拟合",
    },
    ...modelChain.map((model, index) => ({
      id: `refit_stage_${index + 1}_${model}`,
      label: `重拟合第 ${index + 1} 阶段：${BASE_MODEL_LABELS[model]}`,
      description: `在当前完整训练残差上拟合 ${BASE_MODEL_LABELS[model]}，重新计算非负阶段系数并提交该阶段检查点。`,
      actionLabel: `重拟合第 ${index + 1} 阶段`,
    })),
  ];
  return stepPlan.flatMap((step) => step.id === "retrain_chain" ? refitSteps : [step]);
};

export const trimBoostingRefitSteps = (
  stepPlan: GuidedTrainingStepDefinition[],
  stepId: string,
  output: unknown,
) => {
  if (
    !/^refit_stage_\d+_(?:ma|es|arima|lstm)$/.test(stepId)
    || !isRecordValue(output)
    || output.action !== "trim_refit_chain"
  ) {
    return stepPlan;
  }
  const currentIndex = stepPlan.findIndex((step) => step.id === stepId);
  if (currentIndex < 0) return stepPlan;
  return stepPlan.filter((step, index) => (
    index <= currentIndex || !/^refit_stage_\d+_(?:ma|es|arima|lstm)$/.test(step.id)
  ));
};

const replaceUndefinedMapeWithNull = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(replaceUndefinedMapeWithNull);
  if (!isRecordValue(value)) return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [
    key,
    // The former Python implementation used exactly 0 as the undefined-MAPE
    // sentinel. Preserve any nonzero staged MAPE that may describe a different
    // subset instead of erasing every metric with the same key.
    key === "mape" && item === 0 ? null : replaceUndefinedMapeWithNull(item),
  ]));
};

const resultMarksMapeUnavailable = (
  result: GuidedTrainingSessionPayload["result"],
): boolean => isRecordValue(result)
  && isRecordValue(result.results)
  && isRecordValue(result.results.metrics)
  && result.results.metrics.mape === null;

export const buildGuidedTrainingSessionPayload = (
  row: GuidedTrainingSessionRow,
): GuidedTrainingSessionPayload => {
  const completedStepIds = parseJsonColumn<string[]>(row.completed_step_ids, []);
  const rawStepOutputs = parseJsonColumn<Record<string, unknown>>(row.step_outputs, {});
  const result = parseJsonColumn<GuidedTrainingSessionPayload["result"]>(row.result_payload, null);
  // Sessions completed before nullable MAPE was introduced can retain a raw
  // Python `0` in staged outputs even though their canonical final metric is
  // already null. Normalize reads so historical all-zero runs are not presented
  // as a mathematically defined 0% MAPE.
  const stepOutputs = resultMarksMapeUnavailable(result)
    ? replaceUndefinedMapeWithNull(rawStepOutputs) as Record<string, unknown>
    : rawStepOutputs;
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

export const buildRefreshedCompletedGuidedTrainingSessionPayload = async (
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

export const buildCanonicalTrainingBody = (
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

export interface GuidedTrainingIdentity {
  requestHash: string;
  datasetRevision: string;
  runtimeRevision: string;
  baseRunFingerprint: Record<string, string>;
}

export const computeCurrentGuidedTrainingIdentity = async (
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
    options.terminationSignal,
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

export const stableJsonStringify = (value: unknown): string => {
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

export const guidedRequestPayloadMatches = (
  row: Pick<GuidedTrainingSessionRow, "request_payload">,
  canonicalBody: TrainInput,
) => stableJsonStringify(parseJsonColumn<TrainInput>(row.request_payload, {} as TrainInput))
  === stableJsonStringify(canonicalBody);

type GuidedSupersessionCasRow = Pick<
  GuidedTrainingSessionRow,
  | "session_id"
  | "status"
  | "current_step_id"
  | "active_attempt_id"
  | "request_hash"
  | "dataset_revision"
  | "runtime_revision"
  | "protocol_version"
  | "committed_state_path"
  | "committed_state_sha256"
>;

export const markGuidedTrainingSessionSuperseded = async (
  row: GuidedSupersessionCasRow,
  reason: string,
) => {
  const [result] = await queryWithTimeout(pool,
    `UPDATE model_guided_training_sessions
        SET status = 'superseded',
            current_step_id = NULL,
            active_attempt_id = NULL,
            superseded_reason = ?,
            error_message = NULL
      WHERE session_id = ?
        AND status = ?
        AND current_step_id <=> ?
        AND active_attempt_id <=> ?
        AND request_hash = ?
        AND dataset_revision = ?
        AND runtime_revision = ?
        AND protocol_version = ?
        AND committed_state_path <=> ?
        AND committed_state_sha256 <=> ?`,
    [
      reason,
      row.session_id,
      row.status,
      row.current_step_id,
      row.active_attempt_id,
      row.request_hash,
      row.dataset_revision,
      row.runtime_revision,
      row.protocol_version,
      row.committed_state_path,
      row.committed_state_sha256,
    ],
  );
  return Number((result as { affectedRows?: number }).affectedRows ?? 0) === 1;
};

const throwGuidedSupersessionResult = (claimed: boolean, reason: string): never => {
  if (!claimed) {
    throw new ModelConflictError(
      "训练会话状态已变化，请刷新后重试",
      "GUIDED_SESSION_STATE_CHANGED",
    );
  }
  throw new ModelConflictError(reason, "GUIDED_SESSION_SUPERSEDED");
};

export const assertGuidedTrainingSessionArtifactsCurrent = async (
  row: Pick<GuidedTrainingSessionRow, "artifact_dir"> & GuidedSupersessionCasRow,
) => {
  let reason: string | null = null;
  const datasetPath = path.join(row.artifact_dir, GUIDED_DATASET_FILE_NAME);

  try {
    const datasetDigest = crypto.createHash("sha256")
      .update(fs.readFileSync(datasetPath))
      .digest("hex");
    if (datasetDigest !== row.dataset_revision) {
      reason = "训练数据快照完整性校验失败，请重新创建训练会话";
    }
  } catch {
    reason = "训练数据快照已丢失或损坏，请重新创建训练会话";
  }

  const hasCommittedPath = row.committed_state_path !== null;
  const hasCommittedDigest = row.committed_state_sha256 !== null;
  if (!reason && hasCommittedPath !== hasCommittedDigest) {
    reason = "训练检查点记录不完整，请重新创建训练会话";
  }
  if (!reason && hasCommittedPath && hasCommittedDigest) {
    try {
      if (sha256File(row.committed_state_path!) !== row.committed_state_sha256) {
        reason = "训练检查点完整性校验失败，请重新创建训练会话";
      }
    } catch {
      reason = "训练检查点已丢失或损坏，请重新创建训练会话";
    }
  }

  if (!reason) return;
  const claimed = await markGuidedTrainingSessionSuperseded(row, reason);
  return throwGuidedSupersessionResult(claimed, reason);
};

export const assertGuidedTrainingIdentityCurrent = async (
  row: Pick<
    GuidedTrainingSessionRow,
    "experiment_id" | "model_type" | "request_payload"
  > & GuidedSupersessionCasRow,
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
    const claimed = await markGuidedTrainingSessionSuperseded(row, reason);
    return throwGuidedSupersessionResult(claimed, reason);
  }

  if (
    current.requestHash !== row.request_hash
    || current.datasetRevision !== row.dataset_revision
    || current.runtimeRevision !== row.runtime_revision
  ) {
    const reason = "训练数据、实验配置、基础模型或运行环境已变化，请重新创建训练会话";
    const claimed = await markGuidedTrainingSessionSuperseded(row, reason);
    return throwGuidedSupersessionResult(claimed, reason);
  }
  return current;
};

export const assertCompletedGuidedTrainingRunCurrent = async (
  row: Pick<
    GuidedTrainingSessionRow,
    "experiment_id" | "student_id" | "model_type" | "request_payload"
  > & GuidedSupersessionCasRow,
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
    const claimed = await markGuidedTrainingSessionSuperseded(row, reason);
    return throwGuidedSupersessionResult(claimed, reason);
  }
};

export const fetchGuidedTrainingSessionRow = async (
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

export const parseGuidedAttemptReceipt = (
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
    receipt.protocol_version !== GUIDED_TRAINING_PROTOCOL_VERSION
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

export const cleanupIncompleteGuidedAttemptArtifacts = (
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

export const fetchGuidedAttempt = async (
  connection: PoolConnection,
  attemptId: string,
) => {
  const [rows] = await connection.query<GuidedTrainingStepAttemptRow[]>(
    `SELECT *,
            lease_expires_at > NOW() AS lease_is_active
       FROM model_guided_training_step_attempts
      WHERE attempt_id = ?
      LIMIT 1`,
    [attemptId],
  );
  return rows[0] ?? null;
};

export const beginGuidedStepAttempt = async (
  row: GuidedTrainingSessionRow,
  stepId: string,
  timeoutMs: number,
) => {
  const connection = await acquireConnectionWithTimeout(pool);
  let transactionPhase: "before-commit" | "commit-attempted" | "committed" = "before-commit";
  let disposed = false;
  const commitTransaction = async () => {
    transactionPhase = "commit-attempted";
    try {
      await commitConnectionWithOutcomeGuard(connection);
    } catch (error) {
      disposed = true;
      throw error;
    }
    transactionPhase = "committed";
  };
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
          await commitTransaction();
          return { attempt: activeAttempt, receipt, recovered: true };
        }
        if (
          activeAttempt.status === "running"
          && Number(activeAttempt.lease_is_active) === 1
        ) {
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
    const inputStateSha256 = session.committed_state_sha256;
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
    // Read the inserted row while the transaction still owns it. If a
    // post-COMMIT read failed, the database would retain a live attempt lease
    // even though Python never started, blocking retries until lease expiry.
    const attempt = await fetchGuidedAttempt(connection, attemptId);
    if (!attempt) throw new Error("创建训练步骤 attempt 失败");
    await commitTransaction();
    return { attempt, receipt: null, recovered: false };
  } catch (error) {
    if (transactionPhase === "before-commit") {
      await connection.rollback().catch(() => {});
    }
    throw error;
  } finally {
    if (!disposed) connection.release();
  }
};

export const commitGuidedIntermediateStep = async (input: {
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
  let transactionPhase: "before-commit" | "commit-attempted" | "committed" = "before-commit";
  let disposed = false;
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
    transactionPhase = "commit-attempted";
    try {
      await commitConnectionWithOutcomeGuard(connection);
    } catch (error) {
      disposed = true;
      throw error;
    }
    transactionPhase = "committed";
  } catch (error) {
    if (transactionPhase === "before-commit") {
      await connection.rollback().catch(() => {});
    }
    throw error;
  } finally {
    if (!disposed) connection.release();
  }
};

export const recordGuidedAttemptPythonCompleted = async (
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

export const recordGuidedAttemptFailure = async (
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
        const [sessionRows] = await connection.query<RowDataPacket[]>(
          `SELECT session_id
             FROM model_guided_training_sessions
            WHERE session_id = ?
            LIMIT 1
            FOR UPDATE`,
          [sessionId],
        );
        if (!sessionRows[0]) return;

        await connection.query(
          `UPDATE model_guided_training_step_attempts
              SET status = IF(status = 'python_completed', status, 'failed'),
                  error_message = ?
            WHERE attempt_id = ?
              AND session_id = ?
              AND status IN ('running', 'python_completed')`,
          [detail, attemptId, sessionId],
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

export const buildGuidedTrainingPythonParams = async (
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
