import crypto from "crypto";
import fs from "fs";
import path from "path";
import type { PoolConnection, RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import logger from "../../utils/logger";
import {
  acquireConnectionWithTimeout,
  commitConnectionWithOutcomeGuard,
  queryWithTimeout,
} from "./model-db-guard";
import type { ModelType } from "./models.schema";
import {
  buildModelRetrainingRequiredDetails,
  ModelAccessError,
  ModelArtifactError,
  ModelConflictError,
  ModelNotFoundError,
  ModelValidationError,
  isModelCommitOutcomeUnknownError,
} from "./model-errors";
import {
  withModelOperationContext,
  withModelOperationLease,
} from "./model-operation-lease";
import {
  runPythonProductionPreparationStep,
} from "./model-python-runner";
import {
  commitProductionPreparationResult,
  type ModelTrainingRunRow,
} from "./model-run-store";
import type { BaseModelType } from "./model-types";
import { loadDatasetFromDb } from "./model-datasets";
import {
  buildProductionCandidateModelPath,
  computeBaseModelArtifactSha256,
  computeEnsembleArtifactSha256,
  hashFileContents,
  invalidateProductionArtifactForRun,
  removeModelArtifactGroup,
  verifyBacktestBaseArtifactForRun,
  verifyProductionBaseArtifactForRun,
} from "./model-artifacts";
import {
  BASE_MODEL_TIMEOUT_SECONDS,
  type ExperimentModelContextRow,
  assertExperimentModelAccess,
  assertModelReadyForProduction,
  getExperimentTrainingConfig,
  isBaseModelType,
  isDuplicateModelSessionRequestError,
  isRecordValue,
  lockStudentExperimentForModelWrite,
  parseJsonColumn,
  requireActiveModelTrainingRun,
} from "./model-context";
import {
  assertRunRevisionMatches,
  getModelRuntimeRevision,
  hashDatasetSnapshot,
  validateActiveRunAgainstSnapshot,
} from "./model-runtime-identity";
import {
  DEFAULT_PRODUCTION_FORECAST_STEPS,
  GUIDED_DATASET_FILE_NAME,
  MODEL_ORCHESTRATION_HARD_CAP,
  PRODUCTION_PREPARATION_PROTOCOL_VERSION,
  PRODUCTION_PREPARATION_ROOT,
  TRAINING_TIMEOUT,
} from "./model-service-constants";
import {
  removeProductionPreparationArtifactDir,
} from "./model-cleanup.service";

export type ProductionPreparationStatus = "ready" | "running" | "failed" | "completed" | "superseded";

export interface ProductionPreparationStepDefinition {
  id: string;
  label: string;
}

export interface ProductionPreparationSessionRow extends RowDataPacket {
  session_id: string;
  experiment_id: number;
  student_id: number;
  run_id: string;
  model_type: ModelType;
  request_hash: string;
  dataset_revision: string;
  runtime_revision: string;
  protocol_version: number;
  forecast_steps: number;
  status: ProductionPreparationStatus;
  current_step_id: string | null;
  completed_step_ids: unknown;
  step_plan: unknown;
  artifact_dir: string;
  candidate_model_path: string;
  committed_state_path: string | null;
  committed_state_sha256: string | null;
  active_attempt_id: string | null;
  observed_production_revision: string | null;
  production_revision: string | null;
  error_message: string | null;
  superseded_reason: string | null;
  expires_at: Date | string | null;
}

export interface ProductionPreparationAttemptRow extends RowDataPacket {
  attempt_id: string;
  session_id: string;
  step_id: string;
  status: "running" | "committed" | "failed";
  input_state_path: string | null;
  input_state_sha256: string | null;
  candidate_state_path: string;
  candidate_state_sha256: string | null;
  receipt_path: string;
  output_payload: unknown;
  error_message: string | null;
  lease_expires_at: Date | string | null;
  /** Computed by MySQL so DB_DATE_STRINGS and application timezone cannot skew lease ownership. */
  lease_is_active?: number | boolean | string | null;
}

const runProductionAttemptTransaction = async <T>(
  work: (connection: PoolConnection) => Promise<T>,
): Promise<T> => {
  const connection = await acquireConnectionWithTimeout(pool);
  let transactionPhase: "before-commit" | "commit-attempted" | "committed" = "before-commit";
  let disposed = false;
  try {
    await connection.beginTransaction();
    const result = await work(connection);
    transactionPhase = "commit-attempted";
    try {
      await commitConnectionWithOutcomeGuard(connection);
    } catch (error) {
      disposed = true;
      throw error;
    }
    transactionPhase = "committed";
    return result;
  } catch (error) {
    if (transactionPhase === "before-commit") {
      await connection.rollback().catch(() => {});
    }
    throw error;
  } finally {
    if (!disposed) connection.release();
  }
};

export interface ProductionPreparationSessionPayload {
  status: "success";
  results: {
    session_id: string;
    session_status: ProductionPreparationStatus;
    completed_step_ids: string[];
    total_steps: number;
    next_step: ProductionPreparationStepDefinition | null;
    error_message: string | null;
    preparation_token?: string;
    prepared_forecast_steps?: number;
    reused_existing?: boolean;
  };
}

const parseProductionPreparationStepPlan = (value: unknown): ProductionPreparationStepDefinition[] => {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is ProductionPreparationStepDefinition => (
    isRecordValue(item)
    && typeof item.id === "string"
    && item.id.length > 0
    && typeof item.label === "string"
    && item.label.length > 0
  ));
};

const buildProductionPreparationPayload = (
  row: ProductionPreparationSessionRow,
  reusedExisting = false,
): ProductionPreparationSessionPayload => {
  const completedStepIds = parseJsonColumn<string[]>(row.completed_step_ids, []);
  const stepPlan = parseProductionPreparationStepPlan(parseJsonColumn<unknown>(row.step_plan, []));
  const completed = new Set(completedStepIds);
  const nextStep = row.status === "completed" || row.status === "superseded"
    ? null
    : stepPlan.find((step) => !completed.has(step.id)) ?? null;
  return {
    status: "success",
    results: {
      session_id: row.session_id,
      session_status: row.status,
      completed_step_ids: completedStepIds,
      total_steps: stepPlan.length,
      next_step: nextStep,
      error_message: row.error_message,
      ...(row.status === "completed" && row.production_revision
        ? {
          preparation_token: row.production_revision,
          prepared_forecast_steps: Number(row.forecast_steps),
          reused_existing: reusedExisting,
        }
        : {}),
    },
  };
};

const completeProductionPreparationWithReusableArtifact = async (
  row: ProductionPreparationSessionRow,
  stepPlan: ProductionPreparationStepDefinition[],
  activeRun: ModelTrainingRunRow,
  preparationToken: string,
) => {
  const allStepIds = stepPlan.map((step) => step.id);
  const committed = await commitProductionPreparationResult({
    experimentId: row.experiment_id,
    studentId: row.student_id,
    modelType: row.model_type,
    runId: activeRun.run_id,
    sessionId: row.session_id,
    forecastSteps: row.forecast_steps,
    expectedProductionRevision: activeRun.production_revision,
    productionRevision: preparationToken,
    completedStepIds: allStepIds,
    stepPlan,
  });
  if (!committed.committed) {
    throw new ModelConflictError(
      committed.reason === "experiment_completed"
        ? "实验已完成，不能再提交生产准备结果"
        : "模型版本已变化，请重新准备",
      "PRODUCTION_PREPARATION_STALE",
    );
  }
  if (
    row.candidate_model_path !== activeRun.production_artifact_path
    && row.candidate_model_path !== activeRun.artifact_path
  ) {
    removeModelArtifactGroup(row.candidate_model_path);
  }
  const completed = await fetchProductionPreparationSession(row.session_id);
  if (!completed) throw new Error("生产准备会话完成后无法读取");
  return buildProductionPreparationPayload(completed, true);
};

const fetchProductionPreparationSession = async (sessionId: string) => {
  const [rows] = await queryWithTimeout<ProductionPreparationSessionRow[]>(pool,
    `SELECT session_id,
            experiment_id,
            student_id,
            run_id,
            model_type,
            request_hash,
            dataset_revision,
            runtime_revision,
            protocol_version,
            forecast_steps,
            status,
            current_step_id,
            completed_step_ids,
            step_plan,
            artifact_dir,
            candidate_model_path,
            committed_state_path,
            committed_state_sha256,
            active_attempt_id,
            observed_production_revision,
            production_revision,
            error_message,
            superseded_reason,
            expires_at
       FROM model_production_preparation_sessions
      WHERE session_id = ?
      LIMIT 1`,
    [sessionId],
  );
  return rows[0] ?? null;
};

const buildProductionPreparationArtifactDir = (
  experimentId: number,
  studentId: number,
  modelType: ModelType,
  sessionId: string,
) => path.join(
  PRODUCTION_PREPARATION_ROOT,
  `exp${experimentId}-student${studentId}-${modelType}-${sessionId}`,
);

const buildProductionPreparationRequestHash = (input: {
  runId: string;
  datasetRevision: string;
  runtimeRevision: string;
  forecastSteps: number;
}) => crypto.createHash("sha256").update(JSON.stringify(input)).digest("hex");

export interface ProductionPreparationIdentity {
  experiment: ExperimentModelContextRow;
  trainingConfig: ReturnType<typeof getExperimentTrainingConfig>;
  activeRun: ModelTrainingRunRow;
  datasetRevision: string;
  runtimeRevision: string;
  requestHash: string;
}

const markProductionPreparationSessionSuperseded = async (
  row: Pick<ProductionPreparationSessionRow, "session_id">,
  reason: string,
) => {
  await queryWithTimeout(pool,
    `UPDATE model_production_preparation_sessions
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

const computeCurrentProductionPreparationIdentity = async (
  row: Pick<
    ProductionPreparationSessionRow,
    "experiment_id" | "model_type" | "forecast_steps"
  >,
  studentId: number,
): Promise<ProductionPreparationIdentity> => {
  const experiment = await assertExperimentModelAccess(row.experiment_id, studentId);
  assertModelReadyForProduction(experiment, row.model_type);
  const activeRun = await requireActiveModelTrainingRun(
    row.experiment_id,
    studentId,
    row.model_type,
  );
  const trainingConfig = getExperimentTrainingConfig(experiment);
  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
  );
  const datasetJson = JSON.stringify(allData);
  const datasetRevision = hashDatasetSnapshot(datasetJson);
  const runtimeRevision = getModelRuntimeRevision(true);
  assertRunRevisionMatches(activeRun, datasetRevision, runtimeRevision);
  return {
    experiment,
    trainingConfig,
    activeRun,
    datasetRevision,
    runtimeRevision,
    requestHash: buildProductionPreparationRequestHash({
      runId: activeRun.run_id,
      datasetRevision,
      runtimeRevision,
      forecastSteps: Number(row.forecast_steps),
    }),
  };
};

const assertProductionPreparationIdentityCurrent = async (
  row: Pick<
    ProductionPreparationSessionRow,
    | "session_id"
    | "experiment_id"
    | "model_type"
    | "forecast_steps"
    | "run_id"
    | "request_hash"
    | "dataset_revision"
    | "runtime_revision"
  >,
  studentId: number,
): Promise<ProductionPreparationIdentity> => {
  let current: ProductionPreparationIdentity;
  try {
    current = await computeCurrentProductionPreparationIdentity(row, studentId);
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
      ? `生产准备依赖已变化：${error.message}`
      : "生产准备依赖已变化";
    await markProductionPreparationSessionSuperseded(row, reason);
    if (
      error instanceof ModelValidationError
      && error.details?.required_action === "retrain"
    ) {
      throw error;
    }
    throw new ModelConflictError(reason, "PRODUCTION_PREPARATION_STALE");
  }

  if (
    current.activeRun.run_id !== row.run_id
    || current.datasetRevision !== row.dataset_revision
    || current.runtimeRevision !== row.runtime_revision
    || current.requestHash !== row.request_hash
  ) {
    const reason = "训练版本、当前数据集或模型运行环境已变化，请重新训练后再准备生产模型";
    await markProductionPreparationSessionSuperseded(row, reason);
    throw new ModelConflictError(reason, "PRODUCTION_PREPARATION_STALE");
  }
  return current;
};

const verifyProductionSourceArtifactForRun = async (
  modelType: ModelType,
  activeRun: ModelTrainingRunRow,
  terminationSignal?: AbortSignal,
) => {
  try {
    return await verifyBacktestBaseArtifactForRun(
      modelType,
      activeRun,
      terminationSignal,
    );
  } catch (error) {
    if (!(error instanceof ModelArtifactError)) throw error;
    throw new ModelValidationError(
      `${modelType.toUpperCase()} 已训练模型产物已损坏，请重新训练该模型`,
      {
        ...buildModelRetrainingRequiredDetails(modelType),
        code: error.code,
      },
    );
  }
};

const buildProductionStepLabel = (stepId: string) => {
  const modelLabels: Record<string, string> = {
    ma: "移动平均",
    es: "指数平滑",
    arima: "ARIMA",
    lstm: "LSTM",
  };
  if (stepId === "initialize") return "确认生产准备材料";
  if (stepId === "fit_model") return "重拟合生产模型";
  if (stepId === "finalize") return "组装并校验生产产物";
  const member = /^fit_member_(ma|es|arima|lstm)$/.exec(stepId);
  if (member) return `重拟合成员：${modelLabels[member[1]!]}`;
  const stage = /^fit_stage_(\d+)_(ma|es|arima|lstm)$/.exec(stepId);
  if (stage) return `重拟合第 ${stage[1]} 阶段：${modelLabels[stage[2]!]}`;
  throw new ModelValidationError("生产准备返回了未知步骤");
};

export const expandProductionPreparationPlan = (
  currentPlan: ProductionPreparationStepDefinition[],
  output: unknown,
  modelType: ModelType,
) => {
  if (!isRecordValue(output) || !Array.isArray(output.work_items)) {
    throw new ModelValidationError("生产准备初始化未返回有效工作项");
  }
  const workItems = output.work_items.map((item) => {
    if (!isRecordValue(item) || typeof item.id !== "string") {
      throw new ModelValidationError("生产准备初始化返回了无效工作项");
    }
    if (!/^(?:fit_model|fit_member_(?:ma|es|arima|lstm)|fit_stage_\d+_(?:ma|es|arima|lstm))$/.test(item.id)) {
      throw new ModelValidationError("生产准备初始化返回了未知工作项");
    }
    return { id: item.id, label: buildProductionStepLabel(item.id) };
  });
  const deduped = new Set(workItems.map((item) => item.id));
  const directFinalize = modelType === "boosting" && output.direct_finalize === true;
  if (
    deduped.size !== workItems.length
    || (workItems.length === 0 && !directFinalize)
    || (workItems.length > 0 && output.direct_finalize === true)
  ) {
    throw new ModelValidationError("生产准备初始化返回了重复或空工作项");
  }
  return [currentPlan[0]!, ...workItems, currentPlan[currentPlan.length - 1]!];
};

const resolveProductionPreparationStepTimeoutMs = (modelType: ModelType, stepId: string) => {
  const member = (
    stepId === "fit_model" && isBaseModelType(modelType)
      ? modelType
      : /_(ma|es|arima|lstm)$/.exec(stepId)?.[1]
  ) as BaseModelType | undefined;
  if (!member) return TRAINING_TIMEOUT;
  return Math.min(
    MODEL_ORCHESTRATION_HARD_CAP,
    Math.max(TRAINING_TIMEOUT, BASE_MODEL_TIMEOUT_SECONDS[member] * 1000 + 30_000),
  );
};

export async function createOrResumeProductionPreparationSession(
  modelType: ModelType,
  studentId: number,
  experimentId: number,
  forecastSteps = DEFAULT_PRODUCTION_FORECAST_STEPS,
): Promise<ProductionPreparationSessionPayload> {
  return await withModelOperationContext("production_preparation", async (operationContext) => {
    const experiment = await assertExperimentModelAccess(experimentId, studentId);
    assertModelReadyForProduction(experiment, modelType);
    const activeRun = await requireActiveModelTrainingRun(experimentId, studentId, modelType);
    const trainingConfig = getExperimentTrainingConfig(experiment);
    const allData = await loadDatasetFromDb(
      trainingConfig.selected_industry,
      trainingConfig.selected_company,
      trainingConfig.selected_product,
    );
    operationContext.throwIfTerminated();
    const datasetJson = JSON.stringify(allData);
    validateActiveRunAgainstSnapshot(activeRun, datasetJson);
    const datasetRevision = hashDatasetSnapshot(datasetJson);
    const runtimeRevision = getModelRuntimeRevision(true);
    const requestHash = buildProductionPreparationRequestHash({
      runId: activeRun.run_id,
      datasetRevision,
      runtimeRevision,
      forecastSteps,
    });
    const [existingRows] = await queryWithTimeout<ProductionPreparationSessionRow[]>(pool,
      `SELECT session_id,
              experiment_id,
              student_id,
              run_id,
              model_type,
              request_hash,
              dataset_revision,
              runtime_revision,
              protocol_version,
              forecast_steps,
              status,
              current_step_id,
              completed_step_ids,
              step_plan,
              artifact_dir,
              candidate_model_path,
              committed_state_path,
              committed_state_sha256,
              active_attempt_id,
              observed_production_revision,
              production_revision,
              error_message,
              superseded_reason,
              expires_at
         FROM model_production_preparation_sessions
        WHERE experiment_id = ?
          AND student_id = ?
          AND model_type = ?
          AND request_hash = ?
        LIMIT 1`,
      [experimentId, studentId, modelType, requestHash],
    );
    const existing = existingRows[0];
    // Finalization records the production artifact before marking the session
    // completed. Re-read the active run before deciding that a completed
    // session is stale; otherwise a concurrent create request that loaded the
    // pre-finalization run could delete the artifact that just committed.
    const currentActiveRun = existing?.status === "completed"
      ? await requireActiveModelTrainingRun(experimentId, studentId, modelType)
      : activeRun;
    if (
      currentActiveRun.run_id !== activeRun.run_id
      || currentActiveRun.dataset_revision !== activeRun.dataset_revision
      || currentActiveRun.runtime_revision !== activeRun.runtime_revision
    ) {
      throw new ModelConflictError(
        "生产准备期间训练版本已变化，请重试",
        "PRODUCTION_PREPARATION_STALE",
      );
    }
    const completedSessionStillCurrent = existing?.status !== "completed" || (
      Boolean(existing.production_revision)
      && existing.production_revision === currentActiveRun.production_revision
      && currentActiveRun.production_artifact_forecast_steps !== null
      && Number(currentActiveRun.production_artifact_forecast_steps) >= forecastSteps
    );
    if (
      existing
      && existing.protocol_version === PRODUCTION_PREPARATION_PROTOCOL_VERSION
      && existing.status !== "superseded"
      && completedSessionStillCurrent
    ) {
      return buildProductionPreparationPayload(existing);
    }
    if (existing) {
      if (existing.candidate_model_path !== currentActiveRun.production_artifact_path) {
        removeModelArtifactGroup(existing.candidate_model_path);
      }
      if (!removeProductionPreparationArtifactDir(existing.artifact_dir)) {
        throw new ModelValidationError("旧的生产准备会话工件目录不安全，无法重建会话");
      }
      await queryWithTimeout(pool,
        "DELETE FROM model_production_preparation_sessions WHERE session_id = ?",
        [existing.session_id],
      );
    }

    const sessionId = crypto.randomUUID();
    const artifactDir = buildProductionPreparationArtifactDir(
      experimentId,
      studentId,
      modelType,
      sessionId,
    );
    const candidateModelPath = buildProductionCandidateModelPath(
      experimentId,
      studentId,
      modelType,
      activeRun.run_id,
    );
    const stepPlan: ProductionPreparationStepDefinition[] = [
      { id: "initialize", label: buildProductionStepLabel("initialize") },
      { id: "finalize", label: buildProductionStepLabel("finalize") },
    ];
    let insertCommitted = false;
    try {
      fs.mkdirSync(artifactDir, { recursive: true });
      await Bun.write(path.join(artifactDir, GUIDED_DATASET_FILE_NAME), datasetJson);
      operationContext.throwIfTerminated();
      const connection = await acquireConnectionWithTimeout(pool);
      let transactionPhase: "before-commit" | "commit-attempted" | "committed" = "before-commit";
      let disposed = false;
      try {
        await connection.beginTransaction();
        await lockStudentExperimentForModelWrite(connection, experimentId, studentId);
        operationContext.throwIfTerminated();
        const [lockedRunRows] = await connection.query<RowDataPacket[]>(
          `SELECT run_id,
                  request_hash,
                  dataset_revision,
                  runtime_revision,
                  artifact_path,
                  artifact_sha256,
                  production_revision,
                  is_active
             FROM model_training_runs
            WHERE experiment_id = ?
              AND student_id = ?
              AND model_type = ?
              AND is_active = TRUE
            ORDER BY activated_at DESC
            LIMIT 1
            FOR UPDATE`,
          [experimentId, studentId, modelType],
        );
        const lockedRun = lockedRunRows[0];
        if (
          !lockedRun
          || lockedRun.run_id !== activeRun.run_id
          || lockedRun.request_hash !== activeRun.request_hash
          || lockedRun.dataset_revision !== activeRun.dataset_revision
          || lockedRun.runtime_revision !== activeRun.runtime_revision
          || lockedRun.artifact_path !== activeRun.artifact_path
          || (lockedRun.artifact_sha256 ?? null) !== (activeRun.artifact_sha256 ?? null)
          || (lockedRun.production_revision ?? null) !== (activeRun.production_revision ?? null)
          || !Boolean(lockedRun.is_active)
        ) {
          throw new ModelConflictError(
            "生产准备期间训练版本已变化，请重试",
            "PRODUCTION_PREPARATION_STALE",
          );
        }
        operationContext.throwIfTerminated();
        await connection.query<RowDataPacket[]>(
          `INSERT INTO model_production_preparation_sessions
           (session_id, experiment_id, student_id, run_id, model_type, request_hash,
            dataset_revision, runtime_revision, protocol_version, forecast_steps,
            status, current_step_id, completed_step_ids, step_plan, artifact_dir,
            candidate_model_path, observed_production_revision, expires_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', 'initialize', ?, ?, ?, ?, ?, DATE_ADD(NOW(), INTERVAL 7 DAY))`,
          [
            sessionId,
            experimentId,
            studentId,
            activeRun.run_id,
            modelType,
            requestHash,
            datasetRevision,
            runtimeRevision,
            PRODUCTION_PREPARATION_PROTOCOL_VERSION,
            forecastSteps,
            JSON.stringify([]),
            JSON.stringify(stepPlan),
            artifactDir,
            candidateModelPath,
            activeRun.production_revision,
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
        insertCommitted = true;
      } catch (error) {
        if (transactionPhase === "before-commit") {
          await connection.rollback().catch(() => {});
        }
        throw error;
      } finally {
        if (!disposed) connection.release();
      }
    } catch (error) {
      if (!isDuplicateModelSessionRequestError(error)) {
        if (!insertCommitted && !isModelCommitOutcomeUnknownError(error)) {
          removeProductionPreparationArtifactDir(artifactDir);
        }
        throw error;
      }
      removeProductionPreparationArtifactDir(artifactDir);
      operationContext.throwIfTerminated();
      const [racedRows] = await queryWithTimeout<ProductionPreparationSessionRow[]>(pool,
        `SELECT *
           FROM model_production_preparation_sessions
          WHERE experiment_id = ?
            AND student_id = ?
            AND model_type = ?
            AND request_hash = ?
          LIMIT 1`,
        [experimentId, studentId, modelType, requestHash],
      );
      operationContext.throwIfTerminated();
      if (racedRows[0]) return buildProductionPreparationPayload(racedRows[0]);
      throw error;
    }
    const created = await fetchProductionPreparationSession(sessionId);
    if (!created) throw new Error("生产准备会话创建后无法读取");
    return buildProductionPreparationPayload(created);
  });
}

export async function getProductionPreparationSession(
  modelType: ModelType,
  studentId: number,
  sessionId: string,
): Promise<ProductionPreparationSessionPayload> {
  const row = await fetchProductionPreparationSession(sessionId);
  if (!row || row.model_type !== modelType) throw new ModelNotFoundError("未找到生产准备会话");
  if (row.student_id !== studentId) throw new ModelAccessError("无权访问该生产准备会话", 403);
  await assertExperimentModelAccess(row.experiment_id, studentId);
  return buildProductionPreparationPayload(row);
}

const validateProductionPreparationReceipt = async (
  result: Awaited<ReturnType<typeof runPythonProductionPreparationStep>>,
  attempt: ProductionPreparationAttemptRow,
) => {
  if (
    result.protocol_version !== PRODUCTION_PREPARATION_PROTOCOL_VERSION
    || result.attempt_id !== attempt.attempt_id
    || result.step_id !== attempt.step_id
    || result.candidate_state_path !== attempt.candidate_state_path
    || !/^[a-f\d]{64}$/i.test(result.candidate_state_sha256)
  ) {
    throw new ModelValidationError("生产准备步骤返回了无效 checkpoint receipt");
  }
  if (attempt.input_state_path) {
    if (
      result.input_state_path !== attempt.input_state_path
      || result.input_state_sha256 !== attempt.input_state_sha256
      || await hashFileContents(attempt.input_state_path) !== attempt.input_state_sha256
    ) {
      throw new ModelValidationError("生产准备步骤输入 checkpoint 摘要不匹配");
    }
  } else if (result.input_state_path != null || result.input_state_sha256 != null) {
    throw new ModelValidationError("生产准备初始化返回了意外的输入 checkpoint");
  }
  const actualDigest = await hashFileContents(attempt.candidate_state_path);
  if (actualDigest !== result.candidate_state_sha256) {
    throw new ModelValidationError("生产准备 checkpoint 摘要不匹配");
  }
  return result;
};

export const installProductionPreparationAttempt = async (input: {
  session: ProductionPreparationSessionRow;
  attempt: ProductionPreparationAttemptRow;
  stepId: string;
  observedProductionRevision: string | null;
  leaseSeconds: number;
}) => {
  await runProductionAttemptTransaction(async (connection) => {
    const [sessionRows] = await connection.query<ProductionPreparationSessionRow[]>(
      `SELECT *
         FROM model_production_preparation_sessions
        WHERE session_id = ?
        LIMIT 1
        FOR UPDATE`,
      [input.session.session_id],
    );
    const session = sessionRows[0];
    if (
      !session
      || Number(session.student_id) !== Number(input.session.student_id)
      || session.model_type !== input.session.model_type
      || Number(session.protocol_version) !== PRODUCTION_PREPARATION_PROTOCOL_VERSION
      || session.status === "completed"
      || session.status === "superseded"
    ) {
      throw new ModelConflictError(
        "生产准备会话状态已变化，请刷新后重试",
        "PRODUCTION_PREPARATION_STALE",
      );
    }

    const completedStepIds = parseJsonColumn<string[]>(session.completed_step_ids, []);
    const stepPlan = parseProductionPreparationStepPlan(
      parseJsonColumn<unknown>(session.step_plan, []),
    );
    const nextStep = stepPlan.find((step) => !completedStepIds.includes(step.id));
    if (completedStepIds.includes(input.stepId) || nextStep?.id !== input.stepId) {
      throw new ModelConflictError(
        "生产准备步骤已变化，请刷新后重试",
        "PRODUCTION_PREPARATION_STEP_CONFLICT",
      );
    }

    const previousAttemptId = session.active_attempt_id;
    if (previousAttemptId) {
      const [attemptRows] = await connection.query<ProductionPreparationAttemptRow[]>(
        `SELECT *,
                lease_expires_at > NOW() AS lease_is_active
           FROM model_production_preparation_step_attempts
          WHERE attempt_id = ?
            AND session_id = ?
          LIMIT 1`,
        [previousAttemptId, session.session_id],
      );
      const previousAttempt = attemptRows[0];
      if (
        previousAttempt?.status === "running"
        && Number(previousAttempt.lease_is_active) === 1
      ) {
        throw new ModelConflictError(
          "当前生产准备步骤仍在执行",
          "PRODUCTION_PREPARATION_STEP_IN_PROGRESS",
        );
      }
      await connection.query(
        `UPDATE model_production_preparation_step_attempts
            SET status = 'failed',
                error_message = COALESCE(error_message, 'Attempt lease expired or was replaced')
          WHERE attempt_id = ?
            AND session_id = ?
            AND status = 'running'`,
        [previousAttemptId, session.session_id],
      );
    }

    await connection.query(
      `INSERT INTO model_production_preparation_step_attempts
       (attempt_id, session_id, step_id, status, input_state_path,
        input_state_sha256, candidate_state_path, receipt_path, lease_expires_at)
       VALUES (?, ?, ?, 'running', ?, ?, ?, ?, DATE_ADD(NOW(), INTERVAL ? SECOND))`,
      [
        input.attempt.attempt_id,
        session.session_id,
        input.stepId,
        input.attempt.input_state_path,
        input.attempt.input_state_sha256,
        input.attempt.candidate_state_path,
        input.attempt.receipt_path,
        input.leaseSeconds,
      ],
    );
    const [sessionUpdate] = await connection.query(
      `UPDATE model_production_preparation_sessions
          SET status = 'running',
              current_step_id = ?,
              active_attempt_id = ?,
              observed_production_revision = ?,
              error_message = NULL,
              expires_at = DATE_ADD(NOW(), INTERVAL 7 DAY)
        WHERE session_id = ?
          AND status IN ('ready', 'failed', 'running')
          AND active_attempt_id <=> ?`,
      [
        input.stepId,
        input.attempt.attempt_id,
        input.observedProductionRevision,
        session.session_id,
        previousAttemptId,
      ],
    );
    if (Number((sessionUpdate as { affectedRows?: number }).affectedRows ?? 0) !== 1) {
      throw new ModelConflictError(
        "生产准备步骤 attempt 已被替换",
        "PRODUCTION_PREPARATION_STEP_IN_PROGRESS",
      );
    }
  });
};

export const commitProductionPreparationIntermediateAttempt = async (input: {
  sessionId: string;
  attemptId: string;
  status: ProductionPreparationStatus;
  nextStepId: string | null;
  completedStepIds: string[];
  stepPlan: ProductionPreparationStepDefinition[];
  committedStatePath: string;
  committedStateSha256: string;
  outputPayload: unknown;
}) => {
  await runProductionAttemptTransaction(async (connection) => {
    const [sessionRows] = await connection.query<RowDataPacket[]>(
      `SELECT active_attempt_id, status
         FROM model_production_preparation_sessions
        WHERE session_id = ?
        LIMIT 1
        FOR UPDATE`,
      [input.sessionId],
    );
    const session = sessionRows[0];
    if (
      !session
      || session.status !== "running"
      || session.active_attempt_id !== input.attemptId
    ) {
      throw new ModelConflictError(
        "生产准备步骤 attempt 已被替换",
        "PRODUCTION_PREPARATION_STEP_IN_PROGRESS",
      );
    }

    const [attemptUpdate] = await connection.query(
      `UPDATE model_production_preparation_step_attempts
          SET status = 'committed',
              candidate_state_sha256 = ?,
              output_payload = ?,
              error_message = NULL
        WHERE attempt_id = ?
          AND session_id = ?
          AND status = 'running'`,
      [
        input.committedStateSha256,
        JSON.stringify(input.outputPayload ?? {}),
        input.attemptId,
        input.sessionId,
      ],
    );
    if (Number((attemptUpdate as { affectedRows?: number }).affectedRows ?? 0) !== 1) {
      throw new ModelConflictError(
        "生产准备步骤 attempt 已被替换",
        "PRODUCTION_PREPARATION_STEP_IN_PROGRESS",
      );
    }

    const [sessionUpdate] = await connection.query(
      `UPDATE model_production_preparation_sessions
          SET status = ?,
              current_step_id = ?,
              completed_step_ids = ?,
              step_plan = ?,
              committed_state_path = ?,
              committed_state_sha256 = ?,
              active_attempt_id = NULL,
              production_revision = NULL,
              error_message = NULL,
              expires_at = DATE_ADD(NOW(), INTERVAL 7 DAY)
        WHERE session_id = ?
          AND status = 'running'
          AND active_attempt_id = ?`,
      [
        input.status,
        input.nextStepId,
        JSON.stringify(input.completedStepIds),
        JSON.stringify(input.stepPlan),
        input.committedStatePath,
        input.committedStateSha256,
        input.sessionId,
        input.attemptId,
      ],
    );
    if (Number((sessionUpdate as { affectedRows?: number }).affectedRows ?? 0) !== 1) {
      throw new ModelConflictError(
        "生产准备步骤 attempt 已被替换",
        "PRODUCTION_PREPARATION_STEP_IN_PROGRESS",
      );
    }
  });
};

export const recordProductionPreparationAttemptFailure = async (input: {
  sessionId: string;
  attemptId: string;
  stepId: string;
  message: string;
}) => {
  try {
    await runProductionAttemptTransaction(async (connection) => {
      const [sessionRows] = await connection.query<RowDataPacket[]>(
        `SELECT active_attempt_id, status
           FROM model_production_preparation_sessions
          WHERE session_id = ?
          LIMIT 1
          FOR UPDATE`,
        [input.sessionId],
      );
      const session = sessionRows[0];
      if (
        !session
        || session.status !== "running"
        || session.active_attempt_id !== input.attemptId
      ) {
        return;
      }

      await connection.query(
        `UPDATE model_production_preparation_step_attempts
            SET status = 'failed', error_message = ?
          WHERE attempt_id = ?
            AND session_id = ?
            AND status = 'running'`,
        [input.message, input.attemptId, input.sessionId],
      );
      await connection.query(
        `UPDATE model_production_preparation_sessions
            SET status = 'failed',
                current_step_id = ?,
                active_attempt_id = NULL,
                error_message = ?,
                expires_at = DATE_ADD(NOW(), INTERVAL 7 DAY)
          WHERE session_id = ?
            AND status = 'running'
            AND active_attempt_id = ?`,
        [
          input.stepId,
          input.message,
          input.sessionId,
          input.attemptId,
        ],
      );
    });
  } catch (recordError) {
    logger.warn(
      {
        err: recordError,
        sessionId: input.sessionId,
        attemptId: input.attemptId,
      },
      "记录 production preparation attempt 失败状态失败",
    );
  }
};

export async function runProductionPreparationStep(
  modelType: ModelType,
  studentId: number,
  sessionId: string,
  stepId: string,
): Promise<ProductionPreparationSessionPayload> {
  const operationContextName = "production_preparation" as const;
  return await withModelOperationContext(operationContextName, async (operationContext) => {
    const initialRow = await fetchProductionPreparationSession(sessionId);
    if (!initialRow || initialRow.model_type !== modelType) {
      throw new ModelNotFoundError("未找到生产准备会话");
    }
    if (initialRow.student_id !== studentId) throw new ModelAccessError("无权访问该生产准备会话", 403);
    if (initialRow.protocol_version !== PRODUCTION_PREPARATION_PROTOCOL_VERSION || initialRow.status === "superseded") {
      throw new ModelConflictError("生产准备会话已失效，请重新创建", "PRODUCTION_PREPARATION_STALE");
    }
    const completedStepIds = parseJsonColumn<string[]>(initialRow.completed_step_ids, []);
    if (completedStepIds.includes(stepId)) return buildProductionPreparationPayload(initialRow);
    const stepPlan = parseProductionPreparationStepPlan(parseJsonColumn<unknown>(initialRow.step_plan, []));
    const nextStep = stepPlan.find((step) => !completedStepIds.includes(step.id));
    if (!nextStep || nextStep.id !== stepId) {
      throw new ModelConflictError("只能执行生产准备会话的下一步骤", "PRODUCTION_PREPARATION_STEP_CONFLICT");
    }

    const [refreshResult] = await queryWithTimeout(pool,
      `UPDATE model_production_preparation_sessions
          SET expires_at = CASE
                WHEN expires_at IS NULL THEN DATE_ADD(NOW(), INTERVAL 7 DAY)
                ELSE GREATEST(
                  DATE_ADD(expires_at, INTERVAL 1 SECOND),
                  DATE_ADD(NOW(), INTERVAL 7 DAY)
                )
              END
        WHERE session_id = ?
          AND status IN ('ready', 'failed', 'running')
          AND (expires_at IS NULL OR expires_at > NOW())`,
      [sessionId],
    );
    if (Number((refreshResult as { affectedRows?: number }).affectedRows ?? 0) !== 1) {
      throw new ModelConflictError("生产准备会话已过期，请重新创建", "PRODUCTION_PREPARATION_STALE");
    }

    return await withModelOperationLease(
      operationContext,
      { mode: "slot_and_execution", experimentId: initialRow.experiment_id, modelType },
      async (lease) => {
        // A second tab may have waited for the same advisory lock while the
        // first request committed this step. Re-read under the lock so stale
        // checkpoint pointers can never replay or overwrite a later step.
        const currentRow = await fetchProductionPreparationSession(sessionId);
        if (!currentRow || currentRow.model_type !== modelType || currentRow.student_id !== studentId) {
          throw new ModelNotFoundError("未找到生产准备会话");
        }
        if (
          currentRow.protocol_version !== PRODUCTION_PREPARATION_PROTOCOL_VERSION
          || currentRow.status === "superseded"
        ) {
          throw new ModelConflictError("生产准备会话已失效，请重新创建", "PRODUCTION_PREPARATION_STALE");
        }
        const currentCompletedStepIds = parseJsonColumn<string[]>(currentRow.completed_step_ids, []);
        if (currentCompletedStepIds.includes(stepId)) {
          return buildProductionPreparationPayload(currentRow);
        }
        const currentStepPlan = parseProductionPreparationStepPlan(
          parseJsonColumn<unknown>(currentRow.step_plan, []),
        );
        const currentNextStep = currentStepPlan.find((step) => !currentCompletedStepIds.includes(step.id));
        if (!currentNextStep || currentNextStep.id !== stepId) {
          throw new ModelConflictError(
            "只能执行生产准备会话的下一步骤",
            "PRODUCTION_PREPARATION_STEP_CONFLICT",
          );
        }
        let stepIdentity = await assertProductionPreparationIdentityCurrent(
          currentRow,
          studentId,
        );
        const reusable = await getReusableProductionModelArtifactResult(
          modelType,
          currentRow.forecast_steps,
          stepIdentity.experiment,
          stepIdentity.activeRun,
          lease.terminationSignal,
        );
        if (reusable) {
          await lease.verifyOwned();
          return await completeProductionPreparationWithReusableArtifact(
            currentRow,
            currentStepPlan,
            stepIdentity.activeRun,
            reusable.results.preparation_token,
          );
        }
        // A failed reusable-artifact verification clears that artifact from
        // the active run. Re-read before recording the CAS baseline used by
        // this attempt.
        stepIdentity = await assertProductionPreparationIdentityCurrent(
          currentRow,
          studentId,
        );
        const { activeRun } = stepIdentity;

        await verifyProductionSourceArtifactForRun(
          modelType,
          activeRun,
          lease.terminationSignal,
        );
        const datasetSnapshotPath = path.join(currentRow.artifact_dir, GUIDED_DATASET_FILE_NAME);
        const datasetSnapshotSha256 = await hashFileContents(datasetSnapshotPath, lease.terminationSignal);
        if (datasetSnapshotSha256 !== currentRow.dataset_revision) {
          throw new ModelConflictError(
            "生产准备数据快照已损坏，请重新创建会话",
            "PRODUCTION_PREPARATION_STALE",
          );
        }
        if (currentRow.committed_state_path) {
          if (
            !currentRow.committed_state_sha256
            || await hashFileContents(currentRow.committed_state_path, lease.terminationSignal)
              !== currentRow.committed_state_sha256
          ) {
            throw new ModelConflictError(
              "生产准备 checkpoint 已损坏，请重新创建会话",
              "PRODUCTION_PREPARATION_STALE",
            );
          }
        } else if (currentRow.committed_state_sha256) {
          throw new ModelConflictError(
            "生产准备 checkpoint 状态不完整，请重新创建会话",
            "PRODUCTION_PREPARATION_STALE",
          );
        }
        const timeoutMs = resolveProductionPreparationStepTimeoutMs(modelType, stepId);
        const attemptLeaseSeconds = Math.ceil((timeoutMs + 60_000) / 1000);
        const attemptId = crypto.randomUUID();
        const attemptDir = path.join(currentRow.artifact_dir, "attempts", attemptId);
        fs.mkdirSync(attemptDir, { recursive: true });
        const attempt: ProductionPreparationAttemptRow = {
          attempt_id: attemptId,
          session_id: sessionId,
          step_id: stepId,
          status: "running",
          input_state_path: currentRow.committed_state_path,
          input_state_sha256: currentRow.committed_state_sha256,
          candidate_state_path: path.join(attemptDir, "candidate.pkl"),
          candidate_state_sha256: null,
          receipt_path: path.join(attemptDir, "receipt.json"),
          output_payload: null,
          error_message: null,
          lease_expires_at: null,
        } as ProductionPreparationAttemptRow;
        try {
          await installProductionPreparationAttempt({
            session: currentRow,
            attempt,
            stepId,
            observedProductionRevision: activeRun.production_revision,
            leaseSeconds: attemptLeaseSeconds,
          });
        } catch (error) {
          if (!isModelCommitOutcomeUnknownError(error)) {
            fs.rmSync(attemptDir, { recursive: true, force: true });
          }
          throw error;
        }

        try {
          const rawReceipt = await runPythonProductionPreparationStep(
            modelType,
            stepId,
            {
              dataFile: datasetSnapshotPath,
              sourceModel: activeRun.artifact_path,
              saveModel: currentRow.candidate_model_path,
              trainStart: stepIdentity.trainingConfig.data_window_train_start_index,
              trainEnd: stepIdentity.trainingConfig.data_window_evaluate_end_index,
              forecastSteps: currentRow.forecast_steps,
              pretty: true,
            },
            attemptId,
            attempt.input_state_path,
            attempt.candidate_state_path,
            attempt.receipt_path,
            currentRow.artifact_dir,
            timeoutMs,
            lease.terminationSignal,
          );
          const receipt = await validateProductionPreparationReceipt(rawReceipt, attempt);
          await verifyProductionSourceArtifactForRun(
            modelType,
            activeRun,
            lease.terminationSignal,
          );
          await assertProductionPreparationIdentityCurrent(currentRow, studentId);
          await lease.verifyOwned();
          const nextCompleted = [...currentCompletedStepIds, stepId];
          let nextPlan = currentStepPlan;
          if (stepId === "initialize") {
            nextPlan = expandProductionPreparationPlan(
              currentStepPlan,
              receipt.output,
              modelType,
            );
          } else if (
            /^fit_stage_\d+_(?:ma|es|arima|lstm)$/.test(stepId)
            && isRecordValue(receipt.output)
            && receipt.output.trim_remaining === true
          ) {
            nextPlan = currentStepPlan.filter((step) => (
              !/^fit_stage_\d+_(?:ma|es|arima|lstm)$/.test(step.id)
              || currentCompletedStepIds.includes(step.id)
              || step.id === stepId
            ));
            const finalizeStep = currentStepPlan.find((step) => step.id === "finalize");
            if (finalizeStep && !nextPlan.some((step) => step.id === "finalize")) {
              nextPlan.push(finalizeStep);
            }
          }
          const finalResult = isRecordValue(receipt.final_result) ? receipt.final_result : null;
          let productionRevision: string | null = null;
          let replacedProductionArtifactPath: string | null = null;
          if (finalResult) {
            const artifactSha256 = isBaseModelType(modelType)
              ? await computeBaseModelArtifactSha256(modelType, currentRow.candidate_model_path, lease.terminationSignal)
              : await computeEnsembleArtifactSha256(modelType, currentRow.candidate_model_path, lease.terminationSignal);
            const commitIdentity = await assertProductionPreparationIdentityCurrent(
              currentRow,
              studentId,
            );
            await lease.verifyOwned();
            productionRevision = crypto.randomUUID();
            const committed = await commitProductionPreparationResult({
              experimentId: currentRow.experiment_id,
              studentId: currentRow.student_id,
              modelType,
              runId: commitIdentity.activeRun.run_id,
              sessionId,
              forecastSteps: currentRow.forecast_steps,
              expectedProductionRevision: commitIdentity.activeRun.production_revision,
              productionRevision,
              completedStepIds: nextCompleted,
              stepPlan: nextPlan,
              attempt: {
                attemptId,
                artifactPath: currentRow.candidate_model_path,
                artifactSha256,
                committedStatePath: receipt.candidate_state_path,
                committedStateSha256: receipt.candidate_state_sha256,
                outputPayload: receipt.output ?? {},
              },
            });
            if (!committed.committed) {
              throw new ModelConflictError(
                committed.reason === "experiment_completed"
                  ? "实验已完成，不能再提交生产准备结果"
                  : "模型版本已变化，请重新准备",
                "PRODUCTION_PREPARATION_STALE",
              );
            }
            replacedProductionArtifactPath = committed.replacedProductionArtifactPath;
          }
          const status: ProductionPreparationStatus = finalResult ? "completed" : "ready";
          const nextStepId = finalResult
            ? null
            : nextPlan.find((step) => !nextCompleted.includes(step.id))?.id ?? null;
          if (!finalResult) {
            await commitProductionPreparationIntermediateAttempt({
              sessionId,
              attemptId,
              status,
              nextStepId,
              completedStepIds: nextCompleted,
              stepPlan: nextPlan,
              committedStatePath: receipt.candidate_state_path,
              committedStateSha256: receipt.candidate_state_sha256,
              outputPayload: receipt.output ?? {},
            });
          }
          if (
            finalResult
            && replacedProductionArtifactPath
            && replacedProductionArtifactPath !== currentRow.candidate_model_path
            && replacedProductionArtifactPath !== activeRun.artifact_path
          ) {
            removeModelArtifactGroup(replacedProductionArtifactPath);
          }
          const updated = await fetchProductionPreparationSession(sessionId);
          if (!updated) throw new Error("生产准备步骤提交后无法读取会话");
          return buildProductionPreparationPayload(updated);
        } catch (error) {
          if (!isModelCommitOutcomeUnknownError(error)) {
            const message = error instanceof Error ? error.message : "生产准备步骤失败";
            await recordProductionPreparationAttemptFailure({
              sessionId,
              attemptId,
              stepId,
              message: message.slice(0, 4000),
            });
          }
          throw error;
        }
      },
    );
  });
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
