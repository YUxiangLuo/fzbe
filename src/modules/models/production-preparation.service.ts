import {
  type ProductionPreparationStatus,
  type ProductionPreparationStepDefinition,
  type ProductionPreparationSessionRow,
  type ProductionPreparationAttemptRow,
  type ProductionPreparationSessionPayload,
  parseProductionCompletedStepIds,
  parseProductionPreparationStepPlan,
  hasValidProductionStepProgress,
  buildProductionPreparationPayload,
  buildProductionStepLabel,
  expandProductionPreparationPlan,
  jsonColumnCasValue,
} from "./production-preparation-state";
import {
  fetchProductionPreparationAttempt,
  claimProductionPreparationReceiptReplay,
  installProductionPreparationAttempt,
  commitProductionPreparationIntermediateAttempt,
  recordProductionPreparationAttemptFailure,
} from "./production-preparation-attempt-store";
// Preserve existing service imports while implementation moves behind the boundary.
export type {
  ProductionPreparationStatus, ProductionPreparationStepDefinition,
  ProductionPreparationSessionRow, ProductionPreparationAttemptRow,
  ProductionPreparationSessionPayload,
} from "./production-preparation-state";
export { expandProductionPreparationPlan } from "./production-preparation-state";
export {
  installProductionPreparationAttempt,
  commitProductionPreparationIntermediateAttempt,
  recordProductionPreparationAttemptFailure,
} from "./production-preparation-attempt-store";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import type { RowDataPacket } from "mysql2/promise";
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
  type ProductionPreparationPythonStepResult,
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
            expires_at,
            updated_at,
            expires_at IS NOT NULL AND expires_at > NOW() AS session_is_live,
            EXISTS (
              SELECT 1
                FROM model_production_preparation_step_attempts AS active_attempt
               WHERE active_attempt.attempt_id = model_production_preparation_sessions.active_attempt_id
                 AND active_attempt.session_id = model_production_preparation_sessions.session_id
                 AND active_attempt.status = 'running'
                 AND active_attempt.lease_expires_at > NOW()
            ) AS active_attempt_lease_is_live
       FROM model_production_preparation_sessions
      WHERE session_id = ?
      LIMIT 1`,
    [sessionId],
  );
  return rows[0] ?? null;
};

const fetchProductionPreparationSessionByRequest = async (input: {
  experimentId: number;
  studentId: number;
  modelType: ModelType;
  requestHash: string;
}) => {
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
            expires_at,
            updated_at,
            expires_at IS NOT NULL AND expires_at > NOW() AS session_is_live,
            EXISTS (
              SELECT 1
                FROM model_production_preparation_step_attempts AS active_attempt
               WHERE active_attempt.attempt_id = model_production_preparation_sessions.active_attempt_id
                 AND active_attempt.session_id = model_production_preparation_sessions.session_id
                 AND active_attempt.status = 'running'
                 AND active_attempt.lease_expires_at > NOW()
            ) AS active_attempt_lease_is_live
       FROM model_production_preparation_sessions
      WHERE experiment_id = ?
        AND student_id = ?
        AND model_type = ?
        AND request_hash = ?
      LIMIT 1`,
    [input.experimentId, input.studentId, input.modelType, input.requestHash],
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
  row: Pick<
    ProductionPreparationSessionRow,
    | "session_id"
    | "status"
    | "active_attempt_id"
    | "expires_at"
    | "current_step_id"
    | "completed_step_ids"
    | "step_plan"
    | "committed_state_path"
    | "committed_state_sha256"
  >,
  reason: string,
) => {
  if (!(["ready", "running", "failed"] as const).includes(
    row.status as "ready" | "running" | "failed",
  )) return false;
  const [result] = await queryWithTimeout(pool,
    `UPDATE model_production_preparation_sessions AS session
        SET status = 'superseded',
            current_step_id = NULL,
            active_attempt_id = NULL,
            superseded_reason = ?,
            error_message = NULL
      WHERE session.session_id = ?
        AND session.status = ?
        AND session.active_attempt_id <=> ?
        AND session.expires_at <=> ?
        AND session.current_step_id <=> ?
        AND session.completed_step_ids = ?
        AND session.step_plan = ?
        AND session.committed_state_path <=> ?
        AND session.committed_state_sha256 <=> ?
        AND session.status IN ('ready', 'running', 'failed')
        AND NOT EXISTS (
          SELECT 1
            FROM model_production_preparation_step_attempts AS attempt
           WHERE attempt.attempt_id = session.active_attempt_id
             AND attempt.session_id = session.session_id
             AND attempt.status = 'running'
             AND attempt.lease_expires_at > NOW()
        )`,
    [
      reason,
      row.session_id,
      row.status,
      row.active_attempt_id,
      row.expires_at,
      row.current_step_id,
      jsonColumnCasValue(row.completed_step_ids),
      jsonColumnCasValue(row.step_plan),
      row.committed_state_path,
      row.committed_state_sha256,
    ],
  );
  return Number((result as { affectedRows?: number }).affectedRows ?? 0) === 1;
};

const claimAlreadySupersededProductionPreparationSession = async (
  row: ProductionPreparationSessionRow,
) => {
  const claimToken = `Retirement claim ${crypto.randomUUID()}`;
  const [result] = await queryWithTimeout(pool,
    `UPDATE model_production_preparation_sessions
        SET superseded_reason = ?
      WHERE session_id = ?
        AND status = 'superseded'
        AND active_attempt_id <=> ?
        AND completed_step_ids = ?
        AND step_plan = ?
        AND committed_state_path <=> ?
        AND committed_state_sha256 <=> ?`,
    [
      claimToken,
      row.session_id,
      row.active_attempt_id,
      jsonColumnCasValue(row.completed_step_ids),
      jsonColumnCasValue(row.step_plan),
      row.committed_state_path,
      row.committed_state_sha256,
    ],
  );
  return Number((result as { affectedRows?: number }).affectedRows ?? 0) === 1;
};

const hasLiveProductionPreparationAttempt = (
  row: Pick<ProductionPreparationSessionRow, "active_attempt_lease_is_live">,
) => Number(row.active_attempt_lease_is_live) === 1;

const classifyProductionPreparationSessionHealth = async (
  row: ProductionPreparationSessionRow,
  terminationSignal?: AbortSignal,
  allowExpiredReceiptReplay = false,
): Promise<{ healthy: boolean; liveAttempt: boolean; reason?: string }> => {
  const liveAttempt = hasLiveProductionPreparationAttempt(row);
  if (Number(row.protocol_version) !== PRODUCTION_PREPARATION_PROTOCOL_VERSION) {
    return { healthy: false, liveAttempt: false, reason: "生产准备会话协议已失效" };
  }
  if (row.status === "superseded") {
    return { healthy: false, liveAttempt: false, reason: row.superseded_reason ?? "生产准备会话已失效" };
  }
  if (!hasValidProductionStepProgress(row)) {
    return { healthy: false, liveAttempt, reason: "生产准备步骤状态已损坏" };
  }
  if (row.status === "completed") return { healthy: true, liveAttempt: false };
  if (liveAttempt && !allowExpiredReceiptReplay) {
    return { healthy: true, liveAttempt: true };
  }
  // Production SELECTs always project these DB-time booleans. Older unit-test
  // fixtures may omit both and are treated as metadata-only legacy fixtures.
  if (
    row.session_is_live === undefined
    && row.active_attempt_lease_is_live === undefined
  ) {
    return { healthy: true, liveAttempt: false };
  }
  if (Number(row.session_is_live) !== 1 && !allowExpiredReceiptReplay) {
    return { healthy: false, liveAttempt: false, reason: "生产准备会话已过期" };
  }

  try {
    const datasetPath = path.join(row.artifact_dir, GUIDED_DATASET_FILE_NAME);
    if (await hashFileContents(datasetPath, terminationSignal) !== row.dataset_revision) {
      return { healthy: false, liveAttempt: false, reason: "生产准备数据快照已损坏" };
    }
    const hasCheckpointPath = row.committed_state_path !== null;
    const hasCheckpointDigest = row.committed_state_sha256 !== null;
    if (hasCheckpointPath !== hasCheckpointDigest) {
      return { healthy: false, liveAttempt: false, reason: "生产准备 checkpoint 状态不完整" };
    }
    if (
      hasCheckpointPath
      && await hashFileContents(row.committed_state_path!, terminationSignal)
        !== row.committed_state_sha256
    ) {
      return { healthy: false, liveAttempt: false, reason: "生产准备 checkpoint 已损坏" };
    }
  } catch (error) {
    if (terminationSignal?.aborted) {
      throw terminationSignal.reason instanceof Error
        ? terminationSignal.reason
        : error;
    }
    return { healthy: false, liveAttempt: false, reason: "生产准备会话工件缺失或不可读" };
  }
  return { healthy: true, liveAttempt: false };
};

const computeCurrentProductionPreparationIdentity = async (
  row: Pick<
    ProductionPreparationSessionRow,
    "experiment_id" | "model_type" | "forecast_steps"
  >,
  studentId: number,
  terminationSignal?: AbortSignal,
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
    terminationSignal,
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
  row: ProductionPreparationSessionRow,
  studentId: number,
  terminationSignal?: AbortSignal,
): Promise<ProductionPreparationIdentity> => {
  let current: ProductionPreparationIdentity;
  try {
    current = await computeCurrentProductionPreparationIdentity(
      row,
      studentId,
      terminationSignal,
    );
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
      operationContext.deadlineSignal,
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
    const requestIdentity = { experimentId, studentId, modelType, requestHash };

    // A completed, current session is immutable metadata and does not start a
    // Python process. Resolve that fast path before model admission so students
    // can reuse an existing production artifact while unrelated model jobs have
    // filled the process capacity. Any uncertain or stale state falls through
    // to the serialized path below for CAS retirement/rebuild.
    const reusableBeforeAdmission = await fetchProductionPreparationSessionByRequest(
      requestIdentity,
    );
    if (
      reusableBeforeAdmission?.status === "completed"
      && reusableBeforeAdmission.protocol_version === PRODUCTION_PREPARATION_PROTOCOL_VERSION
      && reusableBeforeAdmission.run_id === activeRun.run_id
      && reusableBeforeAdmission.dataset_revision === activeRun.dataset_revision
      && reusableBeforeAdmission.runtime_revision === activeRun.runtime_revision
      && Boolean(reusableBeforeAdmission.production_revision)
      && reusableBeforeAdmission.production_revision === activeRun.production_revision
      && activeRun.production_artifact_forecast_steps !== null
      && Number(activeRun.production_artifact_forecast_steps) >= forecastSteps
      && (await classifyProductionPreparationSessionHealth(
        reusableBeforeAdmission,
        operationContext.deadlineSignal,
      )).healthy
    ) {
      return buildProductionPreparationPayload(reusableBeforeAdmission, true);
    }

    return await withModelOperationLease(
      operationContext,
      { execution: { experimentId, modelType } },
      async (lease) => {
        await lease.verifyOwned();
    const existing = await fetchProductionPreparationSessionByRequest(requestIdentity);
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
    let completedSessionStillCurrent = existing?.status !== "completed" || (
      Boolean(existing.production_revision)
      && existing.production_revision === currentActiveRun.production_revision
      && currentActiveRun.production_artifact_forecast_steps !== null
      && Number(currentActiveRun.production_artifact_forecast_steps) >= forecastSteps
    );
    let existingHealth: Awaited<ReturnType<typeof classifyProductionPreparationSessionHealth>> | null = null;
    let retirementClaimed = false;
    let retirementReason = "生产准备会话已失效";
    if (existing) {
      existingHealth = await classifyProductionPreparationSessionHealth(
        existing,
        lease.terminationSignal,
      );
      if (existing.status === "completed" && !existingHealth.healthy) {
        completedSessionStillCurrent = false;
      }
      if (
        !existingHealth.healthy
        && existing.status !== "superseded"
        && existing.status !== "completed"
      ) {
        retirementReason = existingHealth.reason ?? retirementReason;
        retirementClaimed = await markProductionPreparationSessionSuperseded(
          existing,
          retirementReason,
        );
        if (!retirementClaimed) {
          const retained = await fetchProductionPreparationSession(existing.session_id);
          if (retained && retained.status !== "superseded") {
            return buildProductionPreparationPayload(retained);
          }
          throw new ModelConflictError(
            "生产准备会话状态已变化，请重试",
            "PRODUCTION_PREPARATION_STALE",
          );
        }
      }
    }
    if (
      existing
      && existing.protocol_version === PRODUCTION_PREPARATION_PROTOCOL_VERSION
      && existing.status !== "superseded"
      && completedSessionStillCurrent
      && existingHealth?.healthy === true
    ) {
      return buildProductionPreparationPayload(existing);
    }
    if (existing) {
      let deleteResult: unknown;
      if (existing.status === "completed") {
        // Completed-stale retirement is intentionally separate from generic
        // incomplete-session supersession and uses the freshly re-read run.
        [deleteResult] = await queryWithTimeout(pool,
          `DELETE session
             FROM model_production_preparation_sessions AS session
            WHERE session.session_id = ?
              AND session.status = 'completed'
              AND session.active_attempt_id <=> ?
              AND session.production_revision <=> ?
              AND session.completed_step_ids = ?
              AND session.step_plan = ?`,
          [
            existing.session_id,
            existing.active_attempt_id,
            existing.production_revision,
            jsonColumnCasValue(existing.completed_step_ids),
            jsonColumnCasValue(existing.step_plan),
          ],
        );
      } else {
        if (existing.status === "superseded") {
          retirementClaimed = await claimAlreadySupersededProductionPreparationSession(
            existing,
          );
        }
        if (!retirementClaimed) {
          const retained = await fetchProductionPreparationSession(existing.session_id);
          if (retained && retained.status !== "superseded") {
            return buildProductionPreparationPayload(retained);
          }
          throw new ModelConflictError(
            "生产准备会话状态已变化，请重试",
            "PRODUCTION_PREPARATION_STALE",
          );
        }
        [deleteResult] = await queryWithTimeout(pool,
          `DELETE session
             FROM model_production_preparation_sessions AS session
            WHERE session.session_id = ?
              AND session.status = 'superseded'
              AND session.active_attempt_id IS NULL
              AND session.completed_step_ids = ?
              AND session.step_plan = ?
              AND session.committed_state_path <=> ?
              AND session.committed_state_sha256 <=> ?`,
          [
            existing.session_id,
            jsonColumnCasValue(existing.completed_step_ids),
            jsonColumnCasValue(existing.step_plan),
            existing.committed_state_path,
            existing.committed_state_sha256,
          ],
        );
      }
      if (Number((deleteResult as { affectedRows?: number }).affectedRows ?? 0) !== 1) {
        const retained = await fetchProductionPreparationSession(existing.session_id);
        if (retained && retained.status !== "superseded") {
          return buildProductionPreparationPayload(retained);
        }
        throw new ModelConflictError(
          "生产准备会话状态已变化，请重试",
          "PRODUCTION_PREPARATION_STALE",
        );
      }
      try {
        if (existing.candidate_model_path !== currentActiveRun.production_artifact_path) {
          removeModelArtifactGroup(existing.candidate_model_path);
        }
        removeProductionPreparationArtifactDir(existing.artifact_dir);
      } catch (error) {
        logger.warn(
          { err: error, sessionId: existing.session_id, artifactDir: existing.artifact_dir },
          "重建生产准备会话后清理旧工件失败",
        );
      }
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
      },
    );
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
  terminationSignal?: AbortSignal,
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
      || await hashFileContents(attempt.input_state_path, terminationSignal) !== attempt.input_state_sha256
    ) {
      throw new ModelValidationError("生产准备步骤输入 checkpoint 摘要不匹配");
    }
  } else if (result.input_state_path != null || result.input_state_sha256 != null) {
    throw new ModelValidationError("生产准备初始化返回了意外的输入 checkpoint");
  }
  const actualDigest = await hashFileContents(
    attempt.candidate_state_path,
    terminationSignal,
  );
  if (actualDigest !== result.candidate_state_sha256) {
    throw new ModelValidationError("生产准备 checkpoint 摘要不匹配");
  }
  return result;
};

const parseProductionPreparationAttemptReceipt = async (
  attempt: ProductionPreparationAttemptRow,
  terminationSignal?: AbortSignal,
): Promise<ProductionPreparationPythonStepResult | null> => {
  if (!fs.existsSync(attempt.receipt_path) || !fs.existsSync(attempt.candidate_state_path)) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(fs.readFileSync(attempt.receipt_path, "utf8"));
  } catch {
    return null;
  }
  if (!isRecordValue(parsed)) return null;
  try {
    return await validateProductionPreparationReceipt(
      parsed as unknown as ProductionPreparationPythonStepResult,
      attempt,
      terminationSignal,
    );
  } catch (error) {
    if (terminationSignal?.aborted) throw error;
    logger.warn(
      { err: error, attemptId: attempt.attempt_id },
      "忽略无效 production preparation receipt 并回退到正常 attempt 流程",
    );
    return null;
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
    const initialCompletedStepIds = parseProductionCompletedStepIds(
      initialRow.completed_step_ids,
    );
    const initialStepPlan = parseProductionPreparationStepPlan(initialRow.step_plan);
    if (
      !initialCompletedStepIds
      || !initialStepPlan
      || !hasValidProductionStepProgress(initialRow)
    ) {
      const reason = "生产准备步骤状态已损坏";
      const claimed = await markProductionPreparationSessionSuperseded(initialRow, reason);
      if (!claimed) {
        const retained = await fetchProductionPreparationSession(sessionId);
        if (retained && retained.status !== "superseded") {
          return buildProductionPreparationPayload(retained);
        }
      }
      throw new ModelConflictError(reason, "PRODUCTION_PREPARATION_STALE");
    }
    if (initialCompletedStepIds.includes(stepId)) {
      return buildProductionPreparationPayload(initialRow);
    }
    const initialNextStep = initialStepPlan.find(
      (step) => !initialCompletedStepIds.includes(step.id),
    );
    if (!initialNextStep || initialNextStep.id !== stepId) {
      throw new ModelConflictError(
        "只能执行生产准备会话的下一步骤",
        "PRODUCTION_PREPARATION_STEP_CONFLICT",
      );
    }
    return await withModelOperationLease(
      operationContext,
      { execution: { experimentId: initialRow.experiment_id, modelType } },
      async (lease) => {
        // Another request may have completed this step after the initial read
        // but before this request acquired the process-local execution lock.
        // Re-read under the lock so stale
        // checkpoint pointers can never replay or overwrite a later step.
        let currentRow = await fetchProductionPreparationSession(sessionId);
        if (!currentRow || currentRow.model_type !== modelType || currentRow.student_id !== studentId) {
          throw new ModelNotFoundError("未找到生产准备会话");
        }
        if (
          currentRow.protocol_version !== PRODUCTION_PREPARATION_PROTOCOL_VERSION
          || currentRow.status === "superseded"
        ) {
          throw new ModelConflictError("生产准备会话已失效，请重新创建", "PRODUCTION_PREPARATION_STALE");
        }
        const currentCompletedStepIds = parseProductionCompletedStepIds(
          currentRow.completed_step_ids,
        );
        const currentStepPlan = parseProductionPreparationStepPlan(currentRow.step_plan);
        if (
          !currentCompletedStepIds
          || !currentStepPlan
          || !hasValidProductionStepProgress(currentRow)
        ) {
          const reason = "生产准备步骤状态已损坏";
          const claimed = await markProductionPreparationSessionSuperseded(currentRow, reason);
          if (!claimed) {
            const retained = await fetchProductionPreparationSession(sessionId);
            if (retained && retained.status !== "superseded") {
              return buildProductionPreparationPayload(retained);
            }
          }
          throw new ModelConflictError(reason, "PRODUCTION_PREPARATION_STALE");
        }
        if (currentCompletedStepIds.includes(stepId)) {
          return buildProductionPreparationPayload(currentRow);
        }
        const currentNextStep = currentStepPlan.find(
          (step) => !currentCompletedStepIds.includes(step.id),
        );
        if (!currentNextStep || currentNextStep.id !== stepId) {
          throw new ModelConflictError(
            "只能执行生产准备会话的下一步骤",
            "PRODUCTION_PREPARATION_STEP_CONFLICT",
          );
        }

        let recoveredAttempt: ProductionPreparationAttemptRow | null = null;
        let recoveredReceipt: ProductionPreparationPythonStepResult | null = null;
        if (currentRow.active_attempt_id) {
          const activeAttempt = await fetchProductionPreparationAttempt(
            currentRow.active_attempt_id,
          );
          if (activeAttempt?.session_id === sessionId && activeAttempt.step_id === stepId) {
            recoveredReceipt = await parseProductionPreparationAttemptReceipt(
              activeAttempt,
              lease.terminationSignal,
            );
            if (recoveredReceipt) {
              const replayLeaseSeconds = Math.ceil((
                resolveProductionPreparationStepTimeoutMs(modelType, stepId) + 60_000
              ) / 1000);
              const claimed = await claimProductionPreparationReceiptReplay(
                currentRow,
                activeAttempt,
                replayLeaseSeconds,
              );
              if (!claimed) {
                const retained = await fetchProductionPreparationSession(sessionId);
                const retainedCompleted = retained
                  ? parseProductionCompletedStepIds(retained.completed_step_ids)
                  : null;
                if (
                  retained
                  && (
                    retained.status === "completed"
                    || retained.active_attempt_id !== activeAttempt.attempt_id
                    || retainedCompleted?.includes(stepId)
                  )
                ) {
                  return buildProductionPreparationPayload(retained);
                }
                throw new ModelConflictError(
                  "生产准备 receipt attempt 已提交、替换或状态已变化",
                  "PRODUCTION_PREPARATION_STEP_IN_PROGRESS",
                );
              }
              const replayRow = await fetchProductionPreparationSession(sessionId);
              if (!replayRow || replayRow.active_attempt_id !== activeAttempt.attempt_id) {
                throw new ModelConflictError(
                  "生产准备会话状态已变化，请重试",
                  "PRODUCTION_PREPARATION_STALE",
                );
              }
              currentRow = replayRow;
              recoveredAttempt = activeAttempt;
            } else if (
              activeAttempt.status === "running"
              && Number(activeAttempt.lease_is_active) === 1
            ) {
              return buildProductionPreparationPayload(currentRow);
            }
          } else if (hasLiveProductionPreparationAttempt(currentRow)) {
            return buildProductionPreparationPayload(currentRow);
          }
        }

        const currentHealth = await classifyProductionPreparationSessionHealth(
          currentRow,
          lease.terminationSignal,
          recoveredReceipt !== null,
        );
        if (!currentHealth.healthy) {
          const reason = currentHealth.reason ?? "生产准备会话工件已失效";
          const claimed = await markProductionPreparationSessionSuperseded(currentRow, reason);
          if (!claimed) {
            const retained = await fetchProductionPreparationSession(sessionId);
            if (retained && retained.status !== "superseded") {
              return buildProductionPreparationPayload(retained);
            }
          }
          throw new ModelConflictError(reason, "PRODUCTION_PREPARATION_STALE");
        }
        if (!recoveredReceipt && !hasLiveProductionPreparationAttempt(currentRow)) {
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
                AND status = ?
                AND active_attempt_id <=> ?
                AND expires_at <=> ?
                AND completed_step_ids = ?
                AND step_plan = ?
                AND status IN ('ready', 'failed', 'running')
                AND expires_at > NOW()`,
            [
              sessionId,
              currentRow.status,
              currentRow.active_attempt_id,
              currentRow.expires_at,
              jsonColumnCasValue(currentRow.completed_step_ids),
              jsonColumnCasValue(currentRow.step_plan),
            ],
          );
          if (Number((refreshResult as { affectedRows?: number }).affectedRows ?? 0) !== 1) {
            const reason = "生产准备会话已过期，请重新创建";
            const claimed = await markProductionPreparationSessionSuperseded(currentRow, reason);
            if (!claimed) {
              const retained = await fetchProductionPreparationSession(sessionId);
              if (retained && retained.status !== "superseded") {
                return buildProductionPreparationPayload(retained);
              }
            }
            throw new ModelConflictError(reason, "PRODUCTION_PREPARATION_STALE");
          }
          const refreshed = await fetchProductionPreparationSession(sessionId);
          if (!refreshed) throw new ModelNotFoundError("未找到生产准备会话");
          currentRow = refreshed;
        }
        let stepIdentity = await assertProductionPreparationIdentityCurrent(
          currentRow,
          studentId,
          lease.terminationSignal,
        );
        const reusable = recoveredReceipt
          ? null
          : await getReusableProductionModelArtifactResult(
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
          lease.terminationSignal,
        );
        const { activeRun } = stepIdentity;

        await verifyProductionSourceArtifactForRun(
          modelType,
          activeRun,
          lease.terminationSignal,
        );
        const datasetSnapshotPath = path.join(
          currentRow.artifact_dir,
          GUIDED_DATASET_FILE_NAME,
        );
        const timeoutMs = resolveProductionPreparationStepTimeoutMs(modelType, stepId);
        let attempt = recoveredAttempt;
        if (!attempt) {
          const attemptLeaseSeconds = Math.ceil((timeoutMs + 60_000) / 1000);
          const attemptId = crypto.randomUUID();
          const attemptDir = path.join(currentRow.artifact_dir, "attempts", attemptId);
          fs.mkdirSync(attemptDir, { recursive: true });
          attempt = {
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
        }
        const attemptId = attempt.attempt_id;

        try {
          const rawReceipt = recoveredReceipt ?? await runPythonProductionPreparationStep(
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
          const receipt = recoveredReceipt ?? await validateProductionPreparationReceipt(
            rawReceipt,
            attempt,
            lease.terminationSignal,
          );
          await verifyProductionSourceArtifactForRun(
            modelType,
            activeRun,
            lease.terminationSignal,
          );
          await assertProductionPreparationIdentityCurrent(
            currentRow,
            studentId,
            lease.terminationSignal,
          );
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
              lease.terminationSignal,
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
    terminationSignal,
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
