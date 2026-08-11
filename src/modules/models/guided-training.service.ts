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
import type { ModelType, TrainInput } from "./models.schema";
import {
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
  type ModelOperationContext,
} from "./model-operation-lease";
import {
  runPythonGuidedTrainingStep,
} from "./model-python-runner";
import { transformTrainingResult } from "./model-results";
import {
  buildExperimentTrainingRevision,
  buildModelTrainingRequestHash,
  commitModelTrainingRun,
  getActiveModelTrainingRun,
} from "./model-run-store";
import {
  buildGuidedTrainingArtifactDir,
  buildVersionedModelPath,
} from "./model-runtime";
import {
  type GuidedTrainingSessionPayload,
  type GuidedTrainingStatus,
} from "./model-types";
import { loadDatasetFromDb } from "./model-datasets";
import {
  computeModelArtifactSha256,
  copyModelArtifact,
  getGuidedSessionModelPath,
  hasSavedModelArtifact,
  removeModelArtifactGroup,
  verifyBaseModelArtifactIntegrity,
} from "./model-artifacts";
import {
  assertExperimentModelAccess,
  assertTrainingRequestMatchesExperiment,
  buildBaseRunContext,
  buildEnsemblePythonParams,
  getExperimentTrainingConfig,
  isDuplicateModelSessionRequestError,
  isEnsembleModelType,
  isGuidedModelType,
  lockStudentExperimentForModelWrite,
  parseJsonColumn,
  resolveGuidedStepTimeoutMs,
  validateMovingAverageWindow,
  writeGuidedDatasetSnapshot,
} from "./model-context";
import {
  getModelRuntimeRevision,
  hashDatasetSnapshot,
} from "./model-runtime-identity";
import {
  GUIDED_DATASET_FILE_NAME,
  GUIDED_TRAINING_PROTOCOL_VERSION,
} from "./model-service-constants";
import {
  deleteGuidedTrainingSession,
  removeGuidedTrainingArtifactDir,
} from "./model-cleanup.service";
import {
  type GuidedTrainingSessionRow,
  type GuidedTrainingStepAttemptRow,
  assertCompletedGuidedTrainingRunCurrent,
  assertGuidedTrainingIdentityCurrent,
  beginGuidedStepAttempt,
  buildCanonicalTrainingBody,
  buildGuidedTrainingPythonParams,
  buildGuidedTrainingSessionPayload,
  buildGuidedTrainingStepPlan,
  buildRefreshedCompletedGuidedTrainingSessionPayload,
  cleanupIncompleteGuidedAttemptArtifacts,
  commitGuidedIntermediateStep,
  expandBoostingRefitSteps,
  fetchGuidedTrainingSessionRow,
  getGuidedNextStepId,
  getGuidedTrainingSteps,
  guidedRequestPayloadMatches,
  isRecordValue,
  markGuidedTrainingSessionSuperseded,
  parseGuidedAttemptReceipt,
  recordGuidedAttemptFailure,
  recordGuidedAttemptPythonCompleted,
  shouldTrimBoostingRoundSteps,
  trimBoostingRefitSteps,
  trimBoostingStepPlanAfterEarlyStop,
} from "./guided-training-state";

const createOrResumeGuidedTrainingSessionInternal = async (
  modelType: ModelType,
  studentId: number,
  body: TrainInput,
  operationContext: ModelOperationContext,
  terminationSignal: AbortSignal,
): Promise<GuidedTrainingSessionPayload> => {
  const experiment = await assertExperimentModelAccess(body.experiment_id, studentId);
  operationContext.throwIfTerminated();
  assertTrainingRequestMatchesExperiment(experiment, body);
  const trainingConfig = getExperimentTrainingConfig(experiment);
  const canonicalBody = buildCanonicalTrainingBody(body, trainingConfig);
  if (modelType === "ma") {
    validateMovingAverageWindow(trainingConfig, canonicalBody.moving_average_window);
  }
  if (isEnsembleModelType(modelType)) {
    buildEnsemblePythonParams(experiment, canonicalBody);
  }

  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
  );
  operationContext.throwIfTerminated();
  if (
    trainingConfig.data_window_train_end_index >= allData.length
    || trainingConfig.data_window_evaluate_end_index >= allData.length
  ) {
    throw new ModelValidationError("数据窗口索引超出范围", { dataset_size: allData.length });
  }
  const datasetJson = JSON.stringify(allData);
  const datasetRevision = hashDatasetSnapshot(datasetJson);
  const runtimeRevision = getModelRuntimeRevision(true);
  const baseRunFingerprint = isEnsembleModelType(modelType)
    ? (await buildBaseRunContext(
      canonicalBody.experiment_id,
      studentId,
      canonicalBody,
      datasetRevision,
      runtimeRevision,
      { terminationSignal },
    )).fingerprint
    : {};
  operationContext.throwIfTerminated();
  const experimentRevision = buildExperimentTrainingRevision(modelType, experiment, canonicalBody);
  const requestHash = buildModelTrainingRequestHash({
    modelType,
    body: canonicalBody,
    datasetRevision,
    runtimeRevision,
    experimentRevision,
    baseRunFingerprint,
  });
  const [existingRows] = await queryWithTimeout<GuidedTrainingSessionRow[]>(pool,
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
      WHERE experiment_id = ?
        AND student_id = ?
        AND model_type = ?
        AND request_hash = ?
      LIMIT 1`,
    [canonicalBody.experiment_id, studentId, modelType, requestHash],
  );
  operationContext.throwIfTerminated();

  const existingSession = existingRows[0];
  if (existingSession) {
    if (existingSession.status === "superseded" || existingSession.protocol_version !== GUIDED_TRAINING_PROTOCOL_VERSION) {
      operationContext.throwIfTerminated();
      const deletedSession = await deleteGuidedTrainingSession(existingSession);
      operationContext.throwIfTerminated();
      if (!deletedSession) {
        throw new ModelValidationError("旧的分阶段训练会话清理失败，请稍后重试");
      }
    } else if (existingSession.status === "completed") {
      try {
        await assertCompletedGuidedTrainingRunCurrent(existingSession, studentId, { terminationSignal });
        return await buildRefreshedCompletedGuidedTrainingSessionPayload(existingSession, studentId);
      } catch (error) {
        if (!(error instanceof ModelConflictError) || error.code !== "GUIDED_SESSION_SUPERSEDED") {
          throw error;
        }

        logger.warn(
          {
            sessionId: existingSession.session_id,
            artifactDir: existingSession.artifact_dir,
            modelType,
          },
          "已完成 guided training 会话不再对应当前训练版本，将删除会话并重新创建",
        );
        operationContext.throwIfTerminated();
        const deletedSession = await deleteGuidedTrainingSession(existingSession);
        operationContext.throwIfTerminated();
        if (!deletedSession) {
          throw new ModelValidationError("旧的分阶段训练会话工件清理失败，请稍后重试");
        }
      }
    } else {
      const datasetFile = path.join(existingSession.artifact_dir, GUIDED_DATASET_FILE_NAME);
      if (!fs.existsSync(datasetFile)) {
        operationContext.throwIfTerminated();
        await writeGuidedDatasetSnapshot(existingSession.artifact_dir, datasetJson);
        operationContext.throwIfTerminated();
      }
      operationContext.throwIfTerminated();
      return buildGuidedTrainingSessionPayload(existingSession);
    }
  }

  if (!existingSession) {
    const activeRun = await getActiveModelTrainingRun(
      canonicalBody.experiment_id,
      studentId,
      modelType,
    );
    operationContext.throwIfTerminated();
    if (activeRun?.session_id) {
      const completedSession = await fetchGuidedTrainingSessionRow(activeRun.session_id);
      operationContext.throwIfTerminated();
      if (
        completedSession
        && completedSession.status === "completed"
        && completedSession.protocol_version === GUIDED_TRAINING_PROTOCOL_VERSION
        && completedSession.model_type === modelType
        && completedSession.student_id === studentId
        && guidedRequestPayloadMatches(completedSession, canonicalBody)
      ) {
        try {
          await assertCompletedGuidedTrainingRunCurrent(completedSession, studentId, { terminationSignal });
          return await buildRefreshedCompletedGuidedTrainingSessionPayload(completedSession, studentId);
        } catch (error) {
          if (!(error instanceof ModelConflictError) || error.code !== "GUIDED_SESSION_SUPERSEDED") {
            throw error;
          }
          operationContext.throwIfTerminated();
          const deletedSession = await deleteGuidedTrainingSession(completedSession);
          operationContext.throwIfTerminated();
          if (!deletedSession) {
            throw new ModelValidationError("失效的分阶段训练会话清理失败，请稍后重试");
          }
        }
      }
    }
  }

  const sessionId = crypto.randomUUID();
  const artifactDir = buildGuidedTrainingArtifactDir(canonicalBody.experiment_id, studentId, modelType, sessionId);
  const completedStepIds: string[] = [];
  const stepPlan = buildGuidedTrainingStepPlan(modelType, canonicalBody);
  const currentStepId = getGuidedNextStepId(stepPlan, completedStepIds);
  let insertCommitted = false;
  try {
    operationContext.throwIfTerminated();
    fs.mkdirSync(artifactDir, { recursive: true });
    await writeGuidedDatasetSnapshot(artifactDir, datasetJson);
    operationContext.throwIfTerminated();
    const connection = await acquireConnectionWithTimeout(pool);
    let transactionPhase: "before-commit" | "commit-attempted" | "committed" = "before-commit";
    let disposed = false;
    try {
      await connection.beginTransaction();
      await lockStudentExperimentForModelWrite(
        connection,
        canonicalBody.experiment_id,
        studentId,
      );
      operationContext.throwIfTerminated();
      await connection.query<RowDataPacket[]>(
        `INSERT INTO model_guided_training_sessions
           (session_id,
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
            expires_at)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, ?, NULL, ?, DATE_ADD(NOW(), INTERVAL 7 DAY))`,
        [
          sessionId,
          canonicalBody.experiment_id,
          studentId,
          modelType,
          requestHash,
          datasetRevision,
          runtimeRevision,
          GUIDED_TRAINING_PROTOCOL_VERSION,
          JSON.stringify(canonicalBody),
          currentStepId,
          JSON.stringify(completedStepIds),
          JSON.stringify({}),
          JSON.stringify(stepPlan),
          artifactDir,
        ],
      );
      transactionPhase = "commit-attempted";
      try {
        await commitConnectionWithOutcomeGuard(connection);
      } catch (error) {
        // A lost COMMIT acknowledgement makes the INSERT outcome ambiguous.
        // Never return this session to the pool or delete its artifact dir;
        // the orphan/session reconciler can resolve either outcome safely.
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
    operationContext.throwIfTerminated();
  } catch (error) {
    if (!isDuplicateModelSessionRequestError(error)) {
      // A rolled-back transaction cannot reference this directory. Preserve it
      // only after an acknowledged commit (for a post-commit deadline) or when
      // the COMMIT outcome itself is unknown.
      if (!insertCommitted && !isModelCommitOutcomeUnknownError(error)) {
        fs.rmSync(artifactDir, { recursive: true, force: true });
      }
      throw error;
    }
    fs.rmSync(artifactDir, { recursive: true, force: true });
    operationContext.throwIfTerminated();
    const [racedRows] = await queryWithTimeout<GuidedTrainingSessionRow[]>(pool,
      `SELECT *
         FROM model_guided_training_sessions
        WHERE experiment_id = ?
          AND student_id = ?
          AND model_type = ?
          AND request_hash = ?
        LIMIT 1`,
      [canonicalBody.experiment_id, studentId, modelType, requestHash],
    );
    operationContext.throwIfTerminated();
    if (racedRows[0]) {
      if (racedRows[0].status === "completed") {
        await assertCompletedGuidedTrainingRunCurrent(racedRows[0], studentId, { terminationSignal });
        return await buildRefreshedCompletedGuidedTrainingSessionPayload(racedRows[0], studentId);
      }
      return buildGuidedTrainingSessionPayload(racedRows[0]);
    }
    throw error;
  }

  operationContext.throwIfTerminated();
  const row = await fetchGuidedTrainingSessionRow(sessionId);
  operationContext.throwIfTerminated();
  if (!row) {
    throw new Error("创建分阶段训练会话失败");
  }

  return buildGuidedTrainingSessionPayload(row);
};

export async function createOrResumeGuidedTrainingSession(
  modelType: ModelType,
  studentId: number,
  body: TrainInput,
): Promise<GuidedTrainingSessionPayload> {
  if (!isGuidedModelType(modelType)) {
    throw new ModelValidationError("当前模型暂不支持分阶段训练");
  }

  return await withModelOperationContext("guided_training_session", async (operationContext) => {
    return await withModelOperationLease(
      operationContext,
      { execution: { experimentId: body.experiment_id, modelType } },
      async (lease) => await createOrResumeGuidedTrainingSessionInternal(
        modelType,
        studentId,
        body,
        operationContext,
        lease.terminationSignal,
      ),
    );
  });
}

export async function getGuidedTrainingSession(
  modelType: ModelType,
  studentId: number,
  sessionId: string,
): Promise<GuidedTrainingSessionPayload> {
  if (!isGuidedModelType(modelType)) {
    throw new ModelValidationError("当前模型暂不支持分阶段训练");
  }

  const row = await fetchGuidedTrainingSessionRow(sessionId);
  if (!row || row.model_type !== modelType) {
    throw new ModelNotFoundError("未找到分阶段训练会话");
  }

  if (row.student_id !== studentId) {
    throw new ModelAccessError("权限不足：该训练会话不属于当前用户", 403);
  }

  await assertExperimentModelAccess(row.experiment_id, studentId);
  if (row.status !== "superseded") {
    try {
      // Keep GET metadata reads free of artifact hashing and Python validation.
      // The leased create/resume and step endpoints perform those checks.
      if (row.status === "completed") {
        await assertCompletedGuidedTrainingRunCurrent(row, studentId, { verifyArtifacts: false });
        return await buildRefreshedCompletedGuidedTrainingSessionPayload(row, studentId);
      } else {
        await assertGuidedTrainingIdentityCurrent(row, studentId, { verifyArtifacts: false });
      }
    } catch (error) {
      if (!(error instanceof ModelConflictError) || error.code !== "GUIDED_SESSION_SUPERSEDED") {
        throw error;
      }
      const supersededRow = await fetchGuidedTrainingSessionRow(sessionId);
      if (!supersededRow) throw new ModelNotFoundError("未找到分阶段训练会话");
      return buildGuidedTrainingSessionPayload(supersededRow);
    }
  }
  return buildGuidedTrainingSessionPayload(row);
}

export async function discardGuidedTrainingSession(
  modelType: ModelType,
  studentId: number,
  sessionId: string,
) {
  if (!isGuidedModelType(modelType)) {
    throw new ModelValidationError("当前模型不支持分阶段训练");
  }
  return await withModelOperationContext("guided_training_discard", async (operationContext) => {
    const row = await fetchGuidedTrainingSessionRow(sessionId);
    operationContext.throwIfTerminated();
    if (!row || row.model_type !== modelType) {
      throw new ModelNotFoundError("未找到分阶段训练会话");
    }
    if (row.student_id !== studentId) {
      throw new ModelAccessError("权限不足：该训练会话不属于当前用户", 403);
    }
    if (row.status === "running" || row.status === "completed") {
      throw new ModelConflictError(
        row.status === "running" ? "运行中的训练会话不能删除" : "已完成的训练会话不能删除",
        "GUIDED_SESSION_NOT_DISCARDABLE",
      );
    }

    return await withModelOperationLease(
      operationContext,
      { execution: { experimentId: row.experiment_id, modelType } },
      async (lease) => {
        await lease.verifyOwned();
        const [result] = await queryWithTimeout(pool,
          `DELETE FROM model_guided_training_sessions
            WHERE session_id = ?
              AND student_id = ?
              AND status IN ('ready', 'failed', 'superseded')`,
          [sessionId, studentId],
        );
        if (Number((result as { affectedRows?: number }).affectedRows ?? 0) !== 1) {
          throw new ModelConflictError("训练会话状态已变化，无法删除", "GUIDED_SESSION_NOT_DISCARDABLE");
        }
        try {
          removeGuidedTrainingArtifactDir(row.artifact_dir);
        } catch (error) {
          logger.warn({ err: error, sessionId, artifactDir: row.artifact_dir }, "删除会话后清理 guided 工件失败");
        }
        return { status: "success" as const };
      },
    );
  });
}

export async function runGuidedTrainingStep(
  modelType: ModelType,
  studentId: number,
  sessionId: string,
  stepId: string,
): Promise<GuidedTrainingSessionPayload> {
  if (!isGuidedModelType(modelType)) {
    throw new ModelValidationError("当前模型暂不支持分阶段训练");
  }

  const activeAttemptRef: { current: GuidedTrainingStepAttemptRow | null } = { current: null };
  let stepCommitted = false;

  try {
    return await withModelOperationContext("guided_training_step", async (operationContext) => {
      const initialRow = await fetchGuidedTrainingSessionRow(sessionId);
      operationContext.throwIfTerminated();
      if (!initialRow || initialRow.model_type !== modelType) {
        throw new ModelNotFoundError("未找到分阶段训练会话");
      }
      if (initialRow.student_id !== studentId) {
        throw new ModelAccessError("权限不足：该训练会话不属于当前用户", 403);
      }

      return await withModelOperationLease(
        operationContext,
        { execution: { experimentId: initialRow.experiment_id, modelType } },
        async (lease) => {
          // Another request may have completed this step after the initial
          // metadata read but before this request acquired the process-local lock.
          // Re-read under the lock and use only this fresh row from here on.
          const row = await fetchGuidedTrainingSessionRow(sessionId);
          operationContext.throwIfTerminated();
          if (!row || row.model_type !== modelType) {
            throw new ModelNotFoundError("未找到分阶段训练会话");
          }
          if (row.student_id !== studentId) {
            throw new ModelAccessError("权限不足：该训练会话不属于当前用户", 403);
          }
          if (row.status === "completed") {
            await assertCompletedGuidedTrainingRunCurrent(row, studentId, {
              terminationSignal: lease.terminationSignal,
            });
            return await buildRefreshedCompletedGuidedTrainingSessionPayload(row, studentId);
          }
          if (row.status === "superseded" || row.protocol_version !== GUIDED_TRAINING_PROTOCOL_VERSION) {
            if (row.status !== "superseded") {
              await markGuidedTrainingSessionSuperseded(row, "训练会话协议已升级，请重新创建训练会话");
            }
            throw new ModelConflictError(
              row.superseded_reason ?? "训练会话已失效，请重新创建训练会话",
              "GUIDED_SESSION_SUPERSEDED",
            );
          }

          const body = parseJsonColumn<TrainInput>(row.request_payload, {} as TrainInput);
          const evaluationStart = Number(body.data_window_evaluate_start_index);
          const evaluationEnd = Number(body.data_window_evaluate_end_index);
          const expectedEvaluationLength = evaluationEnd - evaluationStart + 1;
          if (
            !Number.isSafeInteger(evaluationStart)
            || !Number.isSafeInteger(evaluationEnd)
            || !Number.isSafeInteger(expectedEvaluationLength)
            || expectedEvaluationLength <= 0
          ) {
            throw new ModelValidationError("训练会话缺少有效的评估区间");
          }
          const completedStepIds = parseJsonColumn<string[]>(row.completed_step_ids, []);
          if (completedStepIds.includes(stepId)) return buildGuidedTrainingSessionPayload(row);

          const stepPlan = getGuidedTrainingSteps(row);
          const nextStepId = getGuidedNextStepId(stepPlan, completedStepIds);
          if (stepId !== nextStepId) {
            throw new ModelValidationError("只能按训练流程顺序执行下一个阶段");
          }

          await assertGuidedTrainingIdentityCurrent(row, studentId, {
            terminationSignal: lease.terminationSignal,
          });
          const stepTimeoutMs = resolveGuidedStepTimeoutMs(modelType, stepId, body);
          const attemptContext = await beginGuidedStepAttempt(row, stepId, stepTimeoutMs);
          const activeAttempt = attemptContext.attempt;
          activeAttemptRef.current = activeAttempt;
          let receipt = attemptContext.receipt;

          if (!receipt) {
            const sessionModelPath = getGuidedSessionModelPath(row.artifact_dir, modelType);
            const pythonParams = await buildGuidedTrainingPythonParams(
              modelType,
              studentId,
              body,
              row.artifact_dir,
              lease.terminationSignal,
            );
            pythonParams.saveModel = sessionModelPath;
            await runPythonGuidedTrainingStep(
              modelType,
              stepId,
              pythonParams,
              activeAttempt.attempt_id,
              activeAttempt.input_state_path,
              activeAttempt.candidate_state_path,
              activeAttempt.receipt_path,
              row.artifact_dir,
              stepTimeoutMs,
              lease.terminationSignal,
            );
            receipt = parseGuidedAttemptReceipt(activeAttempt);
            if (!receipt) throw new Error("Python 步骤已结束但未生成有效的 checkpoint receipt");
          }

          await lease.verifyOwned();
          await recordGuidedAttemptPythonCompleted(activeAttempt, receipt);
          await assertGuidedTrainingIdentityCurrent(row, studentId, {
            terminationSignal: lease.terminationSignal,
          });

          const nextCompletedStepIds = [...completedStepIds, stepId];
          const stepOutputs = parseJsonColumn<Record<string, unknown>>(row.step_outputs, {});
          const nextStepOutputs = { ...stepOutputs, [stepId]: receipt.output ?? {} };
          let nextStepPlan = stepPlan;
          if (shouldTrimBoostingRoundSteps(modelType, stepId, receipt.output)) {
            nextStepPlan = trimBoostingStepPlanAfterEarlyStop(nextStepPlan, stepId);
            if (isRecordValue(receipt.output)) {
              nextStepPlan = expandBoostingRefitSteps(nextStepPlan, receipt.output);
            }
          }
          if (modelType === "boosting") {
            nextStepPlan = trimBoostingRefitSteps(nextStepPlan, stepId, receipt.output);
          }

          let cleanupUncommittedArtifactPath: string | null = null;
          try {
            let transformedResult: ReturnType<typeof transformTrainingResult> | null = null;
            let versionedArtifactPath: string | null = null;
            let artifactSha256: string | null = null;
            const receiptFinalResult = isRecordValue(receipt.final_result) ? receipt.final_result : null;
            if (receiptFinalResult) {
              const sourceModelPath = getGuidedSessionModelPath(row.artifact_dir, row.model_type);
              versionedArtifactPath = buildVersionedModelPath(
                row.experiment_id,
                row.student_id,
                row.model_type,
                row.session_id,
                "backtest",
              );
              let versionedArtifactWasReusable: boolean;
              if (hasSavedModelArtifact(row.model_type, versionedArtifactPath)) {
                try {
                  const integrity = await verifyBaseModelArtifactIntegrity(
                    row.model_type,
                    versionedArtifactPath,
                    null,
                    lease.terminationSignal,
                  );
                  artifactSha256 = integrity.sha256;
                  versionedArtifactWasReusable = true;
                } catch (error) {
                  if (!(error instanceof ModelArtifactError)) throw error;
                  removeModelArtifactGroup(versionedArtifactPath);
                  versionedArtifactWasReusable = false;
                }
              } else {
                versionedArtifactWasReusable = false;
              }
              if (!versionedArtifactWasReusable) {
                await lease.verifyOwned();
                cleanupUncommittedArtifactPath = versionedArtifactPath;
                await copyModelArtifact(
                  row.model_type,
                  sourceModelPath,
                  versionedArtifactPath,
                  lease.terminationSignal,
                );
                await lease.verifyOwned();
              }
              if (!artifactSha256) {
                // The current guided step has just produced a validated result
                // and copyModelArtifact created this destination from that
                // committed session artifact.  Hash the new output directly;
                // semantic Python validation is reserved for recovering a
                // pre-existing, otherwise untrusted destination above.
                artifactSha256 = await computeModelArtifactSha256(
                  row.model_type,
                  versionedArtifactPath,
                  lease.terminationSignal,
                );
                await lease.verifyOwned();
              }
              receiptFinalResult.saved_model = versionedArtifactPath;
              transformedResult = transformTrainingResult(
                modelType,
                receiptFinalResult,
                expectedEvaluationLength,
              );
            }

            const status: GuidedTrainingStatus = transformedResult ? "completed" : "ready";
            const currentStepId = status === "completed"
              ? null
              : getGuidedNextStepId(nextStepPlan, nextCompletedStepIds);

            await lease.verifyOwned();
            if (transformedResult && versionedArtifactPath) {
              await commitModelTrainingRun({
                runId: row.session_id,
                sessionId: row.session_id,
                experimentId: row.experiment_id,
                studentId: row.student_id,
                modelType: row.model_type,
                requestHash: row.request_hash,
                datasetRevision: row.dataset_revision,
                runtimeRevision: row.runtime_revision,
                body,
                result: transformedResult,
                artifactPath: versionedArtifactPath,
                artifactSha256,
                guidedSession: {
                  sessionId,
                  currentStepId,
                  completedStepIds: nextCompletedStepIds,
                  stepOutputs: nextStepOutputs,
                  stepPlan: nextStepPlan,
                  result: transformedResult,
                  activeAttemptId: activeAttempt.attempt_id,
                  committedStatePath: activeAttempt.candidate_state_path,
                  committedStateSha256: String(receipt.candidate_state_sha256),
                },
              });
            } else {
              await commitGuidedIntermediateStep({
                sessionId,
                attempt: activeAttempt,
                status,
                currentStepId,
                completedStepIds: nextCompletedStepIds,
                stepOutputs: nextStepOutputs,
                stepPlan: nextStepPlan,
                receipt,
              });
            }
            stepCommitted = true;
            cleanupUncommittedArtifactPath = null;

            const updatedRow = await fetchGuidedTrainingSessionRow(sessionId);
            if (!updatedRow) throw new Error("分阶段训练会话已丢失");
            return buildGuidedTrainingSessionPayload(updatedRow);
          } catch (error) {
            if (
              cleanupUncommittedArtifactPath
              && !isModelCommitOutcomeUnknownError(error)
            ) {
              removeModelArtifactGroup(cleanupUncommittedArtifactPath);
            }
            throw error;
          }
        },
      );
    });
  } catch (error) {
    const failedAttempt = activeAttemptRef.current;
    if (
      !stepCommitted
      && failedAttempt
      && !isModelCommitOutcomeUnknownError(error)
    ) {
      // Failure recording is best-effort and must never replace the original
      // model/deadline/lock error returned to the caller.
      await recordGuidedAttemptFailure(
        sessionId,
        failedAttempt.attempt_id,
        stepId,
        error,
      );
      // Invalid/partial candidates are removed for every failure mode. A fully
      // validated receipt is intentionally retained for idempotent recovery.
      cleanupIncompleteGuidedAttemptArtifacts(failedAttempt);
    }
    throw error;
  }
}
