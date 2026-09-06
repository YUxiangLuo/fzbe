import type { PoolConnection, RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import logger from "../../utils/logger";
import { acquireConnectionWithTimeout, commitConnectionWithOutcomeGuard, queryWithTimeout } from "./model-db-guard";
import { ModelConflictError } from "./model-errors";
import { PRODUCTION_PREPARATION_PROTOCOL_VERSION } from "./model-service-constants";
import {
  type ProductionPreparationSessionRow,
  type ProductionPreparationAttemptRow,
  type ProductionPreparationStatus,
  type ProductionPreparationStepDefinition,
  parseProductionCompletedStepIds,
  parseProductionPreparationStepPlan,
  hasValidProductionStepProgress,
  jsonColumnCasValue,
} from "./production-preparation-state";

// Owns attempt transactions, lock ordering and commit-outcome handling.
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

export const fetchProductionPreparationAttempt = async (attemptId: string) => {
  const [rows] = await queryWithTimeout<ProductionPreparationAttemptRow[]>(pool,
    `SELECT *, lease_expires_at > NOW() AS lease_is_active
       FROM model_production_preparation_step_attempts
      WHERE attempt_id = ?
      LIMIT 1`,
    [attemptId],
  );
  return rows[0] ?? null;
};

export const claimProductionPreparationReceiptReplay = async (
  session: ProductionPreparationSessionRow,
  attempt: ProductionPreparationAttemptRow,
  leaseSeconds: number,
) => await runProductionAttemptTransaction(async (connection) => {
  const [sessionRows] = await connection.query<ProductionPreparationSessionRow[]>(
    `SELECT *
       FROM model_production_preparation_sessions
      WHERE session_id = ?
      LIMIT 1
      FOR UPDATE`,
    [session.session_id],
  );
  const lockedSession = sessionRows[0];
  if (
    !lockedSession
    || lockedSession.status !== session.status
    || lockedSession.active_attempt_id !== attempt.attempt_id
    || lockedSession.current_step_id !== session.current_step_id
  ) {
    return false;
  }

  const [attemptRows] = await connection.query<ProductionPreparationAttemptRow[]>(
    `SELECT *
       FROM model_production_preparation_step_attempts
      WHERE attempt_id = ?
        AND session_id = ?
      LIMIT 1
      FOR UPDATE`,
    [attempt.attempt_id, session.session_id],
  );
  const lockedAttempt = attemptRows[0];
  if (
    !lockedAttempt
    || lockedAttempt.step_id !== attempt.step_id
    || !(["running", "failed"] as const).includes(
      lockedAttempt.status as "running" | "failed",
    )
    || lockedAttempt.status !== attempt.status
    || lockedAttempt.input_state_path !== attempt.input_state_path
    || lockedAttempt.input_state_sha256 !== attempt.input_state_sha256
    || lockedAttempt.candidate_state_path !== attempt.candidate_state_path
    || lockedAttempt.candidate_state_sha256 !== attempt.candidate_state_sha256
    || lockedAttempt.receipt_path !== attempt.receipt_path
  ) {
    return false;
  }

  const [attemptUpdate] = await connection.query(
    `UPDATE model_production_preparation_step_attempts
        SET status = 'running',
            error_message = NULL,
            lease_expires_at = DATE_ADD(NOW(), INTERVAL ? SECOND)
      WHERE attempt_id = ?
        AND session_id = ?
        AND step_id = ?
        AND status = ?
        AND input_state_path <=> ?
        AND input_state_sha256 <=> ?
        AND candidate_state_path = ?
        AND candidate_state_sha256 <=> ?
        AND receipt_path = ?`,
    [
      leaseSeconds,
      attempt.attempt_id,
      session.session_id,
      attempt.step_id,
      attempt.status,
      attempt.input_state_path,
      attempt.input_state_sha256,
      attempt.candidate_state_path,
      attempt.candidate_state_sha256,
      attempt.receipt_path,
    ],
  );
  if (Number((attemptUpdate as { affectedRows?: number }).affectedRows ?? 0) !== 1) {
    return false;
  }

  const [sessionUpdate] = await connection.query(
    `UPDATE model_production_preparation_sessions
        SET status = 'running',
            current_step_id = ?,
            error_message = NULL,
            expires_at = DATE_ADD(NOW(), INTERVAL 7 DAY)
      WHERE session_id = ?
        AND status = ?
        AND active_attempt_id = ?
        AND current_step_id <=> ?
        AND completed_step_ids = ?
        AND step_plan = ?
        AND committed_state_path <=> ?
        AND committed_state_sha256 <=> ?
        AND status IN ('ready', 'running', 'failed')`,
    [
      attempt.step_id,
      session.session_id,
      session.status,
      attempt.attempt_id,
      session.current_step_id,
      jsonColumnCasValue(session.completed_step_ids),
      jsonColumnCasValue(session.step_plan),
      session.committed_state_path,
      session.committed_state_sha256,
    ],
  );
  if (Number((sessionUpdate as { affectedRows?: number }).affectedRows ?? 0) !== 1) {
    throw new ModelConflictError(
      "生产准备 receipt replay ownership 已变化",
      "PRODUCTION_PREPARATION_STEP_IN_PROGRESS",
    );
  }
  return true;
});

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

    const completedStepIds = parseProductionCompletedStepIds(session.completed_step_ids);
    const stepPlan = parseProductionPreparationStepPlan(session.step_plan);
    if (!completedStepIds || !stepPlan || !hasValidProductionStepProgress(session)) {
      throw new ModelConflictError(
        "生产准备步骤状态已损坏",
        "PRODUCTION_PREPARATION_STALE",
      );
    }
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
