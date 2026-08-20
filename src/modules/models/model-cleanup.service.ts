import fs from "fs";
import path from "path";
import type { Stats } from "node:fs";
import type { RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import { CLEANUP_RETENTION } from "../../utils/constants";
import logger from "../../utils/logger";
import { PY_DIR } from "../../utils/python";
import { queryWithTimeout } from "./model-db-guard";
import {
  releaseModelExecutionLock,
  tryAcquireModelExecutionLock,
  type ModelExecutionLockHandle,
} from "./model-execution-lock";
import { removeModelArtifactGroup } from "./model-artifacts";
import { GUIDED_MODEL_TYPES } from "./model-types";
import type { ModelType } from "./models.schema";
import { GUIDED_TRAINING_ROOT } from "./model-runtime";
import { PRODUCTION_PREPARATION_ROOT } from "./model-service-constants";

type ProductionPreparationStatus = "ready" | "running" | "failed" | "completed" | "superseded";

interface GuidedTrainingSessionRow extends RowDataPacket {
  session_id: string;
  artifact_dir: string;
}

interface GuidedTrainingCleanupRow extends RowDataPacket {
  session_id: string;
  experiment_id: number;
  model_type: ModelType;
  artifact_dir: string;
  status: string;
}

interface ProductionPreparationCleanupRow extends GuidedTrainingCleanupRow {
  status: ProductionPreparationStatus;
  candidate_model_path: string;
  candidate_is_active: number | boolean | string;
}

interface StaleModelExperimentEligibilityRow extends RowDataPacket {
  can_delete: 0 | 1 | boolean | string;
}

const GUIDED_TRAINING_CLEANUP_BATCH_SIZE = 500;
const PRODUCTION_PREPARATION_CLEANUP_BATCH_SIZE = 500;

const tryAcquireExperimentCleanupLocks = (experimentId: number) => {
  const handles: ModelExecutionLockHandle[] = [];
  for (const modelType of GUIDED_MODEL_TYPES) {
    const handle = tryAcquireModelExecutionLock(experimentId, modelType);
    if (!handle) {
      for (const acquired of handles.reverse()) {
        releaseModelExecutionLock(acquired);
      }
      return null;
    }
    handles.push(handle);
  }
  return {
    release: () => {
      for (const handle of handles.reverse()) {
        releaseModelExecutionLock(handle);
      }
    },
  };
};

const isPathInsideDirectory = (candidatePath: string, rootDirectory: string) => {
  const resolvedCandidate = path.resolve(candidatePath);
  const resolvedRoot = path.resolve(rootDirectory);
  const relativePath = path.relative(resolvedRoot, resolvedCandidate);
  return relativePath.length > 0
    && !relativePath.startsWith("..")
    && !path.isAbsolute(relativePath);
};

export const removeGuidedTrainingArtifactDir = (artifactDir: string) => {
  if (!isPathInsideDirectory(artifactDir, GUIDED_TRAINING_ROOT)) {
    logger.warn(
      { artifactDir, guidedTrainingRoot: GUIDED_TRAINING_ROOT },
      "跳过不安全的 guided training 工件目录清理",
    );
    return false;
  }

  fs.rmSync(artifactDir, { recursive: true, force: true });
  return true;
};

export const removeProductionPreparationArtifactDir = (artifactDir: string) => {
  if (!isPathInsideDirectory(artifactDir, PRODUCTION_PREPARATION_ROOT)) {
    logger.warn(
      { artifactDir, productionPreparationRoot: PRODUCTION_PREPARATION_ROOT },
      "跳过不安全的 production preparation 工件目录清理",
    );
    return false;
  }

  fs.rmSync(artifactDir, { recursive: true, force: true });
  return true;
};

export const deleteGuidedTrainingSession = async (
  row: Pick<GuidedTrainingSessionRow, "session_id" | "artifact_dir">,
) => {
  let canDeleteSessionRow = false;
  try {
    removeGuidedTrainingArtifactDir(row.artifact_dir);
    canDeleteSessionRow = true;
  } catch (error) {
    logger.warn(
      {
        err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
        sessionId: row.session_id,
        artifactDir: row.artifact_dir,
      },
      "删除 guided training 工件目录失败，将保留会话记录以便后续重试",
    );
  }

  if (!canDeleteSessionRow) {
    return false;
  }

  await queryWithTimeout(pool,
    `DELETE FROM model_guided_training_sessions
      WHERE session_id = ?`,
    [row.session_id],
  );
  return true;
};

const cleanupOrphanGuidedTrainingArtifactDirs = async (
  now: Date,
  maxAgeMs: number,
) => {
  let entries: string[];
  try {
    entries = fs.readdirSync(GUIDED_TRAINING_ROOT);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return 0;
    }
    throw error;
  }

  let deletedArtifactDirs = 0;
  for (const entry of entries) {
    const artifactDir = path.join(GUIDED_TRAINING_ROOT, entry);
    let stats: Stats;
    try {
      stats = fs.statSync(artifactDir);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        logger.warn({ err: error, artifactDir }, "读取 guided training 工件目录状态失败");
      }
      continue;
    }

    if (!stats.isDirectory() || now.getTime() - stats.mtimeMs < maxAgeMs) {
      continue;
    }

    const [rows] = await queryWithTimeout<RowDataPacket[]>(pool,
      `SELECT session_id
         FROM model_guided_training_sessions
        WHERE artifact_dir = ?
        LIMIT 1`,
      [artifactDir],
    );
    if (rows.length > 0) {
      continue;
    }

    try {
      if (removeGuidedTrainingArtifactDir(artifactDir)) {
        deletedArtifactDirs += 1;
      }
    } catch (error) {
      logger.warn(
        {
          err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
          artifactDir,
        },
        "清理 orphan guided training 工件目录失败",
      );
    }
  }

  return deletedArtifactDirs;
};

export async function cleanupExpiredGuidedTrainingSessions(
  options: { now?: Date; limit?: number; orphanMaxAgeMs?: number } = {},
) {
  const now = options.now ?? new Date();
  const rawLimit = options.limit ?? GUIDED_TRAINING_CLEANUP_BATCH_SIZE;
  const limit = Number.isInteger(rawLimit) && rawLimit > 0
    ? Math.min(rawLimit, GUIDED_TRAINING_CLEANUP_BATCH_SIZE)
    : GUIDED_TRAINING_CLEANUP_BATCH_SIZE;

  const expirationCutoffExpression = options.now ? "?" : "NOW()";
  const expirationCutoffParams = options.now ? [options.now, options.now] : [];
  const [rows] = await queryWithTimeout<GuidedTrainingCleanupRow[]>(pool,
    `SELECT sessions.session_id,
            sessions.experiment_id,
            sessions.model_type,
            sessions.artifact_dir,
            sessions.status
       FROM model_guided_training_sessions AS sessions
       LEFT JOIN model_guided_training_step_attempts AS attempts
         ON attempts.attempt_id = sessions.active_attempt_id
      WHERE sessions.expires_at IS NOT NULL
        AND sessions.expires_at <= ${expirationCutoffExpression}
        AND NOT (
          sessions.status = 'running'
          AND attempts.status IN ('running', 'python_completed')
          AND attempts.lease_expires_at IS NOT NULL
          AND attempts.lease_expires_at > ${expirationCutoffExpression}
        )
      ORDER BY sessions.expires_at ASC
      LIMIT ${limit}`,
    expirationCutoffParams,
  );

  let deletedArtifactDirs = 0;
  const sessionIdsToDelete: string[] = [];
  for (const row of rows) {
    const executionLock = tryAcquireModelExecutionLock(
      Number(row.experiment_id),
      row.model_type,
    );
    if (!executionLock) continue;
    try {
      if (row.status !== "superseded") {
        const claimCutoffParams = options.now
          ? [row.session_id, options.now, options.now]
          : [row.session_id];
        const [claimResult] = await queryWithTimeout(pool,
          `UPDATE model_guided_training_sessions AS sessions
           LEFT JOIN model_guided_training_step_attempts AS attempts
             ON attempts.attempt_id = sessions.active_attempt_id
              SET sessions.status = 'superseded',
                  sessions.current_step_id = NULL,
                  sessions.active_attempt_id = NULL,
                  sessions.superseded_reason = 'Guided training session expired'
            WHERE sessions.session_id = ?
              AND sessions.expires_at IS NOT NULL
              AND sessions.expires_at <= ${expirationCutoffExpression}
              AND sessions.status <> 'superseded'
              AND NOT (
                sessions.status = 'running'
                AND attempts.status IN ('running', 'python_completed')
                AND attempts.lease_expires_at IS NOT NULL
                AND attempts.lease_expires_at > ${expirationCutoffExpression}
              )`,
          claimCutoffParams,
        );
        if (Number((claimResult as { affectedRows?: number }).affectedRows ?? 0) !== 1) {
          continue;
        }
      }
      if (removeGuidedTrainingArtifactDir(row.artifact_dir)) {
        deletedArtifactDirs += 1;
      }
      sessionIdsToDelete.push(row.session_id);
    } catch (error) {
      logger.warn(
        {
          err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
          sessionId: row.session_id,
          artifactDir: row.artifact_dir,
        },
        "清理过期 guided training 工件目录失败",
      );
    } finally {
      releaseModelExecutionLock(executionLock);
    }
  }

  if (sessionIdsToDelete.length > 0) {
    const placeholders = sessionIdsToDelete.map(() => "?").join(", ");
    await queryWithTimeout(pool,
      `DELETE FROM model_guided_training_sessions
        WHERE session_id IN (${placeholders})`,
      sessionIdsToDelete,
    );
  }

  deletedArtifactDirs += await cleanupOrphanGuidedTrainingArtifactDirs(
    now,
    options.orphanMaxAgeMs ?? CLEANUP_RETENTION.GUIDED_TRAINING_ORPHAN_MS,
  );

  return {
    deletedSessions: sessionIdsToDelete.length,
    deletedArtifactDirs,
  };
}

const cleanupOrphanProductionPreparationArtifactDirs = async (
  now: Date,
  maxAgeMs: number,
) => {
  let entries: string[];
  try {
    entries = fs.readdirSync(PRODUCTION_PREPARATION_ROOT);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return 0;
    throw error;
  }

  let deletedArtifactDirs = 0;
  for (const entry of entries) {
    const artifactDir = path.join(PRODUCTION_PREPARATION_ROOT, entry);
    let stats: Stats;
    try {
      stats = fs.statSync(artifactDir);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        logger.warn({ err: error, artifactDir }, "读取 production preparation 工件目录状态失败");
      }
      continue;
    }
    if (!stats.isDirectory() || now.getTime() - stats.mtimeMs < maxAgeMs) continue;

    const [rows] = await queryWithTimeout<RowDataPacket[]>(pool,
      `SELECT session_id
         FROM model_production_preparation_sessions
        WHERE artifact_dir = ?
        LIMIT 1`,
      [artifactDir],
    );
    if (rows.length > 0) continue;
    try {
      if (removeProductionPreparationArtifactDir(artifactDir)) deletedArtifactDirs += 1;
    } catch (error) {
      logger.warn(
        {
          err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
          artifactDir,
        },
        "清理 orphan production preparation 工件目录失败",
      );
    }
  }
  return deletedArtifactDirs;
};

export async function cleanupExpiredProductionPreparationSessions(
  options: { now?: Date; limit?: number; orphanMaxAgeMs?: number } = {},
) {
  const now = options.now ?? new Date();
  const rawLimit = options.limit ?? PRODUCTION_PREPARATION_CLEANUP_BATCH_SIZE;
  const limit = Number.isInteger(rawLimit) && rawLimit > 0
    ? Math.min(rawLimit, PRODUCTION_PREPARATION_CLEANUP_BATCH_SIZE)
    : PRODUCTION_PREPARATION_CLEANUP_BATCH_SIZE;
  const expirationCutoffExpression = options.now ? "?" : "NOW()";
  const expirationCutoffParams = options.now ? [options.now, options.now] : [];
  const [rows] = await queryWithTimeout<ProductionPreparationCleanupRow[]>(pool,
    `SELECT sessions.session_id,
            sessions.experiment_id,
            sessions.model_type,
            sessions.artifact_dir,
            sessions.status,
            sessions.candidate_model_path,
            EXISTS (
              SELECT 1
                FROM model_training_runs AS active_runs
               WHERE active_runs.is_active = TRUE
                 AND (active_runs.artifact_path = sessions.candidate_model_path
                      OR active_runs.production_artifact_path = sessions.candidate_model_path)
            ) AS candidate_is_active
       FROM model_production_preparation_sessions AS sessions
       LEFT JOIN model_production_preparation_step_attempts AS attempts
         ON attempts.attempt_id = sessions.active_attempt_id
      WHERE sessions.expires_at IS NOT NULL
        AND sessions.expires_at <= ${expirationCutoffExpression}
        AND NOT (
          attempts.status = 'running'
          AND attempts.lease_expires_at IS NOT NULL
          AND attempts.lease_expires_at > ${expirationCutoffExpression}
        )
      ORDER BY sessions.expires_at ASC
      LIMIT ${limit}`,
    expirationCutoffParams,
  );

  let deletedArtifactDirs = 0;
  const sessionIdsToDelete: string[] = [];
  for (const row of rows) {
    const executionLock = tryAcquireModelExecutionLock(
      Number(row.experiment_id),
      row.model_type,
    );
    if (!executionLock) continue;
    try {
      if (row.status !== "superseded") {
        const claimCutoffParams = options.now ? [row.session_id, options.now, options.now] : [row.session_id];
        const completedClaim = row.status === "completed";
        const [claimResult] = await queryWithTimeout(pool,
          `UPDATE model_production_preparation_sessions AS sessions
           LEFT JOIN model_production_preparation_step_attempts AS attempts
             ON attempts.attempt_id = sessions.active_attempt_id
              SET ${completedClaim
                ? "sessions.superseded_reason = 'Completed production preparation cleanup claimed'"
                : `sessions.status = 'superseded',
                  sessions.current_step_id = NULL,
                  sessions.active_attempt_id = NULL,
                  sessions.superseded_reason = 'Production preparation session expired'`}
            WHERE sessions.session_id = ?
              AND sessions.expires_at IS NOT NULL
              AND sessions.expires_at <= ${expirationCutoffExpression}
              AND sessions.status = '${completedClaim ? "completed" : row.status}'
              AND NOT (
                attempts.status = 'running'
                AND attempts.lease_expires_at IS NOT NULL
                AND attempts.lease_expires_at > ${expirationCutoffExpression}
              )`,
          claimCutoffParams,
        );
        if (Number((claimResult as { affectedRows?: number }).affectedRows ?? 0) !== 1) continue;
      }
      if (!removeProductionPreparationArtifactDir(row.artifact_dir)) continue;
      deletedArtifactDirs += 1;
      if (!Boolean(Number(row.candidate_is_active))) {
        removeModelArtifactGroup(row.candidate_model_path);
      }
      sessionIdsToDelete.push(row.session_id);
    } catch (error) {
      logger.warn(
        {
          err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
          sessionId: row.session_id,
          artifactDir: row.artifact_dir,
        },
        "清理过期 production preparation 工件目录失败",
      );
    } finally {
      releaseModelExecutionLock(executionLock);
    }
  }
  if (sessionIdsToDelete.length > 0) {
    const placeholders = sessionIdsToDelete.map(() => "?").join(", ");
    await queryWithTimeout(pool,
      `DELETE FROM model_production_preparation_sessions
        WHERE session_id IN (${placeholders})`,
      sessionIdsToDelete,
    );
  }
  deletedArtifactDirs += await cleanupOrphanProductionPreparationArtifactDirs(
    now,
    options.orphanMaxAgeMs ?? CLEANUP_RETENTION.PRODUCTION_PREPARATION_ORPHAN_MS,
  );
  return { deletedSessions: sessionIdsToDelete.length, deletedArtifactDirs };
}

export function cleanupExperimentModelArtifacts(experimentId: number, studentId: number) {
  const executionLocks = tryAcquireExperimentCleanupLocks(experimentId);
  if (!executionLocks) {
    logger.info(
      { experimentId, studentId },
      "模型任务仍在执行，延后清理实验模型产物",
    );
    return;
  }
  try {
    const trainedModelsDir = path.join(PY_DIR, "trained_models");
    let entries: string[];
    try {
      entries = fs.readdirSync(trainedModelsDir);
    } catch {
      return;
    }

    const prefix = `exp${experimentId}-student${studentId}-`;
    let deleted = 0;
    for (const entry of entries) {
      if (!entry.startsWith(prefix)) continue;
      try {
        fs.unlinkSync(path.join(trainedModelsDir, entry));
        deleted++;
      } catch (error: unknown) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
          logger.warn({ err: error, file: entry }, "删除模型产物失败");
        }
      }
    }

    if (deleted > 0) {
      logger.info({ experimentId, studentId, deleted }, "已清理实验模型产物");
    }
  } finally {
    executionLocks.release();
  }
}

interface ModelArtifactGroup {
  experimentId: number;
  studentId: number;
  prefix: string;
  files: string[];
  newestMtimeMs: number;
}

const MODEL_ARTIFACT_PREFIX_PATTERN = /^exp(\d+)-student(\d+)-/;

const toCleanupAgeSeconds = (maxAgeMs: number) => Math.max(1, Math.ceil(maxAgeMs / 1000));

const isDbTruthy = (value: unknown) =>
  value === true
  || value === 1
  || value === "1"
  || value === BigInt(1);

const shouldDeleteStaleModelArtifactGroup = async (
  group: ModelArtifactGroup,
  maxAgeSeconds: number,
  fileAgeEligible: boolean,
) => {
  const [rows] = await queryWithTimeout<StaleModelExperimentEligibilityRow[]>(pool,
    `SELECT CASE
              WHEN status = 'Completed' THEN 1
              WHEN ? = 1 AND TIMESTAMPDIFF(
                SECOND,
                GREATEST(
                  COALESCE(last_activity_at, CAST('1970-01-01 00:00:00' AS DATETIME)),
                  COALESCE(completion_time, CAST('1970-01-01 00:00:00' AS DATETIME)),
                  COALESCE(start_time, CAST('1970-01-01 00:00:00' AS DATETIME))
                ),
                NOW()
              ) >= ? THEN 1
              ELSE 0
            END AS can_delete
       FROM experiment_status
      WHERE experiment_id = ?
        AND student_id = ?
      LIMIT 1`,
    [fileAgeEligible ? 1 : 0, maxAgeSeconds, group.experimentId, group.studentId],
  );

  // Keep recent orphan files for the normal retention period. Completed
  // experiments are allowed through immediately so a report-time cleanup that
  // skipped a busy execution lock is durably recovered by the next sweep.
  if (!rows[0]) return fileAgeEligible;
  return isDbTruthy(rows[0].can_delete);
};

export async function cleanupStaleModelArtifacts(
  options: { now?: Date; maxAgeMs?: number } = {},
) {
  const trainedModelsDir = path.join(PY_DIR, "trained_models");
  const now = options.now ?? new Date();
  const maxAgeMs = options.maxAgeMs ?? CLEANUP_RETENTION.MODEL_ARTIFACTS_MS;
  const cutoffMs = now.getTime() - maxAgeMs;
  const maxAgeSeconds = toCleanupAgeSeconds(maxAgeMs);
  const result = {
    scannedGroups: 0,
    deletedGroups: 0,
    deletedFiles: 0,
    failedFiles: 0,
  };

  let entries: string[];
  try {
    entries = fs.readdirSync(trainedModelsDir);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return result;
    }
    throw error;
  }

  const groups = new Map<string, ModelArtifactGroup>();
  for (const entry of entries) {
    const match = MODEL_ARTIFACT_PREFIX_PATTERN.exec(entry);
    if (!match) continue;

    const filePath = path.join(trainedModelsDir, entry);
    let stats: Stats;
    try {
      stats = fs.statSync(filePath);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        logger.warn({ err: error, file: entry }, "读取模型产物状态失败");
      }
      continue;
    }
    if (!stats.isFile()) continue;

    const prefix = match[0]!;
    const existing = groups.get(prefix);
    if (existing) {
      existing.files.push(entry);
      existing.newestMtimeMs = Math.max(existing.newestMtimeMs, stats.mtimeMs);
    } else {
      groups.set(prefix, {
        experimentId: Number(match[1]),
        studentId: Number(match[2]),
        prefix,
        files: [entry],
        newestMtimeMs: stats.mtimeMs,
      });
    }
  }

  for (const group of groups.values()) {
    result.scannedGroups += 1;
    const executionLocks = tryAcquireExperimentCleanupLocks(group.experimentId);
    if (!executionLocks) continue;
    try {
      if (!await shouldDeleteStaleModelArtifactGroup(
        group,
        maxAgeSeconds,
        group.newestMtimeMs <= cutoffMs,
      )) {
        continue;
      }

      let deletedInGroup = 0;
      for (const file of group.files) {
        try {
          fs.unlinkSync(path.join(trainedModelsDir, file));
          deletedInGroup += 1;
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
            result.failedFiles += 1;
            logger.warn({ err: error, file }, "删除过期模型产物失败");
          }
        }
      }

      if (deletedInGroup > 0) {
        result.deletedGroups += 1;
        result.deletedFiles += deletedInGroup;
        logger.info(
          {
            experimentId: group.experimentId,
            studentId: group.studentId,
            prefix: group.prefix,
            deletedFiles: deletedInGroup,
          },
          "已清理过期模型产物组",
        );
      }
    } finally {
      executionLocks.release();
    }
  }

  return result;
}
