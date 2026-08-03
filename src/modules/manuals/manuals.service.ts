import { createHash } from "node:crypto";
import { rm } from "node:fs/promises";
import path from "path";
import type { RowDataPacket, ResultSetHeader } from "mysql2";
import { env, pool } from "../../config";
import type { ExperimentManual } from "../../types";
import { validateFile, generateUniqueFilename, FileValidationError } from "../../utils/file-validator";
import { FILE_SIZE_LIMITS } from "../../utils/constants";
import { logWarning } from "../../utils/error-handler";
import {
  purgeStagedManagedPaths,
  reconcileStagedDeletionJournals,
  restoreStagedManagedPaths,
  stageManagedPathsForDeletion,
  type ManagedDeletionJournal,
  type StagedManagedPath,
} from "../../utils/staged-file-deletion";
import {
  tryAcquireManagedDeletionLocks,
  type ManagedDeletionLockLease,
} from "../../utils/managed-deletion-lock";
import {
  acquireManagedDeletionPermit,
} from "../../utils/managed-deletion-admission";
import {
  ensureDatabaseManagedStorageRoot,
  getDatabaseManagedStorageRoot,
  resolveStoredManagedPath,
} from "../../utils/managed-storage";
import type { UpdateManualInput } from "./manuals.schema";
import { databaseDatetimeToIsoString } from "../../utils/database-datetime";

type ManualRow = ExperimentManual & RowDataPacket & { uploader_name?: string };
type ManualFileRow = RowDataPacket & { manual_id: number; file_path: string };

const uploadsRoot = path.resolve(import.meta.dir, "../../../uploads");
const manualsRoot = path.resolve(uploadsRoot, "manuals");
const MANUAL_DELETION_LOCK_TIMEOUT_SECONDS = 10;

const manualStorageRoot = () => getDatabaseManagedStorageRoot(manualsRoot, env);

const manualDeletionLockIdentity = (manualId: number) => ({
  databaseName: env.DB_DATABASE,
  kind: "manual",
  entityIds: [manualId],
  // Different manual rows may intentionally reference the same PDF. The
  // kind-wide lock keeps those cross-ID decisions serialized across processes.
  includeGlobalLock: true,
});

function canonicalPathFingerprint(absolutePath: string): string {
  return createHash("sha256").update(path.resolve(absolutePath)).digest("hex");
}

async function resolveManualFile(
  storedPath: string,
  enforceDisposableNamespace = false,
) {
  return resolveStoredManagedPath({
    storedPath,
    managedRoot: manualsRoot,
    environment: env,
    legacyFallbackDirectory: manualStorageRoot(),
    enforceDisposableNamespace,
  });
}

async function releaseManualDeletionLock(
  lease: ManagedDeletionLockLease,
  context: Record<string, unknown>,
) {
  await lease.release().catch((error) => {
    logWarning("Failed to release manual deletion advisory lock", {
      ...context,
      error: error instanceof Error ? error.message : String(error),
    });
  });
}

function toPublicFilePath(absolutePath: string | null | undefined): string | null {
  if (!absolutePath) return null;
  // Physical disposable-database namespaces are intentionally hidden behind
  // the stable authenticated URL contract.
  return `manuals/${path.basename(absolutePath)}`.replace(/\\/g, "/");
}

function normalizeManualRow<T extends { file_path?: string | null; uploaded_at?: unknown }>(
  row: T,
): Omit<T, "file_path" | "uploaded_at"> & { file_path: string | null; uploaded_at: string | null } {
  return {
    ...row,
    file_path: toPublicFilePath(row.file_path),
    uploaded_at: databaseDatetimeToIsoString(row.uploaded_at, env.DB_TIMEZONE),
  };
}

export interface UploadManualInput {
  file: File;
  fileName: string;
  description?: string;
  uploaderId: number;
}

async function probeUploadedManualAfterAmbiguousCommit(
  manualId: number,
  uploadPath: string,
): Promise<
  | { state: "committed"; manual: ReturnType<typeof normalizeManualRow<ManualRow>> }
  | { state: "not-committed" }
  | { state: "unresolved" }
> {
  const connection = await pool.getConnection().catch((error) => {
    logWarning("Ambiguous manual upload commit probe could not acquire a connection", {
      manualId,
      error: error instanceof Error ? error.message : String(error),
    });
    return null;
  });
  if (!connection) return { state: "unresolved" };

  let released = false;
  try {
    await connection.beginTransaction();
    const [rows] = await connection.query<ManualRow[]>(
      `SELECT m.*, u.full_name as uploader_name
         FROM experiment_manuals m
         LEFT JOIN users u ON m.uploader_id = u.user_id
        WHERE m.manual_id = ? AND m.file_path = ?
        FOR UPDATE`,
      [manualId, uploadPath],
    );
    if (rows[0]) {
      try {
        await connection.commit();
        connection.release();
      } catch {
        // The probe is read-only. Destroying the uncertain session releases
        // its locks, while the row observed under lock remains authoritative.
        connection.destroy();
      }
      released = true;
      return { state: "committed", manual: normalizeManualRow(rows[0]) };
    }

    // Keep the missing-row lock until the orphan file is removed. This closes
    // the narrow ABA window in which a row with the same explicit ID/path
    // could otherwise be inserted after the probe but before cleanup.
    await rm(uploadPath, { force: true });
    try {
      await connection.commit();
      connection.release();
    } catch {
      connection.destroy();
    }
    released = true;
    return { state: "not-committed" };
  } catch (error) {
    try {
      await connection.rollback();
    } catch {
      connection.destroy();
      released = true;
    }
    logWarning("Manual upload file retained because ambiguous commit could not be reconciled", {
      manualId,
      error: error instanceof Error ? error.message : String(error),
    });
    return { state: "unresolved" };
  } finally {
    if (!released) connection.release();
  }
}

export async function uploadManual(input: UploadManualInput) {
  const { file, fileName, description, uploaderId } = input;

  await validateFile(file, "pdf", {
    maxSize: FILE_SIZE_LIMITS.PDF_MAX_SIZE,
    checkMagicBytes: true,
  });

  const storageRoot = await ensureDatabaseManagedStorageRoot(manualsRoot, env);
  const uniqueFilename = generateUniqueFilename(file.name, { directory: storageRoot });
  const uploadPath = path.resolve(storageRoot, uniqueFilename);
  try {
    await Bun.write(uploadPath, file);
  } catch (error) {
    // Bun.write may create and partially populate the destination before an
    // ENOSPC/EIO failure. No database row can own this fresh UUID path yet.
    await rm(uploadPath, { force: true }).catch((cleanupError) => {
      logWarning("Failed to cleanup partially written manual upload", {
        path: uploadPath,
        error: cleanupError instanceof Error ? cleanupError.message : String(cleanupError),
      });
    });
    throw error;
  }

  const connection = await pool.getConnection().catch(async (error) => {
    // No database transaction can own this PDF yet.
    await rm(uploadPath, { force: true }).catch(() => {});
    throw error;
  });
  let transactionPhase: "before-commit" | "commit-attempted" | "committed" = "before-commit";
  let released = false;
  let insertedManualId: number | null = null;
  try {
    await connection.beginTransaction();

    const [result] = await connection.query<ResultSetHeader>(
      "INSERT INTO experiment_manuals (file_name, file_path, description, uploader_id) VALUES (?, ?, ?, ?)",
      [fileName, uploadPath, description || null, uploaderId],
    );
    insertedManualId = result.insertId;

    const [rows] = await connection.query<ManualRow[]>(
      `SELECT m.*, u.full_name as uploader_name
       FROM experiment_manuals m
       LEFT JOIN users u ON m.uploader_id = u.user_id
       WHERE m.manual_id = ?`,
      [result.insertId],
    );

    const manual = normalizeManualRow(rows[0]!);
    transactionPhase = "commit-attempted";
    await connection.commit();
    transactionPhase = "committed";
    return manual;
  } catch (error) {
    if (transactionPhase === "commit-attempted" && insertedManualId !== null) {
      connection.destroy();
      released = true;
      const resolution = await probeUploadedManualAfterAmbiguousCommit(
        insertedManualId,
        uploadPath,
      );
      if (resolution.state === "committed") return resolution.manual;
      // The not-committed branch already removed the orphan while holding its
      // gap lock. The unresolved branch deliberately retains a possible orphan
      // rather than risk a committed DB row whose PDF has been deleted.
    } else if (transactionPhase !== "committed") {
      try {
        await connection.rollback();
      } catch {
        connection.destroy();
        released = true;
      }
      await rm(uploadPath, { force: true }).catch(() => {});
    }
    throw error;
  } finally {
    if (!released) connection.release();
  }
}

export async function getAllManuals() {
  const [rows] = await pool.query<ManualRow[]>(`
    SELECT m.*, u.full_name as uploader_name
    FROM experiment_manuals m
    LEFT JOIN users u ON m.uploader_id = u.user_id
    ORDER BY m.uploaded_at DESC
  `);
  return rows.map(normalizeManualRow);
}

export async function getActiveManual() {
  const [rows] = await pool.query<ManualRow[]>(`
    SELECT m.*, u.full_name as uploader_name
    FROM experiment_manuals m
    LEFT JOIN users u ON m.uploader_id = u.user_id
    WHERE m.is_active = true
    LIMIT 1
  `);
  return rows[0] ? normalizeManualRow(rows[0]) : null;
}

export async function updateManual(manualId: number, input: UpdateManualInput) {
  const { file_name, description, is_active } = input;

  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    if (is_active === true) {
      await connection.query(
        "UPDATE experiment_manuals SET is_active = false WHERE manual_id != ?",
        [manualId],
      );
    }

    const fields: string[] = [];
    const values: unknown[] = [];
    if (file_name) {
      fields.push("file_name = ?");
      values.push(file_name);
    }
    if (description !== undefined) {
      fields.push("description = ?");
      values.push(description);
    }
    if (is_active !== undefined) {
      fields.push("is_active = ?");
      values.push(is_active);
    }

    if (fields.length === 0) {
      await connection.rollback();
      return { error: "没有需要更新的字段" as const };
    }

    values.push(manualId);
    const [result] = await connection.query<ResultSetHeader>(
      `UPDATE experiment_manuals SET ${fields.join(", ")} WHERE manual_id = ?`,
      values,
    );

    if (result.affectedRows === 0) {
      await connection.rollback();
      return { error: "not_found" as const };
    }

    await connection.commit();

    const [rows] = await connection.query<ManualRow[]>(
      `SELECT m.*, u.full_name as uploader_name
       FROM experiment_manuals m
       LEFT JOIN users u ON m.uploader_id = u.user_id
       WHERE m.manual_id = ?`,
      [manualId],
    );

    return { data: normalizeManualRow(rows[0]!) };
  } catch (error) {
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}

export async function deleteManual(manualId: number) {
  await ensureDatabaseManagedStorageRoot(manualsRoot, env);
  const deletionPermit = await acquireManagedDeletionPermit();
  if (!deletionPermit) {
    logWarning("Timed out waiting for manual deletion admission", { manualId });
    return { error: "file_cleanup_failed" as const };
  }

  const deletionLock = await tryAcquireManagedDeletionLocks(
    pool,
    manualDeletionLockIdentity(manualId),
    MANUAL_DELETION_LOCK_TIMEOUT_SECONDS,
  ).catch((error) => {
    deletionPermit.release();
    throw error;
  });
  if (!deletionLock) {
    deletionPermit.release();
    logWarning("Timed out acquiring manual deletion advisory lock", { manualId });
    return { error: "file_cleanup_failed" as const };
  }

  const connection = await pool.getConnection().catch(async (error) => {
    await releaseManualDeletionLock(deletionLock, { manualId });
    deletionPermit.release();
    throw error;
  });
  let stagedPaths: StagedManagedPath[] = [];
  let targetFingerprint = "";
  let transactionPhase: "before-commit" | "commit-attempted" | "committed" = "before-commit";
  let released = false;
  let deletionLockReleased = false;

  try {
    await connection.beginTransaction();
    // Lock the complete reference set so a shared file cannot become
    // unreferenced/referenced between the decision and DELETE. Under InnoDB's
    // default repeatable-read isolation this ordered full-range read also
    // prevents a concurrent insert from racing the path decision.
    const [rows] = await connection.query<ManualFileRow[]>(
      `SELECT manual_id, file_path
         FROM experiment_manuals
        ORDER BY manual_id ASC
        FOR UPDATE`,
    );
    const targetRow = rows.find((row) => Number(row.manual_id) === manualId);
    if (!targetRow) {
      await connection.rollback();
      return { error: "not_found" as const };
    }

    try {
      const targetPath = (await resolveManualFile(targetRow.file_path, true)).absolutePath;
      targetFingerprint = canonicalPathFingerprint(targetPath);
      const survivingReferences = new Set<string>();
      for (const row of rows) {
        if (Number(row.manual_id) === manualId) continue;
        try {
          survivingReferences.add((await resolveManualFile(row.file_path)).absolutePath);
        } catch {
          // An unrelated malformed historic reference must not make a managed
          // target deletable. It simply cannot prove ownership of this target.
        }
      }
      stagedPaths = await stageManagedPathsForDeletion(
        survivingReferences.has(targetPath) ? [] : [targetPath],
        manualStorageRoot(),
        {
          databaseName: env.DB_DATABASE,
          kind: "manual",
          entityIds: [manualId],
          filePathFingerprint: targetFingerprint,
        },
      );
    } catch (error) {
      await connection.rollback();
      logWarning("Failed to stage manual file for deletion", {
        manualId,
        error: error instanceof Error ? error.message : String(error),
      });
      return { error: "file_cleanup_failed" as const };
    }

    const [result] = await connection.query<ResultSetHeader>(
      "DELETE FROM experiment_manuals WHERE manual_id = ?",
      [manualId],
    );
    if (result.affectedRows === 0) {
      await connection.rollback();
      await restoreStagedManagedPaths(stagedPaths);
      return { error: "not_found" as const };
    }

    transactionPhase = "commit-attempted";
    await connection.commit();
    transactionPhase = "committed";
    connection.release();
    released = true;

    // Keep the named lock until the durable purge has completed so an
    // interval reconciler cannot act on this operation mid-cleanup.
    let failedCleanupPaths: string[];
    try {
      failedCleanupPaths = await purgeStagedManagedPaths(stagedPaths);
    } catch (error) {
      failedCleanupPaths = stagedPaths.map((entry) => entry.stagedPath);
      logWarning("Manual deletion purge raised after the row commit", {
        manualId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    if (failedCleanupPaths.length > 0) {
      logWarning("Manual deletion committed with isolated file pending cleanup", {
        manualId,
        failedCleanupPaths,
      });
    }
    return {
      success: true,
      cleanupPending: failedCleanupPaths.length > 0,
    };
  } catch (error) {
    if (transactionPhase !== "committed") {
      if (transactionPhase === "commit-attempted") {
        if (!released) {
          // A connection which threw while acknowledging COMMIT must never be
          // returned to the pool: its transaction state is unknowable.
          connection.destroy();
          released = true;
        }
        const resolution = await reconcileAmbiguousManualCommit(
          manualId,
          targetFingerprint,
          stagedPaths,
        );
        if (resolution.state === "committed") {
          return {
            success: true,
            cleanupPending: resolution.cleanupPending,
          };
        }
        throw error;
      }

      try {
        await connection.rollback();
      } catch {
        connection.destroy();
        released = true;
      }
      if (stagedPaths.length > 0) {
        await restoreStagedManagedPaths(stagedPaths).catch((restoreError) => {
          logWarning("Failed to restore staged manual file after database rollback", {
            manualId,
            error: restoreError instanceof Error ? restoreError.message : String(restoreError),
          });
        });
      }
    }
    throw error;
  } finally {
    if (!released) connection.release();
    if (!deletionLockReleased) {
      await releaseManualDeletionLock(deletionLock, { manualId });
      deletionLockReleased = true;
    }
    deletionPermit.release();
  }
}

type ManualJournalMetadata = {
  manualId: number;
  filePathFingerprint: string;
};

function parseManualJournalMetadata(journal: ManagedDeletionJournal): ManualJournalMetadata {
  if (
    journal.metadata.kind !== "manual"
    || journal.metadata.entityIds.length !== 1
    || !Number.isInteger(journal.metadata.entityIds[0])
    || Number(journal.metadata.entityIds[0]) <= 0
    || typeof journal.metadata.filePathFingerprint !== "string"
    || !/^[0-9a-f]{64}$/.test(journal.metadata.filePathFingerprint)
    || journal.entries.length !== 1
    || canonicalPathFingerprint(journal.entries[0]!.originalPath)
      !== journal.metadata.filePathFingerprint
  ) {
    throw new Error(`Invalid manual deletion journal metadata: ${journal.operationId}`);
  }
  return {
    manualId: Number(journal.metadata.entityIds[0]),
    filePathFingerprint: journal.metadata.filePathFingerprint,
  };
}

async function lockedManualRecoveryDecision(
  rows: ManualFileRow[],
  manualId: number,
  filePathFingerprint: string,
): Promise<{ action: "restore" | "purge"; ownerMatches: boolean }> {
  let ownerMatches = false;
  let anyReferenceMatches = false;
  for (const row of rows) {
    let resolvedPath: string;
    try {
      resolvedPath = (await resolveManualFile(
        row.file_path,
        Number(row.manual_id) === manualId,
      )).absolutePath;
    } catch (error) {
      if (Number(row.manual_id) === manualId) throw error;
      continue;
    }
    const matches = canonicalPathFingerprint(resolvedPath) === filePathFingerprint;
    if (matches) anyReferenceMatches = true;
    if (Number(row.manual_id) === manualId && matches) ownerMatches = true;
  }
  return {
    // The original ID with the same canonical path means DELETE did not
    // commit. A different surviving row may also legitimately share the file.
    // If the ID was reused with a different path (ABA), purge the old file.
    action: ownerMatches || anyReferenceMatches ? "restore" : "purge",
    ownerMatches,
  };
}

async function reconcileAmbiguousManualCommit(
  manualId: number,
  filePathFingerprint: string,
  stagedPaths: StagedManagedPath[],
): Promise<
  | { state: "committed"; cleanupPending: boolean }
  | { state: "not-committed" }
  | { state: "unresolved" }
> {
  const connection = await pool.getConnection().catch((error) => {
    logWarning("Ambiguous manual commit owner probe could not acquire a connection", {
      manualId,
      error: error instanceof Error ? error.message : String(error),
    });
    return null;
  });
  if (!connection) return { state: "unresolved" };

  let released = false;
  try {
    await connection.beginTransaction();
    const [rows] = await connection.query<ManualFileRow[]>(
      `SELECT manual_id, file_path
         FROM experiment_manuals
        ORDER BY manual_id ASC
        FOR UPDATE`,
    );
    const decision = await lockedManualRecoveryDecision(
      rows,
      manualId,
      filePathFingerprint,
    );
    let cleanupPending = false;
    if (stagedPaths.length > 0) {
      if (decision.action === "restore") {
        await restoreStagedManagedPaths(stagedPaths);
      } else {
        try {
          cleanupPending = (await purgeStagedManagedPaths(stagedPaths)).length > 0;
        } catch (error) {
          // The database probe already established that the deletion committed
          // (or the ID belongs to a different incarnation). Retain the journal
          // and truthfully report post-commit cleanup as pending.
          cleanupPending = true;
          logWarning("Ambiguous committed manual deletion purge remains pending", {
            manualId,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }
    }

    try {
      // This read-only transaction intentionally remains open through the
      // filesystem decision. Its FOR UPDATE locks close the ACK-loss race.
      await connection.commit();
      connection.release();
    } catch {
      connection.destroy();
    }
    released = true;
    return decision.ownerMatches
      ? { state: "not-committed" }
      : { state: "committed", cleanupPending };
  } catch (error) {
    try {
      await connection.rollback();
    } catch {
      connection.destroy();
      released = true;
    }
    logWarning("Manual deletion journal retained because ambiguous commit could not be reconciled", {
      manualId,
      error: error instanceof Error ? error.message : String(error),
    });
    return { state: "unresolved" };
  } finally {
    if (!released) connection.release();
  }
}

export async function reconcileManualDeletionJournals() {
  const storageRoot = await ensureDatabaseManagedStorageRoot(manualsRoot, env);
  const deletionPermit = await acquireManagedDeletionPermit();
  if (!deletionPermit) {
    // Startup must be able to distinguish a real empty scan from a scan that
    // never acquired the shared deletion admission slot.
    throw new Error("Timed out waiting to reconcile manual deletion journals");
  }
  try {
    const result = await reconcileStagedDeletionJournals(
      storageRoot,
      async (journal) => {
        if (journal.metadata.databaseName !== env.DB_DATABASE) return "skip";
        const metadata = parseManualJournalMetadata(journal);
        const deletionLock = await tryAcquireManagedDeletionLocks(
          pool,
          manualDeletionLockIdentity(metadata.manualId),
          0,
        );
        if (!deletionLock) return "skip";

        const connection = await pool.getConnection().catch(async (error) => {
          await releaseManualDeletionLock(deletionLock, {
            operationId: journal.operationId,
            manualId: metadata.manualId,
          });
          throw error;
        });
        let released = false;
        try {
          await connection.beginTransaction();
          const [rows] = await connection.query<ManualFileRow[]>(
            `SELECT manual_id, file_path
               FROM experiment_manuals
              ORDER BY manual_id ASC
              FOR UPDATE`,
          );
          const decision = await lockedManualRecoveryDecision(
            rows,
            metadata.manualId,
            metadata.filePathFingerprint,
          );
          return {
            action: decision.action,
            // reconcileStagedDeletionJournals invokes this only after the
            // restore/purge attempt, keeping the row lock for the full action.
            release: async () => {
              if (released) return;
              released = true;
              try {
                await connection.commit();
                connection.release();
              } catch {
                connection.destroy();
              } finally {
                await releaseManualDeletionLock(deletionLock, {
                  operationId: journal.operationId,
                  manualId: metadata.manualId,
                });
              }
            },
          };
        } catch (error) {
          try {
            await connection.rollback();
          } catch {
            connection.destroy();
            released = true;
          }
          if (!released) connection.release();
          released = true;
          await releaseManualDeletionLock(deletionLock, {
            operationId: journal.operationId,
            manualId: metadata.manualId,
          });
          throw error;
        }
      },
    );
    if (result.failedJournals > 0) {
      logWarning("Manual deletion journal reconciliation left failures", {
        failedJournals: result.failedJournals,
        failedJournalNames: result.failedJournalNames,
      });
    }
    return result;
  } finally {
    deletionPermit.release();
  }
}

export { FileValidationError };
