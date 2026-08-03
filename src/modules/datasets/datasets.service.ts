import path from "path";
import { createHash, randomUUID } from "node:crypto";
import { unlink, mkdir, rm, rmdir } from "node:fs/promises";
import type { RowDataPacket, ResultSetHeader } from "mysql2/promise";
import { env, pool } from "../../config";
import {
  CsvEncodingError,
  CsvRowLimitError,
  parseCsvSegments,
  getCsvHeaders,
  readCsvRows,
  type SegmentData,
} from "../../utils/csv-processor";
import {
  FileValidationError,
  fitFilenameToStoragePathLimits,
  generateUniqueFilename,
  sanitizeFilename,
  validateFile,
} from "../../utils/file-validator";
import { logWarning } from "../../utils/error-handler";
import { CSV_LIMITS, FILE_SIZE_LIMITS } from "../../utils/constants";
import {
  classifyManagedDeletionOwnership,
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
  tryAcquireManagedDeletionPermit,
} from "../../utils/managed-deletion-admission";
import {
  ensureDatabaseManagedStorageRoot,
  getDatabaseManagedStorageRoot,
  resolveStoredManagedPath,
} from "../../utils/managed-storage";
import { resolveExistingManagedFile } from "../../utils/managed-file";
import { databaseDatetimeToIsoString } from "../../utils/database-datetime";
import type { UpdateDatasetInput } from "./datasets.schema";
import { normalizeSalesValue, normalizeYearMonth } from "./dataset-values";

interface DatasetRow extends RowDataPacket {
  dataset_id: number;
  data_name: string;
  file_path: string;
  description: string | null;
  uploaded_at: Date | string;
}

interface SegmentRow extends RowDataPacket {
  dataset_id?: number;
  industry_name?: string;
  company_name?: string;
  product_name?: string;
  file_path?: string;
}

interface SegmentFileRow extends RowDataPacket {
  segment_id: number;
  dataset_id: number;
  file_path: string;
}

interface DatasetFileReferenceRow extends RowDataPacket {
  dataset_id: number;
  file_path: string;
}

const uploadsRoot = path.resolve(import.meta.dir, "../../../uploads");
const datasetsRoot = path.resolve(uploadsRoot, "datasets");
const tempUploadsRoot = path.resolve(uploadsRoot, "temp");
const DATASET_DELETION_LOCK_TIMEOUT_SECONDS = 10;

const datasetStorageRoot = () => getDatabaseManagedStorageRoot(datasetsRoot, env);

async function removeEmptyDatasetDirectories(
  entries: readonly Pick<StagedManagedPath, "originalPath">[],
) {
  const storageRoot = path.resolve(datasetStorageRoot());
  const candidateDirectories = new Set<string>();
  for (const entry of entries) {
    let candidate = path.dirname(path.resolve(entry.originalPath));
    while (candidate !== storageRoot && path.dirname(candidate) !== candidate) {
      const relative = path.relative(storageRoot, candidate);
      if (
        relative === ".."
        || relative.startsWith(`..${path.sep}`)
        || path.isAbsolute(relative)
      ) break;
      candidateDirectories.add(candidate);
      candidate = path.dirname(candidate);
    }
  }
  // Deepest first: remove the exact generation's segments directory, then its
  // now-empty generation directory. Never infer a path from a reusable ID.
  for (const directory of [...candidateDirectories].sort((a, b) => b.length - a.length)) {
    await rmdir(directory).catch(() => {});
  }
}

const datasetDeletionLockIdentity = (datasetId: number) => ({
  databaseName: env.DB_DATABASE,
  kind: "dataset",
  entityIds: [datasetId],
  // Every deletion locks all other dataset/segment references. A stable lock
  // prevents cross-process delete(A)/delete(B) from taking those rows in the
  // opposite order while retaining the entity lock for stale-request safety.
  includeGlobalLock: true,
});

async function releaseDatasetDeletionLock(
  lease: ManagedDeletionLockLease,
  context: Record<string, unknown>,
) {
  await lease.release().catch((error) => {
    logWarning("Failed to release dataset deletion advisory lock", {
      ...context,
      error: error instanceof Error ? error.message : String(error),
    });
  });
}

async function purgeDeletedDatasetFiles(
  datasetId: number,
  stagedPaths: StagedManagedPath[],
): Promise<boolean> {
  try {
    const failedCleanupPaths = await purgeStagedManagedPaths(stagedPaths);
    if (failedCleanupPaths.length > 0) {
      logWarning("Dataset deletion committed with isolated files pending cleanup", {
        datasetId,
        failedCleanupPaths,
      });
    }
    return failedCleanupPaths.length > 0;
  } catch (error) {
    // The database outcome is already final. Preserve the durable journal for
    // reconciliation and never turn a committed deletion into a retryable API
    // failure merely because post-commit filesystem cleanup raised.
    logWarning("Dataset deletion purge raised after the row commit", {
      datasetId,
      error: error instanceof Error ? error.message : String(error),
    });
    return stagedPaths.length > 0;
  }
}

async function reconcileAmbiguousDatasetCommit(
  datasetId: number,
  stagedPaths: StagedManagedPath[],
): Promise<
  | { state: "committed"; cleanupPending: boolean }
  | { state: "not-committed" }
  | { state: "unresolved" }
> {
  if (stagedPaths.length === 0) {
    // There is no filesystem evidence to reconcile, but the owner row still
    // resolves the commit acknowledgement ambiguity.
    let owner: Awaited<ReturnType<typeof probeDatasetOwnerWithLock>> | null = null;
    try {
      owner = await probeDatasetOwnerWithLock(datasetId);
      return !owner.exists
        ? { state: "committed", cleanupPending: false }
        : { state: "not-committed" };
    } catch (error) {
      logWarning("Ambiguous dataset commit owner probe failed", {
        datasetId,
        error: error instanceof Error ? error.message : String(error),
      });
      return { state: "unresolved" };
    } finally {
      await owner?.release().catch(() => {});
    }
  }
  let owner: Awaited<ReturnType<typeof probeDatasetOwnerWithLock>> | null = null;
  try {
    owner = await probeDatasetOwnerWithLock(datasetId);
    if (owner.exists) {
      const ownershipMatch = classifyManagedDeletionOwnership(
        stagedPaths,
        owner.referencedPaths,
      );
      if (ownershipMatch === "partial") {
        logWarning("Dataset deletion journal retained after partial owner path match", {
          datasetId,
        });
        return { state: "unresolved" };
      }
      if (ownershipMatch === "none") {
        const cleanupPending = await purgeDeletedDatasetFiles(datasetId, stagedPaths);
        return {
          state: "committed",
          cleanupPending,
        };
      }
      await restoreStagedManagedPaths(stagedPaths);
      return { state: "not-committed" };
    }
    const cleanupPending = await purgeDeletedDatasetFiles(datasetId, stagedPaths);
    return {
      state: "committed",
      cleanupPending,
    };
  } catch (error) {
    // Never guess after an ambiguous commit. Keeping both tombstone and journal
    // lets startup/interval reconciliation make the decision once DB access is
    // healthy again.
    logWarning("Dataset deletion journal retained because ambiguous commit could not be reconciled", {
      datasetId,
      error: error instanceof Error ? error.message : String(error),
    });
    return { state: "unresolved" };
  } finally {
    await owner?.release().catch(() => {});
  }
}

/**
 * A commit acknowledgement failure makes the original transaction unusable.
 * Probe ownership on a fresh transaction and take the same row lock used by
 * deletion before making any restore/purge decision.
 */
async function probeDatasetOwnerWithLock(datasetId: number): Promise<{
  exists: boolean;
  referencedPaths: Set<string>;
  release: () => Promise<void>;
}> {
  const connection = await pool.getConnection();
  let handedOff = false;
  let destroyed = false;
  try {
    await connection.beginTransaction();
    const [rows] = await connection.query<DatasetFileReferenceRow[]>(
      "SELECT dataset_id, file_path FROM datasets WHERE dataset_id = ? LIMIT 1 FOR UPDATE",
      [datasetId],
    );
    let segmentRows: SegmentFileRow[] = [];
    if (rows.length > 0) {
      const [foundSegmentRows] = await connection.query<SegmentFileRow[]>(
        `SELECT segment_id, dataset_id, file_path
           FROM dataset_segments
          WHERE dataset_id = ?
          ORDER BY segment_id ASC
          FOR UPDATE`,
        [datasetId],
      );
      segmentRows = foundSegmentRows;
    }
    const referencedPaths = rows.length > 0
      ? await resolveDatasetOwnerFileReferences(rows, segmentRows)
      : new Set<string>();
    let released = false;
    handedOff = true;
    return {
      exists: rows.length > 0,
      referencedPaths,
      release: async () => {
        if (released) return;
        released = true;
        try {
          await connection.commit();
          connection.release();
        } catch (error) {
          connection.destroy();
          throw error;
        }
      },
    };
  } catch (error) {
    try {
      await connection.rollback();
    } catch {
      connection.destroy();
      destroyed = true;
    }
    throw error;
  } finally {
    if (!handedOff && !destroyed) connection.release();
  }
}

async function resolveDatasetFileForDeletion(storedPath: string) {
  return resolveStoredManagedPath({
    storedPath,
    managedRoot: datasetsRoot,
    environment: env,
    legacyFallbackDirectory: datasetStorageRoot(),
    enforceDisposableNamespace: true,
  });
}

async function resolveSegmentFileForDeletion(datasetId: number, storedPath: string) {
  return resolveStoredManagedPath({
    storedPath,
    managedRoot: datasetsRoot,
    environment: env,
    legacyFallbackDirectory: path.join(datasetStorageRoot(), String(datasetId), "segments"),
    enforceDisposableNamespace: true,
  });
}

async function collectSurvivingDatasetFileReferences(
  datasetRows: DatasetFileReferenceRow[],
  segmentRows: SegmentFileRow[],
): Promise<Set<string>> {
  const references = new Set<string>();
  for (const row of datasetRows) {
    try {
      const resolved = await resolveStoredManagedPath({
        storedPath: row.file_path,
        managedRoot: datasetsRoot,
        environment: env,
        legacyFallbackDirectory: datasetStorageRoot(),
      });
      references.add(resolved.absolutePath);
    } catch {}
  }
  for (const row of segmentRows) {
    try {
      const resolved = await resolveStoredManagedPath({
        storedPath: row.file_path,
        managedRoot: datasetsRoot,
        environment: env,
        legacyFallbackDirectory: path.join(datasetStorageRoot(), String(row.dataset_id), "segments"),
      });
      references.add(resolved.absolutePath);
    } catch {}
  }
  return references;
}

async function resolveDatasetOwnerFileReferences(
  datasetRows: DatasetFileReferenceRow[],
  segmentRows: SegmentFileRow[],
): Promise<Set<string>> {
  const references = new Set<string>();
  for (const row of datasetRows) {
    references.add((await resolveStoredManagedPath({
      storedPath: row.file_path,
      managedRoot: datasetsRoot,
      environment: env,
      legacyFallbackDirectory: datasetStorageRoot(),
    })).absolutePath);
  }
  for (const row of segmentRows) {
    references.add((await resolveStoredManagedPath({
      storedPath: row.file_path,
      managedRoot: datasetsRoot,
      environment: env,
      legacyFallbackDirectory: path.join(datasetStorageRoot(), String(row.dataset_id), "segments"),
    })).absolutePath);
  }
  return references;
}

function toPublicFilePath(absolutePath: string | null | undefined): string | null {
  if (!absolutePath) return null;
  // Physical database namespaces must not leak into or change the stable
  // authenticated download URL.
  return `datasets/${path.basename(absolutePath)}`.replace(/\\/g, "/");
}

function normalizeDatasetRow(row: DatasetRow) {
  return {
    ...row,
    file_path: toPublicFilePath(row.file_path),
    uploaded_at: databaseDatetimeToIsoString(row.uploaded_at, env.DB_TIMEZONE),
  };
}

function isDuplicateDatasetSegmentError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const maybeError = error as { code?: string; errno?: number; message?: string };
  return (
    (maybeError.code === "ER_DUP_ENTRY" || maybeError.errno === 1062) &&
    (maybeError.message?.includes("uniq_dataset_segments_product") ?? false)
  );
}

function escapeCsvCell(value: unknown): string {
  const text = String(value ?? "");
  const escaped = text.replace(/"/g, '""');
  return /[",\r\n]/.test(text) ? `"${escaped}"` : escaped;
}

export function toCsvString(data: Record<string, string>[]): string {
  if (data.length === 0) return "";
  const headers = Object.keys(data[0] ?? {});
  const headerRow = headers.map(escapeCsvCell).join(",");
  const bodyRows = data.map((row) => headers.map((header) => escapeCsvCell(row[header])).join(","));
  return [headerRow, ...bodyRows].join("\n");
}

export function buildDatasetSegmentFilename(input: {
  industryName: string;
  companyName: string;
  productName: string;
}): string {
  const fingerprint = createHash("sha256")
    .update(JSON.stringify([
      input.industryName,
      input.companyName,
      input.productName,
    ]))
    .digest("hex")
    .slice(0, 16);
  // Keep the human-readable part deliberately ASCII and short. Linux limits
  // one filename component by bytes (normally 255), while String#slice counts
  // UTF-16 code units; a suffix of 80 CJK characters/emoji can therefore
  // exceed NAME_MAX even though it appears bounded in JavaScript.
  const readableSuffix = `${input.companyName}-${input.productName}`
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^[._-]+|[._-]+$/g, "")
    .slice(0, 40)
    .replace(/[._-]+$/g, "") || "segment";
  return `${fingerprint}-${readableSuffix}.csv`;
}

export function buildDatasetGenerationPaths(
  storageRoot: string,
  datasetId: number,
  originalFilename: string,
) {
  const generationToken = randomUUID();
  const generationName = `${datasetId}-${generationToken}`;
  const originalPrefix = `${generationName}-`;
  const storedFilename = fitFilenameToStoragePathLimits(
    sanitizeFilename(originalFilename),
    { directory: storageRoot, fixedPrefix: originalPrefix },
  );
  const datasetDirectory = path.join(storageRoot, generationName);
  return {
    generationToken,
    datasetDirectory,
    segmentsDirectory: path.join(datasetDirectory, "segments"),
    originalPath: path.join(storageRoot, storedFilename),
  };
}

export function getDatasetStructureLimitError(
  segmentsMap: ReadonlyMap<string, SegmentData>,
): string | null {
  if (segmentsMap.size > CSV_LIMITS.DATASET_UPLOAD_MAX_SEGMENTS) {
    return `数据集行业/公司/产品组合数量不能超过 ${CSV_LIMITS.DATASET_UPLOAD_MAX_SEGMENTS}`;
  }

  for (const segment of segmentsMap.values()) {
    if (segment.rows.length > CSV_LIMITS.DATASET_SEGMENT_MAX_ROWS) {
      const identity = [segment.industryName, segment.companyName, segment.productName].join("/");
      return `产品分段 "${identity}" 的数据行数不能超过 ${CSV_LIMITS.DATASET_SEGMENT_MAX_ROWS}`;
    }
  }

  return null;
}

type DatasetUploadCommitProbe =
  | { state: "committed"; row: DatasetRow }
  | { state: "not-committed" }
  | { state: "unknown" };

async function probeDatasetUploadCommit(
  datasetId: number,
  expectedOriginalPath: string,
  expectedDatasetDirectory: string,
  expectedSegmentPaths: readonly string[],
): Promise<DatasetUploadCommitProbe> {
  let probeConnection: Awaited<ReturnType<typeof pool.getConnection>> | null = null;
  let transactionStarted = false;
  let connectionDestroyed = false;

  const finishReadOnlyProbe = async () => {
    if (!probeConnection || !transactionStarted) return;
    try {
      await probeConnection.rollback();
    } catch {
      // The observation was made while the row/gap lock was held. A lost ACK
      // cannot invalidate that read, but the uncertain session must never be
      // returned to the pool.
      probeConnection.destroy();
      connectionDestroyed = true;
    } finally {
      transactionStarted = false;
    }
  };
  try {
    probeConnection = await pool.getConnection();
    await probeConnection.beginTransaction();
    transactionStarted = true;
    const [datasetRows] = await probeConnection.query<DatasetRow[]>(
      "SELECT * FROM datasets WHERE dataset_id = ? LIMIT 1 FOR UPDATE",
      [datasetId],
    );
    const datasetRow = datasetRows[0];
    if (!datasetRow) {
      // Keep the missing-row/gap lock through orphan cleanup. Releasing it
      // first would permit an explicit-ID/path ABA insert whose newly owned
      // files the original request could then erase.
      await rm(expectedDatasetDirectory, { recursive: true, force: true });
      await rm(expectedOriginalPath, { force: true });
      await finishReadOnlyProbe();
      return { state: "not-committed" };
    }

    const [segmentRows] = await probeConnection.query<SegmentFileRow[]>(
      `SELECT segment_id, dataset_id, file_path
         FROM dataset_segments
        WHERE dataset_id = ?
        ORDER BY segment_id ASC
        FOR UPDATE`,
      [datasetId],
    );
    const resolvedOriginal = await resolveExistingManagedFile({
      storedPath: datasetRow.file_path,
      managedRoot: datasetsRoot,
      environment: env,
      legacyFallbackDirectory: datasetStorageRoot(),
      enforceDisposableNamespace: true,
    });
    const resolvedSegments = await Promise.all(segmentRows.map((segment) =>
      resolveExistingManagedFile({
        storedPath: segment.file_path,
        managedRoot: datasetsRoot,
        environment: env,
        legacyFallbackDirectory: path.join(
          datasetStorageRoot(),
          String(datasetId),
          "segments",
        ),
        enforceDisposableNamespace: true,
      })
    ));
    const actualSegmentPaths = resolvedSegments
      .map((resolved) => resolved?.absolutePath)
      .filter((value): value is string => Boolean(value))
      .map((value) => path.resolve(value))
      .sort();
    const expectedCanonicalSegments = expectedSegmentPaths
      .map((value) => path.resolve(value))
      .sort();
    const referencesMatch = (
      resolvedOriginal?.absolutePath === path.resolve(expectedOriginalPath)
      && actualSegmentPaths.length === expectedCanonicalSegments.length
      && actualSegmentPaths.every((value, index) => value === expectedCanonicalSegments[index])
    );
    await finishReadOnlyProbe();
    return referencesMatch
      ? { state: "committed", row: datasetRow }
      : { state: "unknown" };
  } catch {
    if (transactionStarted && probeConnection) {
      try {
        await probeConnection.rollback();
      } catch {
        probeConnection.destroy();
        connectionDestroyed = true;
      } finally {
        transactionStarted = false;
      }
    }
    return { state: "unknown" };
  } finally {
    if (probeConnection && !connectionDestroyed) probeConnection.release();
  }
}

export async function uploadDataset(file: File, dataName: string, description?: string) {
  let tempUploadPath: string | null = null;
  let datasetDirectory: string | null = null;
  let permanentOriginalPath: string | null = null;
  const permanentSegmentPaths: string[] = [];

  try {
    await validateFile(file, "csv", { maxSize: FILE_SIZE_LIMITS.DATASET_CSV_MAX_SIZE, checkMagicBytes: false });
  } catch (error) {
    if (error instanceof FileValidationError) return { error: error.message, status: 400 as const };
    throw error;
  }

  const tempDir = await ensureDatabaseManagedStorageRoot(tempUploadsRoot, env);
  const uniqueFilename = generateUniqueFilename(file.name);
  tempUploadPath = path.join(tempDir, uniqueFilename);
  await Bun.write(tempUploadPath, file);

  try {
    const requiredHeaders = ["年份", "月份", "销售数量", "行业名称", "公司名称", "产品名称", "数量单位"];
    const headers = await getCsvHeaders(tempUploadPath);
    const missingHeaders = requiredHeaders.filter((h) => !headers.includes(h));
    if (missingHeaders.length > 0) {
      return { error: "文件格式错误", status: 400 as const, details: { missing_headers: missingHeaders } };
    }

    const segmentsMap = await parseCsvSegments(tempUploadPath);
    const structureLimitError = getDatasetStructureLimitError(segmentsMap);
    if (structureLimitError) {
      return {
        error: structureLimitError,
        status: 400 as const,
      };
    }

    for (const [, segment] of segmentsMap.entries()) {
      const timePoints = new Set<string>();
      let previousTime = "";
      const firstRow = segment.rows[0];
      if (!firstRow) return { error: "数据格式错误", status: 400 as const, details: { message: `产品 "${segment.productName}" 无有效数据` } };

      const expectedUnit = (firstRow["数量单位"] ?? "").trim();

      for (const [i, row] of segment.rows.entries()) {
        const yearRaw = row["年份"]?.trim();
        const monthRaw = row["月份"]?.trim();

        if (!yearRaw || !/^\d{4}$/.test(yearRaw)) {
          return { error: "数据格式错误", status: 400 as const, details: { message: `产品 "${segment.productName}" 第 ${i + 1} 行年份格式错误` } };
        }
        if (!monthRaw || !/^(0?[1-9]|1[0-2])$/.test(monthRaw)) {
          return { error: "数据格式错误", status: 400 as const, details: { message: `产品 "${segment.productName}" 第 ${i + 1} 行月份格式错误` } };
        }

        const normalizedTime = normalizeYearMonth(yearRaw, monthRaw)!;
        const currentTime = `${normalizedTime.year}-${normalizedTime.month}`;

        if (timePoints.has(currentTime)) {
          return { error: "数据重复", status: 400 as const, details: { message: `产品 "${segment.productName}" 存在重复的时间点: ${currentTime}` } };
        }
        timePoints.add(currentTime);

        if (previousTime && currentTime < previousTime) {
          return { error: "时间顺序错误", status: 400 as const, details: { message: `产品 "${segment.productName}" 时间顺序颠倒` } };
        }
        previousTime = currentTime;

        const sales = normalizeSalesValue(row["销售数量"]);
        if (!sales) {
          return { error: "数据格式错误", status: 400 as const, details: { message: `产品 "${segment.productName}" 第 ${i + 1} 行销售数量无效` } };
        }

        const unitRaw = (row["数量单位"] ?? "").trim();
        if (unitRaw !== expectedUnit) {
          return { error: "数据不一致", status: 400 as const, details: { message: `产品 "${segment.productName}" 第 ${i + 1} 行数量单位不一致` } };
        }

        // Segment files are the canonical model input. Keep the original upload
        // untouched for audit, but never persist presentation formatting here.
        row["年份"] = normalizedTime.year;
        row["月份"] = normalizedTime.month;
        row["销售数量"] = sales.canonical;
        row["数量单位"] = unitRaw;
      }
    }

    const connection = await pool.getConnection();
    let transactionPhase: "active" | "commit-attempted" | "committed" = "active";
    let connectionDestroyed = false;
    let insertedDatasetId: number | null = null;
    try {
      await connection.beginTransaction();

      if (segmentsMap.size > 0) {
        const duplicateConditions = Array.from({ length: segmentsMap.size }).map(() => "(industry_name = ? AND company_name = ? AND product_name = ?)").join(" OR ");
        const duplicateValues = Array.from(segmentsMap.values()).flatMap((seg) => [seg.industryName, seg.companyName, seg.productName]);
        const [existing] = await connection.query<SegmentRow[]>(`SELECT industry_name, company_name, product_name FROM dataset_segments WHERE ${duplicateConditions}`, duplicateValues);

        if (existing.length > 0) {
          const preview = existing.slice(0, 5).map((row) => `${row.industry_name}/${row.company_name}/${row.product_name}`).join(", ");
          await connection.rollback();
          return { error: `数据库已经存在相同产品的销售数据: ${preview}${existing.length > 5 ? "..." : ""}`, status: 409 as const };
        }
      }

      const [result] = await connection.query<ResultSetHeader>("INSERT INTO datasets (data_name, file_path, description) VALUES (?, ?, ?)", [dataName, "placeholder", description || null]);
      const datasetId = result.insertId;
      insertedDatasetId = datasetId;

      const permanentStorageDir = getDatabaseManagedStorageRoot(
        datasetsRoot,
        env,
      );
      await ensureDatabaseManagedStorageRoot(datasetsRoot, env);
      const generationPaths = buildDatasetGenerationPaths(
        permanentStorageDir,
        datasetId,
        file.name,
      );
      permanentOriginalPath = generationPaths.originalPath;
      datasetDirectory = generationPaths.datasetDirectory;
      const segmentsDir = generationPaths.segmentsDirectory;
      await mkdir(segmentsDir, { recursive: true });

      await Bun.write(permanentOriginalPath, Bun.file(tempUploadPath));
      await connection.query("UPDATE datasets SET file_path = ? WHERE dataset_id = ?", [permanentOriginalPath, datasetId]);

      if (segmentsMap.size > 0) {
        const segmentValues: unknown[] = [];
        for (const segment of segmentsMap.values()) {
          const segmentFileName = buildDatasetSegmentFilename(segment);
          const segmentFilePath = path.join(segmentsDir, segmentFileName);
          await Bun.write(segmentFilePath, toCsvString(segment.rows));
          permanentSegmentPaths.push(segmentFilePath);
          segmentValues.push(datasetId, segment.industryName, segment.companyName, segment.productName, segmentFilePath);
        }

        const segmentPlaceholders = Array.from({ length: segmentsMap.size }).map(() => "(?, ?, ?, ?, ?)").join(", ");
        await connection.query(`INSERT INTO dataset_segments (dataset_id, industry_name, company_name, product_name, file_path) VALUES ${segmentPlaceholders}`, segmentValues);
      }

      const [rows] = await connection.query<DatasetRow[]>(
        "SELECT * FROM datasets WHERE dataset_id = ?",
        [datasetId],
      );
      const insertedRow = rows[0]!;
      transactionPhase = "commit-attempted";
      await connection.commit();
      transactionPhase = "committed";
      return { data: normalizeDatasetRow(insertedRow) };
    } catch (dbError) {
      if (transactionPhase === "commit-attempted") {
        connection.destroy();
        connectionDestroyed = true;
        if (insertedDatasetId !== null && permanentOriginalPath && datasetDirectory) {
          const probe = await probeDatasetUploadCommit(
            insertedDatasetId,
            permanentOriginalPath,
            datasetDirectory,
            permanentSegmentPaths,
          );
          if (probe.state === "committed") {
            return { data: normalizeDatasetRow(probe.row) };
          }
          // Unknown probes deliberately retain all generated files. A later
          // operator/database recovery can reconcile them without creating a
          // committed row whose canonical files were already destroyed.
        }
        throw dbError;
      }

      if (transactionPhase === "active") {
        try {
          await connection.rollback();
        } catch {
          connection.destroy();
          connectionDestroyed = true;
        }
        if (datasetDirectory) await rm(datasetDirectory, { recursive: true, force: true }).catch(() => {});
        if (permanentOriginalPath) await rm(permanentOriginalPath, { force: true }).catch(() => {});
      }
      if (isDuplicateDatasetSegmentError(dbError)) {
        return { error: "数据库已经存在相同产品的销售数据", status: 409 as const };
      }
      throw dbError;
    } finally {
      if (!connectionDestroyed) connection.release();
    }
  } catch (error) {
    if (error instanceof CsvEncodingError) {
      return {
        error: "文件格式错误",
        status: 400 as const,
        details: { message: error.message },
      };
    }
    if (error instanceof CsvRowLimitError) {
      return { error: error.message, status: 400 as const };
    }
    throw error;
  } finally {
    if (tempUploadPath) await unlink(tempUploadPath).catch(() => {});
  }
}

export async function getAllDatasets() {
  const [rows] = await pool.query<DatasetRow[]>("SELECT * FROM datasets ORDER BY uploaded_at DESC");
  return rows.map(normalizeDatasetRow);
}

export async function getDatasetById(datasetId: number) {
  const [rows] = await pool.query<DatasetRow[]>("SELECT * FROM datasets WHERE dataset_id = ?", [datasetId]);
  if (rows.length === 0) return null;
  return normalizeDatasetRow(rows[0]!);
}

export async function updateDataset(datasetId: number, input: UpdateDatasetInput) {
  const fields: string[] = [];
  const values: unknown[] = [];
  if (input.data_name) { fields.push("data_name = ?"); values.push(input.data_name); }
  if (input.description !== undefined) { fields.push("description = ?"); values.push(input.description); }
  if (fields.length === 0) return { error: "no_fields" as const };

  values.push(datasetId);
  const [result] = await pool.query<ResultSetHeader>(`UPDATE datasets SET ${fields.join(", ")} WHERE dataset_id = ?`, values);
  if (result.affectedRows === 0) return { error: "not_found" as const };

  const [rows] = await pool.query<DatasetRow[]>("SELECT * FROM datasets WHERE dataset_id = ?", [datasetId]);
  return { data: normalizeDatasetRow(rows[0]!) };
}

export async function deleteDataset(datasetId: number) {
  await ensureDatabaseManagedStorageRoot(datasetsRoot, env);
  const deletionPermit = await acquireManagedDeletionPermit();
  if (!deletionPermit) {
    logWarning("Timed out waiting for managed deletion admission", { datasetId });
    return { error: "file_cleanup_failed" as const };
  }

  const deletionLock = await tryAcquireManagedDeletionLocks(
    pool,
    datasetDeletionLockIdentity(datasetId),
    DATASET_DELETION_LOCK_TIMEOUT_SECONDS,
  ).catch((error) => {
    deletionPermit.release();
    throw error;
  });
  if (!deletionLock) {
    deletionPermit.release();
    logWarning("Timed out acquiring dataset deletion advisory lock", { datasetId });
    return { error: "file_cleanup_failed" as const };
  }

  const connection = await pool.getConnection().catch(async (error) => {
    await releaseDatasetDeletionLock(deletionLock, { datasetId });
    deletionPermit.release();
    throw error;
  });
  let stagedPaths: StagedManagedPath[] = [];
  let transactionPhase: "before-commit" | "commit-attempted" | "committed" = "before-commit";
  let released = false;

  try {
    await connection.beginTransaction();
    const [rows] = await connection.query<DatasetRow[]>(
      "SELECT file_path FROM datasets WHERE dataset_id = ? FOR UPDATE",
      [datasetId],
    );
    if (rows.length === 0) {
      await connection.rollback();
      return { error: "not_found" as const };
    }

    const [segmentRows] = await connection.query<SegmentFileRow[]>(
      `SELECT segment_id, dataset_id, file_path
         FROM dataset_segments
        WHERE dataset_id = ?
        ORDER BY segment_id ASC
        FOR UPDATE`,
      [datasetId],
    );
    const [otherDatasetRows] = await connection.query<DatasetFileReferenceRow[]>(
      `SELECT dataset_id, file_path
         FROM datasets
        WHERE dataset_id <> ?
        ORDER BY dataset_id ASC
        FOR UPDATE`,
      [datasetId],
    );
    const [otherSegmentRows] = await connection.query<SegmentFileRow[]>(
      `SELECT segment_id, dataset_id, file_path
         FROM dataset_segments
        WHERE dataset_id <> ?
        ORDER BY segment_id ASC
        FOR UPDATE`,
      [datasetId],
    );
    try {
      const targetPaths = new Set<string>();
      targetPaths.add((await resolveDatasetFileForDeletion(rows[0]!.file_path)).absolutePath);
      for (const segmentRow of segmentRows) {
        targetPaths.add((await resolveSegmentFileForDeletion(
          datasetId,
          segmentRow.file_path,
        )).absolutePath);
      }
      const survivingReferences = await collectSurvivingDatasetFileReferences(
        otherDatasetRows,
        otherSegmentRows,
      );
      stagedPaths = await stageManagedPathsForDeletion(
        [...targetPaths].filter((targetPath) => !survivingReferences.has(targetPath)),
        datasetStorageRoot(),
        { databaseName: env.DB_DATABASE, kind: "dataset", entityIds: [datasetId] },
      );
    } catch (error) {
      await connection.rollback();
      logWarning("Failed to stage dataset files for deletion", {
        datasetId,
        error: error instanceof Error ? error.message : String(error),
      });
      return { error: "file_cleanup_failed" as const };
    }

    const [result] = await connection.query<ResultSetHeader>(
      "DELETE FROM datasets WHERE dataset_id = ?",
      [datasetId],
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

    const cleanupPending = await purgeDeletedDatasetFiles(datasetId, stagedPaths);
    await removeEmptyDatasetDirectories(stagedPaths);
    return {
      success: true,
      cleanupPending,
    };
  } catch (error) {
    if (transactionPhase !== "committed") {
      if (transactionPhase === "commit-attempted") {
        if (!released) {
          connection.destroy();
          released = true;
        }
        // Keep the original entity advisory lock while the fresh DB probe and
        // restore/purge run. Releasing it here would let a second delete commit
        // against a missing original path before an owner-present restore.
        const resolution = await reconcileAmbiguousDatasetCommit(datasetId, stagedPaths);
        if (resolution.state === "committed") {
          await removeEmptyDatasetDirectories(stagedPaths);
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
          logWarning("Failed to restore staged dataset files after database rollback", {
            datasetId,
            error: restoreError instanceof Error ? restoreError.message : String(restoreError),
          });
        });
      }
    }
    throw error;
  } finally {
    if (!released) connection.release();
    await releaseDatasetDeletionLock(deletionLock, { datasetId });
    deletionPermit.release();
  }
}

function datasetIdFromDeletionJournal(journal: ManagedDeletionJournal): number {
  if (journal.metadata.kind !== "dataset" || !Array.isArray(journal.metadata.entityIds)) {
    throw new Error(`Unexpected dataset deletion journal metadata: ${journal.operationId}`);
  }
  const [datasetId, ...extraIds] = journal.metadata.entityIds;
  if (!Number.isInteger(datasetId) || Number(datasetId) <= 0 || extraIds.length > 0) {
    throw new Error(`Invalid dataset deletion journal entity: ${journal.operationId}`);
  }
  return Number(datasetId);
}

export async function reconcileDatasetDeletionJournals() {
  const storageRoot = await ensureDatabaseManagedStorageRoot(datasetsRoot, env);
  const deletionPermit = tryAcquireManagedDeletionPermit();
  if (!deletionPermit) throw new Error("Managed deletion reconciliation is busy");
  try {
    const result = await reconcileStagedDeletionJournals(storageRoot, async (journal) => {
      if (journal.metadata.databaseName !== env.DB_DATABASE) return "skip";
      const datasetId = datasetIdFromDeletionJournal(journal);
      const deletionLock = await tryAcquireManagedDeletionLocks(
        pool,
        datasetDeletionLockIdentity(datasetId),
        0,
      );
      if (!deletionLock) return "skip";

      let owner: Awaited<ReturnType<typeof probeDatasetOwnerWithLock>> | null = null;
      try {
        owner = await probeDatasetOwnerWithLock(datasetId);
        const ownershipMatch = classifyManagedDeletionOwnership(
          journal.entries,
          owner.referencedPaths,
        );
        if (owner.exists && ownershipMatch === "partial") {
          throw new Error(`Partial dataset deletion journal ownership: ${journal.operationId}`);
        }
        const lockedOwner = owner;
        return {
          action: owner.exists && ownershipMatch === "all"
            ? "restore" as const
            : "purge" as const,
          release: async () => {
            try {
              await lockedOwner.release();
              if (!lockedOwner.exists) await removeEmptyDatasetDirectories(journal.entries);
            } finally {
              await releaseDatasetDeletionLock(deletionLock, {
                operationId: journal.operationId,
                datasetId,
              });
            }
          },
        };
      } catch (error) {
        await owner?.release().catch(() => {});
        await releaseDatasetDeletionLock(deletionLock, {
          operationId: journal.operationId,
          datasetId,
        });
        throw error;
      }
    });
    if (result.failedJournals > 0) {
      logWarning("Dataset deletion journal reconciliation left failures", {
        failedJournals: result.failedJournals,
        failedJournalNames: result.failedJournalNames,
      });
    }
    return result;
  } finally {
    deletionPermit.release();
  }
}

export async function getIndustries() {
  const [rows] = await pool.query<SegmentRow[]>("SELECT DISTINCT industry_name FROM dataset_segments ORDER BY industry_name ASC");
  return rows.map((row) => row.industry_name);
}

export async function getCompaniesByIndustry(industry: string) {
  const [rows] = await pool.query<SegmentRow[]>("SELECT DISTINCT company_name FROM dataset_segments WHERE industry_name = ? ORDER BY company_name ASC", [industry]);
  return rows.map((row) => row.company_name);
}

export async function getProductsByCompany(industry: string, company: string) {
  const [rows] = await pool.query<SegmentRow[]>("SELECT DISTINCT product_name FROM dataset_segments WHERE industry_name = ? AND company_name = ? ORDER BY product_name ASC", [industry, company]);
  return rows.map((row) => row.product_name);
}

async function resolveSegmentFileForRead(
  datasetId: number,
  storedPath: string,
): Promise<string | null> {
  const resolved = await resolveExistingManagedFile({
    storedPath,
    managedRoot: datasetsRoot,
    environment: env,
    legacyFallbackDirectory: path.join(
      datasetStorageRoot(),
      String(datasetId),
      "segments",
    ),
    enforceDisposableNamespace: true,
  }).catch(() => null);
  return resolved?.absolutePath ?? null;
}

export async function getProductFields(industry: string, company: string, product: string) {
  const [segments] = await pool.query<SegmentRow[]>("SELECT dataset_id, file_path FROM dataset_segments WHERE industry_name = ? AND company_name = ? AND product_name = ? LIMIT 1", [industry, company, product]);
  const segment = segments[0];
  if (!segment?.file_path || !segment.dataset_id) return { error: "not_found" as const };

  const filePath = await resolveSegmentFileForRead(segment.dataset_id, segment.file_path);
  if (!filePath) return { error: "file_missing" as const };

  const fields = (await getCsvHeaders(filePath))
    .map((h) => h.trim())
    .filter((h) => h.length > 0);

  return { data: fields };
}

export async function getProductSales(industry: string, company: string, product: string) {
  const [segments] = await pool.query<SegmentRow[]>("SELECT dataset_id, file_path FROM dataset_segments WHERE industry_name = ? AND company_name = ? AND product_name = ? LIMIT 1", [industry, company, product]);
  const segment = segments[0];
  if (!segment?.file_path || !segment.dataset_id) return { error: "not_found" as const };

  const filePath = await resolveSegmentFileForRead(segment.dataset_id, segment.file_path);
  if (!filePath) return { error: "file_missing" as const };

  const monthlySales: Array<{ month: string; sales: number }> = [];
  const csvData: string[][] = [];
  let headers: string[] = [];
  let unit = "";

  const rows = await readCsvRows(filePath);
  for (const [rowIndex, row] of rows.entries()) {
    if (headers.length === 0) { headers = Object.keys(row); csvData.push(headers); }
    csvData.push(headers.map((header) => String(row[header] ?? "").trim()));

    const time = normalizeYearMonth(row["年份"], row["月份"]);
    if (!time) {
      return {
        error: "invalid_data" as const,
        details: { row: rowIndex + 1, field: "年份/月份", message: `第 ${rowIndex + 1} 行年月格式无效` },
      };
    }

    const sales = normalizeSalesValue(row["销售数量"]);
    if (!sales) {
      return {
        error: "invalid_data" as const,
        details: { row: rowIndex + 1, field: "销售数量", message: `第 ${rowIndex + 1} 行销售数量无效` },
      };
    }

    if (!unit && row["数量单位"]) unit = String(row["数量单位"]).trim();
    monthlySales.push({ month: `${time.year}-${time.month}`, sales: sales.value });
  }

  return { data: { meta: { industry, company, product, name: product, description: "", unit }, monthlySales, csvData } };
}

export { FileValidationError };
