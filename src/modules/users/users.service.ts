import { randomUUID } from "node:crypto";
import path from "path";
import { unlink } from "node:fs/promises";
import type { RowDataPacket, ResultSetHeader } from "mysql2/promise";
import { env, pool } from "../../config";
import { hashPassword, shouldChangePassword, verifyPassword } from "../../utils/password";
import { logWarning } from "../../utils/error-handler";
import { USER_CONSTANTS, FILE_SIZE_LIMITS, CSV_LIMITS } from "../../utils/constants";
import { CsvEncodingError, parseCsvFileRows } from "../../utils/csv-processor";
import {
  FILESYSTEM_FILENAME_MAX_BYTES,
  fitFilenameToUtf8Bytes,
  sanitizeFilename,
} from "../../utils/file-validator";
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
  acquireManagedDeletionPermit,
  tryAcquireManagedDeletionPermit,
} from "../../utils/managed-deletion-admission";
import {
  ensureDatabaseManagedStorageRoot,
  getDatabaseManagedStorageRoot,
  resolveStoredManagedPath,
} from "../../utils/managed-storage";
import type { CreateTeacherInput, CreateAssistantSingleInput, CreateAssistantInput, UpdateUserInput, ResetPasswordInput, UserSearchQuery, PaginationQuery } from "./users.schema";
import { studentSelfRegisterSchema, teacherAssistantSelfRegisterSchema } from "./users.schema";
import { databaseDatetimeToIsoString } from "../../utils/database-datetime";
import { isPasswordPolicyCompliant } from "../../utils/password-policy";

const { TEACHER_EMAIL_DOMAIN, ASSISTANT_EMAIL_DOMAIN, PHONE_REGEX, MIN_NAME_LENGTH } = USER_CONSTANTS;
const BATCH_CSV_MIME_TYPES = new Set([
  "text/csv",
  "application/vnd.ms-excel",
  "text/plain",
]);
const reportsRoot = path.resolve(import.meta.dir, "../../../uploads/reports");
const tempUploadsRoot = path.resolve(import.meta.dir, "../../../uploads/temp");

const reportStorageRoot = () => getDatabaseManagedStorageRoot(reportsRoot, env);

export function buildBatchImportTempFilename(
  kind: "teachers" | "assistants",
  originalName: string,
): string {
  const safeOriginalName = sanitizeFilename(originalName) || "upload.csv";
  const fixedPrefix = `${kind}-${Date.now()}-${randomUUID()}-`;
  const originalNameBudget = (
    FILESYSTEM_FILENAME_MAX_BYTES - Buffer.byteLength(fixedPrefix, "utf8")
  );
  return `${fixedPrefix}${fitFilenameToUtf8Bytes(safeOriginalName, originalNameBudget)}`;
}

async function reconcileAmbiguousUserCommit(
  userIds: number[],
  stagedPaths: StagedManagedPath[],
  cleanupSummary: DeleteCleanupSummary,
): Promise<"committed" | "not-committed" | "unresolved"> {
  let owner: Awaited<ReturnType<typeof probeUserOwnersWithLock>> | null = null;
  try {
    owner = await probeUserOwnersWithLock(userIds);
    const ownershipMatch = stagedPaths.length > 0
      ? classifyManagedDeletionOwnership(stagedPaths, owner.referencedReportPaths)
      : "none";
    if (ownershipMatch === "partial") {
      logWarning("Report deletion journal retained after partial owner path match", {
        userIds,
      });
      return "unresolved";
    }
    if (owner.ownerCount === userIds.length) {
      // With no staged file generation, the surviving owner rows are the only
      // available commit evidence. Path classification deliberately treats an
      // empty entry list as "none", so short-circuit before consulting it.
      if (stagedPaths.length === 0) return "not-committed";

      if (ownershipMatch === "all") {
        await restoreStagedManagedPaths(stagedPaths);
        return "not-committed";
      }

      // owner rows plus a completely different path generation are
      // observationally ambiguous: either DELETE did not commit and the same
      // owner advanced concurrently, or DELETE committed and the IDs were
      // reused. Purging the now-unreferenced old generation is safe in both
      // worlds, but reporting deletion success is not; that would be a false
      // success in the first world without an immutable user-generation token.
      await purgeDeletedReportFiles(stagedPaths, cleanupSummary);
      return "not-committed";
    }
    if (owner.ownerCount === 0) {
      if (ownershipMatch === "all") {
        await restoreStagedManagedPaths(stagedPaths);
      } else {
        await purgeDeletedReportFiles(stagedPaths, cleanupSummary);
      }
      return "committed";
    }
    // A single transaction cannot commit a proper subset. Never restore files
    // for already-deleted owners or purge files for owners that remain.
    logWarning("Report deletion journal retained after mixed ambiguous-commit owner probe", {
      userIds,
      ownerRows: owner.ownerCount,
    });
    return "unresolved";
  } catch (error) {
    logWarning("Report deletion journal retained because ambiguous commit could not be reconciled", {
      userIds,
      error: error instanceof Error ? error.message : String(error),
    });
    return "unresolved";
  } finally {
    await owner?.release().catch(() => {});
  }
}

async function probeUserOwnersWithLock(userIds: number[]): Promise<{
  ownerCount: number;
  referencedReportPaths: Set<string>;
  release: () => Promise<void>;
}> {
  const placeholders = userIds.map(() => "?").join(", ");
  const connection = await pool.getConnection();
  let handedOff = false;
  let destroyed = false;
  try {
    await connection.beginTransaction();
    // Every workflow that can mutate a student's report/retry state takes the
    // user row first. Doing the same in probes prevents user->report graders
    // from cycling with a report->user reconciliation session.
    const [rows] = await connection.query<RowDataPacket[]>(
      `SELECT user_id
         FROM users
        WHERE user_id IN (${placeholders})
        ORDER BY user_id ASC
        FOR UPDATE`,
      userIds,
    );
    await connection.query<RowDataPacket[]>(
      `SELECT experiment_id
         FROM experiment_status
        WHERE student_id IN (${placeholders})
        ORDER BY experiment_id ASC
        FOR UPDATE`,
      userIds,
    );
    const [reportRows] = await connection.query<ReportFileRow[]>(
      `SELECT /* user-deletion-owner-probe */ report_id, student_id, pdf_file_path
         FROM experiment_reports
        WHERE pdf_file_path IS NOT NULL
        ORDER BY report_id ASC
        FOR UPDATE`,
    );
    const referencedReportPaths = await resolveManagedReportPaths(reportRows, false);
    let released = false;
    handedOff = true;
    return {
      ownerCount: rows.length,
      referencedReportPaths: new Set(referencedReportPaths),
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

function isAllowedBatchCsvMimeType(mimeType: string): boolean {
  if (!mimeType) return true;
  const normalized = mimeType.split(";")[0]?.trim().toLowerCase() ?? "";
  return BATCH_CSV_MIME_TYPES.has(normalized) || normalized.includes("csv");
}

interface UserRow extends RowDataPacket { user_id: number; username: string; full_name: string; email: string; role: string; phone_number: string | null; password_hash: string; created_at: Date | string; }

const normalizeUserCreatedAt = <T extends Record<string, unknown>>(row: T) => ({
  ...row,
  created_at: databaseDatetimeToIsoString(row.created_at, env.DB_TIMEZONE),
});
interface ClassRow extends RowDataPacket { class_id: number; teacher_id: number; }
interface CountRow extends RowDataPacket { total: number; }
interface AssignmentRow extends RowDataPacket { class_id: number; }
interface DeleteUserRow extends RowDataPacket { user_id: number; role: string; }
interface ReportFileRow extends RowDataPacket { report_id: number; student_id: number; pdf_file_path: string | null; }

interface DeleteCleanupSummary {
  deletedClasses?: number;
  removedAssistantAssignments?: number;
  removedEnrollments?: number;
  removedReportFiles?: number;
  reportFileCleanupPending?: number;
}

interface BatchUserCandidate {
  username: string;
  email: string;
  phone: string;
}

type Queryable = {
  query<T extends RowDataPacket[] | ResultSetHeader>(sql: string, values?: unknown[]): Promise<[T, unknown]>;
};

function accumulateCleanupSummary(target: DeleteCleanupSummary, source: DeleteCleanupSummary) {
  if (source.deletedClasses) {
    target.deletedClasses = (target.deletedClasses ?? 0) + source.deletedClasses;
  }
  if (source.removedAssistantAssignments) {
    target.removedAssistantAssignments = (target.removedAssistantAssignments ?? 0) + source.removedAssistantAssignments;
  }
  if (source.removedEnrollments) {
    target.removedEnrollments = (target.removedEnrollments ?? 0) + source.removedEnrollments;
  }
  if (source.removedReportFiles) {
    target.removedReportFiles = (target.removedReportFiles ?? 0) + source.removedReportFiles;
  }
  if (source.reportFileCleanupPending) {
    target.reportFileCleanupPending = (target.reportFileCleanupPending ?? 0) + source.reportFileCleanupPending;
  }
}

async function resolveManagedReportPaths(
  rows: ReportFileRow[],
  enforceDisposableNamespace: boolean,
): Promise<string[]> {
  const managedPaths = new Set<string>();
  for (const row of rows) {
    const storedPath = row.pdf_file_path?.trim();
    if (!storedPath) continue;
    const resolved = await resolveStoredManagedPath({
      storedPath,
      managedRoot: reportsRoot,
      environment: env,
      legacyFallbackDirectory: reportStorageRoot(),
      enforceDisposableNamespace,
    });
    managedPaths.add(resolved.absolutePath);
  }
  return [...managedPaths];
}

async function resolveSurvivingReportReferences(rows: ReportFileRow[]): Promise<Set<string>> {
  const references = new Set<string>();
  for (const row of rows) {
    try {
      for (const resolvedPath of await resolveManagedReportPaths([row], false)) {
        references.add(resolvedPath);
      }
    } catch {
      // A live absolute path outside reportsRoot can never alias a managed
      // target path. It remains untouched and does not block deleting the row.
    }
  }
  return references;
}

async function lockStudentExperimentsAndGetReportRows(
  connection: Queryable,
  studentIds: number[],
): Promise<ReportFileRow[]> {
  if (studentIds.length === 0) return [];
  const placeholders = studentIds.map(() => "?").join(", ");
  // Report submission locks experiment_status before experiment_reports. Match
  // that order so a deletion waits for an in-flight submission, then observes
  // and stages the final committed pdf_file_path instead of orphaning it.
  await connection.query<RowDataPacket[]>(
    `SELECT experiment_id
       FROM experiment_status
      WHERE student_id IN (${placeholders})
      ORDER BY experiment_id ASC
      FOR UPDATE`,
    studentIds,
  );
  const [rows] = await connection.query<ReportFileRow[]>(
    `SELECT report_id, student_id, pdf_file_path
       FROM experiment_reports
      WHERE pdf_file_path IS NOT NULL
      ORDER BY report_id ASC
      FOR UPDATE`,
  );
  return rows;
}

async function purgeDeletedReportFiles(
  stagedPaths: StagedManagedPath[],
  cleanupSummary: DeleteCleanupSummary,
) {
  if (stagedPaths.length > 0) {
    cleanupSummary.removedReportFiles = stagedPaths.length;
  }
  let failedCleanupPaths: string[];
  try {
    failedCleanupPaths = await purgeStagedManagedPaths(stagedPaths);
  } catch (error) {
    // The database outcome is already final. Keep the durable journal for a
    // reconciler and report cleanup-pending instead of turning a committed
    // user deletion into a retryable API failure.
    failedCleanupPaths = stagedPaths.map((entry) => entry.stagedPath);
    logWarning("User deletion report purge raised after the row commit", {
      error: error instanceof Error ? error.message : String(error),
    });
  }
  if (failedCleanupPaths.length > 0) {
    cleanupSummary.reportFileCleanupPending = failedCleanupPaths.length;
    logWarning("User deletion committed with isolated report files pending cleanup", {
      failedCleanupPaths,
    });
  }
}

async function deleteUserWithConnection(connection: Queryable, targetUserId: number, targetRole: string) {
  const cleanupSummary: DeleteCleanupSummary = {};

  switch (targetRole) {
    case "Teacher": {
      const [classRows] = await connection.query<AssignmentRow[]>("SELECT class_id FROM classes WHERE teacher_id = ?", [targetUserId]);
      if (classRows.length > 0) {
        await connection.query("DELETE FROM classes WHERE teacher_id = ?", [targetUserId]);
        cleanupSummary.deletedClasses = classRows.length;
      }
      break;
    }
    case "Assistant": {
      const [assignmentRows] = await connection.query<AssignmentRow[]>("SELECT class_id FROM class_assistants WHERE assistant_id = ?", [targetUserId]);
      if (assignmentRows.length > 0) {
        await connection.query("DELETE FROM class_assistants WHERE assistant_id = ?", [targetUserId]);
        cleanupSummary.removedAssistantAssignments = assignmentRows.length;
      }
      break;
    }
    case "Student": {
      const [enrollmentRows] = await connection.query<AssignmentRow[]>("SELECT class_id FROM enrollments WHERE student_id = ?", [targetUserId]);
      if (enrollmentRows.length > 0) {
        await connection.query("DELETE FROM enrollments WHERE student_id = ?", [targetUserId]);
        cleanupSummary.removedEnrollments = enrollmentRows.length;
      }
      break;
    }
  }

  await connection.query("DELETE FROM users WHERE user_id = ?", [targetUserId]);
  return cleanupSummary;
}

async function findUserConflict(
  db: Queryable,
  input: { username?: string; email?: string; phone?: string | null; excludeUserId?: number },
) {
  const conditions: string[] = [];
  const params: unknown[] = [];

  if (input.username) {
    conditions.push("username = ?");
    params.push(input.username);
  }

  if (input.email) {
    conditions.push("email = ?");
    params.push(input.email);
  }

  if (input.phone) {
    conditions.push("phone_number = ?");
    params.push(input.phone);
  }

  if (conditions.length === 0) return null;

  const whereClause = conditions.length > 1 ? `(${conditions.join(" OR ")})` : conditions[0]!;
  const excludeClause = input.excludeUserId ? " AND user_id <> ?" : "";
  if (input.excludeUserId) params.push(input.excludeUserId);

  const [rows] = await db.query<RowDataPacket[]>(
    `SELECT user_id, username, email, phone_number FROM users WHERE ${whereClause}${excludeClause} LIMIT 20`,
    params,
  );

  for (const row of rows) {
    if (input.username && row.username === input.username) return { error: "dup_username" as const };
    if (input.email && row.email === input.email) return { error: "dup_email" as const };
    if (input.phone && row.phone_number === input.phone) return { error: "dup_phone" as const };
  }

  return null;
}

function mapDuplicateUserError(error: unknown) {
  const err = error as { code?: string; message?: string };
  if (err.code !== "ER_DUP_ENTRY") return null;

  const rawMessage = err.message ?? "";
  const keyMatch = rawMessage.match(/for key '([^']+)'/i);
  const normalizedKey = keyMatch?.[1]?.split(".").pop()?.toLowerCase() ?? null;

  if (normalizedKey?.includes("username") || rawMessage.includes("users.username")) {
    return { error: "dup_username" as const };
  }
  if (normalizedKey?.includes("email") || rawMessage.includes("users.email")) {
    return { error: "dup_email" as const };
  }
  if (normalizedKey?.includes("phone_number") || rawMessage.includes("users.phone_number")) {
    return { error: "dup_phone" as const };
  }

  return null;
}

function mapBatchDuplicateConflicts(error: unknown, rows: BatchUserCandidate[]) {
  const duplicateError = mapDuplicateUserError(error);
  if (!duplicateError) return null;

  const rawMessage = (error as { message?: string }).message ?? "";
  const duplicateValue = rawMessage.match(/Duplicate entry '([^']+)'/i)?.[1];
  const field = duplicateError.error === "dup_username"
    ? "username"
    : duplicateError.error === "dup_email"
      ? "email"
      : "phone";
  const matchedRows = duplicateValue
    ? rows.filter((row) => row[field] === duplicateValue)
    : [];
  const conflictRows = matchedRows.length > 0 ? matchedRows : rows;

  return conflictRows.slice(0, 10).map((row) => ({
    username: row.username,
    email: row.email,
    phone_number: row.phone,
  }));
}

export async function registerSelfUser(role: "Student" | "Teacher" | "Assistant", body: unknown) {
  const schema = role === "Student" ? studentSelfRegisterSchema : teacherAssistantSelfRegisterSchema;
  const result = schema.safeParse(body);
  if (!result.success) return { error: "validation" as const, details: result.error.issues };

  const { username, password, name, email } = result.data;
  const phone = (result.data as { phone?: string }).phone || null;

  const conflict = await findUserConflict(pool, { username, email, phone });
  if (conflict) return conflict;

  const password_hash = await hashPassword(password);

  try {
    const [insertResult] = await pool.query<ResultSetHeader>("INSERT INTO users (username, password_hash, full_name, email, phone_number, role) VALUES (?, ?, ?, ?, ?, ?)", [username, password_hash, name, email, phone, role]);
    return { data: { user_id: insertResult.insertId, username, full_name: name, email, phone_number: phone, role } };
  } catch (error: unknown) {
    const duplicateError = mapDuplicateUserError(error);
    if (duplicateError) return duplicateError;
    throw error;
  }
}

export async function createAssistantSingle(input: CreateAssistantSingleInput) {
  const { username, full_name, email, phone, password } = input;
  const assistantEmail = email ?? `${username}@${ASSISTANT_EMAIL_DOMAIN}`;

  const conflict = await findUserConflict(pool, { username, email: assistantEmail, phone: phone ?? null });
  if (conflict) return conflict;

  const password_hash = await hashPassword(password || phone || username);

  try {
    const [result] = await pool.query<ResultSetHeader>("INSERT INTO users (username, password_hash, full_name, email, phone_number, role) VALUES (?, ?, ?, ?, ?, 'Assistant')", [username, password_hash, full_name, assistantEmail, phone || null]);
    return { data: { user_id: result.insertId, username, full_name, email: assistantEmail, phone_number: phone || null, role: "Assistant" } };
  } catch (error: unknown) {
    const duplicateError = mapDuplicateUserError(error);
    if (duplicateError) return duplicateError;
    throw error;
  }
}

export async function searchUsers(query: UserSearchQuery) {
  const { q, page, limit } = query;
  const term = q.trim();
  if (!term) return { error: "empty_search" as const };

  const offset = (page - 1) * limit;
  const useFullText = term.length > 4 && /^[\w\s@.-]+$/.test(term);

  let countQuery: string;
  let dataQuery: string;
  let countParams: unknown[];
  let dataParams: unknown[];
  const escapedLikeTerm = term
    .replaceAll("=", "==")
    .replaceAll("%", "=%")
    .replaceAll("_", "=_");
  const likeTerm = `%${escapedLikeTerm}%`;

  if (useFullText) {
    const searchTerm = term.split(/\s+/).map((word) => `+${word}*`).join(" ");
    const usernamePrefix = `${escapedLikeTerm}%`;
    const indexedMatches = "SELECT user_id FROM users WHERE username LIKE ? ESCAPE '=' UNION SELECT user_id FROM users WHERE MATCH(full_name, email, phone_number) AGAINST(? IN BOOLEAN MODE)";
    countQuery = `SELECT COUNT(*) AS total FROM (${indexedMatches}) AS matched_users`;
    dataQuery = `SELECT u.user_id, u.username, u.full_name, u.email, u.role, u.created_at, MATCH(u.full_name, u.email, u.phone_number) AGAINST(? IN BOOLEAN MODE) AS relevance FROM users u JOIN (${indexedMatches}) AS matched_users ON matched_users.user_id = u.user_id ORDER BY (u.username = ?) DESC, relevance DESC, u.user_id ASC LIMIT ? OFFSET ?`;
    countParams = [usernamePrefix, searchTerm];
    dataParams = [searchTerm, usernamePrefix, searchTerm, term, limit, offset];
  } else {
    countQuery = "SELECT COUNT(*) AS total FROM users WHERE username LIKE ? ESCAPE '=' OR full_name LIKE ? ESCAPE '=' OR email LIKE ? ESCAPE '=' OR phone_number LIKE ? ESCAPE '='";
    dataQuery = "SELECT user_id, username, full_name, email, role, created_at FROM users WHERE username LIKE ? ESCAPE '=' OR full_name LIKE ? ESCAPE '=' OR email LIKE ? ESCAPE '=' OR phone_number LIKE ? ESCAPE '=' ORDER BY (username = ?) DESC, user_id ASC LIMIT ? OFFSET ?";
    countParams = [likeTerm, likeTerm, likeTerm, likeTerm];
    dataParams = [likeTerm, likeTerm, likeTerm, likeTerm, term, limit, offset];
  }

  const [[countRow]] = await pool.query<CountRow[]>(countQuery, countParams);
  const totalItems = Number(countRow?.total ?? 0);
  const totalPages = Math.ceil(totalItems / limit);

  const [dataRows] = await pool.query<RowDataPacket[]>(dataQuery, dataParams);
  const results = dataRows.map((row) => {
    const { relevance, ...rest } = row;
    return normalizeUserCreatedAt(rest);
  });

  return { data: { data: results, pagination: { currentPage: page, pageSize: limit, totalItems, totalPages }, searchMethod: useFullText ? "fulltext" : "like" } };
}

export async function createAssistantWithClasses(input: CreateAssistantInput, requesterId: number, requesterRole: string) {
  const { username, password, full_name, email, phone_number, class_ids } = input;
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();
    const placeholders = class_ids.map(() => "?").join(",");
    const [classRows] = await connection.query<ClassRow[]>(`SELECT class_id, teacher_id FROM classes WHERE class_id IN (${placeholders})`, class_ids);
    if (classRows.length !== class_ids.length) { await connection.rollback(); return { error: "invalid_classes" as const }; }

    if (requesterRole === "Teacher") {
      for (const row of classRows) {
        if (row.teacher_id !== requesterId) { await connection.rollback(); return { error: "not_class_owner" as const, classId: row.class_id }; }
      }
    }

    const conflict = await findUserConflict(connection, { username, email, phone: phone_number || null });
    if (conflict) { await connection.rollback(); return conflict; }

    const password_hash = await hashPassword(password);
    const [userResult] = await connection.query<ResultSetHeader>("INSERT INTO users (username, password_hash, full_name, email, role, phone_number) VALUES (?, ?, ?, ?, 'Assistant', ?)", [username, password_hash, full_name, email, phone_number || null]);
    const newAssistantId = userResult.insertId;
    const assignmentValues = class_ids.map((classId) => [classId, newAssistantId]);
    await connection.query("INSERT INTO class_assistants (class_id, assistant_id) VALUES ?", [assignmentValues]);
    await connection.commit();
    return { data: { user_id: newAssistantId, username, full_name, email, phone_number: phone_number || null, role: "Assistant", assigned_classes: class_ids } };
  } catch (error: unknown) {
    await connection.rollback();
    const duplicateError = mapDuplicateUserError(error);
    if (duplicateError) return duplicateError;
    throw error;
  } finally { connection.release(); }
}

export async function createTeacher(input: CreateTeacherInput) {
  const { username, full_name, email, phone, password } = input;
  const teacherEmail = email ?? `${username}@${TEACHER_EMAIL_DOMAIN}`;

  const conflict = await findUserConflict(pool, { username, email: teacherEmail, phone: phone ?? null });
  if (conflict) return conflict;

  const password_hash = await hashPassword(password || phone || username);

  try {
    const [result] = await pool.query<ResultSetHeader>("INSERT INTO users (username, password_hash, full_name, email, phone_number, role) VALUES (?, ?, ?, ?, ?, 'Teacher')", [username, password_hash, full_name, teacherEmail, phone || null]);
    return { data: { user_id: result.insertId, username, full_name, email: teacherEmail, phone_number: phone || null, role: "Teacher" } };
  } catch (error: unknown) {
    const duplicateError = mapDuplicateUserError(error);
    if (duplicateError) return duplicateError;
    throw error;
  }
}

export async function batchCreateTeachers(file: File, password: unknown) {
  let normalizedPassword: string | null = null;
  if (password !== undefined && password !== null && password !== "") {
    if (typeof password !== "string") return { error: "invalid_password_type" as const };
    const trimmed = password.trim();
    if (!isPasswordPolicyCompliant(trimmed)) return { error: "password_policy_violation" as const };
    normalizedPassword = trimmed;
  }

  if (!isAllowedBatchCsvMimeType(file.type)) return { error: "invalid_file_type" as const };
  if (file.size > FILE_SIZE_LIMITS.CSV_MAX_SIZE) return { error: "file_too_large" as const };

  const tempDir = await ensureDatabaseManagedStorageRoot(tempUploadsRoot, env);
  const tempCsvPath = path.join(
    tempDir,
    buildBatchImportTempFilename("teachers", file.name),
  );
  await Bun.write(tempCsvPath, file);

  const teacherRows: Array<{ row: number; fullName: string; phone: string; username: string; email: string }> = [];
  const invalidRows: Array<{ row: number; reason: string }> = [];
  const seenPhones = new Set<string>();
  let rowNumber = 0;

  try {
    await parseCsvFileRows(tempCsvPath, (row) => {
      rowNumber += 1;
      // 检查行数限制
      if (rowNumber > CSV_LIMITS.BATCH_IMPORT_MAX_ROWS) {
        invalidRows.push({ row: rowNumber, reason: `超过最大行数限制 ${CSV_LIMITS.BATCH_IMPORT_MAX_ROWS}` });
        return;
      }
      const fullName = row["姓名"]?.trim() ?? "";
      const phone = row["手机号"]?.trim() ?? "";
      if (!fullName || !phone) { invalidRows.push({ row: rowNumber, reason: "姓名或手机号缺失" }); return; }
      if (fullName.length < MIN_NAME_LENGTH) { invalidRows.push({ row: rowNumber, reason: `姓名长度需至少 ${MIN_NAME_LENGTH} 个字符` }); return; }
      if (!PHONE_REGEX.test(phone)) { invalidRows.push({ row: rowNumber, reason: `手机号 ${phone} 格式不正确，应为 6-15 位数字` }); return; }
      if (seenPhones.has(phone)) { invalidRows.push({ row: rowNumber, reason: `CSV 内存在重复手机号 ${phone}` }); return; }
      seenPhones.add(phone);
      teacherRows.push({ row: rowNumber, fullName, phone, username: `prof_${phone}`, email: `prof_${phone}@${TEACHER_EMAIL_DOMAIN}` });
    });

    if (invalidRows.length > 0) return { error: "invalid_rows" as const, invalid_rows: invalidRows.slice(0, 10) };
    if (teacherRows.length === 0) return { error: "no_valid_records" as const };

    const connection = await pool.getConnection();
    try {
      await connection.beginTransaction();
      const usernames = [...new Set(teacherRows.map((r) => r.username))];
      const emails = [...new Set(teacherRows.map((r) => r.email))];
      const phones = [...new Set(teacherRows.map((r) => r.phone))];

      const queries: string[] = [];
      const allParams: unknown[] = [];
      if (usernames.length > 0) { queries.push(`SELECT DISTINCT username, phone_number, email FROM users WHERE username IN (${usernames.map(() => "?").join(", ")})`); allParams.push(...usernames); }
      if (phones.length > 0) { queries.push(`SELECT DISTINCT username, phone_number, email FROM users WHERE phone_number IN (${phones.map(() => "?").join(", ")})`); allParams.push(...phones); }
      if (emails.length > 0) { queries.push(`SELECT DISTINCT username, phone_number, email FROM users WHERE email IN (${emails.map(() => "?").join(", ")})`); allParams.push(...emails); }

      if (queries.length > 0) {
        const [existing] = await connection.query<RowDataPacket[]>(queries.join(" UNION "), allParams);
        if (existing.length > 0) { await connection.rollback(); return { error: "conflicts" as const, conflicts: existing.slice(0, 10) }; }
      }

      let hashedPasswords: string[];
      if (normalizedPassword) {
        const hashed = await hashPassword(normalizedPassword);
        hashedPasswords = teacherRows.map(() => hashed);
      } else {
        hashedPasswords = await Promise.all(teacherRows.map((r) => hashPassword(r.phone)));
      }

      const teacherValues = teacherRows.map((r, i) => [r.username, hashedPasswords[i], r.fullName, r.email, "Teacher", r.phone]);
      const [result] = await connection.query<ResultSetHeader>("INSERT INTO users (username, password_hash, full_name, email, role, phone_number) VALUES ?", [teacherValues]);
      const firstInsertId = result.insertId;
      const createdTeachers = teacherRows.map((r, i) => ({ user_id: firstInsertId + i, username: r.username, full_name: r.fullName, phone_number: r.phone, email: r.email }));
      await connection.commit();
      return { data: { created: createdTeachers.length, teachers: createdTeachers } };
    } catch (error) {
      await connection.rollback();
      const conflicts = mapBatchDuplicateConflicts(error, teacherRows);
      if (conflicts) return { error: "conflicts" as const, conflicts };
      throw error;
    } finally { connection.release(); }
  } catch (error) {
    if (error instanceof CsvEncodingError) {
      return {
        error: "invalid_rows" as const,
        invalid_rows: [{ row: 0, reason: error.message }],
      };
    }
    throw error;
  } finally {
    await unlink(tempCsvPath).catch((err) => logWarning("Failed to cleanup temp teacher CSV", { error: err instanceof Error ? err.message : String(err) }));
  }
}

export async function batchCreateAssistants(file: File, password: unknown) {
  let normalizedPassword: string | null = null;
  if (password !== undefined && password !== null && password !== "") {
    if (typeof password !== "string") return { error: "invalid_password_type" as const };
    const trimmed = password.trim();
    if (!isPasswordPolicyCompliant(trimmed)) return { error: "password_policy_violation" as const };
    normalizedPassword = trimmed;
  }

  if (!isAllowedBatchCsvMimeType(file.type)) return { error: "invalid_file_type" as const };
  if (file.size > FILE_SIZE_LIMITS.CSV_MAX_SIZE) return { error: "file_too_large" as const };

  const tempDir = await ensureDatabaseManagedStorageRoot(tempUploadsRoot, env);
  const tempCsvPath = path.join(
    tempDir,
    buildBatchImportTempFilename("assistants", file.name),
  );
  await Bun.write(tempCsvPath, file);

  const assistantRows: Array<{ row: number; fullName: string; phone: string; username: string; email: string }> = [];
  const invalidRows: Array<{ row: number; reason: string }> = [];
  const seenPhones = new Set<string>();
  let rowNumber = 0;

  try {
    await parseCsvFileRows(tempCsvPath, (row) => {
      rowNumber += 1;
      // 检查行数限制
      if (rowNumber > CSV_LIMITS.BATCH_IMPORT_MAX_ROWS) {
        invalidRows.push({ row: rowNumber, reason: `超过最大行数限制 ${CSV_LIMITS.BATCH_IMPORT_MAX_ROWS}` });
        return;
      }
      const fullName = row["姓名"]?.trim() ?? "";
      const phone = row["手机号"]?.trim() ?? "";
      if (!fullName || !phone) { invalidRows.push({ row: rowNumber, reason: "姓名或手机号缺失" }); return; }
      if (fullName.length < MIN_NAME_LENGTH) { invalidRows.push({ row: rowNumber, reason: `姓名长度需至少 ${MIN_NAME_LENGTH} 个字符` }); return; }
      if (!PHONE_REGEX.test(phone)) { invalidRows.push({ row: rowNumber, reason: `手机号 ${phone} 格式不正确，应为 6-15 位数字` }); return; }
      if (seenPhones.has(phone)) { invalidRows.push({ row: rowNumber, reason: `CSV 内存在重复手机号 ${phone}` }); return; }
      seenPhones.add(phone);
      assistantRows.push({ row: rowNumber, fullName, phone, username: `ta_${phone}`, email: `ta_${phone}@${ASSISTANT_EMAIL_DOMAIN}` });
    });

    if (invalidRows.length > 0) return { error: "invalid_rows" as const, invalid_rows: invalidRows.slice(0, 10) };
    if (assistantRows.length === 0) return { error: "no_valid_records" as const };

    const connection = await pool.getConnection();
    try {
      await connection.beginTransaction();
      const usernames = [...new Set(assistantRows.map((r) => r.username))];
      const emails = [...new Set(assistantRows.map((r) => r.email))];
      const phones = [...new Set(assistantRows.map((r) => r.phone))];

      const queries: string[] = [];
      const allParams: unknown[] = [];
      if (usernames.length > 0) { queries.push(`SELECT DISTINCT username, phone_number, email FROM users WHERE username IN (${usernames.map(() => "?").join(", ")})`); allParams.push(...usernames); }
      if (phones.length > 0) { queries.push(`SELECT DISTINCT username, phone_number, email FROM users WHERE phone_number IN (${phones.map(() => "?").join(", ")})`); allParams.push(...phones); }
      if (emails.length > 0) { queries.push(`SELECT DISTINCT username, phone_number, email FROM users WHERE email IN (${emails.map(() => "?").join(", ")})`); allParams.push(...emails); }

      if (queries.length > 0) {
        const [existing] = await connection.query<RowDataPacket[]>(queries.join(" UNION "), allParams);
        if (existing.length > 0) { await connection.rollback(); return { error: "conflicts" as const, conflicts: existing.slice(0, 10) }; }
      }

      let hashedPasswords: string[];
      if (normalizedPassword) {
        const hashed = await hashPassword(normalizedPassword);
        hashedPasswords = assistantRows.map(() => hashed);
      } else {
        hashedPasswords = await Promise.all(assistantRows.map((r) => hashPassword(r.phone)));
      }

      const assistantValues = assistantRows.map((r, i) => [r.username, hashedPasswords[i], r.fullName, r.email, "Assistant", r.phone]);
      const [result] = await connection.query<ResultSetHeader>("INSERT INTO users (username, password_hash, full_name, email, role, phone_number) VALUES ?", [assistantValues]);
      const firstInsertId = result.insertId;
      const createdAssistants = assistantRows.map((r, i) => ({ user_id: firstInsertId + i, username: r.username, full_name: r.fullName, phone_number: r.phone, email: r.email }));
      await connection.commit();
      return { data: { created: createdAssistants.length, assistants: createdAssistants } };
    } catch (error) {
      await connection.rollback();
      const conflicts = mapBatchDuplicateConflicts(error, assistantRows);
      if (conflicts) return { error: "conflicts" as const, conflicts };
      throw error;
    } finally { connection.release(); }
  } catch (error) {
    if (error instanceof CsvEncodingError) {
      return {
        error: "invalid_rows" as const,
        invalid_rows: [{ row: 0, reason: error.message }],
      };
    }
    throw error;
  } finally {
    await unlink(tempCsvPath).catch((err) => logWarning("Failed to cleanup temp assistant CSV", { error: err instanceof Error ? err.message : String(err) }));
  }
}

export async function getTeacherAssistants(teacherId: number, termId?: number) {
  const [teacherRows] = await pool.query<UserRow[]>(
    "SELECT role FROM users WHERE user_id = ? AND role = 'Teacher'",
    [teacherId],
  );
  if (teacherRows.length === 0) return { error: "not_teacher" as const };

  const conditions = ["c.teacher_id = ?", "u.role = 'Assistant'"];
  const values: unknown[] = [teacherId];
  if (termId !== undefined) {
    conditions.push("c.term_id = ?");
    values.push(termId);
  }

  const [rows] = await pool.query<RowDataPacket[]>(
    `SELECT DISTINCT u.user_id, u.username, u.full_name, u.email, u.phone_number
       FROM users u
       JOIN class_assistants ca ON u.user_id = ca.assistant_id
       JOIN classes c ON ca.class_id = c.class_id
      WHERE ${conditions.join(" AND ")}
      ORDER BY u.user_id DESC`,
    values,
  );
  return { data: rows };
}

export async function getAllUsers(query: PaginationQuery) {
  const { page, limit } = query;
  const offset = (page - 1) * limit;
  const [[countRow], [dataRows]] = await Promise.all([
    pool.query<CountRow[]>("SELECT COUNT(*) as total FROM users"),
    pool.query<RowDataPacket[]>("SELECT user_id, username, full_name, email, role, created_at FROM users ORDER BY user_id ASC LIMIT ? OFFSET ?", [limit, offset]),
  ]);
  const totalItems = countRow[0]?.total ?? 0;
  const totalPages = Math.ceil(totalItems / limit);
  return { data: dataRows.map(normalizeUserCreatedAt), pagination: { currentPage: page, pageSize: limit, totalItems, totalPages } };
}

export async function getAllAssistants() {
  const [rows] = await pool.query<RowDataPacket[]>("SELECT user_id, username, full_name, email, phone_number, role, created_at FROM users WHERE role = 'Assistant'");
  return rows.map(normalizeUserCreatedAt);
}

export async function getUnenrolledStudents() {
  const [rows] = await pool.query<RowDataPacket[]>("SELECT u.user_id, u.username, u.full_name, u.email, u.phone_number, u.created_at FROM users u WHERE u.role = 'Student' AND NOT EXISTS (SELECT 1 FROM enrollments e WHERE e.student_id = u.user_id) ORDER BY u.created_at DESC");
  return rows.map(normalizeUserCreatedAt);
}

export async function getCurrentUser(userId: number) {
  const [[user]] = await pool.query<UserRow[]>("SELECT user_id, username, full_name, email, phone_number, role, created_at, password_hash FROM users WHERE user_id = ?", [userId]);
  if (!user) return { error: "not_found" as const };

  const passwordHash = user.password_hash ?? null;
  const result: Record<string, unknown> = { user_id: user.user_id, username: user.username, full_name: user.full_name, email: user.email, phone_number: user.phone_number, role: user.role, created_at: databaseDatetimeToIsoString(user.created_at, env.DB_TIMEZONE) };

  const shouldAuditPassword = ["Student", "Teacher", "Assistant"].includes(user.role);
  if (shouldAuditPassword && passwordHash) {
    // 获取用于检查的标识符：学生用学号，教师/助教用手机号
    const identifier = user.role === "Student" ? user.username : user.phone_number;
    result.must_change_password = await shouldChangePassword(passwordHash, identifier);
  }

  if (user.role === "Student") {
    const [classRows] = await pool.query<RowDataPacket[]>("SELECT c.class_id, c.class_name, c.class_code, c.teacher_id, teacher.full_name AS teacher_name FROM enrollments e JOIN classes c ON c.class_id = e.class_id LEFT JOIN users teacher ON teacher.user_id = c.teacher_id WHERE e.student_id = ? ORDER BY e.enrollment_id ASC", [userId]);
    result.classes = classRows;
    result.class = classRows[0] ?? null;
  }

  return { data: result };
}

export async function updateCurrentUser(userId: number, input: UpdateUserInput) {
  const { full_name, phone_number, email } = input;

  if (full_name === undefined && phone_number === undefined && email === undefined) {
    const [[user]] = await pool.query<RowDataPacket[]>("SELECT user_id, username, full_name, email, phone_number, role, created_at FROM users WHERE user_id = ?", [userId]);
    return { data: user ? normalizeUserCreatedAt(user) : user };
  }

  const fields: string[] = [];
  const values: unknown[] = [];

  const conflict = await findUserConflict(pool, {
    email,
    phone: phone_number,
    excludeUserId: userId,
  });
  if (conflict) return conflict;

  if (full_name !== undefined) { fields.push("full_name = ?"); values.push(full_name); }
  if (phone_number !== undefined) { fields.push("phone_number = ?"); values.push(phone_number); }
  if (email !== undefined) { fields.push("email = ?"); values.push(email); }
  values.push(userId);

  try {
    await pool.query(`UPDATE users SET ${fields.join(", ")} WHERE user_id = ?`, values);
    const [[user]] = await pool.query<RowDataPacket[]>("SELECT user_id, username, full_name, email, phone_number, role, created_at FROM users WHERE user_id = ?", [userId]);
    return { data: user ? normalizeUserCreatedAt(user) : user };
  } catch (error: unknown) {
    const duplicateError = mapDuplicateUserError(error);
    if (duplicateError) return duplicateError;
    throw error;
  }
}

export async function changePassword(userId: number, currentPassword: string, newPassword: string) {
  const [[user]] = await pool.query<UserRow[]>("SELECT password_hash FROM users WHERE user_id = ?", [userId]);
  if (!user) return { error: "not_found" as const };

  const passwordMatch = await verifyPassword(currentPassword, user.password_hash);
  if (!passwordMatch) return { error: "wrong_password" as const };

  const newPasswordHash = await hashPassword(newPassword);
  await pool.query("UPDATE users SET password_hash = ? WHERE user_id = ?", [newPasswordHash, userId]);
  return { success: true };
}

export async function getUserById(userId: number) {
  const [[user]] = await pool.query<RowDataPacket[]>("SELECT user_id, username, full_name, email, role, created_at FROM users WHERE user_id = ?", [userId]);
  if (!user) return { error: "not_found" as const };
  return { data: normalizeUserCreatedAt(user) };
}

export async function adminChangePassword(targetUserId: number, newPassword: string) {
  const newPasswordHash = await hashPassword(newPassword);
  const [result] = await pool.query<ResultSetHeader>("UPDATE users SET password_hash = ? WHERE user_id = ?", [newPasswordHash, targetUserId]);
  if (result.affectedRows === 0) return { error: "not_found" as const };
  return { success: true };
}

export async function resetStudentPassword(targetUserId: number, requesterId: number, requesterRole: string, input: ResetPasswordInput) {
  const [[targetUser]] = await pool.query<RowDataPacket[]>("SELECT role, username FROM users WHERE user_id = ?", [targetUserId]);
  if (!targetUser) return { error: "not_found" as const };
  if (targetUser.role !== "Student") return { error: "not_student" as const };

  if (requesterRole === "Teacher") {
    const [[row]] = await pool.query<AssignmentRow[]>("SELECT 1 FROM enrollments e JOIN classes c ON e.class_id = c.class_id WHERE e.student_id = ? AND c.teacher_id = ? LIMIT 1", [targetUserId, requesterId]);
    if (!row) return { error: "not_your_student" as const };
  } else if (requesterRole === "Assistant") {
    const [[row]] = await pool.query<AssignmentRow[]>("SELECT 1 FROM enrollments e JOIN class_assistants ca ON e.class_id = ca.class_id WHERE e.student_id = ? AND ca.assistant_id = ? LIMIT 1", [targetUserId, requesterId]);
    if (!row) return { error: "not_assigned_student" as const };
  }

  const password_hash = input.newPassword ? await hashPassword(input.newPassword) : await hashPassword(targetUser.username);
  const [result] = await pool.query<ResultSetHeader>("UPDATE users SET password_hash = ? WHERE user_id = ?", [password_hash, targetUserId]);
  if (result.affectedRows === 0) return { error: "not_found" as const };
  return { success: true, isCustomPassword: Boolean(input.newPassword) };
}

export async function deleteUser(targetUserId: number, requesterId: number) {
  if (requesterId === targetUserId) return { error: "self_delete" as const };
  await ensureDatabaseManagedStorageRoot(reportsRoot, env);

  const deletionPermit = await acquireManagedDeletionPermit();
  if (!deletionPermit) {
    logWarning("Timed out waiting for managed deletion admission", { targetUserId });
    return { error: "file_cleanup_failed" as const };
  }
  const connection = await pool.getConnection().catch((error) => {
    deletionPermit.release();
    throw error;
  });
  let stagedReportPaths: StagedManagedPath[] = [];
  const cleanupSummary: DeleteCleanupSummary = {};
  let transactionPhase: "before-commit" | "commit-attempted" | "committed" = "before-commit";
  let released = false;
  try {
    await connection.beginTransaction();
    const [[userRecord]] = await connection.query<DeleteUserRow[]>(
      "SELECT user_id, role FROM users WHERE user_id = ? FOR UPDATE",
      [targetUserId],
    );
    if (!userRecord) { await connection.rollback(); return { error: "not_found" as const }; }

    const reportRows = userRecord.role === "Student"
      ? await lockStudentExperimentsAndGetReportRows(connection, [targetUserId])
      : [];
    if (userRecord.role === "Student") {
      try {
        const survivingReferences = await resolveSurvivingReportReferences(
          reportRows.filter((row) => row.student_id !== targetUserId),
        );
        const reportPaths = (await resolveManagedReportPaths(
          reportRows.filter((row) => row.student_id === targetUserId),
          true,
        )).filter((reportPath) => !survivingReferences.has(reportPath));
        stagedReportPaths = await stageManagedPathsForDeletion(
          reportPaths,
          reportStorageRoot(),
          {
            databaseName: env.DB_DATABASE,
            kind: "student-reports",
            entityIds: [targetUserId],
          },
        );
      } catch (error) {
        await connection.rollback();
        logWarning("Failed to stage student report files for deletion", {
          targetUserId,
          error: error instanceof Error ? error.message : String(error),
        });
        return { error: "file_cleanup_failed" as const };
      }
    }

    accumulateCleanupSummary(
      cleanupSummary,
      await deleteUserWithConnection(connection, targetUserId, userRecord.role),
    );
    transactionPhase = "commit-attempted";
    await connection.commit();
    transactionPhase = "committed";
    connection.release();
    released = true;
    await purgeDeletedReportFiles(stagedReportPaths, cleanupSummary);
    return { success: true, cleanup: cleanupSummary };
  } catch (error) {
    if (transactionPhase !== "committed") {
      if (transactionPhase === "commit-attempted") {
        if (!released) {
          connection.destroy();
          released = true;
        }
        // Keep the process-wide deletion permit through the fresh owner probe
        // and filesystem resolution; see the dataset equivalent.
        const resolution = await reconcileAmbiguousUserCommit(
          [targetUserId],
          stagedReportPaths,
          cleanupSummary,
        );
        if (resolution === "committed") {
          return { success: true, cleanup: cleanupSummary };
        }
        throw error;
      }

      try {
        await connection.rollback();
      } catch {
        connection.destroy();
        released = true;
      }
      if (stagedReportPaths.length > 0) {
        await restoreStagedManagedPaths(stagedReportPaths).catch((restoreError) => {
          logWarning("Failed to restore staged report files after user deletion rollback", {
            targetUserId,
            error: restoreError instanceof Error ? restoreError.message : String(restoreError),
          });
        });
      }
    }
    throw error;
  } finally {
    if (!released) connection.release();
    deletionPermit.release();
  }
}

export async function batchDeleteUsers(targetUserIds: number[], requesterId: number) {
  if (targetUserIds.includes(requesterId)) return { error: "self_delete" as const };

  const uniqueTargetUserIds = [...new Set(targetUserIds)];
  if (uniqueTargetUserIds.length === 0) {
    return { error: "not_found" as const, missingUserIds: [] as number[] };
  }
  await ensureDatabaseManagedStorageRoot(reportsRoot, env);

  const placeholders = uniqueTargetUserIds.map(() => "?").join(", ");
  const deletionPermit = await acquireManagedDeletionPermit();
  if (!deletionPermit) {
    logWarning("Timed out waiting for managed deletion admission", {
      targetUserIds: uniqueTargetUserIds,
    });
    return { error: "file_cleanup_failed" as const };
  }
  const connection = await pool.getConnection().catch((error) => {
    deletionPermit.release();
    throw error;
  });
  let stagedReportPaths: StagedManagedPath[] = [];
  const cleanupSummary: DeleteCleanupSummary = {};
  let transactionPhase: "before-commit" | "commit-attempted" | "committed" = "before-commit";
  let released = false;

  try {
    await connection.beginTransaction();
    const [userRows] = await connection.query<DeleteUserRow[]>(
      `SELECT user_id, role FROM users WHERE user_id IN (${placeholders}) ORDER BY user_id ASC FOR UPDATE`,
      uniqueTargetUserIds,
    );

    const foundUserIds = new Set(userRows.map((row) => row.user_id));
    const missingUserIds = uniqueTargetUserIds.filter((userId) => !foundUserIds.has(userId));
    if (missingUserIds.length > 0) {
      await connection.rollback();
      return { error: "not_found" as const, missingUserIds };
    }

    const studentIds = userRows
      .filter((userRow) => userRow.role === "Student")
      .map((userRow) => userRow.user_id);
    const reportRows = studentIds.length > 0
      ? await lockStudentExperimentsAndGetReportRows(connection, studentIds)
      : [];
    if (studentIds.length > 0) {
      const studentIdSet = new Set(studentIds);
      try {
        const survivingReferences = await resolveSurvivingReportReferences(
          reportRows.filter((row) => !studentIdSet.has(row.student_id)),
        );
        const reportPaths = (await resolveManagedReportPaths(
          reportRows.filter((row) => studentIdSet.has(row.student_id)),
          true,
        )).filter((reportPath) => !survivingReferences.has(reportPath));
        stagedReportPaths = await stageManagedPathsForDeletion(
          reportPaths,
          reportStorageRoot(),
          {
            databaseName: env.DB_DATABASE,
            kind: "student-reports",
            entityIds: studentIds,
          },
        );
      } catch (error) {
        await connection.rollback();
        logWarning("Failed to stage report files for bulk user deletion", {
          studentIds,
          error: error instanceof Error ? error.message : String(error),
        });
        return { error: "file_cleanup_failed" as const };
      }
    }

    for (const userRow of userRows) {
      const userCleanup = await deleteUserWithConnection(connection, userRow.user_id, userRow.role);
      accumulateCleanupSummary(cleanupSummary, userCleanup);
    }

    transactionPhase = "commit-attempted";
    await connection.commit();
    transactionPhase = "committed";
    connection.release();
    released = true;
    await purgeDeletedReportFiles(stagedReportPaths, cleanupSummary);
    return { success: true, deletedCount: userRows.length, cleanup: cleanupSummary };
  } catch (error) {
    if (transactionPhase !== "committed") {
      if (transactionPhase === "commit-attempted") {
        if (!released) {
          connection.destroy();
          released = true;
        }
        const resolution = await reconcileAmbiguousUserCommit(
          uniqueTargetUserIds,
          stagedReportPaths,
          cleanupSummary,
        );
        if (resolution === "committed") {
          return {
            success: true,
            deletedCount: uniqueTargetUserIds.length,
            cleanup: cleanupSummary,
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
      if (stagedReportPaths.length > 0) {
        await restoreStagedManagedPaths(stagedReportPaths).catch((restoreError) => {
          logWarning("Failed to restore staged report files after bulk user deletion rollback", {
            error: restoreError instanceof Error ? restoreError.message : String(restoreError),
          });
        });
      }
    }
    throw error;
  } finally {
    if (!released) connection.release();
    deletionPermit.release();
  }
}

function studentIdsFromDeletionJournal(journal: ManagedDeletionJournal): number[] {
  if (journal.metadata.kind !== "student-reports" || !Array.isArray(journal.metadata.entityIds)) {
    throw new Error(`Unexpected report deletion journal metadata: ${journal.operationId}`);
  }
  const studentIds = journal.metadata.entityIds;
  if (
    studentIds.length === 0
    || studentIds.some((studentId) => !Number.isInteger(studentId) || Number(studentId) <= 0)
    || new Set(studentIds).size !== studentIds.length
  ) {
    throw new Error(`Invalid report deletion journal entities: ${journal.operationId}`);
  }
  return studentIds.map(Number);
}

export async function reconcileReportDeletionJournals() {
  const storageRoot = await ensureDatabaseManagedStorageRoot(reportsRoot, env);
  const deletionPermit = tryAcquireManagedDeletionPermit();
  if (!deletionPermit) throw new Error("Managed deletion reconciliation is busy");
  try {
    const result = await reconcileStagedDeletionJournals(storageRoot, async (journal) => {
      if (journal.metadata.databaseName !== env.DB_DATABASE) return "skip";
      const studentIds = studentIdsFromDeletionJournal(journal);
      let owner: Awaited<ReturnType<typeof probeUserOwnersWithLock>> | null = null;
      try {
        owner = await probeUserOwnersWithLock(studentIds);
        if (owner.ownerCount !== 0 && owner.ownerCount !== studentIds.length) {
          throw new Error(`Mixed report deletion journal ownership: ${journal.operationId}`);
        }
        const ownershipMatch = classifyManagedDeletionOwnership(
          journal.entries,
          owner.referencedReportPaths,
        );
        if (ownershipMatch === "partial") {
          throw new Error(`Partial report deletion journal ownership: ${journal.operationId}`);
        }
        const lockedOwner = owner;
        return {
          action: ownershipMatch === "all"
            ? "restore" as const
            : "purge" as const,
          release: async () => {
            await lockedOwner.release();
          },
        };
      } catch (error) {
        await owner?.release().catch(() => {});
        throw error;
      }
    });
    if (result.failedJournals > 0) {
      logWarning("Report deletion journal reconciliation left failures", {
        failedJournals: result.failedJournals,
        failedJournalNames: result.failedJournalNames,
      });
    }
    return result;
  } finally {
    deletionPermit.release();
  }
}
