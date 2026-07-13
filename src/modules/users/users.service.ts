import path from "path";
import { mkdir, unlink } from "node:fs/promises";
import type { RowDataPacket, ResultSetHeader } from "mysql2/promise";
import { pool } from "../../config";
import { hashPassword, shouldChangePassword, verifyPassword } from "../../utils/password";
import { logWarning } from "../../utils/error-handler";
import { USER_CONSTANTS, FILE_SIZE_LIMITS, CSV_LIMITS } from "../../utils/constants";
import { CsvEncodingError, parseCsvFileRows } from "../../utils/csv-processor";
import type { CreateTeacherInput, CreateAssistantSingleInput, CreateAssistantInput, UpdateUserInput, ResetPasswordInput, UserSearchQuery, PaginationQuery } from "./users.schema";
import { studentSelfRegisterSchema, teacherAssistantSelfRegisterSchema } from "./users.schema";

const { TEACHER_EMAIL_DOMAIN, ASSISTANT_EMAIL_DOMAIN, PHONE_REGEX, MIN_NAME_LENGTH } = USER_CONSTANTS;
const BATCH_CSV_MIME_TYPES = new Set([
  "text/csv",
  "application/vnd.ms-excel",
  "text/plain",
]);

function isAllowedBatchCsvMimeType(mimeType: string): boolean {
  if (!mimeType) return true;
  const normalized = mimeType.split(";")[0]?.trim().toLowerCase() ?? "";
  return BATCH_CSV_MIME_TYPES.has(normalized) || normalized.includes("csv");
}

interface UserRow extends RowDataPacket { user_id: number; username: string; full_name: string; email: string; role: string; phone_number: string | null; password_hash: string; created_at: Date; }
interface ClassRow extends RowDataPacket { class_id: number; teacher_id: number; }
interface CountRow extends RowDataPacket { total: number; }
interface AssignmentRow extends RowDataPacket { class_id: number; }
interface DeleteUserRow extends RowDataPacket { user_id: number; role: string; }

interface DeleteCleanupSummary {
  deletedClasses?: number;
  removedAssistantAssignments?: number;
  removedEnrollments?: number;
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

  if (normalizedKey === "username" || rawMessage.includes("users.username")) {
    return { error: "dup_username" as const };
  }
  if (normalizedKey === "email" || rawMessage.includes("users.email")) {
    return { error: "dup_email" as const };
  }
  if (normalizedKey === "phone_number" || rawMessage.includes("users.phone_number")) {
    return { error: "dup_phone" as const };
  }

  return null;
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

  let countQuery: string, dataQuery: string, queryParams: unknown[];

  if (useFullText) {
    const searchTerm = term.split(/\s+/).map((word) => `+${word}*`).join(" ");
    countQuery = "SELECT COUNT(*) AS total FROM users WHERE MATCH(full_name, email, phone_number) AGAINST(? IN BOOLEAN MODE)";
    dataQuery = "SELECT user_id, username, full_name, email, role, created_at, MATCH(full_name, email, phone_number) AGAINST(? IN BOOLEAN MODE) AS relevance FROM users WHERE MATCH(full_name, email, phone_number) AGAINST(? IN BOOLEAN MODE) ORDER BY relevance DESC, user_id ASC LIMIT ? OFFSET ?";
    queryParams = [searchTerm];
  } else {
    const likeTerm = `%${term}%`;
    countQuery = "SELECT COUNT(*) AS total FROM users WHERE full_name LIKE ? OR email LIKE ? OR phone_number LIKE ?";
    dataQuery = "SELECT user_id, username, full_name, email, role, created_at FROM users WHERE full_name LIKE ? OR email LIKE ? OR phone_number LIKE ? ORDER BY user_id ASC LIMIT ? OFFSET ?";
    queryParams = [likeTerm, likeTerm, likeTerm];
  }

  const [[countRow]] = await pool.query<CountRow[]>(countQuery, queryParams);
  const totalItems = Number(countRow?.total ?? 0);
  const totalPages = Math.ceil(totalItems / limit);

  const [dataRows] = await pool.query<RowDataPacket[]>(dataQuery, useFullText ? [queryParams[0], queryParams[0], limit, offset] : [...queryParams, limit, offset]);
  const results = dataRows.map((row) => { const { relevance, ...rest } = row; return rest; });

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
    if (trimmed.length < USER_CONSTANTS.MIN_PASSWORD_LENGTH) return { error: "password_too_short" as const };
    normalizedPassword = trimmed;
  }

  if (!isAllowedBatchCsvMimeType(file.type)) return { error: "invalid_file_type" as const };
  if (file.size > FILE_SIZE_LIMITS.CSV_MAX_SIZE) return { error: "file_too_large" as const };

  const tempDir = path.resolve(import.meta.dir, "../../../uploads", "temp");
  await mkdir(tempDir, { recursive: true });
  const tempCsvPath = path.join(tempDir, `teachers-${Date.now()}-${file.name.replace(/[\/\\?%*:|"<>]/g, "-")}`);
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
    } catch (error) { await connection.rollback(); throw error; } finally { connection.release(); }
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
    if (trimmed.length < USER_CONSTANTS.MIN_PASSWORD_LENGTH) return { error: "password_too_short" as const };
    normalizedPassword = trimmed;
  }

  if (!isAllowedBatchCsvMimeType(file.type)) return { error: "invalid_file_type" as const };
  if (file.size > FILE_SIZE_LIMITS.CSV_MAX_SIZE) return { error: "file_too_large" as const };

  const tempDir = path.resolve(import.meta.dir, "../../../uploads", "temp");
  await mkdir(tempDir, { recursive: true });
  const tempCsvPath = path.join(tempDir, `assistants-${Date.now()}-${file.name.replace(/[\/\\?%*:|"<>]/g, "-")}`);
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
    } catch (error) { await connection.rollback(); throw error; } finally { connection.release(); }
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
  return { data: dataRows, pagination: { currentPage: page, pageSize: limit, totalItems, totalPages } };
}

export async function getAllAssistants() {
  const [rows] = await pool.query<RowDataPacket[]>("SELECT user_id, username, full_name, email, phone_number, role, created_at FROM users WHERE role = 'Assistant'");
  return rows;
}

export async function getUnenrolledStudents() {
  const [rows] = await pool.query<RowDataPacket[]>("SELECT u.user_id, u.username, u.full_name, u.email, u.phone_number, u.created_at FROM users u WHERE u.role = 'Student' AND NOT EXISTS (SELECT 1 FROM enrollments e WHERE e.student_id = u.user_id) ORDER BY u.created_at DESC");
  return rows;
}

export async function getCurrentUser(userId: number) {
  const [[user]] = await pool.query<UserRow[]>("SELECT user_id, username, full_name, email, phone_number, role, created_at, password_hash FROM users WHERE user_id = ?", [userId]);
  if (!user) return { error: "not_found" as const };

  const passwordHash = user.password_hash ?? null;
  const result: Record<string, unknown> = { user_id: user.user_id, username: user.username, full_name: user.full_name, email: user.email, phone_number: user.phone_number, role: user.role, created_at: user.created_at };

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

  if (!full_name && !phone_number && !email) {
    const [[user]] = await pool.query<RowDataPacket[]>("SELECT user_id, username, full_name, email, phone_number, role, created_at FROM users WHERE user_id = ?", [userId]);
    return { data: user };
  }

  const fields: string[] = [];
  const values: unknown[] = [];

  const conflict = await findUserConflict(pool, {
    email,
    phone: phone_number,
    excludeUserId: userId,
  });
  if (conflict) return conflict;

  if (full_name) { fields.push("full_name = ?"); values.push(full_name); }
  if (phone_number) { fields.push("phone_number = ?"); values.push(phone_number); }
  if (email) { fields.push("email = ?"); values.push(email); }
  values.push(userId);

  try {
    await pool.query(`UPDATE users SET ${fields.join(", ")} WHERE user_id = ?`, values);
    const [[user]] = await pool.query<RowDataPacket[]>("SELECT user_id, username, full_name, email, phone_number, role, created_at FROM users WHERE user_id = ?", [userId]);
    return { data: user };
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
  return { data: user };
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

  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();
    const [[userRecord]] = await connection.query<DeleteUserRow[]>("SELECT user_id, role FROM users WHERE user_id = ?", [targetUserId]);
    if (!userRecord) { await connection.rollback(); return { error: "not_found" as const }; }

    const cleanupSummary = await deleteUserWithConnection(connection, targetUserId, userRecord.role);
    await connection.commit();
    return { success: true, cleanup: cleanupSummary };
  } catch (error) { await connection.rollback(); throw error; } finally { connection.release(); }
}

export async function batchDeleteUsers(targetUserIds: number[], requesterId: number) {
  if (targetUserIds.includes(requesterId)) return { error: "self_delete" as const };

  const uniqueTargetUserIds = [...new Set(targetUserIds)];
  if (uniqueTargetUserIds.length === 0) {
    return { error: "not_found" as const, missingUserIds: [] as number[] };
  }

  const placeholders = uniqueTargetUserIds.map(() => "?").join(", ");
  const connection = await pool.getConnection();

  try {
    await connection.beginTransaction();
    const [userRows] = await connection.query<DeleteUserRow[]>(
      `SELECT user_id, role FROM users WHERE user_id IN (${placeholders}) ORDER BY user_id ASC`,
      uniqueTargetUserIds,
    );

    const foundUserIds = new Set(userRows.map((row) => row.user_id));
    const missingUserIds = uniqueTargetUserIds.filter((userId) => !foundUserIds.has(userId));
    if (missingUserIds.length > 0) {
      await connection.rollback();
      return { error: "not_found" as const, missingUserIds };
    }

    const cleanupSummary: DeleteCleanupSummary = {};
    for (const userRow of userRows) {
      const userCleanup = await deleteUserWithConnection(connection, userRow.user_id, userRow.role);
      accumulateCleanupSummary(cleanupSummary, userCleanup);
    }

    await connection.commit();
    return { success: true, deletedCount: userRows.length, cleanup: cleanupSummary };
  } catch (error) {
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}
