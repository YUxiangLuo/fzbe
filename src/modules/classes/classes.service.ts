import path from "path";
import { unlink, mkdir } from "node:fs/promises";
import type { RowDataPacket, ResultSetHeader } from "mysql2";
import { pool } from "../../config";
import { validateFile, generateUniqueFilename, FileValidationError } from "../../utils/file-validator";
import { logWarning } from "../../utils/error-handler";
import { hashPassword } from "../../utils/password";
import { USER_CONSTANTS, CSV_LIMITS } from "../../utils/constants";
import { CsvEncodingError, parseCsvFileRows } from "../../utils/csv-processor";
import type { AcademicTermQuery, UpdateClassInput } from "./classes.schema";

interface ClassRow extends RowDataPacket {
  class_id: number;
  class_name: string;
  class_code: string | null;
  academic_year_start: number;
  semester: number;
  teacher_id: number;
  teacher_name?: string;
  created_at: Date;
}

interface UserRow extends RowDataPacket {
  user_id: number;
  username: string;
  full_name: string;
  email: string;
  role: string;
}

interface ClassAssistantRow extends UserRow {
  class_id: number;
}

interface ClassWithAssistants extends ClassRow {
  assistants: UserRow[];
}

interface CountRow extends RowDataPacket {
  count: number;
}

interface AssignmentRow extends RowDataPacket {
  assistant_id?: number;
  enrollment_id?: number;
}

const { STUDENT_ID_REGEX, MIN_NAME_LENGTH, MAX_NAME_LENGTH } = USER_CONSTANTS;

type AcademicTermFilter = AcademicTermQuery;

const hasAcademicTermFilter = (
  filter?: AcademicTermFilter,
): filter is { academic_year_start: number; semester: number } =>
  filter?.academic_year_start !== undefined && filter.semester !== undefined;

const appendAcademicTermClause = (
  clauses: string[],
  values: Array<string | number | null>,
  filter?: AcademicTermFilter,
  alias = "c",
) => {
  if (!hasAcademicTermFilter(filter)) return;
  clauses.push(`${alias}.academic_year_start = ?`);
  values.push(filter.academic_year_start);
  clauses.push(`${alias}.semester = ?`);
  values.push(filter.semester);
};

const isValidStudentId = (value: string) => STUDENT_ID_REGEX.test(value);
const isValidFullName = (value: string) =>
  value.length >= MIN_NAME_LENGTH && value.length <= MAX_NAME_LENGTH;

export async function isAssistantAssignedToClass(classId: number, assistantId: number): Promise<boolean> {
  const [[assignment]] = await pool.query<AssignmentRow[]>(
    "SELECT 1 FROM class_assistants WHERE class_id = ? AND assistant_id = ? LIMIT 1",
    [classId, assistantId],
  );
  return Boolean(assignment);
}

async function attachAssistantsToClasses(classRows: ClassRow[]): Promise<ClassWithAssistants[]> {
  if (classRows.length === 0) return [];

  const classIds = classRows.map((classRow) => classRow.class_id);
  const placeholders = classIds.map(() => "?").join(", ");
  const [assistantRows] = await pool.query<ClassAssistantRow[]>(
    `SELECT ca.class_id, u.user_id, u.username, u.full_name, u.email, u.role
       FROM class_assistants ca
       JOIN users u ON u.user_id = ca.assistant_id
      WHERE ca.class_id IN (${placeholders})
      ORDER BY u.full_name ASC, u.username ASC`,
    classIds,
  );

  const assistantsByClassId = new Map<number, UserRow[]>();
  for (const row of assistantRows) {
    if (!assistantsByClassId.has(row.class_id)) assistantsByClassId.set(row.class_id, []);
    assistantsByClassId.get(row.class_id)!.push({
      user_id: row.user_id,
      username: row.username,
      full_name: row.full_name,
      email: row.email,
      role: row.role,
    } as UserRow);
  }

  return classRows.map((classRow) => ({
    ...classRow,
    assistants: assistantsByClassId.get(classRow.class_id) ?? [],
  }));
}

export interface CreateClassResult {
  class: ClassRow;
  students_created: number;
  students_enrolled: number;
  students: { user_id: number; username: string; full_name: string; email: string }[];
}

export async function createClass(
  className: string,
  classCode: string | null,
  teacherId: number,
  studentListFile: File | undefined,
  academicYearStart: number,
  semester: number,
): Promise<{ error: string; status: 400 | 409; details?: unknown } | { data: CreateClassResult }> {
  let tempCsvPath: string | null = null;

  if (studentListFile) {
    try {
      await validateFile(studentListFile, "csv", { maxSize: 5 * 1024 * 1024, checkMagicBytes: false });
    } catch (error) {
      if (error instanceof FileValidationError) {
        return { error: error.message, status: 400 };
      }
      throw error;
    }
  }

  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    const [result] = await connection.query<ResultSetHeader>(
      "INSERT INTO classes (class_name, class_code, academic_year_start, semester, teacher_id) VALUES (?, ?, ?, ?, ?)",
      [className, classCode || null, academicYearStart, semester, teacherId],
    );
    const classId = result.insertId;

    await connection.query("INSERT INTO grade_weights (class_id) VALUES (?)", [classId]);

    let createdStudentsCount = 0;
    const createdStudents: { user_id: number; username: string; full_name: string; email: string }[] = [];

    if (studentListFile) {
      const tempDir = path.resolve(import.meta.dir, "../../../uploads", "temp");
      await mkdir(tempDir, { recursive: true });
      const uniqueFilename = generateUniqueFilename(studentListFile.name);
      tempCsvPath = path.join(tempDir, uniqueFilename);
      await Bun.write(tempCsvPath, studentListFile);

      const studentRecords: { row: number; studentId: string; fullName: string; email: string }[] = [];
      const invalidRecords: { row: number; reason: string }[] = [];
      const seenStudentIds = new Set<string>();
      let rowNumber = 0;

      await parseCsvFileRows(tempCsvPath, (row) => {
        rowNumber += 1;
        // 检查行数限制
        if (rowNumber > CSV_LIMITS.BATCH_IMPORT_MAX_ROWS) {
          invalidRecords.push({ row: rowNumber, reason: `超过最大行数限制 ${CSV_LIMITS.BATCH_IMPORT_MAX_ROWS}` });
          return;
        }
        const studentId = row["学号"]?.trim() ?? "";
        const fullName = row["姓名"]?.trim() ?? "";

        if (!studentId || !fullName) {
          invalidRecords.push({ row: rowNumber, reason: "学号或姓名缺失" });
          return;
        }
        if (!isValidStudentId(studentId)) {
          invalidRecords.push({ row: rowNumber, reason: `学号 ${studentId} 不符合格式要求（需为8位及以上数字）` });
          return;
        }
        if (!isValidFullName(fullName)) {
          invalidRecords.push({ row: rowNumber, reason: `姓名长度需在 ${MIN_NAME_LENGTH}-${MAX_NAME_LENGTH} 个字符之间` });
          return;
        }
        if (seenStudentIds.has(studentId)) {
          invalidRecords.push({ row: rowNumber, reason: `CSV 内存在重复学号 ${studentId}` });
          return;
        }
        seenStudentIds.add(studentId);
        studentRecords.push({ row: rowNumber, studentId, fullName, email: `${studentId}@student.edu` });
      });

      if (invalidRecords.length > 0) {
        await connection.rollback();
        return { error: "学生名单存在无效记录", status: 400, details: { invalid_rows: invalidRecords.slice(0, 10) } };
      }

      if (studentRecords.length > 0) {
        const studentIds = studentRecords.map((r) => r.studentId);
        const [existingUsers] = await connection.query<UserRow[]>(
          "SELECT username FROM users WHERE username IN (?)",
          [studentIds],
        );

        if (existingUsers.length > 0) {
          await connection.rollback();
          return { error: "事务失败。以下学生已存在于数据库中。", status: 409, details: { existing_students: existingUsers.map((u) => u.username) } };
        }

        const hashedPasswords = await Promise.all(
          studentRecords.map((r) => hashPassword(r.studentId)),
        );

        const studentValues = studentRecords.map((r, i) => [r.studentId, hashedPasswords[i], r.fullName, r.email, "Student"]);
        const [userResult] = await connection.query<ResultSetHeader>(
          "INSERT INTO users (username, password_hash, full_name, email, role) VALUES ?",
          [studentValues],
        );

        const firstUserId = userResult.insertId;
        createdStudentsCount = studentRecords.length;

        const enrollmentValues = studentRecords.map((_, i) => [firstUserId + i, classId]);
        await connection.query("INSERT INTO enrollments (student_id, class_id) VALUES ?", [enrollmentValues]);

        studentRecords.forEach((r, i) => {
          createdStudents.push({ user_id: firstUserId + i, username: r.studentId, full_name: r.fullName, email: r.email });
        });
      }
    }

    await connection.commit();

    const [newClassRows] = await connection.query<ClassRow[]>(
      "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id WHERE c.class_id = ?",
      [classId],
    );

    return { data: { class: newClassRows[0]!, students_created: createdStudentsCount, students_enrolled: createdStudents.length, students: createdStudents } };
  } catch (error: unknown) {
    await connection.rollback();
    if (error instanceof CsvEncodingError) {
      return {
        error: "学生名单存在无效记录",
        status: 400,
        details: { invalid_rows: [{ row: 0, reason: error.message }] },
      };
    }
    const err = error as { code?: string };
    if (err.code === "ER_DUP_ENTRY") {
      return { error: "班级代码已存在", status: 409 };
    }
    throw error;
  } finally {
    connection.release();
    if (tempCsvPath) {
      await unlink(tempCsvPath).catch((err) => logWarning("Failed to cleanup temp CSV file", { error: err instanceof Error ? err.message : String(err) }));
    }
  }
}

export async function getAllClasses() {
  const [rows] = await pool.query<ClassRow[]>(
    "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id",
  );
  return rows;
}

export async function getClassById(classId: number, userId: number, userRole: string) {
  const [classRows] = await pool.query<ClassRow[]>(
    "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id WHERE c.class_id = ?",
    [classId],
  );
  const classInfo = classRows[0];
  if (!classInfo) return { error: "not_found" as const };

  let isAllowed = false;
  if (userRole === "Admin" || userId === classInfo.teacher_id) {
    isAllowed = true;
  } else if (userRole === "Assistant") {
    isAllowed = await isAssistantAssignedToClass(classId, userId);
  } else if (userRole === "Student") {
    const [[enrollment]] = await pool.query<AssignmentRow[]>(
      "SELECT enrollment_id FROM enrollments WHERE student_id = ? AND class_id = ?",
      [userId, classId],
    );
    if (enrollment) isAllowed = true;
  }

  if (!isAllowed) return { error: "forbidden" as const };

  // Students can view class metadata but should not receive full roster/contact lists.
  if (userRole === "Student") {
    return { data: { ...classInfo, students: [], assistants: [] } };
  }

  const [studentRows] = await pool.query<UserRow[]>(
    "SELECT u.user_id, u.username, u.full_name, u.email FROM users u JOIN enrollments e ON u.user_id = e.student_id WHERE e.class_id = ?",
    [classId],
  );
  const [assistantRows] = await pool.query<UserRow[]>(
    "SELECT u.user_id, u.username, u.full_name, u.email FROM users u JOIN class_assistants ca ON u.user_id = ca.assistant_id WHERE ca.class_id = ?",
    [classId],
  );

  return { data: { ...classInfo, students: studentRows, assistants: assistantRows } };
}

export async function updateClass(classId: number, userId: number, userRole: string, input: UpdateClassInput) {
  const [classRows] = await pool.query<ClassRow[]>("SELECT teacher_id FROM classes WHERE class_id = ?", [classId]);
  if (classRows.length === 0) return { error: "not_found" as const };

  const classOwner = classRows[0]!.teacher_id;
  const isAllowed = userRole === "Admin" || userId === classOwner;
  if (!isAllowed) return { error: "forbidden" as const };

  const fields: string[] = [];
  const values: (string | number | null)[] = [];
  if (input.class_name !== undefined) {
    fields.push("class_name = ?");
    values.push(input.class_name);
  }
  if (input.class_code !== undefined) {
    fields.push("class_code = ?");
    values.push(input.class_code || null);
  }
  if (input.academic_year_start !== undefined) {
    fields.push("academic_year_start = ?");
    values.push(input.academic_year_start);
  }
  if (input.semester !== undefined) {
    fields.push("semester = ?");
    values.push(input.semester);
  }
  if (fields.length === 0) return { error: "no_fields" as const };

  values.push(classId);
  try {
    await pool.query(`UPDATE classes SET ${fields.join(", ")} WHERE class_id = ?`, values);
  } catch (error: unknown) {
    const err = error as { code?: string };
    if (err.code === "ER_DUP_ENTRY") return { error: "duplicate" as const };
    throw error;
  }

  const [updatedRows] = await pool.query<ClassRow[]>(
    "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id WHERE c.class_id = ?",
    [classId],
  );
  return { data: updatedRows[0]! };
}

export async function deleteClass(classId: number, userId: number, userRole: string) {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    const [classRows] = await connection.query<ClassRow[]>("SELECT teacher_id FROM classes WHERE class_id = ?", [classId]);
    if (classRows.length === 0) {
      await connection.rollback();
      return { error: "not_found" as const };
    }

    if (userRole !== "Admin" && userId !== classRows[0]!.teacher_id) {
      await connection.rollback();
      return { error: "forbidden" as const };
    }

    const [[assistantCount]] = await connection.query<CountRow[]>("SELECT COUNT(*) as count FROM class_assistants WHERE class_id = ?", [classId]);
    const [[enrollmentCount]] = await connection.query<CountRow[]>("SELECT COUNT(*) as count FROM enrollments WHERE class_id = ?", [classId]);
    const [[gradeWeightsCount]] = await connection.query<CountRow[]>("SELECT COUNT(*) as count FROM grade_weights WHERE class_id = ?", [classId]);

    const [deleteResult] = await connection.query<ResultSetHeader>("DELETE FROM classes WHERE class_id = ?", [classId]);
    if (deleteResult.affectedRows === 0) {
      await connection.rollback();
      return { error: "not_found" as const };
    }

    await connection.commit();
    return {
      data: {
        message: "班级删除成功",
        cleanup: { removedAssistants: assistantCount!.count, removedEnrollments: enrollmentCount!.count, removedGradeWeights: gradeWeightsCount!.count },
      },
    };
  } catch (error) {
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}

export async function getClassesByTeacher(
  teacherId: number,
  userId: number,
  userRole: string,
  termFilter?: AcademicTermFilter,
) {
  if (userRole !== "Admin" && userId !== teacherId) return { error: "forbidden" as const };

  const [teacherRows] = await pool.query<UserRow[]>("SELECT role FROM users WHERE user_id = ? AND role = 'Teacher'", [teacherId]);
  if (teacherRows.length === 0) return { error: "not_teacher" as const };

  const clauses = ["c.teacher_id = ?"];
  const values: Array<string | number | null> = [teacherId];
  appendAcademicTermClause(clauses, values, termFilter);

  const [classRows] = await pool.query<ClassRow[]>(
    `SELECT c.*, u.full_name as teacher_name
       FROM classes c
       JOIN users u ON c.teacher_id = u.user_id
      WHERE ${clauses.join(" AND ")}
      ORDER BY c.academic_year_start DESC, c.semester DESC, c.class_id ASC`,
    values,
  );
  return { data: await attachAssistantsToClasses(classRows) };
}

export async function getClassesByTeacherAssistant(
  teacherId: number,
  assistantId: number,
  userId: number,
  userRole: string,
  termFilter?: AcademicTermFilter,
) {
  if (userRole !== "Admin" && userId !== teacherId) return { error: "forbidden" as const };

  const [teacherRows] = await pool.query<UserRow[]>(
    "SELECT role FROM users WHERE user_id = ? AND role = 'Teacher'",
    [teacherId],
  );
  if (teacherRows.length === 0) return { error: "not_teacher" as const };

  const [assistantRows] = await pool.query<UserRow[]>(
    "SELECT role FROM users WHERE user_id = ? AND role = 'Assistant'",
    [assistantId],
  );
  if (assistantRows.length === 0) return { error: "not_assistant" as const };

  const clauses = ["c.teacher_id = ?", "ca.assistant_id = ?"];
  const values: Array<string | number | null> = [teacherId, assistantId];
  appendAcademicTermClause(clauses, values, termFilter);

  const [classRows] = await pool.query<ClassRow[]>(
    `SELECT c.*, u.full_name as teacher_name
       FROM classes c
       JOIN class_assistants ca ON c.class_id = ca.class_id
       JOIN users u ON c.teacher_id = u.user_id
      WHERE ${clauses.join(" AND ")}
      ORDER BY c.academic_year_start DESC, c.semester DESC, c.class_id ASC`,
    values,
  );
  return { data: await attachAssistantsToClasses(classRows) };
}

export async function getClassesByAssistant(
  assistantId: number,
  userId: number,
  userRole: string,
  termFilter?: AcademicTermFilter,
) {
  if (userRole !== "Admin" && userId !== assistantId) return { error: "forbidden" as const };

  const [assistantRows] = await pool.query<UserRow[]>("SELECT role FROM users WHERE user_id = ? AND role = 'Assistant'", [assistantId]);
  if (assistantRows.length === 0) return { error: "not_assistant" as const };

  const clauses = ["ca.assistant_id = ?"];
  const values: Array<string | number | null> = [assistantId];
  appendAcademicTermClause(clauses, values, termFilter);

  const [classRows] = await pool.query<ClassRow[]>(
    `SELECT c.*, u.full_name as teacher_name
       FROM classes c
       JOIN class_assistants ca ON c.class_id = ca.class_id
       JOIN users u ON c.teacher_id = u.user_id
      WHERE ${clauses.join(" AND ")}
      ORDER BY c.academic_year_start DESC, c.semester DESC, c.class_id ASC`,
    values,
  );
  return { data: await attachAssistantsToClasses(classRows) };
}

export { FileValidationError };
