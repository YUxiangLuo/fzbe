import type { RowDataPacket, ResultSetHeader, Pool, PoolConnection } from "mysql2/promise";
import { pool } from "../../config";
import { getDefaultPasswordHash, hashPassword } from "../../utils/password";
import type { CreateStudentInput } from "./class-members.schema";

const DEFAULT_PASSWORD_HASH = getDefaultPasswordHash();

interface ClassRow extends RowDataPacket {
  class_id: number;
  class_name: string;
  class_code: string | null;
  teacher_id: number;
}

interface UserRow extends RowDataPacket {
  user_id: number;
  username: string;
  full_name: string;
  email: string;
  phone_number: string | null;
  role: string;
  created_at: Date;
}

interface EnrollmentRow extends RowDataPacket {
  class_id: number;
  class_name: string;
}

interface AssignmentRow extends RowDataPacket {
  assistant_id?: number;
}

type Queryable = Pool | PoolConnection;

async function isAssistantAssignedToClass(db: Queryable, classId: number, assistantId: number): Promise<boolean> {
  const [[assignment]] = await db.query<AssignmentRow[]>(
    "SELECT 1 FROM class_assistants WHERE class_id = ? AND assistant_id = ? LIMIT 1",
    [classId, assistantId],
  );
  return Boolean(assignment);
}

export async function checkClassAccess(
  db: Queryable,
  classId: number,
  userId: number,
  userRole: string,
  allowAssistants = true,
): Promise<{ allowed: boolean; teacherId: number | null }> {
  const [classRows] = await db.query<ClassRow[]>("SELECT teacher_id FROM classes WHERE class_id = ?", [classId]);
  if (classRows.length === 0) return { allowed: false, teacherId: null };

  const teacherId = classRows[0]!.teacher_id;
  let isAllowed = userRole === "Admin" || userId === teacherId;

  if (!isAllowed && allowAssistants && userRole === "Assistant") {
    isAllowed = await isAssistantAssignedToClass(db, classId, userId);
  }

  return { allowed: isAllowed, teacherId };
}

export async function createStudentInClass(classId: number, userId: number, userRole: string, input: CreateStudentInput) {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    const { allowed, teacherId } = await checkClassAccess(connection, classId, userId, userRole, true);
    if (teacherId === null) {
      await connection.rollback();
      return { error: "not_found" as const };
    }
    if (!allowed) {
      await connection.rollback();
      return { error: "forbidden" as const };
    }

    let password_hash: string;
    if (input.password) {
      password_hash = await hashPassword(input.password);
    } else {
      password_hash = DEFAULT_PASSWORD_HASH;
    }

    const studentEmail = input.email || `${input.username}@student.edu`;

    const [userResult] = await connection.query<ResultSetHeader>(
      "INSERT INTO users (username, password_hash, full_name, email, role, phone_number) VALUES (?, ?, ?, ?, 'Student', ?)",
      [input.username, password_hash, input.full_name, studentEmail, input.phone_number || null],
    );

    const newStudentId = userResult.insertId;
    await connection.query("INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)", [newStudentId, classId]);

    const [studentRows] = await connection.query<UserRow[]>(
      "SELECT user_id, username, full_name, email, phone_number, role, created_at FROM users WHERE user_id = ? LIMIT 1",
      [newStudentId],
    );

    await connection.commit();
    return { data: studentRows[0]! };
  } catch (error: unknown) {
    await connection.rollback();
    const err = error as { code?: string; message?: string };
    if (err.code === "ER_DUP_ENTRY") {
      if (err.message?.includes("users.username")) return { error: "dup_username" as const };
      if (err.message?.includes("users.email")) return { error: "dup_email" as const };
      return { error: "dup_entry" as const };
    }
    throw error;
  } finally {
    connection.release();
  }
}

export async function enrollExistingStudent(classId: number, studentId: number, userId: number, userRole: string) {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    const { allowed, teacherId } = await checkClassAccess(connection, classId, userId, userRole, true);
    if (teacherId === null) {
      await connection.rollback();
      return { error: "class_not_found" as const };
    }
    if (!allowed) {
      await connection.rollback();
      return { error: "forbidden" as const };
    }

    const [[studentLock]] = await connection.query<UserRow[]>(
      "SELECT user_id, role, full_name FROM users WHERE user_id = ? LIMIT 1 FOR UPDATE",
      [studentId],
    );
    if (!studentLock) {
      await connection.rollback();
      return { error: "student_not_found" as const };
    }
    if (studentLock.role !== "Student") {
      await connection.rollback();
      return { error: "not_student" as const };
    }

    const [existingEnrollments] = await connection.query<EnrollmentRow[]>(
      "SELECT e.class_id, c.class_name FROM enrollments e JOIN classes c ON e.class_id = c.class_id WHERE e.student_id = ? LIMIT 1",
      [studentId],
    );
    if (existingEnrollments.length > 0) {
      await connection.rollback();
      return { error: "already_enrolled" as const, currentClass: existingEnrollments[0]! };
    }

    await connection.query("INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)", [studentId, classId]);
    await connection.commit();

    return { data: { user_id: studentLock.user_id, full_name: studentLock.full_name, class_id: classId } };
  } catch (error: unknown) {
    await connection.rollback();
    const err = error as { code?: string };
    if (err.code === "ER_DUP_ENTRY") return { error: "dup_entry" as const };
    throw error;
  } finally {
    connection.release();
  }
}

export async function getClassStudents(classId: number, userId: number, userRole: string) {
  const { allowed, teacherId } = await checkClassAccess(pool, classId, userId, userRole, true);
  if (teacherId === null) return { error: "not_found" as const };
  if (!allowed) return { error: "forbidden" as const };

  const [students] = await pool.query<UserRow[]>(
    "SELECT u.user_id, u.username, u.full_name, u.email, u.phone_number, u.role, u.created_at FROM users u JOIN enrollments e ON u.user_id = e.student_id WHERE e.class_id = ? ORDER BY u.username ASC",
    [classId],
  );
  return { data: students };
}

export async function removeStudentFromClass(classId: number, studentId: number, userId: number, userRole: string) {
  const { allowed, teacherId } = await checkClassAccess(pool, classId, userId, userRole, true);
  if (teacherId === null) return { error: "class_not_found" as const };
  if (!allowed) return { error: "forbidden" as const };

  const [userRows] = await pool.query<UserRow[]>("SELECT role FROM users WHERE user_id = ?", [studentId]);
  if (userRows.length === 0) return { error: "user_not_found" as const };
  if (userRows[0]!.role !== "Student") return { error: "not_student" as const };

  const [result] = await pool.query<ResultSetHeader>(
    "DELETE FROM enrollments WHERE class_id = ? AND student_id = ?",
    [classId, studentId],
  );
  if (result.affectedRows === 0) return { error: "enrollment_not_found" as const };

  return { success: true };
}

export async function studentJoinClass(classId: number, studentId: number, classCode?: string) {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    await connection.query("SELECT user_id FROM users WHERE user_id = ? LIMIT 1 FOR UPDATE", [studentId]);

    const [existingEnrollments] = await connection.query<EnrollmentRow[]>(
      "SELECT e.class_id, c.class_name FROM enrollments e JOIN classes c ON e.class_id = c.class_id WHERE e.student_id = ? LIMIT 1",
      [studentId],
    );
    if (existingEnrollments.length > 0) {
      await connection.rollback();
      return { error: "already_enrolled" as const, currentClass: existingEnrollments[0]! };
    }

    const [classRows] = await connection.query<ClassRow[]>(
      "SELECT class_id, class_name, class_code, teacher_id FROM classes WHERE class_id = ? LIMIT 1 FOR UPDATE",
      [classId],
    );
    if (classRows.length === 0) {
      await connection.rollback();
      return { error: "class_not_found" as const };
    }

    const classInfo = classRows[0]!;
    if (classInfo.class_code) {
      if (!classCode) {
        await connection.rollback();
        return { error: "code_required" as const };
      }
      if (classCode !== classInfo.class_code) {
        await connection.rollback();
        return { error: "invalid_code" as const };
      }
    }

    await connection.query("INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)", [studentId, classId]);
    await connection.commit();

    return { data: { class_id: classInfo.class_id, class_name: classInfo.class_name, teacher_id: classInfo.teacher_id } };
  } catch (error) {
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}

export async function getJoinableClasses(studentId: number) {
  const [[existingEnrollment]] = await pool.query<EnrollmentRow[]>(
    "SELECT e.class_id FROM enrollments e WHERE e.student_id = ? LIMIT 1",
    [studentId],
  );
  if (existingEnrollment) {
    return { error: "already_enrolled" as const, currentClassId: existingEnrollment.class_id };
  }

  const [rows] = await pool.query<RowDataPacket[]>(
    `SELECT c.class_id, c.class_name, c.class_code IS NOT NULL AS requires_class_code, c.created_at,
            teacher.user_id AS teacher_id, teacher.full_name AS teacher_name, teacher.email AS teacher_email
     FROM classes c LEFT JOIN users teacher ON teacher.user_id = c.teacher_id ORDER BY c.class_id ASC`,
  );
  return { data: rows };
}

export async function addAssistantToClass(classId: number, assistantId: number, userId: number, userRole: string) {
  const { allowed, teacherId } = await checkClassAccess(pool, classId, userId, userRole, false);
  if (teacherId === null) return { error: "class_not_found" as const };
  if (!allowed) return { error: "forbidden" as const };

  const [userRows] = await pool.query<UserRow[]>("SELECT role FROM users WHERE user_id = ?", [assistantId]);
  if (userRows.length === 0 || userRows[0]!.role !== "Assistant") return { error: "not_assistant" as const };

  try {
    await pool.query("INSERT INTO class_assistants (class_id, assistant_id) VALUES (?, ?)", [classId, assistantId]);
    return { success: true };
  } catch (error: unknown) {
    const err = error as { code?: string };
    if (err.code === "ER_DUP_ENTRY") return { error: "dup_entry" as const };
    throw error;
  }
}

export async function getClassAssistants(classId: number, userId: number, userRole: string) {
  const { allowed, teacherId } = await checkClassAccess(pool, classId, userId, userRole, true);
  if (teacherId === null) return { error: "not_found" as const };
  if (!allowed) return { error: "forbidden" as const };

  const [rows] = await pool.query<UserRow[]>(
    "SELECT u.user_id, u.username, u.full_name, u.email FROM users u JOIN class_assistants ca ON u.user_id = ca.assistant_id WHERE ca.class_id = ?",
    [classId],
  );
  return { data: rows };
}

export async function removeAssistantFromClass(classId: number, assistantId: number, userId: number, userRole: string) {
  const { allowed, teacherId } = await checkClassAccess(pool, classId, userId, userRole, false);
  if (teacherId === null) return { error: "class_not_found" as const };
  if (!allowed) return { error: "forbidden" as const };

  const [result] = await pool.query<ResultSetHeader>(
    "DELETE FROM class_assistants WHERE class_id = ? AND assistant_id = ?",
    [classId, assistantId],
  );
  if (result.affectedRows === 0) return { error: "assignment_not_found" as const };

  return { success: true };
}
