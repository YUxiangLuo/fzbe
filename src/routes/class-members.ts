import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool, PoolConnection } from "mysql2/promise";
import { getDefaultPasswordHash } from "../utils";

// Zod Schemas for Validation
const classIdParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, "classId must be a number"),
});

const classAssistantParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, "classId must be a number"),
  assistantId: z.string().regex(/^\d+$/, "assistantId must be a number"),
});

const classStudentParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, "classId must be a number"),
  studentId: z.string().regex(/^\d+$/, "studentId must be a number"),
});

const createStudentSchema = z.object({
  username: z.string().min(1, "学号 is required"),
  full_name: z.string().min(1, "姓名 is required"),
  email: z.string().regex(/^[^\s@]+@[^\s@]+\.[^\s@]+$/, "Invalid email format").optional(),
  phone_number: z.string().optional(),
});

const addAssistantSchema = z.object({
  assistant_id: z.number().int().min(1),
});

const joinClassSchema = z.object({
  class_code: z.string().optional(), // Optional class code for verification
});

const enrollStudentSchema = z.object({
  student_id: z.number().int().positive("Student ID must be a positive integer"),
});

export default (pool: Pool, JWT_SECRET: string) => {
  const classMembersApi = new Hono();
  type Queryable = Pool | PoolConnection;

  const isAssistantAssignedToClass = async (
    db: Queryable,
    classId: number,
    assistantId: number,
  ): Promise<boolean> => {
    const [[assignment]]: any[] = await db.query(
      "SELECT 1 FROM class_assistants WHERE class_id = ? AND assistant_id = ? LIMIT 1",
      [classId, assistantId],
    );
    return Boolean(assignment);
  };

  const checkClassAccess = async (
    db: Queryable,
    classId: number,
    payload: any,
    allowAssistants: boolean = true,
  ): Promise<{ allowed: boolean; teacherId: number | null }> => {
    const [classRows]: any[] = await db.query(
      "SELECT teacher_id FROM classes WHERE class_id = ?",
      [classId],
    );

    if (classRows.length === 0) {
      return { allowed: false, teacherId: null };
    }

    const teacherId = classRows[0].teacher_id;
    let isAllowed = false;

    if (payload.role === "Admin" || payload.sub === teacherId) {
      isAllowed = true;
    } else if (allowAssistants && payload.role === "Assistant") {
      const assigned = await isAssistantAssignedToClass(db, classId, payload.sub);
      if (assigned) {
        isAllowed = true;
      }
    }

    return { allowed: isAllowed, teacherId };
  };

  // POST /api/v1/classes/:classId/students - 创建学生并加入班级
  classMembersApi.post(
    "/api/v1/classes/:classId/students",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    zValidator("json", createStudentSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      const { classId } = c.req.valid("param");
      const classIdNumber = Number.parseInt(classId, 10);
      const { username, full_name, email, phone_number } = c.req.valid("json");

      if (
        payload.role !== "Admin" &&
        payload.role !== "Teacher" &&
        payload.role !== "Assistant"
      ) {
        return c.json({ error: "Forbidden" }, 403);
      }

      const connection = await pool.getConnection();
      try {
        await connection.beginTransaction();

        const { allowed, teacherId } = await checkClassAccess(
          connection,
          classIdNumber,
          payload,
          true,
        );

        if (teacherId === null) {
          await connection.rollback();
          return c.json({ error: "Class not found" }, 404);
        }

        if (!allowed) {
          await connection.rollback();
          return c.json({ error: "Forbidden" }, 403);
        }

        const defaultPasswordHash = getDefaultPasswordHash();
        const studentEmail = email || `${username}@student.edu`;

        const [userResult]: any = await connection.query(
          "INSERT INTO users (username, password_hash, full_name, email, role, phone_number) VALUES (?, ?, ?, ?, 'Student', ?)",
          [
            username,
            defaultPasswordHash,
            full_name,
            studentEmail,
            phone_number || null,
          ],
        );

        const newStudentId = userResult.insertId;

        await connection.query(
          "INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)",
          [newStudentId, classIdNumber],
        );

        const [studentRows]: any[] = await connection.query(
          `SELECT u.user_id,
                  u.username,
                  u.full_name,
                  u.email,
                  u.phone_number,
                  u.role,
                  u.created_at
             FROM users u
            WHERE u.user_id = ?
            LIMIT 1`,
          [newStudentId],
        );

        if (studentRows.length === 0) {
          throw new Error("Failed to load created student record");
        }

        await connection.commit();

        return c.json(studentRows[0], 201);
      } catch (error: any) {
        await connection.rollback();

        if (error.code === "ER_DUP_ENTRY") {
          if (typeof error.message === "string") {
            if (error.message.includes("users.username")) {
              return c.json({ error: "学号已存在" }, 409);
            }
            if (error.message.includes("users.email")) {
              return c.json({ error: "邮箱已存在" }, 409);
            }
          }
          return c.json({ error: "学生创建冲突" }, 409);
        }

        console.error("Error creating student and enrolling in class:", error);
        throw error;
      } finally {
        connection.release();
      }
    },
  );

  // POST /api/v1/classes/:classId/enroll - 教师添加已存在的学生到班级
  classMembersApi.post(
    "/api/v1/classes/:classId/enroll",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    zValidator("json", enrollStudentSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const { student_id } = c.req.valid("json");

        // Only Teacher and Admin can enroll students
        if (payload.role !== "Teacher" && payload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        // Check class access
        const { allowed, teacherId } = await checkClassAccess(
          pool,
          classIdNumber,
          payload,
          false, // Only Teacher and Admin
        );

        if (teacherId === null) {
          return c.json({ error: "Class not found" }, 404);
        }

        if (!allowed) {
          return c.json({ error: "Forbidden: You can only enroll students to your own classes" }, 403);
        }

        // Verify the user exists and is a student
        const [userRows]: any[] = await pool.query(
          "SELECT user_id, role, full_name FROM users WHERE user_id = ?",
          [student_id],
        );

        if (userRows.length === 0) {
          return c.json({ error: "Student not found" }, 404);
        }

        if (userRows[0].role !== "Student") {
          return c.json({ error: "User is not a student" }, 400);
        }

        // Check if student is already in any class
        const [existingEnrollments]: any[] = await pool.query(
          `SELECT e.class_id, c.class_name
           FROM enrollments e
           JOIN classes c ON e.class_id = c.class_id
           WHERE e.student_id = ?
           LIMIT 1`,
          [student_id],
        );

        if (existingEnrollments.length > 0) {
          const currentClass = existingEnrollments[0];
          return c.json(
            {
              error: "Student already enrolled in a class",
              message: "This student is already enrolled in another class",
              current_class: {
                class_id: currentClass.class_id,
                class_name: currentClass.class_name,
              },
            },
            409,
          );
        }

        // Enroll the student
        await pool.query(
          "INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)",
          [student_id, classIdNumber],
        );

        return c.json(
          {
            message: "Student enrolled successfully",
            student: {
              user_id: userRows[0].user_id,
              full_name: userRows[0].full_name,
            },
            class_id: classIdNumber,
          },
          201,
        );
      } catch (error: any) {
        if (error.code === "ER_DUP_ENTRY") {
          return c.json({ error: "Student already enrolled in this class" }, 409);
        }
        console.error("Error enrolling student:", error);
        throw error;
      }
    },
  );

  // GET /api/v1/classes/:classId/students - 获取班级学生
  classMembersApi.get(
    "/api/v1/classes/:classId/students",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);

        const { allowed, teacherId } = await checkClassAccess(
          pool,
          classIdNumber,
          payload,
          true,
        );

        if (teacherId === null) {
          return c.json({ error: "Class not found" }, 404);
        }

        if (!allowed) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const [students] = await pool.query(
          `SELECT u.user_id, u.username, u.full_name, u.email, u.phone_number, u.role, u.created_at
             FROM users u
             JOIN enrollments e ON u.user_id = e.student_id
            WHERE e.class_id = ?
            ORDER BY u.full_name ASC`,
          [classIdNumber],
        );

        return c.json(students);
      } catch (error) {
        console.error("Error fetching students for class:", error);
        throw error;
      }
    },
  );

  // DELETE /api/v1/classes/:classId/students/:studentId - 移除学生
  classMembersApi.delete(
    "/api/v1/classes/:classId/students/:studentId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classStudentParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId, studentId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const studentIdNumber = Number.parseInt(studentId, 10);

        const { allowed, teacherId } = await checkClassAccess(
          pool,
          classIdNumber,
          payload,
          true,
        );

        if (teacherId === null) {
          return c.json({ error: "Class not found" }, 404);
        }

        if (!allowed) {
          return c.json({ error: "Forbidden" }, 403);
        }

        // Verify that the user is actually a student
        const [userRows]: any[] = await pool.query(
          "SELECT role FROM users WHERE user_id = ?",
          [studentIdNumber],
        );

        if (userRows.length === 0) {
          return c.json({ error: "User not found" }, 404);
        }

        if (userRows[0].role !== "Student") {
          return c.json({ error: "User is not a student" }, 400);
        }

        const [result]: any = await pool.query(
          "DELETE FROM enrollments WHERE class_id = ? AND student_id = ?",
          [classIdNumber, studentIdNumber],
        );

        if (result.affectedRows === 0) {
          return c.json({ error: "Enrollment not found" }, 404);
        }

        return c.json({ message: "Student removed from class" });
      } catch (error) {
        console.error("Error removing student from class:", error);
        throw error;
      }
    },
  );

  // POST /api/v1/classes/:classId/join - 学生自主加入班级
  classMembersApi.post(
    "/api/v1/classes/:classId/join",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    zValidator("json", joinClassSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const { class_code } = c.req.valid("json");

        // Verify user is a student
        if (payload.role !== "Student") {
          return c.json({ error: "Only students can join classes" }, 403);
        }

        const studentId = payload.sub;

        // Check if student is already in any class
        const [existingEnrollments]: any[] = await pool.query(
          `SELECT e.class_id, c.class_name, c.class_code
           FROM enrollments e
           JOIN classes c ON e.class_id = c.class_id
           WHERE e.student_id = ?
           LIMIT 1`,
          [studentId],
        );

        if (existingEnrollments.length > 0) {
          const currentClass = existingEnrollments[0];
          return c.json(
            {
              error: "Already enrolled in a class",
              message: "You are already enrolled in another class. Please leave your current class before joining a new one.",
              current_class: {
                class_id: currentClass.class_id,
                class_name: currentClass.class_name,
              },
            },
            409,
          );
        }

        // Check if class exists and get class info
        const [classRows]: any[] = await pool.query(
          "SELECT class_id, class_name, class_code, teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );

        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }

        const classInfo = classRows[0];

        // Verify class code if the class has one
        if (classInfo.class_code) {
          if (!class_code) {
            return c.json({ error: "Class code is required for this class" }, 400);
          }
          if (class_code !== classInfo.class_code) {
            return c.json({ error: "Invalid class code" }, 403);
          }
        }

        // Create enrollment
        await pool.query(
          "INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)",
          [studentId, classIdNumber],
        );

        // Return success with class info
        return c.json(
          {
            message: "Successfully joined class",
            class: {
              class_id: classInfo.class_id,
              class_name: classInfo.class_name,
              teacher_id: classInfo.teacher_id,
            },
          },
          201,
        );
      } catch (error) {
        console.error("Error joining class:", error);
        throw error;
      }
    },
  );

  // POST /api/v1/classes/:classId/assistants - 添加助教
  classMembersApi.post(
    "/api/v1/classes/:classId/assistants",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    zValidator("json", addAssistantSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const { assistant_id } = c.req.valid("json");

        const { allowed, teacherId } = await checkClassAccess(
          pool,
          classIdNumber,
          payload,
          false, // Only Admin and Teacher can add assistants
        );

        if (teacherId === null) {
          return c.json({ error: "Class not found" }, 404);
        }

        if (!allowed) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const [userRows]: any[] = await pool.query(
          "SELECT role FROM users WHERE user_id = ?",
          [assistant_id],
        );
        if (userRows.length === 0 || userRows[0].role !== "Assistant") {
          return c.json({ error: "User is not an assistant" }, 400);
        }

        await pool.query(
          "INSERT INTO class_assistants (class_id, assistant_id) VALUES (?, ?)",
          [classIdNumber, assistant_id],
        );

        return c.json({ message: "Assistant added successfully" }, 201);
      } catch (error: any) {
        if (error.code === "ER_DUP_ENTRY") {
          return c.json({ error: "Assistant already assigned to this class" }, 409);
        }
        console.error("Error adding assistant to class:", error);
        throw error;
      }
    },
  );

  // GET /api/v1/classes/:classId/assistants - 获取班级助教
  classMembersApi.get(
    "/api/v1/classes/:classId/assistants",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);

        const { allowed, teacherId } = await checkClassAccess(
          pool,
          classIdNumber,
          payload,
          true,
        );

        if (teacherId === null) {
          return c.json({ error: "Class not found" }, 404);
        }

        if (!allowed) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const [rows] = await pool.query(
          `SELECT u.user_id, u.username, u.full_name, u.email
           FROM users u
           JOIN class_assistants ca ON u.user_id = ca.assistant_id
           WHERE ca.class_id = ?`,
          [classIdNumber],
        );

        return c.json(rows);
      } catch (error) {
        console.error("Error fetching assistants for class:", error);
        throw error;
      }
    },
  );

  // DELETE /api/v1/classes/:classId/assistants/:assistantId - 移除助教
  classMembersApi.delete(
    "/api/v1/classes/:classId/assistants/:assistantId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classAssistantParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId, assistantId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const assistantIdNumber = Number.parseInt(assistantId, 10);

        const { allowed, teacherId } = await checkClassAccess(
          pool,
          classIdNumber,
          payload,
          false, // Only Admin and Teacher can remove assistants
        );

        if (teacherId === null) {
          return c.json({ error: "Class not found" }, 404);
        }

        if (!allowed) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const [result]: any = await pool.query(
          "DELETE FROM class_assistants WHERE class_id = ? AND assistant_id = ?",
          [classIdNumber, assistantIdNumber],
        );

        if (result.affectedRows === 0) {
          return c.json({ error: "Assignment not found" }, 404);
        }

        return c.json({ message: "Assistant removed successfully" });
      } catch (error) {
        console.error("Error removing assistant from class:", error);
        throw error;
      }
    },
  );

  return classMembersApi;
};
