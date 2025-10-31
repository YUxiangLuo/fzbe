import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool, PoolConnection } from "mysql2/promise";

// Zod Schemas for Validation
const classIdParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, { message: "classId must be a number" }),
});

const classAssistantParamSchema = z.object({
  classId: z.string().regex(/^\d+$/),
  assistantId: z.string().regex(/^\d+$/),
});

const classStudentParamSchema = z.object({
  classId: z.string().regex(/^\d+$/),
  studentId: z.string().regex(/^\d+$/),
});

const createStudentSchema = z.object({
  username: z.string().min(1, { message: "学号 is required" }),
  full_name: z.string().min(1, { message: "姓名 is required" }),
  email: z.string().email().optional(),
  phone_number: z.string().optional(),
});

const addAssistantSchema = z.object({
  assistant_id: z.number().int().min(1),
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

  // POST /api/classes/:classId/students - 创建学生并加入班级
  classMembersApi.post(
    "/api/classes/:classId/students",
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

        const [classRows]: any[] = await connection.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );

        if (classRows.length === 0) {
          await connection.rollback();
          return c.json({ error: "Class not found" }, 404);
        }

        const classOwner = classRows[0].teacher_id;
        let isAllowed = false;
        if (payload.role === "Admin" || payload.sub === classOwner) {
          isAllowed = true;
        } else if (payload.role === "Assistant") {
          const assigned = await isAssistantAssignedToClass(
            connection,
            classIdNumber,
            payload.sub,
          );
          if (assigned) {
            isAllowed = true;
          }
        }

        if (!isAllowed) {
          await connection.rollback();
          return c.json({ error: "Forbidden" }, 403);
        }

        const defaultPasswordHash =
          "$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG";
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

  // GET /api/classes/:classId/students - 获取班级学生
  classMembersApi.get(
    "/api/classes/:classId/students",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);

        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );

        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }

        const classOwner = classRows[0].teacher_id;
        let isAllowed = false;
        if (payload.role === "Admin" || payload.sub === classOwner) {
          isAllowed = true;
        } else if (payload.role === "Assistant") {
          const [[assignment]]: any[] = await pool.query(
            "SELECT 1 FROM class_assistants WHERE class_id = ? AND assistant_id = ?",
            [classIdNumber, payload.sub],
          );
          if (assignment) {
            isAllowed = true;
          }
        }

        if (!isAllowed) {
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

  // DELETE /api/classes/:classId/students/:studentId - 移除学生
  classMembersApi.delete(
    "/api/classes/:classId/students/:studentId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classStudentParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId, studentId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const studentIdNumber = Number.parseInt(studentId, 10);

        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );

        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }

        const classOwner = classRows[0].teacher_id;
        let isAllowed = false;
        if (payload.role === "Admin" || payload.sub === classOwner) {
          isAllowed = true;
        } else if (payload.role === "Assistant") {
          const assigned = await isAssistantAssignedToClass(
            pool,
            classIdNumber,
            payload.sub,
          );
          if (assigned) {
            isAllowed = true;
          }
        }

        if (!isAllowed) {
          return c.json({ error: "Forbidden" }, 403);
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

  // POST /api/classes/:classId/assistants - 添加助教
  classMembersApi.post(
    "/api/classes/:classId/assistants",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    zValidator("json", addAssistantSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const { assistant_id } = c.req.valid("json");

        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );
        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }
        if (payload.role !== "Admin" && payload.sub !== classRows[0].teacher_id) {
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

  // GET /api/classes/:classId/assistants - 获取班级助教
  classMembersApi.get(
    "/api/classes/:classId/assistants",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);

        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );
        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }

        const classOwner = classRows[0].teacher_id;
        let isAllowed = false;
        if (payload.role === "Admin" || payload.sub === classOwner) {
          isAllowed = true;
        } else if (payload.role === "Assistant") {
          const [[assignment]]: any[] = await pool.query(
            "SELECT 1 FROM class_assistants WHERE class_id = ? AND assistant_id = ?",
            [classIdNumber, payload.sub],
          );
          if (assignment) {
            isAllowed = true;
          }
        }

        if (!isAllowed) {
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

  // DELETE /api/classes/:classId/assistants/:assistantId - 移除助教
  classMembersApi.delete(
    "/api/classes/:classId/assistants/:assistantId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classAssistantParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId, assistantId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const assistantIdNumber = Number.parseInt(assistantId, 10);

        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );
        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }
        if (payload.role !== "Admin" && payload.sub !== classRows[0].teacher_id) {
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
