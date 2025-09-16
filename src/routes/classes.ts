import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";

// Zod Schemas for Validation
const paramsSchema = z.object({
  id: z.string().regex(/^\d+$/, { message: "ID must be a number" }),
});

const createClassSchema = z.object({
  class_name: z.string().min(1),
  class_code: z.string().optional(),
});

const enrollSchema = z.object({
  class_code: z.string().min(1, { message: "Class code is required" }),
});

const addAssistantSchema = z.object({
  assistant_id: z.number().int().min(1),
});

export default (pool: Pool, JWT_SECRET: string) => {
  const classesApi = new Hono();

  // 受保护的路由: 创建新班级
  classesApi.post(
    "/",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", createClassSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Teacher" && payload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { class_name, class_code } = c.req.valid("json");
        const teacher_id = payload.sub;

        const [result]: any = await pool.query(
          "INSERT INTO classes (class_name, class_code, teacher_id) VALUES (?, ?, ?)",
          [class_name, class_code || null, teacher_id],
        );

        const newClass = {
          class_id: result.insertId,
          class_name,
          class_code,
          teacher_id,
        };
        return c.json(newClass, 201);
      } catch (error: any) {
        if (error.code === "ER_DUP_ENTRY") {
          return c.json({ error: "Class code already exists" }, 409);
        }
        console.error("Error creating class:", error);
        throw error;
      }
    },
  );

  // 受保护的路由: 学生使用班级代码加入班级
  classesApi.post(
    "/enroll",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", enrollSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Student") {
          return c.json({ error: "Only students can enroll in classes" }, 403);
        }

        const { class_code } = c.req.valid("json");
        const [classRows]: any[] = await pool.query(
          "SELECT class_id FROM classes WHERE class_code = ?",
          [class_code],
        );
        const targetClass = classRows[0];

        if (!targetClass) {
          return c.json({ error: "Class not found" }, 404);
        }

        const student_id = payload.sub;
        await pool.query(
          "INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)",
          [student_id, targetClass.class_id],
        );

        return c.json({ message: "Enrolled successfully" }, 200);
      } catch (error: any) {
        if (error.code === "ER_DUP_ENTRY") {
          return c.json({ error: "Already enrolled in this class" }, 409);
        }
        console.error("Error enrolling in class:", error);
        throw error;
      }
    },
  );

  // 受保护的路由: 获取班级列表
  classesApi.get("/", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const payload = c.get("jwtPayload");
      let query = "";
      const params: (string | number)[] = [payload.sub];

      switch (payload.role) {
        case "Admin":
          query =
            "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id";
          params.pop();
          break;
        case "Teacher":
          query =
            "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id WHERE c.teacher_id = ?";
          break;
        case "Student":
          query =
            "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN enrollments e ON c.class_id = e.class_id JOIN users u ON c.teacher_id = u.user_id WHERE e.student_id = ?";
          break;
        default:
          return c.json({ error: "Forbidden" }, 403);
      }

      const [rows] = await pool.query(query, params);
      return c.json(rows);
    } catch (error) {
      console.error("Error fetching classes:", error);
      throw error;
    }
  });

  // 受保护的路由: 获取单个班级详情 (包括学生列表)
  classesApi.get(
    "/:id",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", paramsSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { id } = c.req.valid("param");
        const classId = parseInt(id, 10);

        const [classRows]: any[] = await pool.query(
          "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id WHERE c.class_id = ?",
          [classId],
        );
        const classInfo = classRows[0];

        if (!classInfo) {
          return c.json({ error: "Class not found" }, 404);
        }

        let isAllowed = false;
        if (payload.role === "Admin" || payload.sub === classInfo.teacher_id) {
          isAllowed = true;
        } else if (payload.role === "Student") {
          const [enrollmentRows]: any[] = await pool.query(
            "SELECT enrollment_id FROM enrollments WHERE student_id = ? AND class_id = ?",
            [payload.sub, classId],
          );
          if (enrollmentRows.length > 0) {
            isAllowed = true;
          }
        }

        if (!isAllowed) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const [studentRows] = await pool.query(
          "SELECT u.user_id, u.username, u.full_name, u.email FROM users u JOIN enrollments e ON u.user_id = e.student_id WHERE e.class_id = ?",
          [classId],
        );

        const response = { ...classInfo, students: studentRows };
        return c.json(response);
      } catch (error) {
        console.error("Error fetching class details:", error);
        throw error;
      }
    },
  );

  // 获取指定教师创建的所有班级 前缀 /classes
  classesApi.get(
    "/teacher/:id/classes",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", paramsSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { id } = c.req.valid("param");
        const teacherId = parseInt(id, 10);

        // Authorization check
        if (payload.role !== "Admin" && payload.sub !== teacherId) {
          return c.json({ error: "Forbidden" }, 403);
        }

        // Check if the user is actually a teacher
        const [teacherRows]: any[] = await pool.query(
          "SELECT role FROM users WHERE user_id = ? AND role = 'Teacher'",
          [teacherId],
        );

        if (teacherRows.length === 0 && payload.role !== "Admin") {
          // Allow admin to query even if the ID is not a teacher,
          // but a teacher must be a teacher.
          return c.json({ error: "Target user is not a teacher" }, 404);
        }

        const [classRows] = await pool.query(
          "SELECT * FROM classes WHERE teacher_id = ?",
          [teacherId],
        );

        return c.json(classRows);
      } catch (error) {
        console.error("Error fetching classes for teacher:", error);
        throw error;
      }
    },
  );

  // --- Assistant Management for a Class ---

  // Add an assistant to a class
  classesApi.post(
    "/:id/assistants",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", paramsSchema),
    zValidator("json", addAssistantSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { id } = c.req.valid("param");
        const classId = parseInt(id, 10);
        const { assistant_id } = c.req.valid("json");

        // 1. Verify class exists and user is the teacher or admin
        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classId],
        );
        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }
        if (payload.role !== "Admin" && payload.sub !== classRows[0].teacher_id) {
          return c.json({ error: "Forbidden" }, 403);
        }

        // 2. Verify the user to be added is an assistant
        const [userRows]: any[] = await pool.query(
          "SELECT role FROM users WHERE user_id = ?",
          [assistant_id],
        );
        if (userRows.length === 0 || userRows[0].role !== "Assistant") {
          return c.json({ error: "User is not an assistant" }, 400);
        }

        // 3. Insert the assignment
        await pool.query(
          "INSERT INTO class_assistants (class_id, assistant_id) VALUES (?, ?)",
          [classId, assistant_id],
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

  // Remove an assistant from a class
  classesApi.delete(
    "/:class_id/assistants/:assistant_id",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", z.object({
        class_id: z.string().regex(/^\d+$/),
        assistant_id: z.string().regex(/^\d+$/),
    })),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { class_id, assistant_id } = c.req.valid("param");
        const classId = parseInt(class_id, 10);
        const assistantId = parseInt(assistant_id, 10);

        // 1. Verify class exists and user is the teacher or admin
        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classId],
        );
        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }
        if (payload.role !== "Admin" && payload.sub !== classRows[0].teacher_id) {
          return c.json({ error: "Forbidden" }, 403);
        }

        // 2. Delete the assignment
        const [result]: any = await pool.query(
          "DELETE FROM class_assistants WHERE class_id = ? AND assistant_id = ?",
          [classId, assistantId],
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

  // Get all assistants for a class
  classesApi.get(
    "/:id/assistants",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", paramsSchema),
    async (c) => {
      try {
        const { id } = c.req.valid("param");
        const classId = parseInt(id, 10);

        const [rows] = await pool.query(
          `SELECT u.user_id, u.username, u.full_name, u.email 
           FROM users u 
           JOIN class_assistants ca ON u.user_id = ca.assistant_id 
           WHERE ca.class_id = ?`,
          [classId],
        );

        return c.json(rows);
      } catch (error) {
        console.error("Error fetching assistants for class:", error);
        throw error;
      }
    },
  );

  return classesApi;
};
