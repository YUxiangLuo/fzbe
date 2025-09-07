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

  return classesApi;
}
