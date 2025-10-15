import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";
import csv from "csv-parser";
import { createReadStream } from "node:fs";
import { unlink, mkdir } from "node:fs/promises";
import path from "path";
import { createUploadValidator } from "../utils";

// Zod Schemas for Validation
const classIdParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, { message: "classId must be a number" }),
});

const teacherIdParamSchema = z.object({
  teacherId: z.string().regex(/^\d+$/, { message: "teacherId must be a number" }),
});

const classAssistantParamSchema = z.object({
  classId: z.string().regex(/^\d+$/),
  assistantId: z.string().regex(/^\d+$/),
});

const studentIdParamSchema = z.object({
  studentId: z.string().regex(/^\d+$/, { message: "studentId must be a number" }),
});

const classStudentParamSchema = z.object({
  classId: z.string().regex(/^\d+$/),
  studentId: z.string().regex(/^\d+$/),
});

const createStudentSchema = z.object({
  username: z.string().min(1, { message: "学号 is required" }), // 学号作为username
  full_name: z.string().min(1, { message: "姓名 is required" }),
  email: z.string().email().optional(), // 可选，如果不提供则自动生成
  phone_number: z.string().optional(),
});

const reportIdParamSchema = z.object({
  reportId: z.string().regex(/^\d+$/),
});

const createClassSchema = z.object({
  class_name: z.string().min(1),
  class_code: z.string().optional(),
});

const updateClassSchema = z
  .object({
    class_name: z.string().min(1).optional(),
    class_code: z.string().optional(),
  })
  .refine((data) => data.class_name !== undefined || data.class_code !== undefined, {
    message: "At least one field (class_name or class_code) must be provided",
  });

const enrollSchema = z.object({
  class_code: z.string().min(1, { message: "Class code is required" }),
});

const addAssistantSchema = z.object({
  assistant_id: z.number().int().min(1),
});

const updateReportSchema = z
  .object({
    grade: z.number().min(0).max(100).optional(),
    feedback: z.string().trim().max(2000).optional(),
  })
  .refine((data) => data.grade !== undefined || data.feedback !== undefined, {
    message: "At least one of grade or feedback must be provided",
  });

const gradeWeightNumeric = z.coerce.number().min(0);

const expFlowKeys = [
  "exp_flow_demand_data_preparation",
  "exp_flow_demand_descriptive_stats",
  "exp_flow_demand_model_selection",
  "exp_flow_demand_generate_results",
  "exp_flow_production_inventory_calc",
  "exp_flow_production_service_level",
  "exp_flow_production_variable_calc",
  "exp_flow_production_plan_creation",
  "exp_flow_report_submission",
];

const gradeWeightFieldNames = [
  "exp_flow",
  ...expFlowKeys,
  "knowledge_test",
  "model_quality",
  "report_quality",
];

const gradeWeightsBaseSchema = z
  .object(
    gradeWeightFieldNames.reduce((acc, key) => {
      acc[key] = gradeWeightNumeric;
      return acc;
    }, {} as Record<string, z.ZodTypeAny>),
  )
  .refine((data) => {
    const expFlowSum = expFlowKeys.reduce((sum, key) => sum + Number(data[key as keyof typeof data] ?? 0), 0);
    return Math.abs(expFlowSum - 100) < 0.0001;
  }, {
    message: "所有实验流程子权重之和必须等于 100",
    path: ["exp_flow"],
  })
  .refine((data) => {
    const flowSum = Number(data.exp_flow ?? 0);
    const otherSum = Number(data.knowledge_test ?? 0) + Number(data.model_quality ?? 0) + Number(data.report_quality ?? 0);
    return Math.abs(flowSum + otherSum - 100) < 0.0001;
  }, {
    message: "实验流程 + 知识点测试 + 模型质量 + 实验报告质量之和必须等于 100",
    path: ["exp_flow"],
  });

const gradeWeightsCreateSchema = gradeWeightsBaseSchema.extend({
  teacher_id: z.coerce.number().int().min(1).optional(),
});

const gradeWeightsUpdateSchema = gradeWeightsBaseSchema.extend({
  teacher_id: z.coerce.number().int().min(1).optional(),
});

const gradeWeightsQuerySchema = z.object({
  teacherId: z.coerce.number().int().min(1).optional(),
});

const normalizeGradeWeightRow = (row: any) => {
  const result: Record<string, number | null> = { teacher_id: row.teacher_id };
  for (const field of gradeWeightFieldNames) {
    result[field] = row[field] !== null && row[field] !== undefined ? Number(row[field]) : null;
  }
  return result;
};

const normalizeExperimentGradeRow = (row: any) => {
  const normalized: Record<string, any> = {
    grade_id: Number(row.grade_id),
    experiment_id: Number(row.experiment_id),
    student_id: Number(row.student_id),
    student_username: row.student_username,
    student_full_name: row.student_full_name,
    grader_id: row.grader_id !== null && row.grader_id !== undefined ? Number(row.grader_id) : null,
    grader_name: row.grader_name ?? null,
    created_at: row.created_at,
  };

  for (const field of gradeWeightFieldNames) {
    normalized[field] = row[field] !== null && row[field] !== undefined ? Number(row[field]) : null;
  }

  return normalized;
};

const csvUploadValidator = createUploadValidator({
  allowedMimeTypes: ["text/csv"],
  maxSizeMb: 50,
});

export default (pool: Pool, JWT_SECRET: string) => {
  const classesApi = new Hono();

  // 受保护的路由: 创建新班级 (支持上传学生名单CSV)
  classesApi.post(
    "/classes",
    jwt({ secret: JWT_SECRET }),
    csvUploadValidator,
    async (c) => {
      let tempCsvPath: string | null = null;

      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Teacher" && payload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const body = c.get("parsedBody");
        const class_name = body["class_name"];
        const class_code = body["class_code"];
        const studentListFile = body["file"]; // file from middleware

        if (!class_name || typeof class_name !== "string") {
          return c.json({ error: "class_name is required" }, 400);
        }

        const teacher_id = payload.sub;
        const connection = await pool.getConnection();

        try {
          await connection.beginTransaction();

          // 1. Create the class
          const [result]: any = await connection.query(
            "INSERT INTO classes (class_name, class_code, teacher_id) VALUES (?, ?, ?)",
            [class_name, typeof class_code === "string" ? class_code || null : null, teacher_id],
          );

          const classId = result.insertId;
          let createdStudentsCount = 0;
          const createdStudents: any[] = [];
          const errors: string[] = [];

          // 2. If a student list CSV file is provided, parse and create students
          if (studentListFile instanceof File) {
            // Save CSV to temporary location
            const tempDir = path.resolve(import.meta.dir, "../../uploads", "temp");
            await mkdir(tempDir, { recursive: true });
            const uniqueFilename = `${Date.now()}-${studentListFile.name}`;
            tempCsvPath = path.join(tempDir, uniqueFilename);
            await Bun.write(tempCsvPath, studentListFile);

            // Fixed password hash
            const defaultPasswordHash = "$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG";

            // Parse CSV and collect student records
            const studentRecords: Array<{ studentId: string; fullName: string; email: string }> = [];

            await new Promise<void>((resolve, reject) => {
              createReadStream(tempCsvPath!)
                .pipe(csv({
                  mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim()
                }))
                .on("data", (row: Record<string, string>) => {
                  const studentId = row["学号"]?.trim();
                  const fullName = row["姓名"]?.trim();

                  if (studentId && fullName) {
                    // Generate email as studentId@student.edu
                    const email = `${studentId}@student.edu`;
                    studentRecords.push({ studentId, fullName, email });
                  }
                })
                .on("end", resolve)
                .on("error", reject);
            });

            // 3. Batch insert students and enroll them in the class
            for (const record of studentRecords) {
              try {
                // Check if student already exists
                const [existingUser]: any[] = await connection.query(
                  "SELECT user_id FROM users WHERE username = ?",
                  [record.studentId]
                );

                let userId: number;

                if (existingUser.length > 0) {
                  // Student already exists, check if enrolled in any other class
                  userId = existingUser[0].user_id;

                  const [enrollments]: any[] = await connection.query(
                    "SELECT class_id FROM enrollments WHERE student_id = ?",
                    [userId]
                  );

                  if (enrollments.length > 0) {
                    // Student is already in another class, reject
                    throw new Error(`学生 ${record.studentId} (${record.fullName}) 已在其他班级中，无法重复添加`);
                  }
                  // Student exists but not enrolled anywhere, can reuse the account
                } else {
                  // Create new student user
                  const [userResult]: any = await connection.query(
                    "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?, ?, ?, ?, 'Student')",
                    [record.studentId, defaultPasswordHash, record.fullName, record.email]
                  );
                  userId = userResult.insertId;
                  createdStudentsCount++;
                }

                // Enroll student in the class
                await connection.query(
                  "INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)",
                  [userId, classId]
                );
                createdStudents.push({
                  user_id: userId,
                  username: record.studentId,
                  full_name: record.fullName,
                  email: record.email
                });
              } catch (studentError: any) {
                if (studentError.code === "ER_DUP_ENTRY") {
                  errors.push(`学生 ${record.studentId} 已存在但无法加入班级`);
                } else {
                  errors.push(`创建学生 ${record.studentId} 失败: ${studentError.message}`);
                }
              }
            }
          }

          await connection.commit();

          const [newClassRows]: any[] = await connection.query(
            "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id WHERE c.class_id = ?",
            [classId],
          );

          return c.json({
            class: newClassRows[0],
            students_created: createdStudentsCount,
            students_enrolled: createdStudents.length,
            students: createdStudents,
            errors: errors.length > 0 ? errors : undefined
          }, 201);

        } catch (error) {
          await connection.rollback();
          throw error;
        } finally {
          connection.release();
        }
      } catch (error: any) {
        if (error.code === "ER_DUP_ENTRY") {
          return c.json({ error: "Class code already exists" }, 409);
        }
        console.error("Error creating class:", error);
        throw error;
      } finally {
        // Cleanup temporary CSV file
        if (tempCsvPath) {
          await unlink(tempCsvPath).catch((err) =>
            console.error("Failed to cleanup temp CSV file:", err)
          );
        }
      }
    },
  );

  // 受保护的路由: 获取班级列表
  classesApi.get("/classes", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin") {
        return c.json({ error: "Forbidden" }, 403);
      }

      const [rows] = await pool.query(
        "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id",
      );
      return c.json(rows);
    } catch (error) {
      console.error("Error fetching classes:", error);
      throw error;
    }
  });

  // 受保护的路由: 获取单个班级详情 (包括学生列表和助教列表)
  classesApi.get(
    "/classes/:classId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);

        const [classRows]: any[] = await pool.query(
          "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id WHERE c.class_id = ?",
          [classIdNumber],
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
            [payload.sub, classIdNumber],
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
          [classIdNumber],
        );

        const [assistantRows] = await pool.query(
          `SELECT u.user_id, u.username, u.full_name, u.email
             FROM users u
             JOIN class_assistants ca ON u.user_id = ca.assistant_id
            WHERE ca.class_id = ?`,
          [classIdNumber],
        );

        const response = { ...classInfo, students: studentRows, assistants: assistantRows };
        return c.json(response);
      } catch (error) {
        console.error("Error fetching class details:", error);
        throw error;
      }
    },
  );

  classesApi.put(
    "/classes/:classId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    zValidator("json", updateClassSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const { class_name, class_code } = c.req.valid("json");

        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );
        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }

        const classOwner = classRows[0].teacher_id;
        if (payload.role !== "Admin" && payload.sub !== classOwner) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const fields: string[] = [];
        const values: (string | number | null)[] = [];
        if (class_name !== undefined) {
          fields.push("class_name = ?");
          values.push(class_name);
        }
        if (class_code !== undefined) {
          fields.push("class_code = ?");
          values.push(class_code || null);
        }

        if (fields.length === 0) {
          return c.json({ error: "No fields to update" }, 400);
        }

        values.push(classIdNumber);

        try {
          const sql = `UPDATE classes SET ${fields.join(", ")} WHERE class_id = ?`;
          await pool.query(sql, values);
        } catch (error: any) {
          if (error.code === "ER_DUP_ENTRY") {
            return c.json({ error: "Class code already exists" }, 409);
          }
          throw error;
        }

        const [updatedRows]: any[] = await pool.query(
          "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id WHERE c.class_id = ?",
          [classIdNumber],
        );

        return c.json(updatedRows[0]);
      } catch (error) {
        console.error("Error updating class:", error);
        throw error;
      }
    },
  );

  // 获取指定教师创建的所有班级
  classesApi.get(
    "/teachers/:teacherId/classes",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", teacherIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { teacherId } = c.req.valid("param");
        const teacherIdNumber = Number.parseInt(teacherId, 10);

        // Authorization check
        if (payload.role !== "Admin" && payload.sub !== teacherIdNumber) {
          return c.json({ error: "Forbidden" }, 403);
        }

        // Check if the user is actually a teacher
        const [teacherRows]: any[] = await pool.query(
          "SELECT role FROM users WHERE user_id = ? AND role = 'Teacher'",
          [teacherIdNumber],
        );

        if (teacherRows.length === 0 && payload.role !== "Admin") {
          // Allow admin to query even if the ID is not a teacher,
          // but a teacher must be a teacher.
          return c.json({ error: "Target user is not a teacher" }, 404);
        }

        const [classRows] = await pool.query(
          "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id WHERE c.teacher_id = ?",
          [teacherIdNumber],
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
    "/classes/:classId/assistants",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    zValidator("json", addAssistantSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const { assistant_id } = c.req.valid("json");

        // 1. Verify class exists and user is the teacher or admin
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

  // Remove an assistant from a class
  classesApi.delete(
    "/classes/:classId/assistants/:assistantId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classAssistantParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId, assistantId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const assistantIdNumber = Number.parseInt(assistantId, 10);

        // 1. Verify class exists and user is the teacher or admin
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

        // 2. Delete the assignment
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

  // Get all assistants for a class
  classesApi.get(
    "/classes/:classId/assistants",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
      try {
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);

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

  // 创建学生并加入指定班级
  classesApi.post(
    "/classes/:classId/students",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    zValidator("json", createStudentSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin" && payload.role !== "Teacher") {
        return c.json({ error: "Forbidden" }, 403);
      }

      const { classId } = c.req.valid("param");
      const classIdNumber = Number.parseInt(classId, 10);
      const { username, full_name, email, phone_number } = c.req.valid("json");

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
        if (payload.role === "Teacher" && payload.sub !== classOwner) {
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

  classesApi.get(
    "/classes/:classId/students",
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
        if (payload.role !== "Admin" && payload.sub !== classOwner) {
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

  classesApi.get(
    "/classes/:classId/reports",
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
        if (payload.role !== "Admin" && payload.sub !== classOwner) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const [reports] = await pool.query(
          `SELECT er.report_id,
                  er.experiment_id,
                  er.student_id,
                  student.username AS student_username,
                  student.full_name AS student_full_name,
                  er.report_content,
                  er.pdf_file_path,
                  er.submitted_at,
                  er.grade,
                  er.feedback,
                  er.graded_by,
                  grader.full_name AS graded_by_full_name,
                  es.status AS experiment_status,
                  es.selected_industry,
                  es.selected_company,
                  es.selected_product
             FROM experiment_reports er
             JOIN experiment_status es
               ON er.experiment_id = es.experiment_id
              AND er.student_id = es.student_id
             JOIN enrollments e
               ON e.student_id = er.student_id
             JOIN users student
               ON student.user_id = er.student_id
             LEFT JOIN users grader
               ON grader.user_id = er.graded_by
            WHERE e.class_id = ?
            ORDER BY er.student_id ASC, er.submitted_at DESC`,
          [classIdNumber],
        );

        return c.json(reports);
      } catch (error) {
        console.error("Error fetching class reports:", error);
        throw error;
      }
    },
  );

  classesApi.get(
    "/classes/:classId/experiment-status",
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
        if (payload.role !== "Admin" && payload.sub !== classOwner) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const [statuses] = await pool.query(
          `SELECT es.experiment_id,
                  es.student_id,
                  student.username AS student_username,
                  student.full_name AS student_full_name,
                  es.start_time,
                  es.completion_time,
                  es.last_activity_at,
                  es.status,
                  es.selected_industry,
                  es.selected_company,
                  es.selected_product,
                  es.selected_best_model
             FROM experiment_status es
             JOIN enrollments e
               ON e.student_id = es.student_id
             JOIN users student
               ON student.user_id = es.student_id
            WHERE e.class_id = ?
            ORDER BY es.student_id ASC, es.start_time DESC`,
          [classIdNumber],
        );

        return c.json(statuses);
      } catch (error) {
        console.error("Error fetching class experiment status:", error);
        throw error;
      }
    },
  );

  classesApi.get(
    "/grade-weights",
    jwt({ secret: JWT_SECRET }),
    zValidator("query", gradeWeightsQuerySchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { teacherId } = c.req.valid("query");

        let targetTeacherId: number;

        if (payload.role === "Teacher") {
          targetTeacherId = payload.sub;
          if (teacherId && teacherId !== targetTeacherId) {
            return c.json({ error: "Forbidden" }, 403);
          }
        } else if (payload.role === "Admin") {
          if (!teacherId) {
            return c.json({ error: "teacherId is required for admin" }, 400);
          }
          targetTeacherId = teacherId;
        } else {
          return c.json({ error: "Forbidden" }, 403);
        }

        const [rows] = await pool.query(
          `SELECT teacher_id, ${gradeWeightFieldNames.join(", ")}
             FROM grade_weights
            WHERE teacher_id = ?
            LIMIT 1`,
          [targetTeacherId],
        );

        const record = (rows as any[])[0];
        if (!record) {
          return c.json({ error: "Grade weights not found" }, 404);
        }

        return c.json(normalizeGradeWeightRow(record));
      } catch (error) {
        console.error("Error fetching grade weights:", error);
        throw error;
      }
    },
  );

  classesApi.post(
    "/grade-weights",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", gradeWeightsCreateSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const body = c.req.valid("json");

        let teacherId = body.teacher_id;

        if (payload.role === "Teacher") {
          teacherId = payload.sub;
          if (body.teacher_id && body.teacher_id !== teacherId) {
            return c.json({ error: "Forbidden" }, 403);
          }
        } else if (payload.role === "Admin") {
          if (!teacherId) {
            return c.json({ error: "teacher_id is required" }, 400);
          }
        } else {
          return c.json({ error: "Forbidden" }, 403);
        }

        if (!teacherId) {
          return c.json({ error: "teacher_id is required" }, 400);
        }

        const [[existingRow]]: any[] = await pool.query(
          "SELECT teacher_id FROM grade_weights WHERE teacher_id = ?",
          [teacherId],
        );
        if (existingRow) {
          return c.json({ error: "Grade weights already exist" }, 409);
        }

        const insertSql = `INSERT INTO grade_weights (teacher_id, ${gradeWeightFieldNames.join(", ")}) VALUES (${["?", ...gradeWeightFieldNames.map(() => "?")].join(", ")})`;

        const values = [teacherId, ...gradeWeightFieldNames.map((field) => Number(body[field as keyof typeof body]))];

        await pool.query(insertSql, values);

        const [rows] = await pool.query(
          `SELECT teacher_id, ${gradeWeightFieldNames.join(", ")}
             FROM grade_weights
            WHERE teacher_id = ?
            LIMIT 1`,
          [teacherId],
        );

        return c.json(normalizeGradeWeightRow((rows as any[])[0]), 201);
      } catch (error) {
        console.error("Error creating grade weights:", error);
        throw error;
      }
    },
  );

  classesApi.put(
    "/grade-weights",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", gradeWeightsUpdateSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const body = c.req.valid("json");

        let teacherId = body.teacher_id;

        if (payload.role === "Teacher") {
          teacherId = payload.sub;
          if (body.teacher_id && body.teacher_id !== teacherId) {
            return c.json({ error: "Forbidden" }, 403);
          }
        } else if (payload.role === "Admin") {
          if (!teacherId) {
            return c.json({ error: "teacher_id is required" }, 400);
          }
        } else {
          return c.json({ error: "Forbidden" }, 403);
        }

        if (!teacherId) {
          return c.json({ error: "teacher_id is required" }, 400);
        }

        const [[existingRow]]: any[] = await pool.query(
          "SELECT teacher_id FROM grade_weights WHERE teacher_id = ?",
          [teacherId],
        );

        if (!existingRow) {
          return c.json({ error: "Grade weights not found" }, 404);
        }

        const updateSql = `UPDATE grade_weights SET ${gradeWeightFieldNames.map((field) => `${field} = ?`).join(", ")} WHERE teacher_id = ?`;
        const values = [...gradeWeightFieldNames.map((field) => Number(body[field as keyof typeof body])), teacherId];

        await pool.query(updateSql, values);

        const [rows] = await pool.query(
          `SELECT teacher_id, ${gradeWeightFieldNames.join(", ")}
             FROM grade_weights
            WHERE teacher_id = ?
            LIMIT 1`,
          [teacherId],
        );

        return c.json(normalizeGradeWeightRow((rows as any[])[0]));
      } catch (error) {
        console.error("Error updating grade weights:", error);
        throw error;
      }
    },
  );

  classesApi.get(
    "/classes/:classId/latest-grades",
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
        if (payload.role !== "Admin" && payload.sub !== classOwner) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const [rows] = await pool.query(
          `SELECT eg.grade_id,
                  eg.experiment_id,
                  eg.student_id,
                  student.username AS student_username,
                  student.full_name AS student_full_name,
                  eg.grader_id,
                  grader.full_name AS grader_name,
                  eg.created_at,
                  ${gradeWeightFieldNames.map((field) => `eg.${field}`).join(", ")}
             FROM experiment_grade eg
             JOIN experiment_status es
               ON es.experiment_id = eg.experiment_id
             JOIN enrollments e
               ON e.student_id = es.student_id
             JOIN users student
               ON student.user_id = eg.student_id
             LEFT JOIN users grader
               ON grader.user_id = eg.grader_id
             JOIN (
                  SELECT eg.student_id,
                         MAX(eg.created_at) AS latest_created
                    FROM experiment_grade eg
                    JOIN experiment_status es
                      ON es.experiment_id = eg.experiment_id
                    JOIN enrollments e
                      ON e.student_id = es.student_id
                   WHERE e.class_id = ?
                   GROUP BY eg.student_id
             ) latest
               ON latest.student_id = eg.student_id
              AND latest.latest_created = eg.created_at
            WHERE e.class_id = ?
            ORDER BY eg.student_id ASC`,
          [classIdNumber, classIdNumber],
        );

        const normalized = (rows as any[]).map(normalizeExperimentGradeRow);
        return c.json(normalized);
      } catch (error) {
        console.error("Error fetching latest experiment grades:", error);
        throw error;
      }
    },
  );

  classesApi.put(
    "/reports/:reportId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", reportIdParamSchema),
    zValidator("json", updateReportSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin" && payload.role !== "Teacher") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { reportId } = c.req.valid("param");
        const reportIdNumber = Number.parseInt(reportId, 10);

        const [reportRows]: any[] = await pool.query(
          "SELECT report_id, experiment_id, student_id FROM experiment_reports WHERE report_id = ?",
          [reportIdNumber],
        );

        if (reportRows.length === 0) {
          return c.json({ error: "Report not found" }, 404);
        }

        const reportStudentId = reportRows[0].student_id;

        if (payload.role === "Teacher") {
          const [authRows]: any[] = await pool.query(
            `SELECT 1
               FROM classes c
               JOIN enrollments e ON e.class_id = c.class_id
              WHERE c.teacher_id = ?
                AND e.student_id = ?
              LIMIT 1`,
            [payload.sub, reportStudentId],
          );

          if (authRows.length === 0) {
            return c.json({ error: "Forbidden" }, 403);
          }
        }

        const { grade, feedback } = c.req.valid("json");
        const updateFields: string[] = [];
        const updateValues: (number | string | null)[] = [];

        if (grade !== undefined) {
          updateFields.push("grade = ?");
          updateValues.push(grade);
        }
        if (feedback !== undefined) {
          updateFields.push("feedback = ?");
          updateValues.push(feedback);
        }

        updateFields.push("graded_by = ?");
        updateValues.push(payload.sub);

        updateValues.push(reportIdNumber);

        const updateSql = `
          UPDATE experiment_reports
             SET ${updateFields.join(", ")}
           WHERE report_id = ?
        `;

        const [updateResult]: any = await pool.query(updateSql, updateValues);

        if (updateResult.affectedRows === 0) {
          return c.json({ error: "Report not found or not updated" }, 404);
        }

        const [updatedRows] = await pool.query(
          `SELECT er.report_id,
                  er.experiment_id,
                  er.student_id,
                  student.username AS student_username,
                  student.full_name AS student_full_name,
                  er.report_content,
                  er.pdf_file_path,
                  er.submitted_at,
                  er.grade,
                  er.feedback,
                  er.graded_by,
                  grader.full_name AS graded_by_full_name,
                  es.status AS experiment_status,
                  es.selected_industry,
                  es.selected_company,
                  es.selected_product
             FROM experiment_reports er
             JOIN experiment_status es
               ON er.experiment_id = es.experiment_id
              AND er.student_id = es.student_id
             JOIN users student
               ON student.user_id = er.student_id
             LEFT JOIN users grader
               ON grader.user_id = er.graded_by
            WHERE er.report_id = ?
            LIMIT 1`,
          [reportIdNumber],
        );

        const updatedReport = (updatedRows as any[])[0];
        return c.json(updatedReport);
      } catch (error) {
        console.error("Error updating class report grading:", error);
        throw error;
      }
    },
  );

  classesApi.get(
    "/classes/:classId/step-events",
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

        if (payload.role === "Teacher") {
          if (payload.sub !== classRows[0].teacher_id) {
            return c.json({ error: "Forbidden" }, 403);
          }
        } else if (payload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const sql = `
WITH ClassStudents AS (
  -- 1. Get all students in the class
  SELECT u.user_id AS student_id, u.username, u.full_name
  FROM users u JOIN enrollments e ON u.user_id = e.student_id
  WHERE e.class_id = ? AND u.role = 'Student'
),
LatestStudentExperiment AS (
  -- 2. Find the latest experiment for each student
  SELECT es.experiment_id, es.student_id,
         ROW_NUMBER() OVER(PARTITION BY es.student_id ORDER BY es.start_time DESC) as rn
  FROM experiment_status es
  WHERE es.student_id IN (SELECT student_id FROM ClassStudents)
),
NumberedEvents AS (
  -- 3. Get all events for the latest experiments, ordered chronologically
  SELECT
    lse.student_id,
    ese.experiment_id,
    ese.step_order,
    ese.event_type,
    ese.event_timestamp,
    -- Use COALESCE on a 2-argument LAG for better MariaDB compatibility
    COALESCE(LAG(ese.step_order, 1) OVER(PARTITION BY lse.student_id ORDER BY ese.event_timestamp), 0) as prev_step_order
  FROM experiment_step_events ese
  JOIN LatestStudentExperiment lse ON ese.experiment_id = lse.experiment_id AND lse.rn = 1
),
JumpBacks AS (
  -- 4. Identify all "jump back" events
  SELECT student_id, event_timestamp, step_order
  FROM NumberedEvents
  WHERE event_type = 'STARTED' AND step_order < prev_step_order
),
LatestResetPoint AS (
  -- 5. Find the most recent "jump back", which is our reset point
  SELECT student_id, event_timestamp as reset_timestamp, step_order as reset_step_order
  FROM (
    SELECT student_id, event_timestamp, step_order,
           ROW_NUMBER() OVER(PARTITION BY student_id ORDER BY event_timestamp DESC) as rn
    FROM JumpBacks
  ) ranked_jumps
  WHERE rn = 1
),
ValidEvents AS (
  -- 6. Filter events to get only the "valid" ones for the final sequence
  SELECT ne.student_id, ne.experiment_id, ne.step_order, ne.event_type, ne.event_timestamp
  FROM NumberedEvents ne
  LEFT JOIN LatestResetPoint lrp ON ne.student_id = lrp.student_id
  WHERE
    -- Condition A: The student never jumped back, so all events are valid.
    lrp.student_id IS NULL OR
    -- Condition B: The event is for a step BEFORE the reset point. These are preserved.
    ne.step_order < lrp.reset_step_order OR
    -- Condition C: The event happened AT or AFTER the reset. This is the new sequence.
    ne.event_timestamp >= lrp.reset_timestamp
),
PivotedSteps AS (
  -- 7. Pivot the valid events into started/completed columns
  SELECT
    student_id,
    experiment_id,
    step_order,
    MAX(CASE WHEN event_type = 'STARTED' THEN event_timestamp END) AS started_at,
    MAX(CASE WHEN event_type = 'COMPLETED' THEN event_timestamp END) AS completed_at
  FROM ValidEvents
  GROUP BY student_id, experiment_id, step_order
)
-- 8. Aggregate everything into the final JSON structure
SELECT
  cs.student_id,
  cs.username,
  cs.full_name,
  lse.experiment_id,
  -- Rely on the mysql2 driver to auto-parse the JSON string
  CASE
    WHEN COUNT(ps.step_order) > 0 THEN
      CONCAT('[',
        GROUP_CONCAT(
          JSON_OBJECT(
            'step_order', ps.step_order,
            'started_at', ps.started_at,
            'completed_at', ps.completed_at
          ) ORDER BY ps.step_order SEPARATOR ','
        ),
      ']')
    ELSE NULL
  END AS steps
FROM ClassStudents cs
LEFT JOIN LatestStudentExperiment lse ON cs.student_id = lse.student_id AND lse.rn = 1
LEFT JOIN PivotedSteps ps ON cs.student_id = ps.student_id
GROUP BY cs.student_id, cs.username, cs.full_name, lse.experiment_id
ORDER BY cs.full_name;
        `;

        const [results] = await pool.query(sql, [classIdNumber]);

        // mysql2/promise driver automatically parses JSON strings from the database
        return c.json(results);
      } catch (error) {
        console.error("Error fetching class step events:", error);
        throw error;
      }
    },
  );

  classesApi.delete(
    "/classes/:classId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      const { classId } = c.req.valid("param");
      const classIdNumber = Number.parseInt(classId, 10);

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
        if (payload.role !== "Admin" && payload.sub !== classOwner) {
          await connection.rollback();
          return c.json({ error: "Forbidden" }, 403);
        }

        const [[assistantCountRow]]: any[] = await connection.query(
          "SELECT COUNT(*) as count FROM class_assistants WHERE class_id = ?",
          [classIdNumber],
        );
        const [[enrollmentCountRow]]: any[] = await connection.query(
          "SELECT COUNT(*) as count FROM enrollments WHERE class_id = ?",
          [classIdNumber],
        );
        const [deleteResult]: any = await connection.query(
          "DELETE FROM classes WHERE class_id = ?",
          [classIdNumber],
        );

        if (deleteResult.affectedRows === 0) {
          await connection.rollback();
          return c.json({ error: "Class not found" }, 404);
        }

        await connection.commit();

        return c.json({
          message: "Class deleted successfully",
          cleanup: {
            removedAssistants: assistantCountRow.count,
            removedEnrollments: enrollmentCountRow.count,
            removedGradeWeights: 0,
          },
        });
      } catch (error) {
        await connection.rollback();
        console.error("Error deleting class:", error);
        throw error;
      } finally {
        connection.release();
      }
    },
  );

  classesApi.delete(
    "/classes/:classId/students/:studentId",
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
        if (payload.role !== "Admin" && payload.sub !== classOwner) {
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

  return classesApi;
};
