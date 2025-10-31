import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";
import csv from "csv-parser";
import { createReadStream } from "node:fs";
import { unlink, mkdir } from "node:fs/promises";
import path from "path";

// Zod Schemas for Validation
const classIdParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, { message: "classId must be a number" }),
});

const teacherIdParamSchema = z.object({
  teacherId: z.string().regex(/^\d+$/, { message: "teacherId must be a number" }),
});

const assistantIdParamSchema = z.object({
  assistantId: z.string().regex(/^\d+$/, { message: "assistantId must be a number" }),
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

export default (pool: Pool, JWT_SECRET: string) => {
  const classesApi = new Hono();

  // POST /api/classes - 创建班级 (支持上传学生名单CSV)
  classesApi.post(
    "/api/classes",
    jwt({ secret: JWT_SECRET }),
    async (c) => {
      let tempCsvPath: string | null = null;

      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Teacher" && payload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const body = await c.req.parseBody();
        const class_name = body["class_name"];
        const class_code = body["class_code"];
        const studentListFile = body["student_list"];

        if (studentListFile instanceof File) {
          const maxSize = 5 * 1024 * 1024; // 5MB
          if (studentListFile.size > maxSize) {
            return c.json({ error: "CSV file size cannot exceed 5MB" }, 413);
          }
        }

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

          // 2. If a student list CSV file is provided, parse and create students
          if (studentListFile instanceof File) {
            const tempDir = path.resolve(import.meta.dir, "../../uploads", "temp");
            await mkdir(tempDir, { recursive: true });
            const uniqueFilename = `${Date.now()}-${studentListFile.name}`;
            tempCsvPath = path.join(tempDir, uniqueFilename);
            await Bun.write(tempCsvPath, studentListFile);

            const studentRecords: Array<{ studentId: string; fullName: string; email: string }> = [];
            await new Promise<void>((resolve, reject) => {
              createReadStream(tempCsvPath!)
                .pipe(csv({ mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim() }))
                .on("data", (row: Record<string, string>) => {
                  const studentId = row["学号"]?.trim();
                  const fullName = row["姓名"]?.trim();
                  if (studentId && fullName) {
                    const email = `${studentId}@student.edu`;
                    studentRecords.push({ studentId, fullName, email });
                  }
                })
                .on("end", resolve)
                .on("error", reject);
            });

            if (studentRecords.length > 0) {
              const studentIds = studentRecords.map(record => record.studentId);
              const [existingUsers]: any[] = await connection.query(
                "SELECT username FROM users WHERE username IN (?)",
                [studentIds]
              );

              if (existingUsers.length > 0) {
                const existingUsernames = existingUsers.map((u: any) => u.username);
                await connection.rollback();
                connection.release();
                return c.json({
                  error: "Transaction failed. The following students already exist in the database.",
                  existing_students: existingUsernames
                }, 409);
              }

              const defaultPasswordHash = "$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG";

              for (const record of studentRecords) {
                const [userResult]: any = await connection.query(
                  "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?, ?, ?, ?, 'Student')",
                  [record.studentId, defaultPasswordHash, record.fullName, record.email]
                );
                const userId = userResult.insertId;
                createdStudentsCount++;

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
            errors: undefined
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
        if (tempCsvPath) {
          await unlink(tempCsvPath).catch((err) =>
            console.error("Failed to cleanup temp CSV file:", err)
          );
        }
      }
    },
  );

  // GET /api/classes - 获取所有班级
  classesApi.get("/api/classes", jwt({ secret: JWT_SECRET }), async (c) => {
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

  // GET /api/classes/:classId - 获取班级详情
  classesApi.get(
    "/api/classes/:classId",
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

  // PUT /api/classes/:classId - 更新班级
  classesApi.put(
    "/api/classes/:classId",
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

  // DELETE /api/classes/:classId - 删除班级
  classesApi.delete(
    "/api/classes/:classId",
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
        const [[gradeWeightsCountRow]]: any[] = await connection.query(
          "SELECT COUNT(*) as count FROM grade_weights WHERE class_id = ?",
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
            removedGradeWeights: gradeWeightsCountRow.count,
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

  // GET /api/teachers/:teacherId/classes - 获取教师的班级
  classesApi.get(
    "/api/teachers/:teacherId/classes",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", teacherIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { teacherId } = c.req.valid("param");
        const teacherIdNumber = Number.parseInt(teacherId, 10);

        if (payload.role !== "Admin" && payload.sub !== teacherIdNumber) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const [teacherRows]: any[] = await pool.query(
          "SELECT role FROM users WHERE user_id = ? AND role = 'Teacher'",
          [teacherIdNumber],
        );

        if (teacherRows.length === 0 && payload.role !== "Admin") {
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

  // GET /api/assistants/:assistantId/classes - 获取助教的班级
  classesApi.get(
    "/api/assistants/:assistantId/classes",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", assistantIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { assistantId } = c.req.valid("param");
        const assistantIdNumber = Number.parseInt(assistantId, 10);

        if (payload.role !== "Admin" && payload.sub !== assistantIdNumber) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const [assistantRows]: any[] = await pool.query(
          "SELECT role FROM users WHERE user_id = ? AND role = 'Assistant'",
          [assistantIdNumber],
        );

        if (assistantRows.length === 0 && payload.role !== "Admin") {
          return c.json({ error: "Target user is not an assistant" }, 404);
        }

        const [classRows] = await pool.query(
          `SELECT c.*, u.full_name as teacher_name
             FROM classes c
             JOIN class_assistants ca ON c.class_id = ca.class_id
             JOIN users u ON c.teacher_id = u.user_id
            WHERE ca.assistant_id = ?`,
          [assistantIdNumber],
        );

        return c.json(classRows);
      } catch (error) {
        console.error("Error fetching classes for assistant:", error);
        throw error;
      }
    },
  );

  return classesApi;
};
