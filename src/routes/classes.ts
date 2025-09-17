import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";

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

export default (pool: Pool, JWT_SECRET: string) => {
  const classesApi = new Hono();

  // 受保护的路由: 创建新班级
  classesApi.post(
    "/classes",
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

        const [newClassRows]: any[] = await pool.query(
          "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id WHERE c.class_id = ?",
          [result.insertId],
        );

        return c.json(newClassRows[0], 201);
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
    "/classes/enroll",
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

  // 受保护的路由: 获取单个班级详情 (包括学生列表)
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
                  es.best_model
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

        const [events] = await pool.query(
          `SELECT ese.event_id,
                  ese.experiment_id,
                  ese.student_id,
                  e.class_id,
                  ese.step_order,
                  ese.event_type,
                  ese.event_timestamp,
                  u.username,
                  u.full_name
             FROM experiment_step_events ese
             JOIN enrollments e ON e.student_id = ese.student_id
             JOIN users u ON u.user_id = ese.student_id
            WHERE e.class_id = ?
            ORDER BY ese.student_id ASC, ese.experiment_id ASC, ese.event_timestamp ASC`,
          [classIdNumber],
        );

        return c.json(events);
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
        const [[gradeWeightCountRow]]: any[] = await connection.query(
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
            removedGradeWeights: gradeWeightCountRow.count,
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
