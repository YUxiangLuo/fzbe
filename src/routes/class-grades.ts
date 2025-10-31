import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool, PoolConnection } from "mysql2/promise";

// Zod Schemas for Validation
const classIdParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, { message: "classId must be a number" }),
});

const reportIdParamSchema = z.object({
  reportId: z.string().regex(/^\d+$/, { message: "reportId must be a number" }),
});

const gradeWeightsSchema = z.object({
  exp_flow_weight: z.number().min(0).max(100).optional(),
  model_quality_weight: z.number().min(0).max(100).optional(),
  knowledge_test_weight: z.number().min(0).max(100).optional(),
  report_quality_weight: z.number().min(0).max(100).optional(),
}).refine((data) => {
  // At least one field must be provided
  return Object.values(data).some(v => v !== undefined);
}, {
  message: "At least one weight field must be provided",
});

const updateGradeWeightsSchema = z.object({
  exp_flow_weight: z.number().min(0).max(100).optional(),
  model_quality_weight: z.number().min(0).max(100).optional(),
  knowledge_test_weight: z.number().min(0).max(100).optional(),
  report_quality_weight: z.number().min(0).max(100).optional(),
}).refine((data) => {
  // At least one field must be provided
  return Object.values(data).some(v => v !== undefined);
}, {
  message: "At least one weight field must be provided",
});

const gradeReportSchema = z.object({
  report_quality: z.number().min(0).max(100).optional(),
  feedback: z.string().trim().optional(),
}).refine((data) => {
  // At least one field must be provided
  return Object.values(data).some(v => v !== undefined);
}, {
  message: "At least one grade field must be provided",
});

export default (pool: Pool, JWT_SECRET: string) => {
  const classGradesApi = new Hono();
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

  const isAssistantAssignedToStudent = async (
    db: Queryable,
    studentId: number,
    assistantId: number,
  ): Promise<boolean> => {
    const [[assignment]]: any[] = await db.query(
      `SELECT 1
         FROM enrollments e
         JOIN class_assistants ca ON e.class_id = ca.class_id
        WHERE e.student_id = ? AND ca.assistant_id = ?
        LIMIT 1`,
      [studentId, assistantId],
    );
    return Boolean(assignment);
  };

  const isTeacherOfStudent = async (
    db: Queryable,
    studentId: number,
    teacherId: number,
  ): Promise<boolean> => {
    const [[row]]: any[] = await db.query(
      `SELECT 1
         FROM enrollments e
         JOIN classes c ON e.class_id = c.class_id
        WHERE e.student_id = ? AND c.teacher_id = ?
        LIMIT 1`,
      [studentId, teacherId],
    );
    return Boolean(row);
  };

  // GET /api/classes/:classId/reports - 获取班级报告
  classGradesApi.get(
    "/api/classes/:classId/reports",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);

        // Check if user is teacher/admin of this class
        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );

        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }

        let isAllowed = false;
        if (payload.role === "Admin" || payload.sub === classRows[0].teacher_id) {
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

        // Get all reports for students in this class
        const [reports]: any[] = await pool.query(
          `SELECT
             er.report_id,
             er.experiment_id,
             er.student_id,
             u.username,
             u.full_name,
             er.submitted_at,
             er.pdf_file_path,
             eg.knowledge_test,
             eg.report_quality,
             eg.grader_id,
             grader.full_name as grader_name
           FROM experiment_reports er
           JOIN users u ON er.student_id = u.user_id
           LEFT JOIN experiment_grade eg ON er.experiment_id = eg.experiment_id AND er.student_id = eg.student_id
           LEFT JOIN users grader ON eg.grader_id = grader.user_id
           JOIN enrollments e ON u.user_id = e.student_id
           WHERE e.class_id = ?
           ORDER BY er.submitted_at DESC`,
          [classIdNumber],
        );

        return c.json(reports);
      } catch (error) {
        console.error("Error fetching class reports:", error);
        throw error;
      }
    },
  );

  // PUT /api/reports/:reportId - 批改报告
  classGradesApi.put(
    "/api/reports/:reportId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", reportIdParamSchema),
    zValidator("json", gradeReportSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (
          payload.role !== "Teacher" &&
          payload.role !== "Admin" &&
          payload.role !== "Assistant"
        ) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { reportId } = c.req.valid("param");
        const reportIdNumber = Number.parseInt(reportId, 10);
        const gradeData = c.req.valid("json");

        const connection = await pool.getConnection();
        try {
          await connection.beginTransaction();

          // Get report details
          const [reportRows]: any[] = await connection.query(
            "SELECT experiment_id, student_id FROM experiment_reports WHERE report_id = ?",
            [reportIdNumber],
          );

          if (reportRows.length === 0) {
            await connection.rollback();
            return c.json({ error: "Report not found" }, 404);
          }

          const { experiment_id, student_id } = reportRows[0];

          let authorized = false;
          if (payload.role === "Admin") {
            authorized = true;
          } else if (payload.role === "Teacher") {
            const ownsStudent = await isTeacherOfStudent(
              connection,
              student_id,
              payload.sub,
            );
            if (ownsStudent) {
              authorized = true;
            }
          } else if (payload.role === "Assistant") {
            const assigned = await isAssistantAssignedToStudent(
              connection,
              student_id,
              payload.sub,
            );
            if (assigned) {
              authorized = true;
            }
          }

          if (!authorized) {
            await connection.rollback();
            return c.json({ error: "Forbidden" }, 403);
          }

          // Build update query for experiment_grade
          const fields: string[] = [];
          const values: any[] = [];


          if (gradeData.report_quality !== undefined) {
            fields.push("report_quality = ?");
            values.push(gradeData.report_quality);
          }
          if (gradeData.feedback !== undefined) {
            fields.push("feedback = ?");
            values.push(gradeData.feedback);
          }

          fields.push("grader_id = ?");
          values.push(payload.sub);

          values.push(experiment_id);
          values.push(student_id);

          const sql = `
            UPDATE experiment_grade
            SET ${fields.join(", ")}
            WHERE experiment_id = ? AND student_id = ?
          `;

          const [result]: any = await connection.query(sql, values);

          if (result.affectedRows === 0) {
            await connection.rollback();
            return c.json({ error: "Grade record not found" }, 404);
          }

          await connection.commit();

          // Fetch updated grade
          const [updatedGrades]: any[] = await connection.query(
            "SELECT * FROM experiment_grade WHERE experiment_id = ? AND student_id = ?",
            [experiment_id, student_id],
          );

          return c.json(updatedGrades[0]);
        } catch (error) {
          await connection.rollback();
          throw error;
        } finally {
          connection.release();
        }
      } catch (error) {
        console.error("Error grading report:", error);
        throw error;
      }
    },
  );

  // GET /api/classes/:classId/experiment-status - 获取班级实验状态
  classGradesApi.get(
    "/api/classes/:classId/experiment-status",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);

        // Check if user is teacher/admin of this class
        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );

        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }

        let isAllowed = false;
        if (payload.role === "Admin" || payload.sub === classRows[0].teacher_id) {
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

        // Get experiment status for all students in class
        const [experimentStatus]: any[] = await pool.query(
          `SELECT
             u.user_id,
             u.username,
             u.full_name,
             e.enrollment_id,
             COUNT(DISTINCT es.experiment_id) AS total_experiments,
             SUM(CASE WHEN es.status = 'Completed' THEN 1 ELSE 0 END) AS completed_experiments,
             SUM(CASE WHEN er.report_id IS NOT NULL THEN 1 ELSE 0 END) AS submitted_reports,
             MAX(COALESCE(es.last_activity_at, es.start_time)) AS last_experiment_date
           FROM enrollments e
           JOIN users u ON e.student_id = u.user_id
           LEFT JOIN experiment_status es ON u.user_id = es.student_id
           LEFT JOIN experiment_reports er ON es.experiment_id = er.experiment_id
           WHERE e.class_id = ?
           GROUP BY u.user_id, e.enrollment_id
           ORDER BY u.username ASC`,
          [classIdNumber],
        );

        return c.json(experimentStatus);
      } catch (error) {
        console.error("Error fetching experiment status:", error);
        throw error;
      }
    },
  );

  // GET /api/classes/:classId/grade-weights - 获取成绩权重
  classGradesApi.get(
    "/api/classes/:classId/grade-weights",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);

        // Check if user is teacher/admin of this class
        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );

        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }

        let isAllowed = false;
        if (payload.role === "Admin" || payload.sub === classRows[0].teacher_id) {
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

        const [weights]: any[] = await pool.query(
          "SELECT * FROM grade_weights WHERE class_id = ?",
          [classIdNumber],
        );

        if (weights.length === 0) {
          return c.json({ error: "Grade weights not found for this class" }, 404);
        }

        return c.json(weights[0]);
      } catch (error) {
        console.error("Error fetching grade weights:", error);
        throw error;
      }
    },
  );

  // POST /api/classes/:classId/grade-weights - 创建成绩权重
  classGradesApi.post(
    "/api/classes/:classId/grade-weights",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    zValidator("json", gradeWeightsSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (
          payload.role !== "Teacher" &&
          payload.role !== "Admin" &&
          payload.role !== "Assistant"
        ) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const weightData = c.req.valid("json");

        // Check if user is teacher/admin of this class
        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );

        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }

        let isAllowed = false;
        if (payload.role === "Admin" || payload.sub === classRows[0].teacher_id) {
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

        // Check if weights already exist
        const [existing]: any[] = await pool.query(
          "SELECT weight_id FROM grade_weights WHERE class_id = ?",
          [classIdNumber],
        );

        if (existing.length > 0) {
          return c.json({ error: "Grade weights already exist for this class" }, 409);
        }

        const fields: string[] = ["class_id"];
        const values: any[] = [classIdNumber];
        const placeholders: string[] = ["?"];

        if (weightData.exp_flow_weight !== undefined) {
          fields.push("exp_flow_weight");
          values.push(weightData.exp_flow_weight);
          placeholders.push("?");
        }
        if (weightData.model_quality_weight !== undefined) {
          fields.push("model_quality_weight");
          values.push(weightData.model_quality_weight);
          placeholders.push("?");
        }
        if (weightData.knowledge_test_weight !== undefined) {
          fields.push("knowledge_test_weight");
          values.push(weightData.knowledge_test_weight);
          placeholders.push("?");
        }
        if (weightData.report_quality_weight !== undefined) {
          fields.push("report_quality_weight");
          values.push(weightData.report_quality_weight);
          placeholders.push("?");
        }

        const sql = `
          INSERT INTO grade_weights (${fields.join(", ")})
          VALUES (${placeholders.join(", ")})
        `;

        const [result]: any = await pool.query(sql, values);

        const [newWeights]: any[] = await pool.query(
          "SELECT * FROM grade_weights WHERE weight_id = ?",
          [result.insertId],
        );

        return c.json(newWeights[0], 201);
      } catch (error) {
        console.error("Error creating grade weights:", error);
        throw error;
      }
    },
  );

  // PUT /api/classes/:classId/grade-weights - 更新成绩权重
  classGradesApi.put(
    "/api/classes/:classId/grade-weights",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    zValidator("json", updateGradeWeightsSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (
          payload.role !== "Teacher" &&
          payload.role !== "Admin" &&
          payload.role !== "Assistant"
        ) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);
        const weightData = c.req.valid("json");

        // Check if user is teacher/admin of this class
        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );

        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }

        let isAllowed = false;
        if (payload.role === "Admin" || payload.sub === classRows[0].teacher_id) {
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

        // Check if weights exist
        const [existing]: any[] = await pool.query(
          "SELECT weight_id FROM grade_weights WHERE class_id = ?",
          [classIdNumber],
        );

        if (existing.length === 0) {
          return c.json({ error: "Grade weights not found for this class" }, 404);
        }

        const fields: string[] = [];
        const values: any[] = [];

        if (weightData.exp_flow_weight !== undefined) {
          fields.push("exp_flow_weight = ?");
          values.push(weightData.exp_flow_weight);
        }
        if (weightData.model_quality_weight !== undefined) {
          fields.push("model_quality_weight = ?");
          values.push(weightData.model_quality_weight);
        }
        if (weightData.knowledge_test_weight !== undefined) {
          fields.push("knowledge_test_weight = ?");
          values.push(weightData.knowledge_test_weight);
        }
        if (weightData.report_quality_weight !== undefined) {
          fields.push("report_quality_weight = ?");
          values.push(weightData.report_quality_weight);
        }

        values.push(classIdNumber);

        const sql = `
          UPDATE grade_weights
          SET ${fields.join(", ")}
          WHERE class_id = ?
        `;

        await pool.query(sql, values);

        const [updatedWeights]: any[] = await pool.query(
          "SELECT * FROM grade_weights WHERE class_id = ?",
          [classIdNumber],
        );

        return c.json(updatedWeights[0]);
      } catch (error) {
        console.error("Error updating grade weights:", error);
        throw error;
      }
    },
  );

  // GET /api/classes/:classId/latest-grades - 获取最新成绩
  classGradesApi.get(
    "/api/classes/:classId/latest-grades",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);

        // Check if user is teacher/admin of this class
        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );

        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }

        let isAllowed = false;
        if (payload.role === "Admin" || payload.sub === classRows[0].teacher_id) {
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

        // Get latest grades for all students in class
        const [latestGrades]: any[] = await pool.query(
          `SELECT
             u.user_id,
             u.username,
             u.full_name,
             eg.experiment_id,
             eg.exp_flow,
             eg.model_quality,
             eg.knowledge_test,
             eg.report_quality,
             eg.feedback,
             eg.created_at AS graded_at,
             ROUND(
               COALESCE(eg.exp_flow, 0) * COALESCE(gw.exp_flow_weight, 0) / 100 +
               COALESCE(eg.model_quality, 0) * COALESCE(gw.model_quality_weight, 0) / 100 +
               COALESCE(eg.knowledge_test, 0) * COALESCE(gw.knowledge_test_weight, 0) / 100 +
               COALESCE(eg.report_quality, 0) * COALESCE(gw.report_quality_weight, 0) / 100,
               2
             ) AS final_score
           FROM enrollments e
           JOIN users u ON e.student_id = u.user_id
           LEFT JOIN (
             SELECT student_id, MAX(created_at) AS max_graded_at
             FROM experiment_grade
             WHERE created_at IS NOT NULL
             GROUP BY student_id
           ) latest ON u.user_id = latest.student_id
           LEFT JOIN experiment_grade eg
             ON latest.student_id = eg.student_id
             AND latest.max_graded_at = eg.created_at
           LEFT JOIN grade_weights gw ON e.class_id = gw.class_id
           WHERE e.class_id = ?
           ORDER BY u.username ASC`,
          [classIdNumber],
        );

        return c.json(latestGrades);
      } catch (error) {
        console.error("Error fetching latest grades:", error);
        throw error;
      }
    },
  );

  // GET /api/classes/:classId/step-events - 获取步骤事件
  classGradesApi.get(
    "/api/classes/:classId/step-events",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { classId } = c.req.valid("param");
        const classIdNumber = Number.parseInt(classId, 10);

        // Check if user is teacher/admin of this class
        const [classRows]: any[] = await pool.query(
          "SELECT teacher_id FROM classes WHERE class_id = ?",
          [classIdNumber],
        );

        if (classRows.length === 0) {
          return c.json({ error: "Class not found" }, 404);
        }

        let isAllowed = false;
        if (payload.role === "Admin" || payload.sub === classRows[0].teacher_id) {
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

        // Get step events for experiments of students in class
        const [stepEvents]: any[] = await pool.query(
          `SELECT
             se.event_id,
             se.experiment_id,
             se.student_id,
             u.username,
             u.full_name,
             se.step_order,
             se.event_type,
             se.event_timestamp
           FROM experiment_step_events se
           JOIN experiment_status es ON se.experiment_id = es.experiment_id
           JOIN users u ON se.student_id = u.user_id
           JOIN enrollments e ON u.user_id = e.student_id
           WHERE e.class_id = ?
           ORDER BY se.event_timestamp DESC`,
          [classIdNumber],
        );

        return c.json(stepEvents);
      } catch (error) {
        console.error("Error fetching step events:", error);
        throw error;
      }
    },
  );

  return classGradesApi;
};
