import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool, PoolConnection } from "mysql2/promise";
import { access, copyFile, mkdtemp, rm } from "node:fs/promises";
import path from "path";
import os from "node:os";

// Zod Schemas for Validation
const classIdParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, { message: "classId must be a number" }),
});

const reportIdParamSchema = z.object({
  reportId: z.string().regex(/^\d+$/, { message: "reportId must be a number" }),
});

const EXP_FLOW_FIELDS = [
  "exp_flow_demand_data_preparation",
  "exp_flow_demand_descriptive_stats",
  "exp_flow_demand_model_selection",
  "exp_flow_demand_generate_results",
  "exp_flow_production_inventory_calc",
  "exp_flow_production_service_level",
  "exp_flow_production_variable_calc",
  "exp_flow_production_plan_creation",
  "exp_flow_report_submission",
] as const;

const computeExperimentScores = (
  row: Record<string, any>,
  weights: Record<string, number>,
) => {
  if (row.experiment_id === null || row.experiment_id === undefined) {
    return { expFlowScore: null, finalScore: null };
  }

  let expFlowTotal = 0;
  for (const field of EXP_FLOW_FIELDS) {
    const stepScore = Number(row[field] ?? 0);
    const weight = Number(weights[field] ?? 0);
    if (Number.isFinite(stepScore) && Number.isFinite(weight)) {
      expFlowTotal += stepScore * weight;
    }
  }
  const expFlowScore = expFlowTotal / 100;

  const knowledge = Number(row.knowledge_test ?? 0);
  const modelQuality = Number(row.model_quality ?? 0);
  const reportQuality = Number(row.report_quality ?? 0);
  const expWeight = Number(weights.exp_flow_weight ?? 0);
  const knowledgeWeight = Number(weights.knowledge_test_weight ?? 0);
  const modelWeight = Number(weights.model_quality_weight ?? 0);
  const reportWeight = Number(weights.report_quality_weight ?? 0);

  const finalScore =
    (Number.isFinite(expFlowScore) ? expFlowScore : 0) * (expWeight / 100) +
    (Number.isFinite(knowledge) ? knowledge : 0) * (knowledgeWeight / 100) +
    (Number.isFinite(modelQuality) ? modelQuality : 0) * (modelWeight / 100) +
    (Number.isFinite(reportQuality) ? reportQuality : 0) * (reportWeight / 100);

  return {
    expFlowScore: Number.isFinite(expFlowScore) ? expFlowScore : null,
    finalScore: Number.isFinite(finalScore) ? finalScore : null,
  };
};

const LATEST_GRADE_SELECTION_SQL = `
  SELECT
    u.user_id,
    u.username,
    u.full_name,
    u.email,
    latest_report.report_id,
    latest_report.submitted_at AS report_submitted_at,
    latest_report.pdf_file_path,
    COALESCE(eg_report.experiment_id, eg_latest.experiment_id) AS experiment_id,
    COALESCE(eg_report.created_at, eg_latest.created_at) AS graded_at,
    latest_report.feedback AS feedback,
    COALESCE(eg_report.exp_flow_demand_data_preparation, eg_latest.exp_flow_demand_data_preparation) AS exp_flow_demand_data_preparation,
    COALESCE(eg_report.exp_flow_demand_descriptive_stats, eg_latest.exp_flow_demand_descriptive_stats) AS exp_flow_demand_descriptive_stats,
    COALESCE(eg_report.exp_flow_demand_model_selection, eg_latest.exp_flow_demand_model_selection) AS exp_flow_demand_model_selection,
    COALESCE(eg_report.exp_flow_demand_generate_results, eg_latest.exp_flow_demand_generate_results) AS exp_flow_demand_generate_results,
    COALESCE(eg_report.exp_flow_production_inventory_calc, eg_latest.exp_flow_production_inventory_calc) AS exp_flow_production_inventory_calc,
    COALESCE(eg_report.exp_flow_production_service_level, eg_latest.exp_flow_production_service_level) AS exp_flow_production_service_level,
    COALESCE(eg_report.exp_flow_production_variable_calc, eg_latest.exp_flow_production_variable_calc) AS exp_flow_production_variable_calc,
    COALESCE(eg_report.exp_flow_production_plan_creation, eg_latest.exp_flow_production_plan_creation) AS exp_flow_production_plan_creation,
    COALESCE(eg_report.exp_flow_report_submission, eg_latest.exp_flow_report_submission) AS exp_flow_report_submission,
    COALESCE(eg_report.knowledge_test, eg_latest.knowledge_test) AS knowledge_test,
    COALESCE(eg_report.model_quality, eg_latest.model_quality) AS model_quality,
    COALESCE(eg_report.report_quality, eg_latest.report_quality) AS report_quality
  FROM enrollments e
  JOIN users u ON e.student_id = u.user_id
  LEFT JOIN (
    SELECT er.*
    FROM experiment_reports er
    JOIN (
      SELECT student_id, MAX(submitted_at) AS max_submitted_at
      FROM experiment_reports
      GROUP BY student_id
    ) recent
      ON recent.student_id = er.student_id
     AND recent.max_submitted_at = er.submitted_at
  ) latest_report ON latest_report.student_id = u.user_id
  LEFT JOIN experiment_grade eg_report
    ON latest_report.experiment_id IS NOT NULL
   AND eg_report.student_id = u.user_id
   AND eg_report.experiment_id = latest_report.experiment_id
  LEFT JOIN (
    SELECT eg_inner.*
    FROM experiment_grade eg_inner
    JOIN (
      SELECT student_id, MAX(created_at) AS max_graded_at
      FROM experiment_grade
      GROUP BY student_id
    ) latest_grade
      ON latest_grade.student_id = eg_inner.student_id
     AND latest_grade.max_graded_at = eg_inner.created_at
  ) eg_latest ON eg_latest.student_id = u.user_id
  WHERE e.class_id = ?
  ORDER BY u.username ASC
`;

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

  // GET /api/v1/classes/:classId/reports - 获取班级报告
  classGradesApi.get(
    "/api/v1/classes/:classId/reports",
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

        const [reports]: any[] = await pool.query(
          `SELECT
             u.user_id,
             u.username,
             u.full_name,
             latest.report_id,
             latest.experiment_id,
             latest.submitted_at,
             latest.pdf_file_path,
             latest.grade,
             latest.feedback
           FROM enrollments e
           JOIN users u ON e.student_id = u.user_id
           LEFT JOIN (
             SELECT er.*
             FROM experiment_reports er
             JOIN (
               SELECT student_id, MAX(submitted_at) AS max_submitted_at
               FROM experiment_reports
               GROUP BY student_id
             ) recent
               ON recent.student_id = er.student_id
              AND recent.max_submitted_at = er.submitted_at
           ) latest ON latest.student_id = u.user_id
           WHERE e.class_id = ?
           ORDER BY u.username ASC`,
          [classIdNumber],
        );

        return c.json(reports);
      } catch (error) {
        console.error("Error fetching class reports:", error);
        throw error;
      }
    },
  );

  // PUT /api/v1/reports/:reportId - 批改报告
  classGradesApi.put(
    "/api/v1/reports/:reportId",
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

          if (
            gradeData.report_quality === undefined &&
            gradeData.feedback === undefined
          ) {
            await connection.rollback();
            return c.json({ error: "No fields provided to update" }, 400);
          }

          if (gradeData.report_quality !== undefined) {
            const [gradeResult]: any = await connection.query(
              `
                UPDATE experiment_grade
                   SET report_quality = ?
                 WHERE experiment_id = ? AND student_id = ?
              `,
              [gradeData.report_quality, experiment_id, student_id],
            );

            if (gradeResult.affectedRows === 0) {
              await connection.rollback();
              return c.json({ error: "Grade record not found" }, 404);
            }
          }

          const reportUpdateFields: string[] = [];
          const reportUpdateValues: any[] = [];

          if (gradeData.report_quality !== undefined) {
            reportUpdateFields.push("grade = ?");
            reportUpdateValues.push(gradeData.report_quality);
          }
          if (gradeData.feedback !== undefined) {
            reportUpdateFields.push("feedback = ?");
            reportUpdateValues.push(gradeData.feedback);
          }

          if (reportUpdateFields.length > 0) {
            reportUpdateValues.push(reportIdNumber);
            await connection.query(
              `
                UPDATE experiment_reports
                   SET ${reportUpdateFields.join(", ")}
                 WHERE report_id = ?
              `,
              reportUpdateValues,
            );
          }

          await connection.commit();

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

  // GET /api/v1/classes/:classId/experiment-runs - 获取班级实验状态
  classGradesApi.get(
    "/api/v1/classes/:classId/experiment-runs",
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

  // GET /api/v1/classes/:classId/grading-policy - 获取成绩权重
  classGradesApi.get(
    "/api/v1/classes/:classId/grading-policy",
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

  // GET /api/v1/classes/:classId/report-archive - 导出班级报告ZIP
  classGradesApi.get(
    "/api/v1/classes/:classId/report-archive",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
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

      const [reports]: any[] = await pool.query(
        `SELECT
           er.pdf_file_path,
           er.experiment_id,
           er.student_id,
           u.username,
           u.full_name
         FROM enrollments e
         JOIN users u ON e.student_id = u.user_id
         JOIN experiment_reports er ON er.student_id = e.student_id
        WHERE e.class_id = ?
          AND er.pdf_file_path IS NOT NULL`,
        [classIdNumber],
      );

      if (reports.length === 0) {
        return c.json({ error: "No reports found for this class" }, 404);
      }

      const tempDir = await mkdtemp(path.join(os.tmpdir(), "class-report-export-"));
      const copiedFiles: string[] = [];
      const usedNames = new Set<string>();

      try {
        for (const report of reports) {
          const originalPath: string | null = report.pdf_file_path;
          if (!originalPath) {
            continue;
          }

          try {
            await access(originalPath);
          } catch {
            continue;
          }

          const rawName =
            (report.username as string | undefined) ||
            (report.full_name as string | undefined) ||
            `student_${report.student_id}`;
          const baseName = `${rawName}_experiment_${report.experiment_id}`
            .replace(/\s+/g, "_")
            .replace(/[^a-zA-Z0-9_\-\.]/g, "_");

          let candidate = `${baseName}.pdf`;
          let suffix = 1;
          while (usedNames.has(candidate)) {
            candidate = `${baseName}_${suffix}.pdf`;
            suffix += 1;
          }
          usedNames.add(candidate);

          const destination = path.join(tempDir, candidate);
          await copyFile(originalPath, destination);
          copiedFiles.push(destination);
        }

        if (copiedFiles.length === 0) {
          return c.json({ error: "No report files available for download" }, 404);
        }

        const zipFileName = `class-${classIdNumber}-reports.zip`;
        const zipPath = path.join(tempDir, zipFileName);

        const zipArgs = [
          "zip",
          "-j",
          zipPath,
          ...copiedFiles.map((filePath) => path.basename(filePath)),
        ];
        let zipProcess;
        try {
          zipProcess = Bun.spawn(zipArgs, {
            cwd: tempDir,
            stdout: "pipe",
            stderr: "pipe",
          });
        } catch (spawnError) {
          throw new Error(
            `Failed to start zip command. Ensure 'zip' is installed on the server: ${spawnError}`,
          );
        }

        const exitCode = await zipProcess.exited;
        if (exitCode !== 0) {
          const errorText = await new Response(zipProcess.stderr).text();
          throw new Error(`zip command failed: ${errorText}`);
        }

        const zipFile = Bun.file(zipPath);
        const zipBytes = await zipFile.arrayBuffer();

        return c.newResponse(zipBytes, {
          status: 200,
          headers: {
            "Content-Type": "application/zip",
            "Content-Disposition": `attachment; filename="${zipFileName}"`,
          },
        });
      } finally {
        await rm(tempDir, { recursive: true, force: true });
      }
    },
  );

  // GET /api/v1/classes/:classId/grade-export.csv - 导出最新实验成绩CSV
  classGradesApi.get(
    "/api/v1/classes/:classId/grade-export.csv",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", classIdParamSchema),
    async (c) => {
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

      const [weightRows]: any[] = await pool.query(
        "SELECT * FROM grade_weights WHERE class_id = ?",
        [classIdNumber],
      );

      if (weightRows.length === 0) {
        return c.json({ error: "Grade weights not configured for this class" }, 404);
      }

      const weights = weightRows[0] as Record<string, number>;

      const [gradeRows]: any[] = await pool.query(
        LATEST_GRADE_SELECTION_SQL,
        [classIdNumber],
      );

      if (gradeRows.length === 0) {
        return c.json({ error: "No enrolled students found for this class" }, 404);
      }

      const toCsvValue = (value: unknown) => {
        if (value === null || value === undefined) {
          return "";
        }
        let str: string;
        if (value instanceof Date) {
          str = value.toISOString();
        } else {
          str = String(value);
        }
        if (/[",\n]/.test(str)) {
          return `"${str.replace(/"/g, '""')}"`;
        }
        return str;
      };

      const formatNumeric = (value: unknown) => {
        if (value === null || value === undefined) {
          return "";
        }
        const num = Number(value);
        if (!Number.isFinite(num)) {
          return "";
        }
        return num.toFixed(2);
      };

      const header = [
        "student_id",
        "username",
        "full_name",
        "email",
        "experiment_id",
        "graded_at",
        ...EXP_FLOW_FIELDS,
        "knowledge_test",
        "model_quality",
        "report_quality",
        "exp_flow_score",
        "final_score",
        "feedback",
      ];

      const csvLines: string[] = [header.join(",")];

      for (const row of gradeRows) {
        const studentId = row.user_id;
        const graded = row.graded_at instanceof Date
          ? row.graded_at.toISOString()
          : row.graded_at ?? "";

        const { expFlowScore, finalScore } = computeExperimentScores(row, weights);
        const expFlowScoreStr =
          expFlowScore === null ? "" : Number(expFlowScore).toFixed(2);
        const finalScoreStr =
          finalScore === null ? "" : Number(finalScore).toFixed(2);

        const rowValues: string[] = [
          toCsvValue(studentId),
          toCsvValue(row.username),
          toCsvValue(row.full_name),
          toCsvValue(row.email),
          toCsvValue(row.experiment_id),
          toCsvValue(graded),
          ...EXP_FLOW_FIELDS.map((field) => formatNumeric(row[field])),
          formatNumeric(row.knowledge_test),
          formatNumeric(row.model_quality),
          formatNumeric(row.report_quality),
          toCsvValue(expFlowScoreStr),
          toCsvValue(finalScoreStr),
          toCsvValue(row.feedback),
        ];

        csvLines.push(rowValues.join(","));
      }

      const csvContent = "\uFEFF" + csvLines.join("\n");
      const fileName = `class-${classIdNumber}-grades.csv`;

      return c.newResponse(csvContent, {
        status: 200,
        headers: {
          "Content-Type": "text/csv; charset=utf-8",
          "Content-Disposition": `attachment; filename="${fileName}"`,
        },
      });
    },
  );

  // PUT /api/v1/classes/:classId/grading-policy - 更新成绩权重
  classGradesApi.put(
    "/api/v1/classes/:classId/grading-policy",
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

  // GET /api/v1/classes/:classId/grade-summaries - 获取最新成绩
  classGradesApi.get(
    "/api/v1/classes/:classId/grade-summaries",
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

        const [weightRows]: any[] = await pool.query(
          "SELECT * FROM grade_weights WHERE class_id = ?",
          [classIdNumber],
        );

        if (weightRows.length === 0) {
          return c.json({ error: "Grade weights not configured for this class" }, 404);
        }

        const weights = weightRows[0] as Record<string, number>;

        const [gradeRows]: any[] = await pool.query(
          LATEST_GRADE_SELECTION_SQL,
          [classIdNumber],
        );

        const toNumberOrNull = (value: unknown) =>
          value === null || value === undefined ? null : Number(value);

        const latestGrades = (gradeRows as any[]).map((row: any) => {
          const { expFlowScore, finalScore } = computeExperimentScores(row, weights);
          const expFlowScoreRounded =
            expFlowScore === null ? null : Number(expFlowScore.toFixed(2));
          const finalScoreRounded =
            finalScore === null ? null : Number(finalScore.toFixed(2));
          return {
            student_id: row.user_id,
            username: row.username,
            full_name: row.full_name,
            experiment_id: row.experiment_id,
            exp_flow_score: expFlowScoreRounded,
            model_quality: toNumberOrNull(row.model_quality),
            knowledge_test: toNumberOrNull(row.knowledge_test),
            report_quality: toNumberOrNull(row.report_quality),
            feedback: row.feedback ?? null,
            graded_at: row.graded_at ?? null,
            final_score: finalScoreRounded,
          };
        });

        return c.json(latestGrades);
      } catch (error) {
        console.error("Error fetching latest grades:", error);
        throw error;
      }
    },
  );

  // GET /api/v1/classes/:classId/experiment-events - 获取步骤事件
  classGradesApi.get(
    "/api/v1/classes/:classId/experiment-events",
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
