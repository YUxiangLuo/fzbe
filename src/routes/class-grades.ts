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

const STEP_COUNT = 7;

const toIsoString = (value: unknown): string | null => {
  if (!value) {
    return null;
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const parsed = new Date(trimmed);
    return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
  }
  const parsed = new Date(value as any);
  return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
};

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

type RawStepEvent = {
  event_id: number;
  experiment_id: number;
  student_id: number;
  step_order: number;
  event_type: "STARTED" | "COMPLETED";
  event_timestamp: unknown;
};

type NormalizedStepEvent = {
  event_id: number;
  experiment_id: number;
  student_id: number;
  step_order: number;
  event_type: "STARTED" | "COMPLETED";
  event_timestamp: string | null;
};

type StepSummary = {
  step_order: number;
  started_at: string | null;
  completed_at: string | null;
  latest_event_type: "STARTED" | "COMPLETED" | null;
};

const createEmptyStepSummaries = (): StepSummary[] =>
  Array.from({ length: STEP_COUNT }, (_, index) => ({
    step_order: index + 1,
    started_at: null,
    completed_at: null,
    latest_event_type: null,
  }));

const buildNormalizedStepTimeline = (events: RawStepEvent[]) => {
  if (!events.length) {
    return { timeline: [] as NormalizedStepEvent[], steps: createEmptyStepSummaries() };
  }

  const steps = createEmptyStepSummaries();
  const timeline: NormalizedStepEvent[] = [];
  let highestRetainedStep = 0;

  const recomputeHighest = () => {
    highestRetainedStep = 0;
    for (const item of timeline) {
      if (item.step_order > highestRetainedStep) {
        highestRetainedStep = item.step_order;
      }
    }
  };

  const dropStepsFrom = (step: number) => {
    for (let i = timeline.length - 1; i >= 0; i--) {
      if (timeline[i].step_order >= step) {
        timeline.splice(i, 1);
      }
    }
    for (let s = step; s <= STEP_COUNT; s++) {
      const idx = s - 1;
      steps[idx] = {
        step_order: s,
        started_at: null,
        completed_at: null,
        latest_event_type: null,
      };
    }
    recomputeHighest();
  };

  for (const rawEvent of events) {
    const step = Number(rawEvent.step_order);
    if (!Number.isInteger(step) || step < 1 || step > STEP_COUNT) {
      continue;
    }

    const normalizedEvent: NormalizedStepEvent = {
      event_id: rawEvent.event_id,
      experiment_id: rawEvent.experiment_id,
      student_id: rawEvent.student_id,
      step_order: step,
      event_type: rawEvent.event_type,
      event_timestamp: toIsoString(rawEvent.event_timestamp),
    };

    if (rawEvent.event_type === "STARTED") {
      if (step <= highestRetainedStep) {
        dropStepsFrom(step);
      }
      timeline.push(normalizedEvent);
      const stepRecord = steps[step - 1];
      stepRecord.started_at = normalizedEvent.event_timestamp;
      stepRecord.completed_at = null;
      stepRecord.latest_event_type = "STARTED";
      if (step > highestRetainedStep) {
        highestRetainedStep = step;
      }
      continue;
    }

    if (rawEvent.event_type === "COMPLETED") {
      for (let i = timeline.length - 1; i >= 0; i--) {
        if (
          timeline[i].step_order === step &&
          timeline[i].event_type === "COMPLETED"
        ) {
          timeline.splice(i, 1);
          break;
        }
      }
      timeline.push(normalizedEvent);
      const stepRecord = steps[step - 1];
      stepRecord.completed_at = normalizedEvent.event_timestamp;
      stepRecord.latest_event_type = "COMPLETED";
      if (step > highestRetainedStep) {
        highestRetainedStep = step;
      }
    }
  }

  return { timeline, steps };
};

const LATEST_GRADE_SELECTION_SQL = `
  SELECT
    u.user_id,
    u.username,
    u.full_name,
    u.email,
    latest.experiment_id,
    latest.start_time AS experiment_started_at,
    grade.exp_flow_demand_data_preparation,
    grade.exp_flow_demand_descriptive_stats,
    grade.exp_flow_demand_model_selection,
    grade.exp_flow_demand_generate_results,
    grade.exp_flow_production_inventory_calc,
    grade.exp_flow_production_service_level,
    grade.exp_flow_production_variable_calc,
    grade.exp_flow_production_plan_creation,
    grade.exp_flow_report_submission,
    grade.knowledge_test,
    grade.model_quality,
    grade.report_quality
  FROM enrollments e
  JOIN users u ON e.student_id = u.user_id
  LEFT JOIN (
    SELECT *
    FROM (
      SELECT
        es.*,
        ROW_NUMBER() OVER (
          PARTITION BY es.student_id
          ORDER BY COALESCE(es.last_activity_at, es.completion_time, es.start_time) DESC,
                   es.experiment_id DESC
        ) AS rn
      FROM experiment_status es
    ) ranked
    WHERE ranked.rn = 1
  ) latest ON latest.student_id = u.user_id
  LEFT JOIN experiment_grade grade
    ON grade.student_id = latest.student_id
   AND grade.experiment_id = latest.experiment_id
  WHERE e.class_id = ?
  ORDER BY u.username ASC
`;

const optionalGradeValue = z
  .number()
  .min(0, { message: "grade must be between 0 and 100" })
  .max(100, { message: "grade must be between 0 and 100" })
  .optional();

const optionalFeedbackValue = z
  .string()
  .transform((value) => {
    const trimmed = value.trim();
    return trimmed.length ? trimmed : null;
  })
  .optional();

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

const gradeReportSchema = z
  .object({
    grade: optionalGradeValue,
    feedback: optionalFeedbackValue,
  })
  .refine(
    (data) => data.grade !== undefined || data.feedback !== undefined,
    {
      message: "At least one grade field must be provided",
    },
  );

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

        // Optimized: Use window function instead of self-join with GROUP BY
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
             latest.feedback,
             grader.full_name AS grader_name
           FROM enrollments e
           JOIN users u ON e.student_id = u.user_id
           LEFT JOIN (
             SELECT * FROM (
               SELECT
                 er.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY er.student_id
                   ORDER BY er.submitted_at DESC, er.report_id DESC
                 ) AS rn
               FROM experiment_reports er
             ) ranked
             WHERE ranked.rn = 1
           ) latest ON latest.student_id = u.user_id
           LEFT JOIN users grader ON latest.graded_by = grader.user_id
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
        const { grade, feedback } = c.req.valid("json");

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

          if (grade !== undefined) {
            const [gradeResult]: any = await connection.query(
              `
                UPDATE experiment_grade
                   SET report_quality = ?
                 WHERE experiment_id = ? AND student_id = ?
              `,
              [grade, experiment_id, student_id],
            );

            if (gradeResult.affectedRows === 0) {
              await connection.rollback();
              return c.json({ error: "Grade record not found" }, 404);
            }
          }

          const reportUpdateFields: string[] = [];
          const reportUpdateValues: any[] = [];

          if (grade !== undefined) {
            reportUpdateFields.push("grade = ?");
            reportUpdateValues.push(grade);
          }
          if (feedback !== undefined) {
            reportUpdateFields.push("feedback = ?");
            reportUpdateValues.push(feedback);
          }

          if (reportUpdateFields.length > 0) {
            const graderId = Number.parseInt(String(payload.sub), 10);
            reportUpdateFields.push("graded_by = ?");
            reportUpdateValues.push(Number.isFinite(graderId) ? graderId : null);
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

          const [[updatedReport]]: any[] = await connection.query(
            `
              SELECT
                u.user_id,
                u.username,
                u.full_name,
                er.report_id,
                er.experiment_id,
                er.submitted_at,
                er.pdf_file_path,
                er.grade,
                er.feedback,
                grader.full_name AS grader_name
              FROM experiment_reports er
              JOIN users u ON u.user_id = er.student_id
              LEFT JOIN users grader ON er.graded_by = grader.user_id
              WHERE er.report_id = ?
            `,
            [reportIdNumber],
          );

          return c.json(
            updatedReport
              ? {
                  user_id: updatedReport.user_id,
                  username: updatedReport.username ?? null,
                  full_name: updatedReport.full_name ?? null,
                  report_id: updatedReport.report_id,
                  experiment_id: updatedReport.experiment_id,
                  submitted_at: updatedReport.submitted_at ?? null,
                  pdf_file_path: updatedReport.pdf_file_path ?? null,
                  grade: updatedReport.grade ?? null,
                  feedback: updatedReport.feedback ?? null,
                  grader_name: updatedReport.grader_name ?? null,
                }
              : null,
          );
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

        const [studentRows]: any[] = await pool.query(
          `SELECT
             u.user_id,
             u.username,
             u.full_name
           FROM enrollments e
           JOIN users u ON e.student_id = u.user_id
           WHERE e.class_id = ?
           ORDER BY u.username ASC`,
          [classIdNumber],
        );

        if (studentRows.length === 0) {
          return c.json([]);
        }

        const studentIds = Array.from(
          new Set(
            studentRows.map((row: any) => Number.parseInt(String(row.user_id), 10)),
          ),
        );
        const statusesByStudent = new Map<number, any[]>();

        if (studentIds.length > 0) {
          const placeholders = studentIds.map(() => "?").join(", ");
          const [statusRows]: any[] = await pool.query(
            `
              SELECT
                experiment_id,
                student_id,
                status,
                current_step,
                highest_completed_step,
                total_active_duration_seconds,
                selected_industry,
                selected_company,
                selected_product,
                start_time,
                last_activity_at,
                completion_time
              FROM experiment_status
              WHERE student_id IN (${placeholders})
              ORDER BY student_id ASC, experiment_id ASC
            `,
            studentIds,
          );

          for (const row of statusRows) {
            const studentId = Number.parseInt(String(row.student_id), 10);
            const normalized = {
              experiment_id: row.experiment_id,
              status: row.status,
              current_step: row.current_step,
              highest_completed_step: row.highest_completed_step,
              total_active_duration_seconds: row.total_active_duration_seconds,
              selected_industry: row.selected_industry ?? null,
              selected_company: row.selected_company ?? null,
              selected_product: row.selected_product ?? null,
              start_time: toIsoString(row.start_time),
              last_activity_at: toIsoString(row.last_activity_at),
              completion_time: toIsoString(row.completion_time),
            };
            if (!statusesByStudent.has(studentId)) {
              statusesByStudent.set(studentId, []);
            }
            statusesByStudent.get(studentId)!.push(normalized);
          }
        }

        const responsePayload = studentRows.map(
          (student: { user_id: number; username: string; full_name: string }) => {
            const studentId = Number.parseInt(String(student.user_id), 10);
            return {
              student_id: studentId,
              username: student.username,
              full_name: student.full_name,
              experiments: statusesByStudent.get(studentId) ?? [],
            };
          },
        );

        return c.json(responsePayload);
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
        ...EXP_FLOW_FIELDS,
        "knowledge_test",
        "model_quality",
        "report_quality",
        "exp_flow_score",
        "final_score",
      ];

      const csvLines: string[] = [header.join(",")];

      for (const row of gradeRows) {
        const studentId = row.user_id;

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
          ...EXP_FLOW_FIELDS.map((field) => formatNumeric(row[field])),
          formatNumeric(row.knowledge_test),
          formatNumeric(row.model_quality),
          formatNumeric(row.report_quality),
          toCsvValue(expFlowScoreStr),
          toCsvValue(finalScoreStr),
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

        const toNumberOrNull = (value: unknown) => {
          if (value === null || value === undefined) {
            return null;
          }
          const num = Number(value);
          return Number.isFinite(num) ? num : null;
        };

        const latestGrades = (gradeRows as any[]).map((row: any) => {
          const { expFlowScore, finalScore } = computeExperimentScores(row, weights);
          return {
            student_id: row.user_id,
            username: row.username,
            full_name: row.full_name,
            experiment_id: row.experiment_id,
            exp_flow_score:
              expFlowScore === null ? null : Number(expFlowScore.toFixed(2)),
            model_quality: toNumberOrNull(row.model_quality),
            knowledge_test: toNumberOrNull(row.knowledge_test),
            report_quality: toNumberOrNull(row.report_quality),
            final_score: finalScore === null ? null : Number(finalScore.toFixed(2)),
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

        const [studentExperimentRows]: any[] = await pool.query(
          `
            WITH ranked AS (
              SELECT
                es.*,
                ROW_NUMBER() OVER (
                  PARTITION BY es.student_id
                  ORDER BY COALESCE(es.last_activity_at, es.completion_time, es.start_time) DESC,
                           es.experiment_id DESC
                ) AS rn
              FROM experiment_status es
              JOIN enrollments e ON es.student_id = e.student_id
              WHERE e.class_id = ?
            )
            SELECT
              u.user_id AS student_id,
              u.username,
              u.full_name,
              latest.experiment_id,
              latest.status,
              latest.current_step,
              latest.highest_completed_step,
              latest.start_time,
              latest.last_activity_at,
              latest.completion_time
            FROM enrollments e
            JOIN users u ON e.student_id = u.user_id
            LEFT JOIN (
              SELECT *
              FROM ranked
              WHERE rn = 1
            ) latest ON latest.student_id = u.user_id
            WHERE e.class_id = ?
            ORDER BY u.username ASC
          `,
          [classIdNumber, classIdNumber],
        );

        const experimentIds = Array.from(
          new Set(
            (studentExperimentRows as any[])
              .map((row: any) => row.experiment_id)
              .filter((id: unknown): id is number => Number.isInteger(id)),
          ),
        );

        let eventsByExperiment = new Map<number, RawStepEvent[]>();

        if (experimentIds.length > 0) {
          const placeholders = experimentIds.map(() => "?").join(", ");
          const [eventRows]: any[] = await pool.query(
            `
              SELECT
                se.event_id,
                se.experiment_id,
                se.student_id,
                se.step_order,
                se.event_type,
                se.event_timestamp
              FROM experiment_step_events se
              WHERE se.experiment_id IN (${placeholders})
              ORDER BY se.experiment_id ASC, se.event_timestamp ASC, se.event_id ASC
            `,
            experimentIds,
          );

          const rows = eventRows as RawStepEvent[];
          eventsByExperiment = rows.reduce(
            (acc: Map<number, RawStepEvent[]>, event: RawStepEvent) => {
              if (!acc.has(event.experiment_id)) {
                acc.set(event.experiment_id, []);
              }
              acc.get(event.experiment_id)!.push(event);
              return acc;
            },
            new Map<number, RawStepEvent[]>(),
          );
        }

        const responsePayload = (studentExperimentRows as any[]).map((row: any) => {
          const experimentId = Number.isInteger(row.experiment_id)
            ? (row.experiment_id as number)
            : null;
          const timelineData = experimentId
            ? buildNormalizedStepTimeline(
              eventsByExperiment.get(experimentId) ?? [],
            )
            : { timeline: [] as NormalizedStepEvent[], steps: createEmptyStepSummaries() };

          return {
            student_id: row.student_id,
            username: row.username,
            full_name: row.full_name,
            experiment_id: experimentId,
            status: row.status ?? null,
            current_step: row.current_step ?? null,
            highest_completed_step: row.highest_completed_step ?? null,
            start_time: toIsoString(row.start_time),
            last_activity_at: toIsoString(row.last_activity_at),
            completion_time: toIsoString(row.completion_time),
            timeline: timelineData.timeline,
            steps: timelineData.steps,
          };
        });

        return c.json(responsePayload);
      } catch (error) {
        console.error("Error fetching step events:", error);
        throw error;
      }
    },
  );

  return classGradesApi;
};
