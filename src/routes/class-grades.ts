import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool, PoolConnection } from "mysql2/promise";
import { access, copyFile, mkdtemp, rm, mkdir } from "node:fs/promises";
import path from "path";
import os from "node:os";
import { handleUnexpectedError } from "../utils/error-handler";

// Zod Schemas for Validation
const classIdParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, { message: "classId 必须是数字" }),
});

const reportIdParamSchema = z.object({
  reportId: z.string().regex(/^\d+$/, { message: "reportId 必须是数字" }),
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
const EXPORTS_DIR = path.resolve(import.meta.dir, "../../uploads", "exports");

const EXPERIMENT_GRADE_FIELDS = [
  "exp_flow_demand_data_preparation",
  "exp_flow_demand_descriptive_stats",
  "exp_flow_demand_model_selection",
  "exp_flow_demand_generate_results",
  "exp_flow_production_inventory_calc",
  "exp_flow_production_service_level",
  "exp_flow_production_variable_calc",
  "exp_flow_production_plan_creation",
  "exp_flow_report_submission",
  "model_quality",
] as const;

const EXPERIMENT_GRADE_SELECT_FRAGMENT = EXPERIMENT_GRADE_FIELDS.length
  ? `,\n             ${EXPERIMENT_GRADE_FIELDS.map(
    (column) => `eg.${column} AS grade_${column}`,
  ).join(",\n             ")}`
  : "";

const EXPERIMENT_GRADE_MUTABLE_FIELDS = [...EXPERIMENT_GRADE_FIELDS] as const;

const toNumberOrNull = (value: unknown): number | null => {
  if (value === null || value === undefined) {
    return null;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
};

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
  const hasExperiment =
    row.experiment_id !== null && row.experiment_id !== undefined;

  const expFlowComponents = EXP_FLOW_FIELDS.map((field) => {
    const score = hasExperiment ? toNumberOrNull(row[field]) : null;
    const weight = Number(weights[field] ?? 0);
    const weighted =
      score === null || !Number.isFinite(weight)
        ? null
        : (score * weight) / 100;
    return {
      field,
      score,
      weight,
      weighted_score: weighted,
    };
  });

  const expFlowScoreRaw = expFlowComponents.reduce((sum, component) => {
    if (component.weighted_score === null) {
      return sum;
    }
    return sum + component.weighted_score;
  }, 0);
  const expFlowScore = hasExperiment
    ? expFlowComponents.some((component) => component.weighted_score !== null)
      ? expFlowScoreRaw
      : null
    : null;

  const knowledge = hasExperiment ? toNumberOrNull(row.knowledge_test) : null;
  const modelQuality = hasExperiment ? toNumberOrNull(row.model_quality) : null;
  const reportQuality = hasExperiment ? toNumberOrNull(row.report_quality) : null;
  const expWeight = Number(weights.exp_flow_weight ?? 0);
  const knowledgeWeight = Number(weights.knowledge_test_weight ?? 0);
  const modelWeight = Number(weights.model_quality_weight ?? 0);
  const reportWeight = Number(weights.report_quality_weight ?? 0);

  const buildFinalComponent = (
    score: number | null,
    weight: number,
  ): { score: number | null; weight: number; weighted_score: number | null } => {
    const normalizedWeight = Number.isFinite(weight) ? weight : 0;
    return {
      score,
      weight: normalizedWeight,
      weighted_score:
        score === null ? null : (score * normalizedWeight) / 100,
    };
  };

  const finalScoreComponents = {
    exp_flow: buildFinalComponent(expFlowScore, expWeight),
    knowledge_test: buildFinalComponent(knowledge, knowledgeWeight),
    model_quality: buildFinalComponent(modelQuality, modelWeight),
    report_quality: buildFinalComponent(reportQuality, reportWeight),
  };

  const finalScoreRaw = Object.values(finalScoreComponents).reduce(
    (sum, component) => {
      if (component.weighted_score === null) {
        return sum;
      }
      return sum + component.weighted_score;
    },
    0,
  );
  const finalScore = hasExperiment
    ? Object.values(finalScoreComponents).some(
      (component) => component.weighted_score !== null,
    )
      ? finalScoreRaw
      : null
    : null;

  return {
    expFlowScore: Number.isFinite(expFlowScore) ? expFlowScore : null,
    finalScore: Number.isFinite(finalScore) ? finalScore : null,
    expFlowComponents,
    finalScoreComponents,
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
      const timelineEntry = timeline[i];
      if (timelineEntry && timelineEntry.step_order >= step) {
        timeline.splice(i, 1);
      }
    }
    for (let s = step; s <= STEP_COUNT; s++) {
      const idx = s - 1;
      if (idx < 0 || idx >= steps.length) {
        continue;
      }
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
      const recordIndex = step - 1;
      const stepRecord = steps[recordIndex];
      if (!stepRecord) {
        continue;
      }
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
        const existing = timeline[i];
        if (
          existing &&
          existing.step_order === step &&
          existing.event_type === "COMPLETED"
        ) {
          timeline.splice(i, 1);
          break;
        }
      }
      timeline.push(normalizedEvent);
      const recordIndex = step - 1;
      const stepRecord = steps[recordIndex];
      if (!stepRecord) {
        continue;
      }
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
    grade.report_quality,
    er.status AS report_status
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
  LEFT JOIN experiment_reports er
    ON er.student_id = latest.student_id
   AND er.experiment_id = latest.experiment_id
  WHERE e.class_id = ?
  ORDER BY u.username ASC
`;

const createOptionalGradeValue = () =>
  z
    .number()
    .min(0, { message: "分数必须在 0 到 100 之间" })
    .max(100, { message: "分数必须在 0 到 100 之间" })
    .optional();

const optionalGradeValue = createOptionalGradeValue();

const optionalFeedbackValue = z
  .string()
  .transform((value) => {
    const trimmed = value.trim();
    return trimmed.length ? trimmed : null;
  })
  .optional();

const experimentGradeFieldSchemas = EXPERIMENT_GRADE_MUTABLE_FIELDS.reduce(
  (acc, field) => {
    acc[field] = createOptionalGradeValue();
    return acc;
  },
  {} as Record<
    (typeof EXPERIMENT_GRADE_MUTABLE_FIELDS)[number],
    ReturnType<typeof createOptionalGradeValue>
  >,
);

const experimentGradeUpdateSchema = z
  .object(experimentGradeFieldSchemas)
  .superRefine((data, ctx) => {
    const hasValue = EXPERIMENT_GRADE_MUTABLE_FIELDS.some(
      (field) => data[field] !== undefined,
    );
    if (!hasValue) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "至少需要提供一个实验成绩字段",
      });
    }
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
  message: "至少需要提供一个权重字段",
});

const optionalStatusValue = z.enum(["graded", "rejected"]).optional();

const gradeReportSchema = z
  .object({
    grade: optionalGradeValue,
    feedback: optionalFeedbackValue,
    status: optionalStatusValue,
    experiment_grade: experimentGradeUpdateSchema,
  })
  .refine(
    (data) =>
      data.grade !== undefined ||
      data.feedback !== undefined ||
      data.status !== undefined ||
      data.experiment_grade !== undefined,
    {
      message: "至少需要提供分数、反馈、状态或实验成绩中的一项",
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
          return c.json({ error: "班级不存在" }, 404);
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
          return c.json({ error: "权限不足" }, 403);
        }

        const [reportRows]: any[] = await pool.query(
          `SELECT
             u.user_id,
             u.username,
             u.full_name,
             er.report_id,
             latest.experiment_id,
             er.status,
             er.submitted_at,
             er.pdf_file_path,
             er.grade,
             er.feedback,
             grader.full_name AS grader_name${EXPERIMENT_GRADE_SELECT_FRAGMENT}
           FROM enrollments e
           JOIN users u ON e.student_id = u.user_id
           LEFT JOIN (
             SELECT
               ranked.student_id,
               ranked.experiment_id
             FROM (
               SELECT
                 es.student_id,
                 es.experiment_id,
                 ROW_NUMBER() OVER (
                   PARTITION BY es.student_id
                   ORDER BY es.experiment_id DESC
                 ) AS rn
               FROM experiment_status es
               JOIN enrollments e2 ON e2.student_id = es.student_id
               WHERE e2.class_id = ?
             ) ranked
             WHERE ranked.rn = 1
           ) latest ON latest.student_id = e.student_id
           LEFT JOIN experiment_reports er
             ON er.student_id = latest.student_id
            AND er.experiment_id = latest.experiment_id
           LEFT JOIN experiment_grade eg
             ON eg.student_id = latest.student_id
            AND eg.experiment_id = latest.experiment_id
           LEFT JOIN users grader ON er.graded_by = grader.user_id
           WHERE e.class_id = ?
           ORDER BY u.username ASC`,
          [classIdNumber, classIdNumber],
        );

        const reports = (reportRows as Record<string, any>[]).map((row) => {
          const sanitizedRow: Record<string, any> = { ...row };
          const gradePayload: Record<string, any> = {};
          let hasGrade = false;

          for (const column of EXPERIMENT_GRADE_FIELDS) {
            const alias = `grade_${column}`;
            const value = sanitizedRow[alias];
            gradePayload[column] = value ?? null;
            if (value !== null && value !== undefined) {
              hasGrade = true;
            }
            delete sanitizedRow[alias];
          }

          return {
            ...sanitizedRow,
            experiment_grade: hasGrade ? gradePayload : null,
          };
        });

        return c.json(reports);
      } catch (error) {
        return handleUnexpectedError(c, error, "获取班级报告失败");
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
          return c.json({ error: "权限不足" }, 403);
        }

        const { reportId } = c.req.valid("param");
        const reportIdNumber = Number.parseInt(reportId, 10);
        const {
          grade,
          feedback,
          status,
          experiment_grade: experimentGradeInput,
        } = c.req.valid("json");

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
            return c.json({ error: "报告不存在" }, 404);
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
            return c.json({ error: "权限不足" }, 403);
          }

          const experimentGradeUpdateFields: string[] = [];
          const experimentGradeUpdateValues: any[] = [];

          if (experimentGradeInput) {
            for (const field of EXPERIMENT_GRADE_MUTABLE_FIELDS) {
              const value = experimentGradeInput[field];
              if (value !== undefined) {
                experimentGradeUpdateFields.push(`${field} = ?`);
                experimentGradeUpdateValues.push(value);
              }
            }
          }

          if (grade !== undefined) {
            experimentGradeUpdateFields.push("report_quality = ?");
            experimentGradeUpdateValues.push(grade);
          }

          if (experimentGradeUpdateFields.length > 0) {
            const [gradeResult]: any = await connection.query(
              `
                UPDATE experiment_grade
                   SET ${experimentGradeUpdateFields.join(", ")}
                 WHERE experiment_id = ? AND student_id = ?
              `,
              [...experimentGradeUpdateValues, experiment_id, student_id],
            );

            if (gradeResult.affectedRows === 0) {
              await connection.rollback();
              return c.json({ error: "成绩记录不存在" }, 404);
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
          if (status !== undefined) {
            reportUpdateFields.push("status = ?");
            reportUpdateValues.push(status);
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
                er.status,
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

          const gradeSelectList = EXPERIMENT_GRADE_MUTABLE_FIELDS.join(", ");
          const [[updatedExperimentGrade]]: any[] = await connection.query(
            `
              SELECT ${gradeSelectList}
                FROM experiment_grade
               WHERE experiment_id = ? AND student_id = ?
               LIMIT 1
            `,
            [experiment_id, student_id],
          );

          const experimentGradeResponse = updatedExperimentGrade
            ? EXPERIMENT_GRADE_FIELDS.reduce<Record<string, number | null>>(
              (acc, field) => {
                const value = updatedExperimentGrade[field];
                acc[field] =
                  value === null || value === undefined ? null : Number(value);
                return acc;
              },
              {},
            )
            : null;

          return c.json(
            updatedReport
              ? {
                user_id: updatedReport.user_id,
                username: updatedReport.username ?? null,
                full_name: updatedReport.full_name ?? null,
                report_id: updatedReport.report_id,
                experiment_id: updatedReport.experiment_id,
                status: updatedReport.status ?? null,
                submitted_at: updatedReport.submitted_at ?? null,
                pdf_file_path: updatedReport.pdf_file_path ?? null,
                grade: updatedReport.grade ?? null,
                feedback: updatedReport.feedback ?? null,
                grader_name: updatedReport.grader_name ?? null,
                experiment_grade: experimentGradeResponse,
              }
              : null,
          );
        } catch (error) {
          await connection.rollback();
          return handleUnexpectedError(c, error, "批改报告失败");
        } finally {
          connection.release();
        }
      } catch (error) {
        return handleUnexpectedError(c, error, "批改报告失败");
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
          return c.json({ error: "班级不存在" }, 404);
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
          return c.json({ error: "权限不足" }, 403);
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

        const responsePayload = studentRows
          .map(
            (student: {
              user_id: number;
              username: string;
              full_name: string;
            }) => {
              const studentId = Number.parseInt(String(student.user_id), 10);
              return {
                student_id: studentId,
                username: student.username,
                full_name: student.full_name,
                experiments: statusesByStudent.get(studentId) ?? [],
              };
            },
          )
          .sort((a: { username: string }, b: { username: string }) => a.username.localeCompare(b.username));

        return c.json(responsePayload);
      } catch (error) {
        return handleUnexpectedError(c, error, "获取实验状态失败");
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
          return c.json({ error: "班级不存在" }, 404);
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
          return c.json({ error: "权限不足" }, 403);
        }

        const [weights]: any[] = await pool.query(
          "SELECT * FROM grade_weights WHERE class_id = ?",
          [classIdNumber],
        );

        if (weights.length === 0) {
          return c.json({ error: "未找到该班级的成绩权重" }, 404);
        }

        return c.json(weights[0]);
      } catch (error) {
        return handleUnexpectedError(c, error, "获取成绩权重失败");
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
        return c.json({ error: "班级不存在" }, 404);
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
        return c.json({ error: "权限不足" }, 403);
      }

      const [reports]: any[] = await pool.query(
        `SELECT
           er.pdf_file_path,
           er.experiment_id,
           er.student_id,
           u.username,
           u.full_name
         FROM (
           SELECT
             es.student_id,
             MAX(es.experiment_id) AS latest_experiment_id
           FROM experiment_status es
           JOIN enrollments e ON es.student_id = e.student_id
          WHERE e.class_id = ?
          GROUP BY es.student_id
         ) latest
         JOIN experiment_reports er
           ON er.student_id = latest.student_id
          AND er.experiment_id = latest.latest_experiment_id
         JOIN users u ON u.user_id = latest.student_id
        WHERE er.pdf_file_path IS NOT NULL
          AND er.status != 'rejected'`,
        [classIdNumber],
      );

      if (reports.length === 0) {
        return c.json({ error: "未找到该班级的有效报告（可能是因为学生尚未提交或报告已被驳回）" }, 404);
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
          return c.json({ error: "没有可下载的报告文件" }, 404);
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
            `启动 zip 命令失败。请确保服务器已安装 'zip': ${spawnError}`,
          );
        }

        const exitCode = await zipProcess.exited;
        if (exitCode !== 0) {
          const errorText = await new Response(zipProcess.stderr).text();
          throw new Error(`zip 命令执行失败: ${errorText}`);
        }

        await mkdir(EXPORTS_DIR, { recursive: true });
        const exportFileName = `class-${classIdNumber}-reports-${Date.now()}.zip`;
        const exportFilePath = path.join(EXPORTS_DIR, exportFileName);
        await copyFile(zipPath, exportFilePath);
        const relativePath = `uploads/exports/${exportFileName}`;

        return c.json({
          file_path: relativePath,
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
        return c.json({ error: "班级不存在" }, 404);
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
        return c.json({ error: "权限不足" }, 403);
      }

      const [weightRows]: any[] = await pool.query(
        "SELECT * FROM grade_weights WHERE class_id = ?",
        [classIdNumber],
      );

      if (weightRows.length === 0) {
        return c.json({ error: "该班级未配置成绩权重" }, 404);
      }

      const weights = weightRows[0] as Record<string, number>;

      const [gradeRows]: any[] = await pool.query(
        LATEST_GRADE_SELECTION_SQL,
        [classIdNumber],
      );

      if (gradeRows.length === 0) {
        return c.json({ error: "该班级未找到已加入的学生" }, 404);
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
        "学号",
        "姓名",
        "实验ID",
        "数据准备",
        "描述性统计",
        "预测模型选择",
        "生成预测结果",
        "库存变量参数计算",
        "服务水平参数计算",
        "生产变量参数计算",
        "制定生产计划",
        "提交实验报告",
        "实验流程",
        "知识点测试",
        "模型选择",
        "实验报告",
        "报告状态",
        "最终得分",
      ];

      const csvLines: string[] = [header.join(",")];

      for (const row of gradeRows) {
        const studentId = row.user_id;
        const isRejected = row.report_status === "rejected";

        const { expFlowScore, finalScore } = computeExperimentScores(row, weights);
        const expFlowScoreStr =
          expFlowScore === null ? "" : Number(expFlowScore).toFixed(2);
        const finalScoreStr =
          finalScore === null ? "" : Number(finalScore).toFixed(2);

        // Helper to conditionally return value or empty string based on rejection status
        const val = (v: string) => (isRejected ? "" : v);

        let statusText = "未提交";
        if (row.report_status === "submitted") statusText = "已提交";
        else if (row.report_status === "graded") statusText = "已评分";
        else if (row.report_status === "rejected") statusText = "已驳回";

        const rowValues: string[] = [
          toCsvValue(row.username),
          toCsvValue(row.full_name),
          toCsvValue(row.experiment_id),
          ...EXP_FLOW_FIELDS.map((field) => val(formatNumeric(row[field]))),
          val(toCsvValue(expFlowScoreStr)),
          val(formatNumeric(row.knowledge_test)),
          val(formatNumeric(row.model_quality)),
          val(formatNumeric(row.report_quality)),
          toCsvValue(statusText),
          val(toCsvValue(finalScoreStr)),
        ];

        csvLines.push(rowValues.join(","));
      }

      const csvContent = "\uFEFF" + csvLines.join("\n");

      await mkdir(EXPORTS_DIR, { recursive: true });
      const exportFileName = `class-${classIdNumber}-grades-${Date.now()}.csv`;
      const exportFilePath = path.join(EXPORTS_DIR, exportFileName);
      await Bun.write(exportFilePath, csvContent);
      const relativePath = `uploads/exports/${exportFileName}`;

      return c.json({
        file_path: relativePath,
      });
    },
  );

  // GET /api/v1/classes/:classId/report-export.csv - 导出班级实验报告CSV
  classGradesApi.get(
    "/api/v1/classes/:classId/report-export.csv",
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
        return c.json({ error: "班级不存在" }, 404);
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
        return c.json({ error: "权限不足" }, 403);
      }

      const [reportRows]: any[] = await pool.query(
        `SELECT
           u.user_id,
           u.username,
           u.full_name,
           er.report_id,
           latest.experiment_id,
           er.status,
           er.submitted_at,
           er.pdf_file_path,
           er.grade,
           er.feedback,
           grader.full_name AS grader_name
         FROM enrollments e
         JOIN users u ON e.student_id = u.user_id
         LEFT JOIN (
           SELECT
             ranked.student_id,
             ranked.experiment_id
           FROM (
             SELECT
               es.student_id,
               es.experiment_id,
               ROW_NUMBER() OVER (
                 PARTITION BY es.student_id
                 ORDER BY es.experiment_id DESC
               ) AS rn
             FROM experiment_status es
             JOIN enrollments e2 ON e2.student_id = es.student_id
             WHERE e2.class_id = ?
           ) ranked
           WHERE ranked.rn = 1
         ) latest ON latest.student_id = e.student_id
         LEFT JOIN experiment_reports er
           ON er.student_id = latest.student_id
          AND er.experiment_id = latest.experiment_id
         LEFT JOIN users grader ON er.graded_by = grader.user_id
         WHERE e.class_id = ?
         ORDER BY u.username ASC`,
        [classIdNumber, classIdNumber],
      );

      if (reportRows.length === 0) {
        return c.json({ error: "该班级未找到已加入的学生" }, 404);
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

      const formatDateTime = (value: unknown) => {
        if (value === null || value === undefined) {
          return "";
        }
        if (value instanceof Date) {
          return value.toISOString().replace('T', ' ').substring(0, 19);
        }
        const dateStr = String(value);
        try {
          const date = new Date(dateStr);
          if (!isNaN(date.getTime())) {
            return date.toISOString().replace('T', ' ').substring(0, 19);
          }
        } catch (e) {
          // ignore
        }
        return dateStr;
      };

      const header = [
        "学号",
        "姓名",
        "实验ID",
        "报告ID",
        "报告状态",
        "提交时间",
        "报告得分",
        "评阅人",
        "评语",
      ];

      const csvLines: string[] = [header.join(",")];

      for (const row of reportRows) {
        let statusText = "未提交";
        if (row.status === "submitted") statusText = "已提交";
        else if (row.status === "graded") statusText = "已评分";
        else if (row.status === "rejected") statusText = "已驳回";

        const rowValues: string[] = [
          toCsvValue(row.username),
          toCsvValue(row.full_name),
          toCsvValue(row.experiment_id),
          toCsvValue(row.report_id),
          toCsvValue(statusText),
          toCsvValue(formatDateTime(row.submitted_at)),
          formatNumeric(row.grade),
          toCsvValue(row.grader_name),
          toCsvValue(row.feedback),
        ];

        csvLines.push(rowValues.join(","));
      }

      const csvContent = "\uFEFF" + csvLines.join("\n");

      await mkdir(EXPORTS_DIR, { recursive: true });
      const exportFileName = `class-${classIdNumber}-reports-${Date.now()}.csv`;
      const exportFilePath = path.join(EXPORTS_DIR, exportFileName);
      await Bun.write(exportFilePath, csvContent);
      const relativePath = `uploads/exports/${exportFileName}`;

      return c.json({
        file_path: relativePath,
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
          return c.json({ error: "权限不足" }, 403);
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
          return c.json({ error: "班级不存在" }, 404);
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
          return c.json({ error: "权限不足" }, 403);
        }

        // Check if weights exist
        const [existing]: any[] = await pool.query(
          "SELECT weight_id FROM grade_weights WHERE class_id = ?",
          [classIdNumber],
        );

        if (existing.length === 0) {
          return c.json({ error: "未找到该班级的成绩权重" }, 404);
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
        return handleUnexpectedError(c, error, "更新成绩权重失败");
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
          return c.json({ error: "班级不存在" }, 404);
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
          return c.json({ error: "权限不足" }, 403);
        }

        const [weightRows]: any[] = await pool.query(
          "SELECT * FROM grade_weights WHERE class_id = ?",
          [classIdNumber],
        );

        if (weightRows.length === 0) {
          return c.json({ error: "该班级未配置成绩权重" }, 404);
        }

        const weights = weightRows[0] as Record<string, number>;

        const [gradeRows]: any[] = await pool.query(
          LATEST_GRADE_SELECTION_SQL,
          [classIdNumber],
        );

        const latestGrades = (gradeRows as any[]).map((row: any) => {
          const {
            expFlowScore,
            finalScore,
            expFlowComponents,
            finalScoreComponents,
          } = computeExperimentScores(row, weights);
          const formatScore = (value: number | null) =>
            value === null ? null : Number(value.toFixed(2));
          const expFlowBreakdown = expFlowComponents.map((component) => ({
            field: component.field,
            score:
              component.score === null
                ? null
                : Number(component.score.toFixed(2)),
            weight: component.weight,
            weighted_score: formatScore(component.weighted_score),
          }));
          const formatFinalComponent = (component: {
            score: number | null;
            weight: number;
            weighted_score: number | null;
          }) => ({
            score: formatScore(component.score),
            weight: component.weight,
            weighted_score: formatScore(component.weighted_score),
          });
          const finalScoreBreakdown = {
            exp_flow: formatFinalComponent(finalScoreComponents.exp_flow),
            knowledge_test: formatFinalComponent(
              finalScoreComponents.knowledge_test,
            ),
            model_quality: formatFinalComponent(
              finalScoreComponents.model_quality,
            ),
            report_quality: formatFinalComponent(
              finalScoreComponents.report_quality,
            ),
          };
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
            report_status: row.report_status ?? null,
            final_score: finalScore === null ? null : Number(finalScore.toFixed(2)),
            exp_flow_breakdown: expFlowBreakdown,
            final_score_breakdown: finalScoreBreakdown,
          };
        });

        return c.json(latestGrades);
      } catch (error) {
        return handleUnexpectedError(c, error, "获取最新成绩失败");
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
          return c.json({ error: "班级不存在" }, 404);
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
          return c.json({ error: "权限不足" }, 403);
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
        return handleUnexpectedError(c, error, "获取实验步骤事件失败");
      }
    },
  );

  return classGradesApi;
};
