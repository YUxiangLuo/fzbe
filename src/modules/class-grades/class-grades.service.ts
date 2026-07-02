import path from "path";
import os from "node:os";
import { access, copyFile, mkdtemp, rm, mkdir } from "node:fs/promises";
import type { RowDataPacket, ResultSetHeader, Pool, PoolConnection } from "mysql2/promise";
import { pool } from "../../config";
import { EXP_FLOW_FIELDS, EXPERIMENT_GRADE_FIELDS, EXPERIMENT_GRADE_MUTABLE_FIELDS, type GradeReportInput, type UpdateGradeWeightsInput } from "./class-grades.schema";

const STEP_COUNT = 7;
const EXPORTS_DIR = path.resolve(import.meta.dir, "../../../uploads", "exports");

const TOP_LEVEL_WEIGHT_FIELDS = [
  "exp_flow_weight",
  "knowledge_test_weight",
  "model_quality_weight",
  "report_quality_weight",
] as const;

function toPublicReportPath(filePath: string | null | undefined): string | null {
  if (!filePath) return null;
  return `reports/${path.basename(filePath)}`;
}

type Queryable = Pool | PoolConnection;

interface ClassRow extends RowDataPacket { teacher_id: number; }
interface AssignmentRow extends RowDataPacket { }
interface ClassSummarySourceRow extends RowDataPacket {
  class_id: number;
  class_name: string;
  term_id: number;
  academic_year: string;
  semester: 1 | 2;
  term_label: string;
  is_current_term: number | boolean;
}
interface TeacherClassRow extends ClassSummarySourceRow {}
interface AssistantClassRow extends ClassSummarySourceRow {}
interface UserRoleRow extends RowDataPacket { role: string; }

export interface TeacherClassGradeSummary {
  class_id: number;
  class_name: string;
  term_id: number;
  academic_year: string;
  semester: 1 | 2;
  term_label: string;
  is_current_term: boolean;
  total_students: number;
  graded_count: number;
  submitted_count: number;
  rejected_count: number;
  not_submitted_count: number;
  average_score: number | null;
}

export async function isAssistantAssignedToClass(db: Queryable, classId: number, assistantId: number): Promise<boolean> {
  const [[assignment]] = await db.query<AssignmentRow[]>("SELECT 1 FROM class_assistants WHERE class_id = ? AND assistant_id = ? LIMIT 1", [classId, assistantId]);
  return Boolean(assignment);
}

async function isAssistantAssignedToStudent(db: Queryable, studentId: number, assistantId: number): Promise<boolean> {
  const [[assignment]] = await db.query<AssignmentRow[]>("SELECT 1 FROM enrollments e JOIN class_assistants ca ON e.class_id = ca.class_id WHERE e.student_id = ? AND ca.assistant_id = ? LIMIT 1", [studentId, assistantId]);
  return Boolean(assignment);
}

async function isTeacherOfStudent(db: Queryable, studentId: number, teacherId: number): Promise<boolean> {
  const [[row]] = await db.query<AssignmentRow[]>("SELECT 1 FROM enrollments e JOIN classes c ON e.class_id = c.class_id WHERE e.student_id = ? AND c.teacher_id = ? LIMIT 1", [studentId, teacherId]);
  return Boolean(row);
}

export async function checkClassAccess(classId: number, userId: number, userRole: string, allowStudents = false) {
  const [classRows] = await pool.query<ClassRow[]>("SELECT teacher_id FROM classes WHERE class_id = ?", [classId]);
  if (classRows.length === 0) return { error: "not_found" as const };

  let isAllowed = userRole === "Admin" || userId === classRows[0]!.teacher_id;
  if (!isAllowed && userRole === "Assistant") isAllowed = await isAssistantAssignedToClass(pool, classId, userId);
  if (!isAllowed && allowStudents && userRole === "Student") {
    const [[enrollment]] = await pool.query<AssignmentRow[]>("SELECT enrollment_id FROM enrollments WHERE student_id = ? AND class_id = ?", [userId, classId]);
    if (enrollment) isAllowed = true;
  }

  if (!isAllowed) return { error: "forbidden" as const };
  return { allowed: true, teacherId: classRows[0]!.teacher_id };
}

const toNumberOrNull = (value: unknown): number | null => {
  if (value === null || value === undefined) return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
};

const toIsoString = (value: unknown): string | null => {
  if (!value) return null;
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const parsed = new Date(trimmed);
    return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
  }
  const parsed = new Date(value as string);
  return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
};

const sumWeightFields = (weights: Record<string, unknown>, fields: readonly string[]) =>
  fields.reduce((sum, field) => sum + Number(weights[field] ?? 0), 0);

export function validateGradeWeightTotals(weights: Record<string, unknown>) {
  const topLevelTotal = sumWeightFields(weights, TOP_LEVEL_WEIGHT_FIELDS);
  const flowDetailTotal = sumWeightFields(weights, EXP_FLOW_FIELDS);
  const topLevelValid = Number.isFinite(topLevelTotal) && Math.abs(topLevelTotal - 100) < 0.0001;
  const flowDetailValid = Number.isFinite(flowDetailTotal) && Math.abs(flowDetailTotal - 100) < 0.0001;

  return {
    valid: topLevelValid && flowDetailValid,
    topLevelTotal,
    flowDetailTotal,
    topLevelValid,
    flowDetailValid,
  };
}

export const buildCsvContent = (csvLines: string[]) => `\uFEFF${csvLines.join("\r\n")}\r\n`;

export function computeExperimentScores(row: Record<string, unknown>, weights: Record<string, number>) {
  const hasExperiment = row.experiment_id !== null && row.experiment_id !== undefined;

  const expFlowComponents = EXP_FLOW_FIELDS.map((field) => {
    const score = hasExperiment ? toNumberOrNull(row[field]) : null;
    const weight = Number(weights[field] ?? 0);
    const weighted = score === null || !Number.isFinite(weight) ? null : (score * weight) / 100;
    return { field, score, weight, weighted_score: weighted };
  });

  const expFlowScoreRaw = expFlowComponents.reduce((sum, c) => c.weighted_score === null ? sum : sum + c.weighted_score, 0);
  const expFlowScore = hasExperiment ? (expFlowComponents.some((c) => c.weighted_score !== null) ? expFlowScoreRaw : null) : null;

  const knowledge = hasExperiment ? toNumberOrNull(row.knowledge_test) : null;
  const modelQuality = hasExperiment ? toNumberOrNull(row.model_quality) : null;
  const reportQuality = hasExperiment ? toNumberOrNull(row.report_quality) : null;

  const buildFinalComponent = (score: number | null, weight: number) => {
    const normalizedWeight = Number.isFinite(weight) ? weight : 0;
    return { score, weight: normalizedWeight, weighted_score: score === null ? null : (score * normalizedWeight) / 100 };
  };

  const finalScoreComponents = {
    exp_flow: buildFinalComponent(expFlowScore, Number(weights.exp_flow_weight ?? 0)),
    knowledge_test: buildFinalComponent(knowledge, Number(weights.knowledge_test_weight ?? 0)),
    model_quality: buildFinalComponent(modelQuality, Number(weights.model_quality_weight ?? 0)),
    report_quality: buildFinalComponent(reportQuality, Number(weights.report_quality_weight ?? 0)),
  };

  const finalScoreRaw = Object.values(finalScoreComponents).reduce((sum, c) => c.weighted_score === null ? sum : sum + c.weighted_score, 0);
  const finalScore = hasExperiment ? (Object.values(finalScoreComponents).some((c) => c.weighted_score !== null) ? finalScoreRaw : null) : null;

  return { expFlowScore: Number.isFinite(expFlowScore) ? expFlowScore : null, finalScore: Number.isFinite(finalScore) ? finalScore : null, expFlowComponents, finalScoreComponents };
}

const EXPERIMENT_GRADE_SELECT_FRAGMENT = EXPERIMENT_GRADE_FIELDS.length ? `,\n             ${EXPERIMENT_GRADE_FIELDS.map((column) => `eg.${column} AS grade_${column}`).join(",\n             ")}` : "";

const CLASS_SUMMARY_SELECT_FRAGMENT = `
  SELECT c.class_id,
         c.class_name,
         c.term_id,
         at.academic_year,
         at.semester,
         CONCAT(at.academic_year, ' 第', at.semester, '学期') AS term_label,
         at.is_active AS is_current_term
    FROM classes c
    JOIN academic_terms at ON at.term_id = c.term_id`;

const appendTermFilter = (conditions: string[], values: unknown[], termId?: number) => {
  if (termId !== undefined) {
    conditions.push("c.term_id = ?");
    values.push(termId);
  }
};

export async function getClassReports(classId: number) {
  const [reportRows] = await pool.query<RowDataPacket[]>(`SELECT u.user_id, u.username, u.full_name, er.report_id, latest.experiment_id, er.status, er.submitted_at, er.pdf_file_path, er.grade, er.feedback, grader.full_name AS grader_name${EXPERIMENT_GRADE_SELECT_FRAGMENT}
    FROM enrollments e JOIN users u ON e.student_id = u.user_id
    LEFT JOIN (SELECT ranked.student_id, ranked.experiment_id FROM (SELECT es.student_id, es.experiment_id, ROW_NUMBER() OVER (PARTITION BY es.student_id ORDER BY es.experiment_id DESC) AS rn FROM experiment_status es JOIN enrollments e2 ON e2.student_id = es.student_id WHERE e2.class_id = ?) ranked WHERE ranked.rn = 1) latest ON latest.student_id = e.student_id
    LEFT JOIN experiment_reports er ON er.student_id = latest.student_id AND er.experiment_id = latest.experiment_id
    LEFT JOIN experiment_grade eg ON eg.student_id = latest.student_id AND eg.experiment_id = latest.experiment_id
    LEFT JOIN users grader ON er.graded_by = grader.user_id WHERE e.class_id = ? ORDER BY u.username ASC`, [classId, classId]);

  return reportRows.map((row) => {
    const sanitizedRow: Record<string, unknown> = { ...row };
    const gradePayload: Record<string, unknown> = {};
    let hasGrade = false;
    for (const column of EXPERIMENT_GRADE_FIELDS) {
      const alias = `grade_${column}`;
      const value = sanitizedRow[alias];
      gradePayload[column] = value ?? null;
      if (value !== null && value !== undefined) hasGrade = true;
      delete sanitizedRow[alias];
    }
    return {
      ...sanitizedRow,
      pdf_file_path: toPublicReportPath(sanitizedRow.pdf_file_path as string | null | undefined),
      experiment_grade: hasGrade ? gradePayload : null,
    };
  });
}

export async function gradeReport(reportId: number, userId: number, userRole: string, input: GradeReportInput) {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();
    const [[reportRow]] = await connection.query<RowDataPacket[]>("SELECT experiment_id, student_id FROM experiment_reports WHERE report_id = ?", [reportId]);
    if (!reportRow) { await connection.rollback(); return { error: "not_found" as const }; }

    const { experiment_id, student_id } = reportRow;

    let authorized = userRole === "Admin";
    if (!authorized && userRole === "Teacher") authorized = await isTeacherOfStudent(connection, student_id, userId);
    if (!authorized && userRole === "Assistant") authorized = await isAssistantAssignedToStudent(connection, student_id, userId);
    if (!authorized) { await connection.rollback(); return { error: "forbidden" as const }; }

    const experimentGradeUpdateFields: string[] = [];
    const experimentGradeUpdateValues: unknown[] = [];

    if (input.experiment_grade) {
      for (const field of EXPERIMENT_GRADE_MUTABLE_FIELDS) {
        const value = input.experiment_grade[field];
        if (value !== undefined) { experimentGradeUpdateFields.push(`${field} = ?`); experimentGradeUpdateValues.push(value); }
      }
    }
    if (input.grade !== undefined) { experimentGradeUpdateFields.push("report_quality = ?"); experimentGradeUpdateValues.push(input.grade); }

    if (experimentGradeUpdateFields.length > 0) {
      const [gradeResult] = await connection.query<ResultSetHeader>(`UPDATE experiment_grade SET ${experimentGradeUpdateFields.join(", ")} WHERE experiment_id = ? AND student_id = ?`, [...experimentGradeUpdateValues, experiment_id, student_id]);
      if (gradeResult.affectedRows === 0) { await connection.rollback(); return { error: "grade_not_found" as const }; }
    }

    const reportUpdateFields: string[] = [];
    const reportUpdateValues: unknown[] = [];
    if (input.grade !== undefined) { reportUpdateFields.push("grade = ?"); reportUpdateValues.push(input.grade); }
    if (input.feedback !== undefined) { reportUpdateFields.push("feedback = ?"); reportUpdateValues.push(input.feedback); }
    if (input.status !== undefined) { reportUpdateFields.push("status = ?"); reportUpdateValues.push(input.status); }

    if (reportUpdateFields.length > 0) {
      reportUpdateFields.push("graded_by = ?");
      reportUpdateValues.push(userId);
      reportUpdateValues.push(reportId);
      await connection.query(`UPDATE experiment_reports SET ${reportUpdateFields.join(", ")} WHERE report_id = ?`, reportUpdateValues);
    }

    await connection.commit();

    const [[updatedReport]] = await connection.query<RowDataPacket[]>(`SELECT u.user_id, u.username, u.full_name, er.report_id, er.experiment_id, er.status, er.submitted_at, er.pdf_file_path, er.grade, er.feedback, grader.full_name AS grader_name FROM experiment_reports er JOIN users u ON u.user_id = er.student_id LEFT JOIN users grader ON er.graded_by = grader.user_id WHERE er.report_id = ?`, [reportId]);
    const [[updatedExperimentGrade]] = await connection.query<RowDataPacket[]>(`SELECT ${EXPERIMENT_GRADE_MUTABLE_FIELDS.join(", ")} FROM experiment_grade WHERE experiment_id = ? AND student_id = ? LIMIT 1`, [experiment_id, student_id]);

    const experimentGradeResponse = updatedExperimentGrade ? EXPERIMENT_GRADE_FIELDS.reduce<Record<string, number | null>>((acc, field) => { acc[field] = updatedExperimentGrade[field] ?? null; return acc; }, {}) : null;

    return {
      data: {
        ...updatedReport,
        pdf_file_path: toPublicReportPath(updatedReport?.pdf_file_path as string | null | undefined),
        experiment_grade: experimentGradeResponse,
      },
    };
  } catch (error) {
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}

export async function getClassExperimentRuns(classId: number) {
  const [studentRows] = await pool.query<RowDataPacket[]>("SELECT u.user_id, u.username, u.full_name FROM enrollments e JOIN users u ON e.student_id = u.user_id WHERE e.class_id = ? ORDER BY u.username ASC", [classId]);
  if (studentRows.length === 0) return [];

  const studentIds = studentRows.map((row) => row.user_id);
  const statusesByStudent = new Map<number, unknown[]>();

  if (studentIds.length > 0) {
    const placeholders = studentIds.map(() => "?").join(", ");
    const [statusRows] = await pool.query<RowDataPacket[]>(`SELECT experiment_id, student_id, status, current_step, highest_completed_step, total_active_duration_seconds, selected_industry, selected_company, selected_product, start_time, last_activity_at, completion_time FROM experiment_status WHERE student_id IN (${placeholders}) ORDER BY student_id ASC, experiment_id ASC`, studentIds);

    for (const row of statusRows) {
      const studentId = row.student_id;
      const normalized = { experiment_id: row.experiment_id, status: row.status, current_step: row.current_step, highest_completed_step: row.highest_completed_step, total_active_duration_seconds: row.total_active_duration_seconds, selected_industry: row.selected_industry ?? null, selected_company: row.selected_company ?? null, selected_product: row.selected_product ?? null, start_time: toIsoString(row.start_time), last_activity_at: toIsoString(row.last_activity_at), completion_time: toIsoString(row.completion_time) };
      if (!statusesByStudent.has(studentId)) statusesByStudent.set(studentId, []);
      statusesByStudent.get(studentId)!.push(normalized);
    }
  }

  return studentRows.map((student) => ({ student_id: student.user_id, username: student.username, full_name: student.full_name, experiments: statusesByStudent.get(student.user_id) ?? [] })).sort((a, b) => a.username.localeCompare(b.username));
}

export async function getGradingPolicy(classId: number) {
  const [weights] = await pool.query<RowDataPacket[]>("SELECT * FROM grade_weights WHERE class_id = ?", [classId]);
  if (weights.length === 0) return null;
  return weights[0];
}

export async function updateGradingPolicy(classId: number, input: UpdateGradeWeightsInput) {
  const [existing] = await pool.query<RowDataPacket[]>("SELECT * FROM grade_weights WHERE class_id = ?", [classId]);
  if (existing.length === 0) return { error: "not_found" as const };

  const mergedWeights = { ...existing[0], ...input };
  const weightValidation = validateGradeWeightTotals(mergedWeights);
  if (!weightValidation.valid) return { error: "invalid_weight_total" as const, details: weightValidation };

  const editableWeightFields = [
    "exp_flow_weight",
    ...EXP_FLOW_FIELDS,
    "model_quality_weight",
    "knowledge_test_weight",
    "report_quality_weight",
  ] as const;

  const fields: string[] = [];
  const values: unknown[] = [];
  for (const field of editableWeightFields) {
    const value = input[field];
    if (value !== undefined) {
      fields.push(`${field} = ?`);
      values.push(value);
    }
  }

  values.push(classId);
  await pool.query(`UPDATE grade_weights SET ${fields.join(", ")} WHERE class_id = ?`, values);

  const [updatedWeights] = await pool.query<RowDataPacket[]>("SELECT * FROM grade_weights WHERE class_id = ?", [classId]);
  return { data: updatedWeights[0] };
}

const LATEST_GRADE_SELECTION_SQL = `SELECT u.user_id, u.username, u.full_name, u.email,
       latest.experiment_id, latest.start_time AS experiment_started_at,
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
    SELECT ranked.*
      FROM (
        SELECT es.*,
               ROW_NUMBER() OVER (PARTITION BY es.student_id ORDER BY es.experiment_id DESC) AS rn
          FROM experiment_status es
          JOIN enrollments e2 ON e2.student_id = es.student_id
         WHERE e2.class_id = ?
      ) ranked
     WHERE ranked.rn = 1
  ) latest ON latest.student_id = u.user_id
  LEFT JOIN experiment_grade grade ON grade.student_id = latest.student_id AND grade.experiment_id = latest.experiment_id
  LEFT JOIN experiment_reports er ON er.student_id = latest.student_id AND er.experiment_id = latest.experiment_id
 WHERE e.class_id = ?
 ORDER BY u.username ASC`;

export async function getGradeSummaries(classId: number) {
  const [weightRows] = await pool.query<RowDataPacket[]>("SELECT * FROM grade_weights WHERE class_id = ?", [classId]);
  if (weightRows.length === 0) return { error: "no_weights" as const };

  const weights = weightRows[0] as Record<string, number>;
  const [gradeRows] = await pool.query<RowDataPacket[]>(LATEST_GRADE_SELECTION_SQL, [classId, classId]);

  const formatScore = (value: number | null) => value === null ? null : Number(value.toFixed(2));

  return {
    data: gradeRows.map((row) => {
      const reportStatus = row.report_status ?? null;
      const hasFinalGrade = reportStatus === "graded";
      const scoreRow = hasFinalGrade ? row : { ...row, report_quality: null };
      const { expFlowScore, finalScore, expFlowComponents, finalScoreComponents } = computeExperimentScores(scoreRow, weights);
      const expFlowBreakdown = expFlowComponents.map((c) => ({ field: c.field, score: formatScore(c.score), weight: c.weight, weighted_score: formatScore(c.weighted_score) }));
      const formatFinalComponent = (c: { score: number | null; weight: number; weighted_score: number | null }) => ({ score: formatScore(c.score), weight: c.weight, weighted_score: formatScore(c.weighted_score) });

      return {
        student_id: row.user_id, username: row.username, full_name: row.full_name, experiment_id: row.experiment_id,
        exp_flow_score: formatScore(expFlowScore), model_quality: toNumberOrNull(row.model_quality), knowledge_test: toNumberOrNull(row.knowledge_test), report_quality: hasFinalGrade ? toNumberOrNull(row.report_quality) : null, report_status: reportStatus, final_score: hasFinalGrade ? formatScore(finalScore) : null,
        exp_flow_breakdown: expFlowBreakdown,
        final_score_breakdown: { exp_flow: formatFinalComponent(finalScoreComponents.exp_flow), knowledge_test: formatFinalComponent(finalScoreComponents.knowledge_test), model_quality: formatFinalComponent(finalScoreComponents.model_quality), report_quality: formatFinalComponent(finalScoreComponents.report_quality) },
      };
    }),
  };
}

function buildTeacherClassSummary(
  classRow: ClassSummarySourceRow,
  gradeRows: Array<{ report_status: string | null; final_score: number | null }>,
): TeacherClassGradeSummary {
  const gradedCount = gradeRows.filter((row) => row.report_status === "graded").length;
  const submittedCount = gradeRows.filter((row) => row.report_status === "submitted").length;
  const rejectedCount = gradeRows.filter((row) => row.report_status === "rejected").length;
  const notSubmittedCount = gradeRows.filter((row) => !row.report_status).length;
  const validScores = gradeRows
    .filter((row) => row.report_status === "graded" && row.final_score !== null)
    .map((row) => row.final_score as number);

  const averageScore = validScores.length > 0
    ? validScores.reduce((sum, score) => sum + score, 0) / validScores.length
    : null;

  return {
    class_id: classRow.class_id,
    class_name: classRow.class_name,
    term_id: classRow.term_id,
    academic_year: classRow.academic_year,
    semester: classRow.semester,
    term_label: classRow.term_label,
    is_current_term: Boolean(classRow.is_current_term),
    total_students: gradeRows.length,
    graded_count: gradedCount,
    submitted_count: submittedCount,
    rejected_count: rejectedCount,
    not_submitted_count: notSubmittedCount,
    average_score: averageScore,
  };
}

const createEmptyTeacherClassSummary = (classRow: ClassSummarySourceRow): TeacherClassGradeSummary => ({
  class_id: classRow.class_id,
  class_name: classRow.class_name,
  term_id: classRow.term_id,
  academic_year: classRow.academic_year,
  semester: classRow.semester,
  term_label: classRow.term_label,
  is_current_term: Boolean(classRow.is_current_term),
  total_students: 0,
  graded_count: 0,
  submitted_count: 0,
  rejected_count: 0,
  not_submitted_count: 0,
  average_score: null,
});

async function buildClassGradeSummaries(classRows: ClassSummarySourceRow[]) {
  const summaries = await Promise.all(
    classRows.map(async (classRow) => {
      try {
        const result = await getGradeSummaries(classRow.class_id);
        if ("error" in result) return createEmptyTeacherClassSummary(classRow);

        const summaryRows = result.data.map((row) => ({
          report_status: row.report_status ?? null,
          final_score: toNumberOrNull(row.final_score),
        }));

        return buildTeacherClassSummary(classRow, summaryRows);
      } catch {
        return createEmptyTeacherClassSummary(classRow);
      }
    }),
  );

  return summaries;
}

export async function getTeacherClassGradeSummaries(teacherId: number, userId: number, userRole: string, termId?: number) {
  if (userRole !== "Admin" && userId !== teacherId) return { error: "forbidden" as const };

  const [teacherRows] = await pool.query<UserRoleRow[]>(
    "SELECT role FROM users WHERE user_id = ? AND role = 'Teacher'",
    [teacherId],
  );
  if (teacherRows.length === 0) return { error: "not_teacher" as const };

  const conditions = ["c.teacher_id = ?"];
  const values: unknown[] = [teacherId];
  appendTermFilter(conditions, values, termId);

  const [classRows] = await pool.query<TeacherClassRow[]>(
    `${CLASS_SUMMARY_SELECT_FRAGMENT}
      WHERE ${conditions.join(" AND ")}
      ORDER BY c.class_id ASC`,
    values,
  );
  if (classRows.length === 0) return { data: [] as TeacherClassGradeSummary[] };

  const summaries = await buildClassGradeSummaries(classRows);
  return { data: summaries };
}

export async function getAssistantClassGradeSummaries(assistantId: number, userId: number, userRole: string, termId?: number) {
  if (userRole !== "Admin" && userId !== assistantId) return { error: "forbidden" as const };

  const [assistantRows] = await pool.query<UserRoleRow[]>(
    "SELECT role FROM users WHERE user_id = ? AND role = 'Assistant'",
    [assistantId],
  );
  if (assistantRows.length === 0) return { error: "not_assistant" as const };

  const conditions = ["ca.assistant_id = ?"];
  const values: unknown[] = [assistantId];
  appendTermFilter(conditions, values, termId);

  const [classRows] = await pool.query<AssistantClassRow[]>(
    `${CLASS_SUMMARY_SELECT_FRAGMENT}
       JOIN class_assistants ca ON c.class_id = ca.class_id
      WHERE ${conditions.join(" AND ")}
      ORDER BY c.class_id ASC`,
    values,
  );
  if (classRows.length === 0) return { data: [] as TeacherClassGradeSummary[] };

  const summaries = await buildClassGradeSummaries(classRows);
  return { data: summaries };
}

export async function exportReportArchive(classId: number) {
  const [reports] = await pool.query<RowDataPacket[]>(`SELECT er.pdf_file_path, er.experiment_id, er.student_id, u.username, u.full_name FROM (SELECT es.student_id, MAX(es.experiment_id) AS latest_experiment_id FROM experiment_status es JOIN enrollments e ON es.student_id = e.student_id WHERE e.class_id = ? GROUP BY es.student_id) latest JOIN experiment_reports er ON er.student_id = latest.student_id AND er.experiment_id = latest.latest_experiment_id JOIN users u ON u.user_id = latest.student_id WHERE er.pdf_file_path IS NOT NULL AND er.status != 'rejected'`, [classId]);

  if (reports.length === 0) return { error: "no_reports" as const };

  const tempDir = await mkdtemp(path.join(os.tmpdir(), "class-report-export-"));
  const copiedFiles: string[] = [];
  const usedNames = new Set<string>();

  try {
    for (const report of reports) {
      const originalPath = report.pdf_file_path as string | null;
      if (!originalPath) continue;
      try { await access(originalPath); } catch { continue; }

      const rawName = (report.username as string) || (report.full_name as string) || `student_${report.student_id}`;
      const baseName = `${rawName}_experiment_${report.experiment_id}`.replace(/\s+/g, "_").replace(/[^a-zA-Z0-9_\-\.]/g, "_");
      let candidate = `${baseName}.pdf`;
      let suffix = 1;
      while (usedNames.has(candidate)) { candidate = `${baseName}_${suffix}.pdf`; suffix += 1; }
      usedNames.add(candidate);

      const destination = path.join(tempDir, candidate);
      await copyFile(originalPath, destination);
      copiedFiles.push(destination);
    }

    if (copiedFiles.length === 0) return { error: "no_files" as const };

    const zipFileName = `class-${classId}-reports.zip`;
    const zipPath = path.join(tempDir, zipFileName);
    const zipProcess = Bun.spawn(["zip", "-j", zipPath, ...copiedFiles.map((f) => path.basename(f))], { cwd: tempDir, stdout: "pipe", stderr: "pipe" });
    const exitCode = await zipProcess.exited;
    if (exitCode !== 0) throw new Error(`zip 命令执行失败`);

    await mkdir(EXPORTS_DIR, { recursive: true });
    const exportFileName = `class-${classId}-reports-${Date.now()}.zip`;
    const exportFilePath = path.join(EXPORTS_DIR, exportFileName);
    await copyFile(zipPath, exportFilePath);

    return { data: { file_path: `exports/${exportFileName}` } };
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

export async function exportGradeCsv(classId: number) {
  const [weightRows] = await pool.query<RowDataPacket[]>("SELECT * FROM grade_weights WHERE class_id = ?", [classId]);
  if (weightRows.length === 0) return { error: "no_weights" as const };

  const weights = weightRows[0] as Record<string, number>;
  const [gradeRows] = await pool.query<RowDataPacket[]>(LATEST_GRADE_SELECTION_SQL, [classId, classId]);
  if (gradeRows.length === 0) return { error: "no_students" as const };

  const toCsvValue = (value: unknown) => { if (value === null || value === undefined) return ""; let str = value instanceof Date ? value.toISOString() : String(value); if (/[",\n]/.test(str)) return `"${str.replace(/"/g, '""')}"`; return str; };
  const formatNumeric = (value: unknown) => { if (value === null || value === undefined) return ""; const num = Number(value); return Number.isFinite(num) ? num.toFixed(2) : ""; };

  const header = ["学号", "姓名", "实验ID", "数据准备", "描述性统计", "预测模型选择", "生成预测结果", "库存变量参数计算", "服务水平参数计算", "生产变量参数计算", "制定生产计划", "提交实验报告", "实验流程", "知识点测试", "模型选择", "实验报告", "报告状态", "最终得分"];
  const csvLines: string[] = [header.join(",")];

  for (const row of gradeRows) {
    const isRejected = row.report_status === "rejected";
    const isGraded = row.report_status === "graded";
    const scoreRow = isGraded ? row : { ...row, report_quality: null };
    const { expFlowScore, finalScore } = computeExperimentScores(scoreRow, weights);
    const val = (v: string) => isRejected ? "" : v;
    let statusText = "未提交";
    if (row.report_status === "submitted") statusText = "待评分";
    else if (row.report_status === "graded") statusText = "已评分";
    else if (row.report_status === "rejected") statusText = "已驳回";

    const reportQuality = isGraded ? formatNumeric(row.report_quality) : "";
    const finalScoreValue = isGraded && finalScore !== null ? finalScore.toFixed(2) : "";
    const rowValues = [toCsvValue(row.username), toCsvValue(row.full_name), toCsvValue(row.experiment_id), ...EXP_FLOW_FIELDS.map((field) => val(formatNumeric(row[field]))), val(expFlowScore === null ? "" : expFlowScore.toFixed(2)), val(formatNumeric(row.knowledge_test)), val(formatNumeric(row.model_quality)), val(reportQuality), toCsvValue(statusText), val(finalScoreValue)];
    csvLines.push(rowValues.join(","));
  }

  const csvContent = buildCsvContent(csvLines);
  await mkdir(EXPORTS_DIR, { recursive: true });
  const exportFileName = `class-${classId}-grades-${Date.now()}.csv`;
  const exportFilePath = path.join(EXPORTS_DIR, exportFileName);
  await Bun.write(exportFilePath, csvContent);

  return { data: { file_path: `exports/${exportFileName}` } };
}

export async function exportReportCsv(classId: number) {
  const [reportRows] = await pool.query<RowDataPacket[]>(`SELECT u.user_id, u.username, u.full_name, er.report_id, latest.experiment_id, er.status, er.submitted_at, er.pdf_file_path, er.grade, er.feedback, grader.full_name AS grader_name FROM enrollments e JOIN users u ON e.student_id = u.user_id LEFT JOIN (SELECT ranked.student_id, ranked.experiment_id FROM (SELECT es.student_id, es.experiment_id, ROW_NUMBER() OVER (PARTITION BY es.student_id ORDER BY es.experiment_id DESC) AS rn FROM experiment_status es JOIN enrollments e2 ON e2.student_id = es.student_id WHERE e2.class_id = ?) ranked WHERE ranked.rn = 1) latest ON latest.student_id = e.student_id LEFT JOIN experiment_reports er ON er.student_id = latest.student_id AND er.experiment_id = latest.experiment_id LEFT JOIN users grader ON er.graded_by = grader.user_id WHERE e.class_id = ? ORDER BY u.username ASC`, [classId, classId]);

  if (reportRows.length === 0) return { error: "no_students" as const };

  const toCsvValue = (value: unknown) => { if (value === null || value === undefined) return ""; let str = value instanceof Date ? value.toISOString() : String(value); if (/[",\n]/.test(str)) return `"${str.replace(/"/g, '""')}"`; return str; };
  const formatNumeric = (value: unknown) => { if (value === null || value === undefined) return ""; const num = Number(value); return Number.isFinite(num) ? num.toFixed(2) : ""; };
  const formatDateTime = (value: unknown) => { if (value === null || value === undefined) return ""; if (value instanceof Date) return value.toISOString().replace("T", " ").substring(0, 19); try { const date = new Date(String(value)); if (!isNaN(date.getTime())) return date.toISOString().replace("T", " ").substring(0, 19); } catch {} return String(value); };

  const header = ["学号", "姓名", "实验ID", "报告ID", "报告状态", "提交时间", "报告得分", "评阅人", "评语"];
  const csvLines: string[] = [header.join(",")];

  for (const row of reportRows) {
    let statusText = "未提交";
    if (row.status === "submitted") statusText = "待评分";
    else if (row.status === "graded") statusText = "已评分";
    else if (row.status === "rejected") statusText = "已驳回";

    const rowValues = [toCsvValue(row.username), toCsvValue(row.full_name), toCsvValue(row.experiment_id), toCsvValue(row.report_id), toCsvValue(statusText), toCsvValue(formatDateTime(row.submitted_at)), formatNumeric(row.grade), toCsvValue(row.grader_name), toCsvValue(row.feedback)];
    csvLines.push(rowValues.join(","));
  }

  const csvContent = buildCsvContent(csvLines);
  await mkdir(EXPORTS_DIR, { recursive: true });
  const exportFileName = `class-${classId}-reports-${Date.now()}.csv`;
  const exportFilePath = path.join(EXPORTS_DIR, exportFileName);
  await Bun.write(exportFilePath, csvContent);

  return { data: { file_path: `exports/${exportFileName}` } };
}

type RawStepEvent = { event_id: number; experiment_id: number; student_id: number; step_order: number; event_type: "STARTED" | "COMPLETED"; event_timestamp: unknown };
type NormalizedStepEvent = { event_id: number; experiment_id: number; student_id: number; step_order: number; event_type: "STARTED" | "COMPLETED"; event_timestamp: string | null };
type StepSummary = { step_order: number; started_at: string | null; completed_at: string | null; latest_event_type: "STARTED" | "COMPLETED" | null };

const createEmptyStepSummaries = (): StepSummary[] => Array.from({ length: STEP_COUNT }, (_, i) => ({ step_order: i + 1, started_at: null, completed_at: null, latest_event_type: null }));

function buildNormalizedStepTimeline(events: RawStepEvent[]) {
  if (!events.length) return { timeline: [] as NormalizedStepEvent[], steps: createEmptyStepSummaries() };

  const steps = createEmptyStepSummaries();
  const timeline: NormalizedStepEvent[] = [];
  let highestRetainedStep = 0;

  const recomputeHighest = () => { highestRetainedStep = 0; for (const item of timeline) if (item.step_order > highestRetainedStep) highestRetainedStep = item.step_order; };
  const dropStepsFrom = (step: number) => {
    for (let i = timeline.length - 1; i >= 0; i--) if (timeline[i]!.step_order >= step) timeline.splice(i, 1);
    for (let s = step; s <= STEP_COUNT; s++) { const idx = s - 1; if (idx >= 0 && idx < steps.length) steps[idx] = { step_order: s, started_at: null, completed_at: null, latest_event_type: null }; }
    recomputeHighest();
  };

  for (const rawEvent of events) {
    const step = Number(rawEvent.step_order);
    if (!Number.isInteger(step) || step < 1 || step > STEP_COUNT) continue;

    const normalizedEvent: NormalizedStepEvent = { event_id: rawEvent.event_id, experiment_id: rawEvent.experiment_id, student_id: rawEvent.student_id, step_order: step, event_type: rawEvent.event_type, event_timestamp: toIsoString(rawEvent.event_timestamp) };

    if (rawEvent.event_type === "STARTED") {
      if (step <= highestRetainedStep) dropStepsFrom(step);
      timeline.push(normalizedEvent);
      const stepRecord = steps[step - 1]!;
      stepRecord.started_at = normalizedEvent.event_timestamp;
      stepRecord.completed_at = null;
      stepRecord.latest_event_type = "STARTED";
      if (step > highestRetainedStep) highestRetainedStep = step;
    } else if (rawEvent.event_type === "COMPLETED") {
      for (let i = timeline.length - 1; i >= 0; i--) if (timeline[i]!.step_order === step && timeline[i]!.event_type === "COMPLETED") { timeline.splice(i, 1); break; }
      timeline.push(normalizedEvent);
      const stepRecord = steps[step - 1]!;
      stepRecord.completed_at = normalizedEvent.event_timestamp;
      stepRecord.latest_event_type = "COMPLETED";
      if (step > highestRetainedStep) highestRetainedStep = step;
    }
  }

  return { timeline, steps };
}

export async function getExperimentEvents(classId: number) {
  const [studentExperimentRows] = await pool.query<RowDataPacket[]>(`WITH ranked AS (SELECT es.*, ROW_NUMBER() OVER (PARTITION BY es.student_id ORDER BY COALESCE(es.last_activity_at, es.completion_time, es.start_time) DESC, es.experiment_id DESC) AS rn FROM experiment_status es JOIN enrollments e ON es.student_id = e.student_id WHERE e.class_id = ?) SELECT u.user_id AS student_id, u.username, u.full_name, latest.experiment_id, latest.status, latest.current_step, latest.highest_completed_step, latest.start_time, latest.last_activity_at, latest.completion_time FROM enrollments e JOIN users u ON e.student_id = u.user_id LEFT JOIN (SELECT * FROM ranked WHERE rn = 1) latest ON latest.student_id = u.user_id WHERE e.class_id = ? ORDER BY u.username ASC`, [classId, classId]);

  const experimentIds = Array.from(new Set(studentExperimentRows.map((row) => row.experiment_id).filter((id): id is number => Number.isInteger(id))));
  let eventsByExperiment = new Map<number, RawStepEvent[]>();

  if (experimentIds.length > 0) {
    const placeholders = experimentIds.map(() => "?").join(", ");
    const [eventRows] = await pool.query<RowDataPacket[]>(`SELECT se.event_id, se.experiment_id, se.student_id, se.step_order, se.event_type, se.event_timestamp FROM experiment_step_events se WHERE se.experiment_id IN (${placeholders}) ORDER BY se.experiment_id ASC, se.event_timestamp ASC, se.event_id ASC`, experimentIds);
    eventsByExperiment = (eventRows as RawStepEvent[]).reduce((acc, event) => { if (!acc.has(event.experiment_id)) acc.set(event.experiment_id, []); acc.get(event.experiment_id)!.push(event); return acc; }, new Map<number, RawStepEvent[]>());
  }

  return studentExperimentRows.map((row) => {
    const experimentId = Number.isInteger(row.experiment_id) ? row.experiment_id : null;
    const timelineData = experimentId ? buildNormalizedStepTimeline(eventsByExperiment.get(experimentId) ?? []) : { timeline: [], steps: createEmptyStepSummaries() };
    return { student_id: row.student_id, username: row.username, full_name: row.full_name, experiment_id: experimentId, status: row.status ?? null, current_step: row.current_step ?? null, highest_completed_step: row.highest_completed_step ?? null, start_time: toIsoString(row.start_time), last_activity_at: toIsoString(row.last_activity_at), completion_time: toIsoString(row.completion_time), timeline: timelineData.timeline, steps: timelineData.steps };
  });
}
