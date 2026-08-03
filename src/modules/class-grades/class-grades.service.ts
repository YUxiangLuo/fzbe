import path from "path";
import os from "node:os";
import { createHash, randomUUID } from "node:crypto";
import { copyFile, mkdtemp, rm } from "node:fs/promises";
import type { RowDataPacket, ResultSetHeader, Pool, PoolConnection } from "mysql2/promise";
import { env, pool } from "../../config";
import { CLEANUP_RETENTION } from "../../utils/constants";
import { cleanupDirectoryEntriesOlderThan } from "../../utils/disk-cleanup";
import {
  FILESYSTEM_FILENAME_MAX_BYTES,
  fitFilenameToUtf8Bytes,
  sanitizeFilename,
} from "../../utils/file-validator";
import {
  ensureDatabaseManagedStorageRoot,
  getDatabaseManagedStorageRoot,
} from "../../utils/managed-storage";
import { resolveExistingManagedFile } from "../../utils/managed-file";
import { databaseDatetimeToIsoString } from "../../utils/database-datetime";
import { EXP_FLOW_FIELDS, EXPERIMENT_GRADE_FIELDS, EXPERIMENT_GRADE_MUTABLE_FIELDS, type GradeReportInput, type UpdateGradeWeightsInput } from "./class-grades.schema";

const STEP_COUNT = 7;
const EXPORTS_DIR = path.resolve(import.meta.dir, "../../../uploads", "exports");
const REPORTS_DIR = path.resolve(import.meta.dir, "../../../uploads", "reports");
const exportsStorageRoot = () => getDatabaseManagedStorageRoot(EXPORTS_DIR, env);
const GENERATED_EXPORT_UUID_PATTERN = "[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}";
const GENERATED_EXPORT_TOKEN_PATTERN = `\\d+(?:-${GENERATED_EXPORT_UUID_PATTERN})?`;
export const GENERATED_EXPORT_FILE_PATTERN = new RegExp(
  `^class-(\\d+)-(?:(?:reports-${GENERATED_EXPORT_TOKEN_PATTERN}\\.(?:zip|csv))|(?:grades-${GENERATED_EXPORT_TOKEN_PATTERN}\\.csv))$`,
);
const ZIP_EXPORT_TIMEOUT_MS = 60_000;
const REDO_IN_PROGRESS_STATUS = "redo_in_progress" as const;
const REVIEW_REVISION_EXPERIMENT_GRADE_FIELDS = [
  ...EXPERIMENT_GRADE_MUTABLE_FIELDS,
  "report_quality",
] as const;

export function buildGeneratedExportFilename(
  classId: number,
  kind: "reports" | "grades",
  extension: "zip" | "csv",
  timestamp = Date.now(),
  generationToken = randomUUID(),
): string {
  if (kind === "grades" && extension !== "csv") {
    throw new Error("成绩导出仅支持 CSV 格式");
  }
  return `class-${classId}-${kind}-${timestamp}-${generationToken}.${extension}`;
}

export function parseClassIdFromExportFilename(filename: string): number | null {
  const match = filename.match(GENERATED_EXPORT_FILE_PATTERN);
  if (!match?.[1]) return null;
  const classId = Number(match[1]);
  return Number.isSafeInteger(classId) && classId > 0 ? classId : null;
}

interface ReportArchiveEntryIdentity {
  username?: string | null;
  fullName?: string | null;
  studentId: number;
  experimentId: number;
}

/**
 * Allocates a traversal-safe ZIP entry name that remains within the usual
 * filesystem NAME_MAX limit measured in UTF-8 bytes. The stable database
 * identities stay outside the truncatable display-name prefix, so long or
 * colliding usernames cannot make two students overwrite one another in the
 * temporary export directory.
 */
export function allocateReportArchiveEntryFilename(
  identity: ReportArchiveEntryIdentity,
  usedNames: Set<string>,
): string {
  const studentId = Number(identity.studentId);
  const experimentId = Number(identity.experimentId);
  if (!Number.isSafeInteger(studentId) || studentId <= 0) {
    throw new Error("报告归档学生 ID 无效");
  }
  if (!Number.isSafeInteger(experimentId) || experimentId <= 0) {
    throw new Error("报告归档实验 ID 无效");
  }

  const rawName = identity.username || identity.fullName || `student_${studentId}`;
  // Dots have no identifying value here and removing them makes the resulting
  // component incapable of becoming either `.` or `..`. sanitizeFilename also
  // strips every path separator before the name is joined to the temp root.
  const safeStem = sanitizeFilename(rawName)
    .replace(/\.+/g, "_")
    .replace(/^[-_]+|[-_]+$/g, "") || "student";

  for (let collision = 0; ; collision += 1) {
    const collisionSuffix = collision === 0 ? "" : `_${collision}`;
    const identitySuffix = `_student_${studentId}_experiment_${experimentId}${collisionSuffix}.pdf`;
    const suffixBytes = Buffer.byteLength(identitySuffix, "utf8");
    const stemByteBudget = FILESYSTEM_FILENAME_MAX_BYTES - suffixBytes;
    if (stemByteBudget <= 0) {
      throw new Error("报告归档唯一标识超过文件名长度限制");
    }

    // Reuse the shared code-point-aware truncator. Supplying a temporary
    // extension lets it bound just the display stem while preserving Unicode
    // characters without splitting a UTF-8 sequence.
    const boundedStemWithExtension = fitFilenameToUtf8Bytes(
      `${safeStem}.pdf`,
      stemByteBudget + Buffer.byteLength(".pdf", "utf8"),
    );
    const boundedStem = path.basename(boundedStemWithExtension, ".pdf");
    const candidate = `${boundedStem}${identitySuffix}`;
    if (!usedNames.has(candidate)) {
      usedNames.add(candidate);
      return candidate;
    }
  }
}

async function createZipArchive(
  zipPath: string,
  sourceBasenames: readonly string[],
  workingDirectory: string,
): Promise<void> {
  const zipProcess = Bun.spawn(
    // `--` keeps a valid but dash-prefixed student identifier from being
    // interpreted as another zip option.
    ["zip", "-q", "-j", zipPath, "--", ...sourceBasenames],
    { cwd: workingDirectory, stdout: "ignore", stderr: "pipe" },
  );
  // Start consuming stderr immediately; otherwise a full pipe can block a
  // large archive before the parent ever awaits process completion.
  const stderrPromise = new Response(zipProcess.stderr).text().catch(() => "");
  let timedOut = false;
  let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<number>((resolve) => {
    timeoutHandle = setTimeout(() => {
      timedOut = true;
      zipProcess.kill("SIGKILL");
      resolve(-1);
    }, ZIP_EXPORT_TIMEOUT_MS);
  });
  const exitCode = await Promise.race([zipProcess.exited, timeout]);
  if (timeoutHandle) clearTimeout(timeoutHandle);

  if (timedOut) {
    // Give the killed child a short bounded window to close its pipes without
    // turning export cleanup into an unbounded wait.
    await Promise.race([
      zipProcess.exited,
      new Promise((resolve) => setTimeout(resolve, 1_000)),
    ]);
    throw new Error(`zip 命令执行超时（${ZIP_EXPORT_TIMEOUT_MS / 1_000} 秒）`);
  }

  const stderr = (await stderrPromise).trim();
  if (exitCode !== 0) {
    const detail = stderr ? `: ${stderr.slice(0, 500)}` : "";
    throw new Error(`zip 命令执行失败（退出码 ${exitCode}）${detail}`);
  }
}

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

export async function resolveStoredReportPdfPath(
  filePath: string | null | undefined,
  options: { reportsDir?: string } = {},
): Promise<string | null> {
  if (!filePath) return null;
  const managedRoot = path.resolve(options.reportsDir ?? REPORTS_DIR);
  const resolved = await resolveExistingManagedFile({
    storedPath: filePath,
    managedRoot,
    environment: env,
    legacyFallbackDirectory: getDatabaseManagedStorageRoot(managedRoot, env),
    enforceDisposableNamespace: true,
  }).catch(() => null);
  return resolved?.absolutePath ?? null;
}

export async function isReportPdfAvailable(
  filePath: string | null | undefined,
  options: { reportsDir?: string } = {},
): Promise<boolean> {
  return Boolean(await resolveStoredReportPdfPath(filePath, options));
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

/**
 * Build an opaque optimistic-concurrency token from every value a report
 * reviewer can overwrite. Numeric database strings are normalized so the
 * list and locked-row queries produce the same token across mysql2 settings.
 */
export function buildReportReviewRevision(
  report: Record<string, unknown>,
  experimentGrade: Record<string, unknown> | null | undefined,
): string {
  const canonicalExperimentGrade = REVIEW_REVISION_EXPERIMENT_GRADE_FIELDS.reduce<Record<string, number | null>>(
    (result, field) => {
      result[field] = toNumberOrNull(experimentGrade?.[field]);
      return result;
    },
    {},
  );
  const canonicalState = {
    version: 1,
    report: {
      status: report.status === null || report.status === undefined ? null : String(report.status),
      grade: toNumberOrNull(report.grade),
      feedback: report.feedback === null || report.feedback === undefined ? null : String(report.feedback),
      graded_by: toNumberOrNull(report.graded_by),
    },
    experiment_grade: canonicalExperimentGrade,
  };

  return createHash("sha256").update(JSON.stringify(canonicalState)).digest("hex");
}

const toIsoString = (value: unknown): string | null => {
  return databaseDatetimeToIsoString(value, env.DB_TIMEZONE);
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

const SPREADSHEET_FORMULA_PREFIX = /^(?:[\t\r\n\u0000]|\s*[=+\-@＝＋－＠])/u;

/**
 * Encode an untrusted value as one CSV cell.
 *
 * Quoting/doubling prevents delimiter injection. A leading tab keeps formula-like
 * values as text when the human-facing CSV is opened in Excel/LibreOffice.
 */
export const encodeCsvCell = (value: unknown) => {
  if (value === null || value === undefined) return "";
  const rawValue = value instanceof Date ? value.toISOString() : String(value);
  const safeValue = SPREADSHEET_FORMULA_PREFIX.test(rawValue) ? `\t${rawValue}` : rawValue;
  if (safeValue !== rawValue || /[",\r\n]/.test(safeValue)) {
    return `"${safeValue.replace(/"/g, '""')}"`;
  }
  return safeValue;
};

/**
 * Format a database date-time string without changing its wall-clock value.
 * Date instances are an instant rather than a wall-clock value, so their
 * deterministic fallback representation is UTC. Export queries should prefer
 * a DATE_FORMAT alias when the database wall-clock value is required.
 */
export const formatCsvDateTime = (value: unknown) => {
  if (value === null || value === undefined || value === "") return "";

  const rawValue = String(value).trim();
  const databaseDateTime = rawValue.match(
    /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:\.\d+)?$/,
  );
  if (databaseDateTime) return `${databaseDateTime[1]} ${databaseDateTime[2]}`;

  const date = value instanceof Date ? value : new Date(rawValue);
  if (Number.isNaN(date.getTime())) return rawValue;
  return date.toISOString().replace("T", " ").slice(0, 19);
};

const formatNumeric = (value: unknown) => {
  if (value === null || value === undefined) return "";
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(2) : "";
};

const reportStatusText = (status: unknown) => {
  if (status === "submitted") return "待评分";
  if (status === "graded") return "已评分";
  if (status === "rejected") return "已驳回";
  if (status === REDO_IN_PROGRESS_STATUS) return "重做中";
  return "未提交";
};

const GRADE_EXPORT_HEADER = ["学号", "姓名", "实验ID", "数据准备", "描述性统计", "预测模型选择", "生成预测结果", "库存变量参数计算", "服务水平参数计算", "生产变量参数计算", "制定生产计划", "提交实验报告", "实验流程", "知识点测试", "模型选择", "实验报告", "报告状态", "报告评语", "最终得分"];

export function buildGradeCsvLines(gradeRows: Array<Record<string, unknown>>, weights: Record<string, number>) {
  const csvLines: string[] = [GRADE_EXPORT_HEADER.join(",")];

  for (const row of gradeRows) {
    const isGraded = row.report_status === "graded";
    const { expFlowScore, finalScore } = computeExperimentScores(row, weights);
    const officialScore = (value: string) => isGraded ? value : "";

    const reportQuality = isGraded ? formatNumeric(row.report_quality) : "";
    const finalScoreValue = isGraded && finalScore !== null ? finalScore.toFixed(2) : "";
    const rowValues = [encodeCsvCell(row.username), encodeCsvCell(row.full_name), encodeCsvCell(row.experiment_id), ...EXP_FLOW_FIELDS.map((field) => officialScore(formatNumeric(row[field]))), officialScore(expFlowScore === null ? "" : expFlowScore.toFixed(2)), officialScore(formatNumeric(row.knowledge_test)), officialScore(formatNumeric(row.model_quality)), officialScore(reportQuality), encodeCsvCell(reportStatusText(row.report_status)), encodeCsvCell(row.report_feedback), officialScore(finalScoreValue)];
    csvLines.push(rowValues.join(","));
  }

  return csvLines;
}

const EXPERIMENT_GRADE_SELECT_FRAGMENT = REVIEW_REVISION_EXPERIMENT_GRADE_FIELDS.length ? `,\n             ${REVIEW_REVISION_EXPERIMENT_GRADE_FIELDS.map((column) => `eg.${column} AS grade_${column}`).join(",\n             ")}` : "";

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
  const [reportRows] = await pool.query<RowDataPacket[]>(`SELECT u.user_id,
           u.username,
           u.full_name,
           er.report_id,
           latest.experiment_id,
           latest.retry_of_experiment_id,
           CASE
             WHEN er.report_id IS NULL
              AND latest.retry_of_experiment_id IS NOT NULL
              AND parent_er.status = 'rejected'
             THEN '${REDO_IN_PROGRESS_STATUS}'
             ELSE er.status
           END AS status,
           er.submitted_at,
           er.pdf_file_path,
           er.grade,
           er.graded_by AS review_graded_by,
           CASE
             WHEN er.report_id IS NULL
              AND latest.retry_of_experiment_id IS NOT NULL
              AND parent_er.status = 'rejected'
             THEN parent_er.feedback
             ELSE er.feedback
           END AS feedback,
           CASE
             WHEN er.report_id IS NULL
              AND latest.retry_of_experiment_id IS NOT NULL
              AND parent_er.status = 'rejected'
             THEN parent_grader.full_name
             ELSE grader.full_name
           END AS grader_name,
           parent_er.report_id AS rejected_report_parent_report_id,
           parent_er.experiment_id AS rejected_report_experiment_id,
           parent_er.status AS rejected_report_status,
           parent_er.feedback AS rejected_report_feedback,
           parent_er.submitted_at AS rejected_report_submitted_at,
           parent_er.pdf_file_path AS rejected_report_pdf_file_path,
           parent_grader.full_name AS rejected_report_grader_name${EXPERIMENT_GRADE_SELECT_FRAGMENT}
    FROM enrollments e JOIN users u ON e.student_id = u.user_id
    LEFT JOIN (SELECT ranked.student_id, ranked.experiment_id, ranked.retry_of_experiment_id FROM (SELECT es.student_id, es.experiment_id, es.retry_of_experiment_id, ROW_NUMBER() OVER (PARTITION BY es.student_id ORDER BY es.experiment_id DESC) AS rn FROM experiment_status es JOIN enrollments e2 ON e2.student_id = es.student_id WHERE e2.class_id = ?) ranked WHERE ranked.rn = 1) latest ON latest.student_id = e.student_id
    LEFT JOIN experiment_reports er ON er.student_id = latest.student_id AND er.experiment_id = latest.experiment_id
    LEFT JOIN experiment_status parent ON parent.experiment_id = latest.retry_of_experiment_id AND parent.student_id = latest.student_id
    LEFT JOIN experiment_reports parent_er ON parent_er.student_id = parent.student_id AND parent_er.experiment_id = parent.experiment_id
    LEFT JOIN experiment_grade eg ON eg.student_id = latest.student_id AND eg.experiment_id = latest.experiment_id
    LEFT JOIN users grader ON er.graded_by = grader.user_id
    LEFT JOIN users parent_grader ON parent_er.graded_by = parent_grader.user_id
    WHERE e.class_id = ? ORDER BY u.username ASC`, [classId, classId]);

  return await Promise.all(reportRows.map(async (row) => {
    const sanitizedRow: Record<string, unknown> = { ...row };
    sanitizedRow.submitted_at = toIsoString(sanitizedRow.submitted_at);
    const pdfFilePath = sanitizedRow.pdf_file_path as string | null | undefined;
    const rejectedPdfFilePath = sanitizedRow.rejected_report_pdf_file_path as string | null | undefined;
    // Keep the immediate rejected parent available throughout the child
    // lifecycle. Reviewers need the original rejection requirements while
    // comparing a submitted or graded retry, not only before resubmission.
    const rejectedReport = (
      sanitizedRow.retry_of_experiment_id !== null
      && sanitizedRow.retry_of_experiment_id !== undefined
      && sanitizedRow.rejected_report_parent_report_id !== null
      && sanitizedRow.rejected_report_parent_report_id !== undefined
      && sanitizedRow.rejected_report_experiment_id !== null
      && sanitizedRow.rejected_report_experiment_id !== undefined
      && sanitizedRow.rejected_report_status === "rejected"
    ) ? {
      report_id: Number(sanitizedRow.rejected_report_parent_report_id),
      experiment_id: Number(sanitizedRow.rejected_report_experiment_id),
      status: "rejected" as const,
      feedback: sanitizedRow.rejected_report_feedback ?? null,
      grader_name: sanitizedRow.rejected_report_grader_name ?? null,
      submitted_at: toIsoString(sanitizedRow.rejected_report_submitted_at),
      pdf_file_path: toPublicReportPath(rejectedPdfFilePath),
      pdf_file_available: await isReportPdfAvailable(rejectedPdfFilePath),
    } : null;
    delete sanitizedRow.rejected_report_parent_report_id;
    delete sanitizedRow.rejected_report_experiment_id;
    delete sanitizedRow.rejected_report_status;
    delete sanitizedRow.rejected_report_feedback;
    delete sanitizedRow.rejected_report_submitted_at;
    delete sanitizedRow.rejected_report_pdf_file_path;
    delete sanitizedRow.rejected_report_grader_name;
    const hasReport = sanitizedRow.report_id !== null && sanitizedRow.report_id !== undefined;
    const reviewExperimentGrade = REVIEW_REVISION_EXPERIMENT_GRADE_FIELDS.reduce<Record<string, unknown>>((result, column) => {
      result[column] = sanitizedRow[`grade_${column}`] ?? null;
      return result;
    }, {});
    const reviewRevision = hasReport ? buildReportReviewRevision({
      status: sanitizedRow.status,
      grade: sanitizedRow.grade,
      feedback: sanitizedRow.feedback,
      graded_by: sanitizedRow.review_graded_by,
    }, reviewExperimentGrade) : null;
    delete sanitizedRow.review_graded_by;
    const gradePayload: Record<string, unknown> = {};
    let hasGrade = false;
    for (const column of EXPERIMENT_GRADE_FIELDS) {
      const alias = `grade_${column}`;
      const value = sanitizedRow[alias];
      gradePayload[column] = value ?? null;
      if (value !== null && value !== undefined) hasGrade = true;
      delete sanitizedRow[alias];
    }
    delete sanitizedRow.grade_report_quality;
    return {
      ...sanitizedRow,
      review_revision: reviewRevision,
      pdf_file_path: toPublicReportPath(pdfFilePath),
      pdf_file_available: await isReportPdfAvailable(pdfFilePath),
      rejected_report: rejectedReport,
      // A freshly-created retry has an experiment_grade row whose database
      // defaults are zero. Those defaults are not a formal score object before
      // a report exists; submitted/graded reports still expose their automatic
      // evidence so reviewers can inspect and revise it.
      experiment_grade: hasReport && hasGrade ? gradePayload : null,
    };
  }));
}

export async function gradeReport(reportId: number, userId: number, userRole: string, input: GradeReportInput) {
  const connection = await pool.getConnection();
  try {
    // Resolve the immutable student identity before the transaction. The
    // transaction then locks both FK-related users in primary-key order before
    // the report: the student owns the report and the reviewer is written to
    // graded_by. Locking only the student would leave a late report -> reviewer
    // edge that cycles with ordered batch user deletion.
    const [[reportIdentity]] = await connection.query<RowDataPacket[]>(
      "SELECT student_id FROM experiment_reports WHERE report_id = ? LIMIT 1",
      [reportId],
    );
    if (!reportIdentity) return { error: "not_found" as const };

    await connection.beginTransaction();
    const studentId = Number(reportIdentity.student_id);
    const lockedUserIds = [...new Set([studentId, userId])].sort((left, right) => left - right);
    const userPlaceholders = lockedUserIds.map(() => "?").join(", ");
    const [lockedUsers] = await connection.query<RowDataPacket[]>(
      `SELECT user_id, role
         FROM users
        WHERE user_id IN (${userPlaceholders})
        ORDER BY user_id ASC
        FOR UPDATE`,
      lockedUserIds,
    );
    const lockedUsersById = new Map(
      lockedUsers.map((row) => [Number(row.user_id), row]),
    );
    const studentRow = lockedUsersById.get(studentId);
    if (!studentRow || studentRow.role !== "Student") {
      await connection.rollback();
      return { error: "not_found" as const };
    }
    const reviewerRow = lockedUsersById.get(userId);
    if (
      !reviewerRow
      || reviewerRow.role !== userRole
      || !["Teacher", "Admin", "Assistant"].includes(userRole)
    ) {
      await connection.rollback();
      return { error: "forbidden" as const };
    }

    const [[reportRow]] = await connection.query<RowDataPacket[]>(
      "SELECT experiment_id, student_id, status, grade, feedback, graded_by FROM experiment_reports WHERE report_id = ? FOR UPDATE",
      [reportId],
    );
    if (!reportRow) { await connection.rollback(); return { error: "not_found" as const }; }
    if (Number(reportRow.student_id) !== Number(reportIdentity.student_id)) {
      await connection.rollback();
      return { error: "status_conflict" as const, current_status: String(reportRow.status) };
    }

    const { experiment_id, student_id, status: currentStatus } = reportRow;

    let authorized = userRole === "Admin";
    if (!authorized && userRole === "Teacher") authorized = await isTeacherOfStudent(connection, student_id, userId);
    if (!authorized && userRole === "Assistant") authorized = await isAssistantAssignedToStudent(connection, student_id, userId);
    if (!authorized) { await connection.rollback(); return { error: "forbidden" as const }; }

    const [[latestExperiment]] = await connection.query<RowDataPacket[]>(
      `SELECT experiment_id
         FROM experiment_status
        WHERE student_id = ?
        ORDER BY experiment_id DESC
        LIMIT 1`,
      [student_id],
    );
    if (Number(latestExperiment?.experiment_id) !== Number(experiment_id)) {
      await connection.rollback();
      return { error: "status_conflict" as const, current_status: currentStatus as string };
    }

    const [[experimentGradeRow]] = await connection.query<RowDataPacket[]>(
      `SELECT grade_id, ${REVIEW_REVISION_EXPERIMENT_GRADE_FIELDS.join(", ")}
         FROM experiment_grade
        WHERE experiment_id = ? AND student_id = ?
        FOR UPDATE`,
      [experiment_id, student_id],
    );
    if (!experimentGradeRow) {
      await connection.rollback();
      return { error: "grade_not_found" as const };
    }

    // Status alone cannot distinguish two reviewers who both opened the same
    // graded report. Compare the complete locked review snapshot as well, so
    // any report or mutable component change makes the stale write fail.
    const currentReviewRevision = buildReportReviewRevision(reportRow, experimentGradeRow);
    if (
      String(currentStatus) !== input.expected_status
      || currentReviewRevision !== input.expected_review_revision
    ) {
      await connection.rollback();
      return { error: "status_conflict" as const, current_status: String(currentStatus) };
    }

    // A rejected report is immutable. A graded report may be revised, but it
    // cannot be converted into a rejection through this ordinary review API.
    // The locked expected-status comparison makes the first submitted ->
    // terminal transition win while preserving explicit graded revisions.
    if (currentStatus === "rejected" || (input.status === "rejected" && currentStatus !== "submitted")) {
      await connection.rollback();
      return { error: "status_conflict" as const, current_status: currentStatus as string };
    }

    if (input.status === "rejected") {
      await connection.query(
        "UPDATE experiment_grade SET report_quality = NULL WHERE experiment_id = ? AND student_id = ?",
        [experiment_id, student_id],
      );

      const [reportResult] = await connection.query<ResultSetHeader>(
        "UPDATE experiment_reports SET grade = NULL, feedback = ?, status = 'rejected', graded_by = ? WHERE report_id = ? AND status = 'submitted'",
        [input.feedback, userId, reportId],
      );
      if (reportResult.affectedRows === 0) {
        await connection.rollback();
        return { error: "status_conflict" as const, current_status: currentStatus as string };
      }
    } else {
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
        await connection.query(`UPDATE experiment_grade SET ${experimentGradeUpdateFields.join(", ")} WHERE experiment_id = ? AND student_id = ?`, [...experimentGradeUpdateValues, experiment_id, student_id]);
      }

      const reportUpdateFields: string[] = [];
      const reportUpdateValues: unknown[] = [];
      if (input.grade !== undefined) { reportUpdateFields.push("grade = ?"); reportUpdateValues.push(input.grade); }
      if (input.feedback !== undefined) { reportUpdateFields.push("feedback = ?"); reportUpdateValues.push(input.feedback); }
      if (input.status !== undefined) { reportUpdateFields.push("status = ?"); reportUpdateValues.push(input.status); }

      if (reportUpdateFields.length > 0) {
        reportUpdateFields.push("graded_by = ?");
        reportUpdateValues.push(userId);
        reportUpdateValues.push(reportId, currentStatus);
        await connection.query(`UPDATE experiment_reports SET ${reportUpdateFields.join(", ")} WHERE report_id = ? AND status = ?`, reportUpdateValues);
      }
    }

    const [[updatedReport]] = await connection.query<RowDataPacket[]>(`SELECT u.user_id, u.username, u.full_name, er.report_id, er.experiment_id, er.status, er.submitted_at, er.pdf_file_path, er.grade, er.feedback, er.graded_by AS review_graded_by, grader.full_name AS grader_name FROM experiment_reports er JOIN users u ON u.user_id = er.student_id LEFT JOIN users grader ON er.graded_by = grader.user_id WHERE er.report_id = ?`, [reportId]);
    const [[updatedExperimentGrade]] = await connection.query<RowDataPacket[]>(`SELECT ${REVIEW_REVISION_EXPERIMENT_GRADE_FIELDS.join(", ")} FROM experiment_grade WHERE experiment_id = ? AND student_id = ? LIMIT 1`, [experiment_id, student_id]);

    const experimentGradeResponse = updatedExperimentGrade ? EXPERIMENT_GRADE_FIELDS.reduce<Record<string, number | null>>((acc, field) => { acc[field] = updatedExperimentGrade[field] ?? null; return acc; }, {}) : null;
    const updatedReviewRevision = buildReportReviewRevision({
      ...updatedReport,
      graded_by: updatedReport?.review_graded_by,
    }, updatedExperimentGrade);
    const publicUpdatedReport = { ...updatedReport };
    delete publicUpdatedReport.review_graded_by;
    publicUpdatedReport.submitted_at = toIsoString(publicUpdatedReport.submitted_at);

    await connection.commit();

    return {
      data: {
        ...publicUpdatedReport,
        review_revision: updatedReviewRevision,
        pdf_file_path: toPublicReportPath(updatedReport?.pdf_file_path as string | null | undefined),
        pdf_file_available: await isReportPdfAvailable(updatedReport?.pdf_file_path as string | null | undefined),
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
       latest.experiment_id,
       latest.retry_of_experiment_id,
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
       CASE
         WHEN er.report_id IS NULL
          AND latest.retry_of_experiment_id IS NOT NULL
          AND parent_er.status = 'rejected'
         THEN '${REDO_IN_PROGRESS_STATUS}'
         ELSE er.status
       END AS report_status,
       CASE
         WHEN er.report_id IS NULL
          AND latest.retry_of_experiment_id IS NOT NULL
          AND parent_er.status = 'rejected'
         THEN parent_er.feedback
         ELSE er.feedback
       END AS report_feedback
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
  LEFT JOIN experiment_reports parent_er ON parent_er.student_id = latest.student_id AND parent_er.experiment_id = latest.retry_of_experiment_id
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
      const { expFlowScore, finalScore, expFlowComponents, finalScoreComponents } = computeExperimentScores(row, weights);
      const expFlowBreakdown = expFlowComponents.map((c) => ({
        field: c.field,
        score: hasFinalGrade ? formatScore(c.score) : null,
        weight: c.weight,
        weighted_score: hasFinalGrade ? formatScore(c.weighted_score) : null,
      }));
      const formatFinalComponent = (c: { score: number | null; weight: number; weighted_score: number | null }) => ({
        score: hasFinalGrade ? formatScore(c.score) : null,
        weight: c.weight,
        weighted_score: hasFinalGrade ? formatScore(c.weighted_score) : null,
      });

      return {
        student_id: row.user_id, username: row.username, full_name: row.full_name, experiment_id: row.experiment_id,
        retry_of_experiment_id: toNumberOrNull(row.retry_of_experiment_id),
        exp_flow_score: hasFinalGrade ? formatScore(expFlowScore) : null,
        model_quality: hasFinalGrade ? toNumberOrNull(row.model_quality) : null,
        knowledge_test: hasFinalGrade ? toNumberOrNull(row.knowledge_test) : null,
        report_quality: hasFinalGrade ? toNumberOrNull(row.report_quality) : null,
        report_status: reportStatus,
        final_score: hasFinalGrade ? formatScore(finalScore) : null,
        report_feedback: row.report_feedback ?? null,
        exp_flow_breakdown: expFlowBreakdown,
        final_score_breakdown: { exp_flow: formatFinalComponent(finalScoreComponents.exp_flow), knowledge_test: formatFinalComponent(finalScoreComponents.knowledge_test), model_quality: formatFinalComponent(finalScoreComponents.model_quality), report_quality: formatFinalComponent(finalScoreComponents.report_quality) },
      };
    }),
  };
}

export function buildTeacherClassSummary(
  classRow: ClassSummarySourceRow,
  gradeRows: Array<{ report_status: string | null; final_score: number | null }>,
): TeacherClassGradeSummary {
  const gradedCount = gradeRows.filter((row) => row.report_status === "graded").length;
  const submittedCount = gradeRows.filter((row) => row.report_status === "submitted").length;
  const rejectedCount = gradeRows.filter((row) =>
    row.report_status === "rejected" || row.report_status === REDO_IN_PROGRESS_STATUS
  ).length;
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
      const readablePath = await resolveStoredReportPdfPath(originalPath);
      if (!readablePath) continue;

      const candidate = allocateReportArchiveEntryFilename({
        username: report.username as string | null,
        fullName: report.full_name as string | null,
        studentId: Number(report.student_id),
        experimentId: Number(report.experiment_id),
      }, usedNames);

      const destination = path.join(tempDir, candidate);
      await copyFile(readablePath, destination);
      copiedFiles.push(destination);
    }

    if (copiedFiles.length === 0) return { error: "no_files" as const };

    const zipFileName = `class-${classId}-reports.zip`;
    const zipPath = path.join(tempDir, zipFileName);
    await createZipArchive(
      zipPath,
      copiedFiles.map((filePath) => path.basename(filePath)),
      tempDir,
    );

    const exportsDir = await ensureDatabaseManagedStorageRoot(EXPORTS_DIR, env);
    const exportFileName = buildGeneratedExportFilename(classId, "reports", "zip");
    const exportFilePath = path.join(exportsDir, exportFileName);
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

  const csvContent = buildCsvContent(buildGradeCsvLines(gradeRows, weights));
  const exportsDir = await ensureDatabaseManagedStorageRoot(EXPORTS_DIR, env);
  const exportFileName = buildGeneratedExportFilename(classId, "grades", "csv");
  const exportFilePath = path.join(exportsDir, exportFileName);
  await Bun.write(exportFilePath, csvContent);

  return { data: { file_path: `exports/${exportFileName}` } };
}

const REPORT_EXPORT_HEADER = ["学号", "姓名", "实验ID", "报告ID", "报告状态", "提交时间", "报告得分", "评阅人", "评语"];

export function buildReportCsvLines(reportRows: Array<Record<string, unknown>>) {
  const csvLines: string[] = [REPORT_EXPORT_HEADER.join(",")];

  for (const row of reportRows) {
    const rowValues = [
      encodeCsvCell(row.username),
      encodeCsvCell(row.full_name),
      encodeCsvCell(row.experiment_id),
      encodeCsvCell(row.report_id),
      encodeCsvCell(reportStatusText(row.status)),
      encodeCsvCell(formatCsvDateTime(row.submitted_at_text ?? row.submitted_at)),
      row.status === "graded" ? formatNumeric(row.grade) : "",
      encodeCsvCell(row.grader_name),
      encodeCsvCell(row.feedback),
    ];
    csvLines.push(rowValues.join(","));
  }

  return csvLines;
}

export async function exportReportCsv(classId: number) {
  const [reportRows] = await pool.query<RowDataPacket[]>(`SELECT u.user_id,
           u.username,
           u.full_name,
           er.report_id,
           latest.experiment_id,
           latest.retry_of_experiment_id,
           CASE
             WHEN er.report_id IS NULL
              AND latest.retry_of_experiment_id IS NOT NULL
              AND parent_er.status = 'rejected'
             THEN '${REDO_IN_PROGRESS_STATUS}'
             ELSE er.status
           END AS status,
           er.submitted_at,
           DATE_FORMAT(er.submitted_at, '%Y-%m-%d %H:%i:%s') AS submitted_at_text,
           er.pdf_file_path,
           er.grade,
           CASE
             WHEN er.report_id IS NULL
              AND latest.retry_of_experiment_id IS NOT NULL
              AND parent_er.status = 'rejected'
             THEN parent_er.feedback
             ELSE er.feedback
           END AS feedback,
           CASE
             WHEN er.report_id IS NULL
              AND latest.retry_of_experiment_id IS NOT NULL
              AND parent_er.status = 'rejected'
             THEN parent_grader.full_name
             ELSE grader.full_name
           END AS grader_name
      FROM enrollments e
      JOIN users u ON e.student_id = u.user_id
      LEFT JOIN (
        SELECT ranked.student_id, ranked.experiment_id, ranked.retry_of_experiment_id
          FROM (
            SELECT es.student_id,
                   es.experiment_id,
                   es.retry_of_experiment_id,
                   ROW_NUMBER() OVER (PARTITION BY es.student_id ORDER BY es.experiment_id DESC) AS rn
              FROM experiment_status es
              JOIN enrollments e2 ON e2.student_id = es.student_id
             WHERE e2.class_id = ?
          ) ranked
         WHERE ranked.rn = 1
      ) latest ON latest.student_id = e.student_id
      LEFT JOIN experiment_reports er ON er.student_id = latest.student_id AND er.experiment_id = latest.experiment_id
      LEFT JOIN experiment_reports parent_er ON parent_er.student_id = latest.student_id AND parent_er.experiment_id = latest.retry_of_experiment_id
      LEFT JOIN users grader ON er.graded_by = grader.user_id
      LEFT JOIN users parent_grader ON parent_er.graded_by = parent_grader.user_id
     WHERE e.class_id = ?
     ORDER BY u.username ASC`, [classId, classId]);

  if (reportRows.length === 0) return { error: "no_students" as const };

  const csvContent = buildCsvContent(buildReportCsvLines(reportRows));
  const exportsDir = await ensureDatabaseManagedStorageRoot(EXPORTS_DIR, env);
  const exportFileName = buildGeneratedExportFilename(classId, "reports", "csv");
  const exportFilePath = path.join(exportsDir, exportFileName);
  await Bun.write(exportFilePath, csvContent);

  return { data: { file_path: `exports/${exportFileName}` } };
}

export async function cleanupExpiredExportFiles(
  options: { now?: Date; maxAgeMs?: number; directory?: string } = {},
) {
  return cleanupDirectoryEntriesOlderThan(options.directory ?? exportsStorageRoot(), {
    now: options.now,
    maxAgeMs: options.maxAgeMs ?? CLEANUP_RETENTION.EXPORT_FILES_MS,
    entryKind: "file",
    namePattern: GENERATED_EXPORT_FILE_PATTERN,
  });
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
  // Attempts are an append-only lineage. A freshly-created retry may not have
  // activity timestamps yet, so recency must follow the attempt id instead of
  // the most recently active row. This keeps progress aligned with reports,
  // grades, statistics, and exports after a rejection.
  const [studentExperimentRows] = await pool.query<RowDataPacket[]>(`WITH ranked AS (SELECT es.*, ROW_NUMBER() OVER (PARTITION BY es.student_id ORDER BY es.experiment_id DESC) AS rn FROM experiment_status es JOIN enrollments e ON es.student_id = e.student_id WHERE e.class_id = ?) SELECT u.user_id AS student_id, u.username, u.full_name, latest.experiment_id, latest.status, latest.current_step, latest.highest_completed_step, latest.start_time, latest.last_activity_at, latest.completion_time FROM enrollments e JOIN users u ON e.student_id = u.user_id LEFT JOIN (SELECT * FROM ranked WHERE rn = 1) latest ON latest.student_id = u.user_id WHERE e.class_id = ? ORDER BY u.username ASC`, [classId, classId]);

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
