import path from "path";
import { mkdir, rm } from "node:fs/promises";
import type { RowDataPacket, ResultSetHeader } from "mysql2";
import { marked } from "marked";
import puppeteer from "puppeteer-core";
import { env, pool } from "../../config";
import { sanitizeHTML, HTMLSanitizationError } from "../../utils/html-sanitizer";
import { CONTENT_LIMITS, CONCURRENCY_LIMITS, TIMEOUTS, PATHS } from "../../utils/constants";
import { Semaphore } from "../../utils/semaphore";
import logger from "../../utils/logger";
import { withDeadlockRetry } from "../../utils/db-retry";
import {
  buildExperimentFlowGradeValues,
  getReportSubmissionEligibilityError,
} from "./report-submission-policy";
import {
  parseCompatibleProductionForecastResults,
  validateCompletedProductionPlan,
} from "../experiments/experiments.schema";
import { cleanupExperimentModelArtifacts } from "../models/models.service";
import {
  ensureDatabaseManagedStorageRoot,
  getDatabaseManagedStorageRoot,
  isPathInsideRoot,
  resolveStoredManagedPath,
} from "../../utils/managed-storage";
import { resolveExistingManagedFile } from "../../utils/managed-file";
import { fitFilenameToStoragePathLimits } from "../../utils/file-validator";
import { databaseDatetimeToIsoString } from "../../utils/database-datetime";

const pdfSemaphore = new Semaphore(
  CONCURRENCY_LIMITS.MAX_CONCURRENT_PDF_JOBS,
  CONCURRENCY_LIMITS.MAX_PENDING_PDF_JOBS,
);
const reportsManagedRoot = path.resolve(import.meta.dir, "../../../uploads", "reports");
const PDF_BROWSER_CLOSE_GRACE_MS = 2_000;

export class PdfGenerationTimeoutError extends Error {
  readonly code = "PDF_GENERATION_TIMEOUT";

  constructor(public readonly timeoutMs: number, cause?: unknown) {
    super(`PDF generation timed out after ${timeoutMs}ms`);
    this.name = "PdfGenerationTimeoutError";
    if (cause !== undefined) {
      (this as Error & { cause?: unknown }).cause = cause;
    }
  }
}

export function getPdfQueueStats() {
  return {
    activeJobs: pdfSemaphore.active,
    pendingJobs: pdfSemaphore.pending,
    maxConcurrentJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_PDF_JOBS,
    maxPendingJobs: CONCURRENCY_LIMITS.MAX_PENDING_PDF_JOBS,
    queueTimeoutMs: TIMEOUTS.PDF_QUEUE,
    generationTimeoutMs: TIMEOUTS.PDF_GENERATION,
  };
}

export async function holdPdfQueueSlotForTest(holdMs: number): Promise<void> {
  await pdfSemaphore.acquire(TIMEOUTS.PDF_QUEUE);
  try {
    await new Promise((resolve) => setTimeout(resolve, holdMs));
  } finally {
    pdfSemaphore.release();
  }
}

const MODEL_TYPES = [
  "ma", "exp", "arima", "lstm",
  "ensemble_weighted", "ensemble_boosting", "ensemble_stacking",
] as const;

type ModelType = (typeof MODEL_TYPES)[number];

interface ModelMetric {
  type: ModelType;
  rmse: number;
  mae: number;
  r2: number;
}

interface ExperimentRow extends RowDataPacket {
  experiment_id: number;
  student_id: number;
  retry_of_experiment_id: number | null;
  status: string;
  current_step: number;
  highest_completed_step: number;
  start_time: Date | string | null;
  last_activity_at: Date | string | null;
  completion_time: Date | string | null;
  selected_industry: string | null;
  selected_company: string | null;
  selected_product: string | null;
  selected_best_model: string | null;
  quiz_about_model_completed: boolean;
  quiz_about_plan_completed: boolean;
  production_plan_completed: boolean;
  data_window_train_start_index: number | null;
  data_window_train_end_index: number | null;
  data_window_evaluate_start_index: number | null;
  data_window_evaluate_end_index: number | null;
  production_forecast_periods: number | null;
  production_initial_inventory: number | null;
  production_target_service_level: number | null;
  production_safety_stock_z_score: number | null;
  production_forecast_results: unknown;
  production_forecast_evidence: string | null;
  production_mps_table: unknown;
  production_capacity_mode: string | null;
  production_capacity_scenario: string | null;
  production_capacity: number | null;
  production_custom_capacity: number | null;
  moving_average_completed: boolean;
  moving_average_metrics_rmse: number | null;
  moving_average_metrics_mae: number | null;
  moving_average_metrics_r2: number | null;
  exponential_smoothing_completed: boolean;
  exponential_smoothing_metrics_rmse: number | null;
  exponential_smoothing_metrics_mae: number | null;
  exponential_smoothing_metrics_r2: number | null;
  arima_completed: boolean;
  arima_metrics_rmse: number | null;
  arima_metrics_mae: number | null;
  arima_metrics_r2: number | null;
  lstm_completed: boolean;
  lstm_metrics_rmse: number | null;
  lstm_metrics_mae: number | null;
  lstm_metrics_r2: number | null;
  ensemble_weighted_completed: boolean;
  ensemble_weighted_metrics_rmse: number | null;
  ensemble_weighted_metrics_mae: number | null;
  ensemble_weighted_metrics_r2: number | null;
  ensemble_boosting_completed: boolean;
  ensemble_boosting_metrics_rmse: number | null;
  ensemble_boosting_metrics_mae: number | null;
  ensemble_boosting_metrics_r2: number | null;
  ensemble_stacking_completed: boolean;
  ensemble_stacking_metrics_rmse: number | null;
  ensemble_stacking_metrics_mae: number | null;
  ensemble_stacking_metrics_r2: number | null;
}

interface ReportRow extends RowDataPacket {
  report_id: number;
  experiment_id: number;
  student_id: number;
  status: string;
  submitted_at: Date | string | null;
  grade: number | null;
  feedback: string | null;
  pdf_file_path?: string | null;
  graded_by: number | null;
}

function toPublicReportPath(filePath: string | null | undefined): string | null {
  if (!filePath) return null;
  return `reports/${path.basename(filePath)}`;
}

async function isStoredReportPdfAvailable(filePath: string | null | undefined): Promise<boolean> {
  if (!filePath) return false;
  return resolveExistingManagedFile({
    storedPath: filePath,
    managedRoot: reportsManagedRoot,
    environment: env,
    legacyFallbackDirectory: getDatabaseManagedStorageRoot(reportsManagedRoot, env),
    enforceDisposableNamespace: true,
  }).then(Boolean).catch(() => false);
}

async function buildLatestReportRecord(report: ReportRow) {
  return {
    ...report,
    submitted_at: databaseDatetimeToIsoString(report.submitted_at, env.DB_TIMEZONE),
    pdf_file_path: toPublicReportPath(report.pdf_file_path),
    pdf_file_available: await isStoredReportPdfAvailable(report.pdf_file_path),
  };
}

async function getRejectedParentReport(
  studentId: number,
  retryOfExperimentId: number | null,
) {
  if (retryOfExperimentId == null) return null;

  const [[rejectedReport]] = await pool.query<ReportRow[]>(
    `SELECT er.*
       FROM experiment_reports er
       JOIN experiment_status parent ON parent.experiment_id = er.experiment_id
      WHERE er.experiment_id = ?
        AND er.student_id = ?
        AND parent.student_id = ?
        AND er.status = 'rejected'
      LIMIT 1`,
    [retryOfExperimentId, studentId, studentId],
  );
  return rejectedReport ? buildLatestReportRecord(rejectedReport) : null;
}

async function resolveGeneratedReportCleanupPath(
  pdfPath: string | null | undefined,
  allowedRoot = reportsManagedRoot,
): Promise<string | null> {
  if (!pdfPath) return null;
  const resolvedAllowedRoot = path.resolve(allowedRoot);
  const databaseReportsRoot = getDatabaseManagedStorageRoot(reportsManagedRoot, env);
  const isManagedProductionRoot = (
    resolvedAllowedRoot === reportsManagedRoot
    || resolvedAllowedRoot === databaseReportsRoot
  );
  let candidatePath: string;
  if (path.isAbsolute(pdfPath) && isPathInsideRoot(pdfPath, resolvedAllowedRoot)) {
    candidatePath = path.resolve(pdfPath);
  } else if (isManagedProductionRoot) {
    const resolved = await resolveStoredManagedPath({
      storedPath: pdfPath,
      managedRoot: reportsManagedRoot,
      environment: env,
      legacyFallbackDirectory: databaseReportsRoot,
      enforceDisposableNamespace: true,
    });
    candidatePath = resolved.absolutePath;
  } else {
    throw new Error(`Report cleanup path is outside its generation root: ${pdfPath}`);
  }

  // Lexical containment is insufficient for destructive operations: an
  // ancestor such as reports/linked may be a symlink to an external directory.
  // Reuse the component-by-component regular-file validator before rm.
  const existing = await resolveExistingManagedFile({
    storedPath: candidatePath,
    managedRoot: resolvedAllowedRoot,
    environment: env,
    legacyFallbackDirectory: resolvedAllowedRoot,
    enforceDisposableNamespace: false,
  });
  return existing?.absolutePath ?? null;
}

export async function cleanupGeneratedReportPdf(
  pdfPath: string | null | undefined,
  allowedRoot = reportsManagedRoot,
): Promise<void> {
  if (!pdfPath) return;

  try {
    const cleanupPath = await resolveGeneratedReportCleanupPath(pdfPath, allowedRoot);
    if (cleanupPath) await rm(cleanupPath, { force: true });
  } catch (error) {
    logger.warn({ err: error, pdfPath }, "清理报告 PDF 失败");
  }
}

export async function cleanupReplacedReportPdf(
  previousPdfPath: string,
  allowedRoot: string,
): Promise<void> {
  try {
    const cleanupPath = await resolveGeneratedReportCleanupPath(previousPdfPath, allowedRoot);
    if (!cleanupPath) return;
    const [survivingRows] = await pool.query<Array<RowDataPacket & { pdf_file_path: string | null }>>(
      `SELECT pdf_file_path
         FROM experiment_reports
        WHERE pdf_file_path IS NOT NULL`,
    );
    for (const row of survivingRows) {
      if (!row.pdf_file_path) continue;
      // Exact equality is a safe fast path even if an old external value can
      // no longer be resolved by the current deployment.
      if (row.pdf_file_path === previousPdfPath) return;
      const survivingPath = await resolveGeneratedReportCleanupPath(
        row.pdf_file_path,
        allowedRoot,
      ).catch(() => null);
      if (survivingPath === cleanupPath) return;
    }
    await rm(cleanupPath, { force: true });
  } catch (error) {
    // A failed reference scan must retain the old PDF. Deleting on uncertainty
    // could break another legacy report which shares the same physical file.
    logger.warn({ err: error, previousPdfPath }, "检查旧报告 PDF 引用失败，已保留文件");
  }
}

function getBestModel(experiment: ExperimentRow): ModelType | null {
  const models: ModelMetric[] = [];

  const addModel = (type: ModelType, prefix: string) => {
    const key = (k: string) => `${prefix}_${k}` as keyof ExperimentRow;
    if (!experiment[key("completed")]) return;

    const rmse = Number(experiment[key("metrics_rmse")]);
    const mae = Number(experiment[key("metrics_mae")]);
    const r2 = Number(experiment[key("metrics_r2")]);

    if (!isNaN(rmse) && !isNaN(mae) && !isNaN(r2)) {
      models.push({ type, rmse, mae, r2 });
    }
  };

  addModel("ma", "moving_average");
  addModel("exp", "exponential_smoothing");
  addModel("arima", "arima");
  addModel("lstm", "lstm");
  addModel("ensemble_weighted", "ensemble_weighted");
  addModel("ensemble_boosting", "ensemble_boosting");
  addModel("ensemble_stacking", "ensemble_stacking");

  if (models.length === 0) return null;

  const modelsWithScore = models.map((m) => ({ ...m, score: m.rmse + m.mae }));
  const minScore = Math.min(...modelsWithScore.map((m) => m.score));
  const threshold = minScore === 0 ? 0 : minScore * 1.01;
  const candidates = modelsWithScore.filter((m) => m.score <= threshold);
  candidates.sort((a, b) => b.r2 - a.r2);

  return candidates[0]!.type;
}

async function markdownToSanitizedHtml(markdown: string): Promise<string> {
  const parsed = await marked.parse(markdown);
  if (typeof parsed !== "string") {
    throw new HTMLSanitizationError("Markdown 转换为 HTML 失败");
  }
  return sanitizeHTML(parsed);
}

function buildPdfDocument(content: string): string {
  return `<!DOCTYPE html>
  <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>Experiment Report</title>
      <style>
        :root { color-scheme: light; }
        body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
          margin: 0; padding: 40px; line-height: 1.6; color: #1f2933; background-color: #ffffff;
        }
        .report-content h1, .report-content h2, .report-content h3,
        .report-content h4, .report-content h5, .report-content h6 {
          margin-top: 1.8rem; margin-bottom: 0.6rem;
        }
        .report-content pre {
          background-color: #f8fafc; padding: 14px; border-radius: 6px; overflow-x: auto;
          font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 0.9rem;
        }
        .report-content code {
          font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
          background: #f1f5f9; padding: 2px 4px; border-radius: 4px;
        }
        .report-content table { border-collapse: collapse; width: 100%; margin: 1.2rem 0; }
        .report-content thead { display: table-header-group; }
        .report-content tr { break-inside: avoid; page-break-inside: avoid; }
        .report-content th, .report-content td { border: 1px solid #cbd5f5; padding: 8px 10px; text-align: left; }
        .report-content ul, .report-content ol { padding-left: 1.4rem; }
        .report-content blockquote { border-left: 4px solid #94a3b8; padding-left: 12px; color: #475569; margin: 1rem 0; }
        .report-content hr { border: none; border-top: 1px solid #e2e8f0; margin: 2rem 0; }
        .report-content .report-appendix-break { break-before: page; page-break-before: always; height: 0; }
      </style>
    </head>
    <body><main class="report-content">${content}</main></body>
  </html>`;
}

export interface SubmitReportResult {
  message: string;
  report_id: number;
  pdf_path: string;
}

export interface SubmitReportOptions {
  /** Override the PDF directory for isolated tests and maintenance jobs. */
  reportsDir?: string;
}

type SubmitReportSuccess = {
  data: SubmitReportResult;
  isUpdate: boolean;
};

interface ReportCommitCandidate {
  result: SubmitReportSuccess;
  previousPdfPath: string | null;
  expectedReportId: number;
  pdfPath: string;
}

class AmbiguousReportCommitError extends Error {
  constructor(
    public readonly originalError: unknown,
    public readonly candidate: ReportCommitCandidate,
  ) {
    super("Report transaction commit acknowledgement was not received");
    this.name = "AmbiguousReportCommitError";
  }
}

type ReportCommitProbeState = "committed" | "not-committed" | "unknown";

async function probeAmbiguousReportCommit(
  experimentId: number,
  studentId: number,
  candidate: ReportCommitCandidate,
): Promise<ReportCommitProbeState> {
  const connection = await pool.getConnection();
  let phase: "active" | "commit-attempted" | "committed" = "active";
  let disposed = false;
  try {
    await connection.query("SET TRANSACTION ISOLATION LEVEL READ COMMITTED");
    await connection.beginTransaction();
    await connection.query<RowDataPacket[]>(
      "SELECT user_id FROM users WHERE user_id = ? LIMIT 1 FOR UPDATE",
      [studentId],
    );
    const [[experiment]] = await connection.query<RowDataPacket[]>(
      `SELECT experiment_id, student_id, status
         FROM experiment_status
        WHERE experiment_id = ? AND student_id = ?
        LIMIT 1
        FOR UPDATE`,
      [experimentId, studentId],
    );
    const [[report]] = await connection.query<ReportRow[]>(
      `SELECT report_id, experiment_id, student_id, pdf_file_path, status
         FROM experiment_reports
        WHERE experiment_id = ? AND student_id = ?
        LIMIT 1
        FOR UPDATE`,
      [experimentId, studentId],
    );

    const observedState: ReportCommitProbeState = (() => {
      const sameCommittedReport = (
        Number(report?.report_id) === candidate.expectedReportId
        && report?.pdf_file_path === candidate.pdfPath
        && ["submitted", "graded", "rejected"].includes(String(report?.status))
        && String(experiment?.status) === "Completed"
      );
      if (sameCommittedReport) return "committed";

      // Absence, or the exact pre-transaction report version, proves this PDF
      // was not committed. Any mixed/newer state is intentionally unknown: a
      // concurrent resubmission or review may have advanced it before the
      // locking read acquired its locks.
      if (!report) return "not-committed";
      if (
        Number(report.report_id) === candidate.expectedReportId
        && report.pdf_file_path === candidate.previousPdfPath
        && report.pdf_file_path !== candidate.pdfPath
      ) {
        return "not-committed";
      }
      return "unknown";
    })();

    phase = "commit-attempted";
    try {
      await connection.commit();
    } catch (commitError) {
      // This transaction is read-only: the state was observed while holding
      // all relevant row/gap locks, so losing the COMMIT acknowledgement does
      // not invalidate that observation. The session itself is uncertain and
      // must never be returned to the pool.
      disposed = true;
      try {
        connection.destroy();
      } catch {}
      logger.warn(
        { err: commitError, experimentId, studentId, observedState },
        "报告提交探针 COMMIT 回执丢失，采用锁内观测结果",
      );
      return observedState;
    }
    phase = "committed";
    disposed = true;
    try {
      connection.release();
    } catch (releaseError) {
      connection.destroy();
      logger.warn({ err: releaseError }, "释放报告提交探针连接失败");
    }

    return observedState;
  } catch (error) {
    if (phase === "commit-attempted") {
      disposed = true;
      try {
        connection.destroy();
      } catch {}
    } else {
      try {
        await connection.rollback();
      } catch {
        disposed = true;
        try {
          connection.destroy();
        } catch {}
      }
    }
    throw error;
  } finally {
    if (!disposed) connection.release();
  }
}

export async function submitReport(
  experimentId: number,
  studentId: number,
  markdownContent: string,
  options: SubmitReportOptions = {},
): Promise<{ error: string; status: 400 | 403 | 404 } | { data: SubmitReportResult; isUpdate: boolean }> {
  // 输入长度限制
  if (markdownContent.length > CONTENT_LIMITS.MARKDOWN_MAX_LENGTH) {
    return {
      error: `报告内容过长，最大允许 ${CONTENT_LIMITS.MARKDOWN_MAX_LENGTH} 字符`,
      status: 400,
    };
  }

  const [[precheckExperiment]] = await pool.query<ExperimentRow[]>(
    `SELECT * FROM experiment_status WHERE experiment_id = ? LIMIT 1`,
    [experimentId],
  );

  if (!precheckExperiment) {
    return { error: "实验不存在", status: 404 };
  }

  if (precheckExperiment.student_id !== studentId) {
    return { error: "权限不足：该实验不属于当前用户", status: 403 };
  }

  let allowCompletedWithoutReport = false;
  if (precheckExperiment.status === "Completed") {
    const [precheckReports] = await pool.query<RowDataPacket[]>(
      `SELECT report_id
         FROM experiment_reports
        WHERE experiment_id = ? AND student_id = ?
        LIMIT 1`,
      [experimentId, studentId],
    );
    allowCompletedWithoutReport = precheckReports.length === 0;
  }

  const precheckEligibilityError = getReportSubmissionEligibilityError(
    precheckExperiment,
    {
      allowCompletedWithoutReport,
      allowLegacyV1Write: env.ALLOW_LEGACY_PRODUCTION_PLAN_WRITES,
    },
  );
  if (precheckEligibilityError) {
    return { error: precheckEligibilityError, status: 400 };
  }

  const sanitizedReportContent = await markdownToSanitizedHtml(markdownContent);
  const htmlDocument = buildPdfDocument(sanitizedReportContent);

  const reportsDir = options.reportsDir
    ? path.resolve(options.reportsDir)
    : await ensureDatabaseManagedStorageRoot(reportsManagedRoot, env);
  if (options.reportsDir) {
    // Isolated tests and maintenance callers supply their own trusted root;
    // production storage always goes through the symlink-safe category helper.
    await mkdir(reportsDir, { recursive: true });
  }

  // Date.now() alone is not a unique ownership boundary: two submissions for
  // the same experiment can finish PDF generation in the same millisecond.
  // If they shared a path, the losing transaction would clean up the winner's
  // committed PDF. Keep the timestamp for operator readability, but give each
  // request an unguessable, independently cleanable artifact.
  const uniqueFilename = fitFilenameToStoragePathLimits("submission.pdf", {
    directory: reportsDir,
    fixedPrefix: `report-exp${experimentId}-student${studentId}-${Date.now()}-${crypto.randomUUID()}-`,
  });
  const pdfPath = path.join(reportsDir, uniqueFilename);

  await pdfSemaphore.acquire(TIMEOUTS.PDF_QUEUE);
  try {
    let browser: Awaited<ReturnType<typeof puppeteer.launch>> | null = null;
    let generationTimedOut = false;
    const timeoutError = new PdfGenerationTimeoutError(TIMEOUTS.PDF_GENERATION);
    const forceKillBrowser = (
      target: Awaited<ReturnType<typeof puppeteer.launch>> | null,
    ) => {
      try {
        target?.process?.()?.kill("SIGKILL");
      } catch (error) {
        logger.warn(
          { err: error, experimentId, studentId },
          "强制终止 PDF Chromium 进程时发生异常",
        );
      }
    };

    const generation = (async () => {
      browser = await puppeteer.launch({
        executablePath: PATHS.CHROMIUM_PATH,
        headless: true,
        timeout: TIMEOUTS.PDF_GENERATION,
        args: [
          "--no-sandbox",
          "--disable-setuid-sandbox",
          "--disable-gpu",
          "--disable-dev-shm-usage",
        ],
      });

      if (generationTimedOut) {
        forceKillBrowser(browser);
        throw timeoutError;
      }

      try {
        const page = await browser.newPage();
        await page.setExtraHTTPHeaders({ "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline';" });
        await page.setJavaScriptEnabled(false);
        await page.setContent(htmlDocument, { waitUntil: "domcontentloaded", timeout: 10000 });
        await page.pdf({
          path: pdfPath,
          format: "A4",
          printBackground: true,
          timeout: TIMEOUTS.PDF_GENERATION,
        });
      } finally {
        const closingBrowser = browser;
        const closeWork = closingBrowser.close();
        const closeOutcome = closeWork.then(
          () => true,
          () => false,
        );
        let closeTimer: ReturnType<typeof setTimeout> | undefined;
        const closeDeadline = new Promise<false>((resolve) => {
          closeTimer = setTimeout(() => resolve(false), PDF_BROWSER_CLOSE_GRACE_MS);
          closeTimer.unref?.();
        });
        try {
          const closedGracefully = await Promise.race([closeOutcome, closeDeadline]);
          if (!closedGracefully) {
            forceKillBrowser(closingBrowser);
          }
        } finally {
          if (closeTimer) clearTimeout(closeTimer);
        }
      }
    })();
    void generation.catch(() => {});

    let generationTimer: ReturnType<typeof setTimeout> | undefined;
    const generationDeadline = new Promise<never>((_, reject) => {
      generationTimer = setTimeout(() => {
        generationTimedOut = true;
        forceKillBrowser(browser);
        reject(timeoutError);
      }, TIMEOUTS.PDF_GENERATION);
      generationTimer.unref?.();
    });

    try {
      await Promise.race([generation, generationDeadline]);
    } catch (error) {
      if (error === timeoutError) {
        await Promise.race([
          generation.catch(() => {}),
          Bun.sleep(PDF_BROWSER_CLOSE_GRACE_MS),
        ]);
      }
      throw error;
    } finally {
      if (generationTimer) clearTimeout(generationTimer);
    }
  } catch (error) {
    const normalizedError = error instanceof PdfGenerationTimeoutError
      ? error
      : error instanceof Error && error.name === "TimeoutError"
        ? new PdfGenerationTimeoutError(TIMEOUTS.PDF_GENERATION, error)
        : error;
    logger.error({ err: normalizedError, experimentId, studentId }, "PDF 生成失败");
    await cleanupGeneratedReportPdf(pdfPath, reportsDir);
    throw normalizedError;
  } finally {
    pdfSemaphore.release();
  }

  const finalizeCommittedSubmission = (
    result: SubmitReportSuccess,
    previousPdfPath: string | null,
  ) => {
    if (previousPdfPath && previousPdfPath !== pdfPath) {
      Promise.resolve().then(() => cleanupReplacedReportPdf(previousPdfPath, reportsDir));
    }

    // 异步清理该实验产生的所有模型训练文件，不阻塞响应
    Promise.resolve().then(() => {
      try {
        cleanupExperimentModelArtifacts(experimentId, studentId);
      } catch (error) {
        logger.error({ err: error, experimentId, studentId }, "清理实验模型产物失败");
      }
    });
    return result;
  };

  try {
    const transactionResult = await withDeadlockRetry(async () => {
      const connection = await pool.getConnection();
      let transactionPhase: "before-commit" | "commit-attempted" | "committed" = "before-commit";
      let disposed = false;
      try {
        await connection.query("SET TRANSACTION ISOLATION LEVEL READ COMMITTED");
        await connection.beginTransaction();
        // User is the common first lock for report submission, grading, retry,
        // and user deletion. The explicit lock also acquires the FK parent
        // before experiment/report rows, avoiding an implicit users lock late
        // in INSERT ... experiment_reports.
        await connection.query<RowDataPacket[]>(
          "SELECT user_id FROM users WHERE user_id = ? LIMIT 1 FOR UPDATE",
          [studentId],
        );

        const [[experiment]] = await connection.query<ExperimentRow[]>(
          `SELECT * FROM experiment_status WHERE experiment_id = ? LIMIT 1 FOR UPDATE`,
          [experimentId],
        );

        if (!experiment) {
          await connection.rollback();
          return { result: { error: "实验不存在", status: 404 as const }, previousPdfPath: null };
        }

        if (experiment.student_id !== studentId) {
          await connection.rollback();
          return { result: { error: "权限不足：该实验不属于当前用户", status: 403 as const }, previousPdfPath: null };
        }

        let existingReports: ReportRow[] | undefined;
        if (experiment.status === "Completed") {
          const [rows] = await connection.query<ReportRow[]>(
            `SELECT report_id, pdf_file_path
               FROM experiment_reports
              WHERE experiment_id = ? AND student_id = ?
              LIMIT 1
              FOR UPDATE`,
            [experimentId, studentId],
          );
          existingReports = rows;
        }

        const eligibilityError = getReportSubmissionEligibilityError(
          experiment,
          {
            allowCompletedWithoutReport: existingReports?.length === 0,
            allowLegacyV1Write: env.ALLOW_LEGACY_PRODUCTION_PLAN_WRITES,
          },
        );
        if (eligibilityError) {
          await connection.rollback();
          return { result: { error: eligibilityError, status: 400 as const }, previousPdfPath: null };
        }

        const isCompletedRecovery = (
          experiment.status === "Completed"
          && existingReports?.length === 0
        );
        // A completed row is immutable through the experiment APIs. If report
        // creation was interrupted after that terminal write, validate it as
        // the terminal snapshot it actually is; coercing it back to an active
        // row creates a state that can neither be repaired nor submitted.
        const productionPlanValidation = validateCompletedProductionPlan(
          experiment,
          {
            allowLegacyV1Write: env.ALLOW_LEGACY_PRODUCTION_PLAN_WRITES,
          },
        );
        if (!productionPlanValidation.ok) {
          await connection.rollback();
          return {
            result: {
              error: productionPlanValidation.message,
              status: 400 as const,
            },
            previousPdfPath: null,
          };
        }
        const evidenceClaims = productionPlanValidation.ok
          ? productionPlanValidation.evidenceClaims
          : null;
        const forecastVersion = parseCompatibleProductionForecastResults(
          experiment.production_forecast_results,
        )?.version;
        const isLegacyTransitionPlan = (
          experiment.status !== "Completed"
          && env.ALLOW_LEGACY_PRODUCTION_PLAN_WRITES
          && forecastVersion === "v1"
        );
        if (!evidenceClaims && !isCompletedRecovery && !isLegacyTransitionPlan) {
          await connection.rollback();
          return {
            result: {
              error: "生产预测凭证缺失或无效，请重新生成并保存生产计划后再提交报告。",
              status: 400 as const,
            },
            previousPdfPath: null,
          };
        }
        if (evidenceClaims) {
          const [activeRunRows] = await connection.query<RowDataPacket[]>(
            `SELECT run_id, production_revision
               FROM model_training_runs
              WHERE experiment_id = ?
                AND student_id = ?
                AND model_type = ?
                AND is_active = TRUE
              ORDER BY activated_at DESC
              LIMIT 1
              FOR UPDATE`,
            [experimentId, studentId, evidenceClaims.model_type],
          );
          const activeRun = activeRunRows[0];
          if (
            !activeRun
            || activeRun.run_id !== evidenceClaims.run_id
            || activeRun.production_revision !== evidenceClaims.production_revision
          ) {
            await connection.rollback();
            return {
              result: {
                error: "生产预测对应的模型版本已失效，请重新准备、预测并保存生产计划后再提交报告。",
                status: 400 as const,
              },
              previousPdfPath: null,
            };
          }
        }

        if (!existingReports) {
          const [rows] = await connection.query<ReportRow[]>(
            `SELECT report_id, pdf_file_path
               FROM experiment_reports
              WHERE experiment_id = ? AND student_id = ?
              LIMIT 1
              FOR UPDATE`,
            [experimentId, studentId],
          );
          existingReports = rows;
        }
        const existingReport = existingReports[0];
        const previousPdfPath = existingReport?.pdf_file_path ?? null;

        const [reportResult] = await connection.query<ResultSetHeader>(
          `INSERT INTO experiment_reports (experiment_id, student_id, report_content, pdf_file_path, status, submitted_at)
           VALUES (?, ?, ?, ?, 'submitted', NOW())
           ON DUPLICATE KEY UPDATE
             report_content = ?,
             pdf_file_path = ?,
             status = 'submitted',
             submitted_at = NOW()`,
          [
            experimentId,
            studentId,
            sanitizedReportContent,
            pdfPath,
            sanitizedReportContent,
            pdfPath,
          ],
        );

        const calculatedBest = getBestModel(experiment);
        const selectedBest = experiment.selected_best_model;
        let modelQualityScore = 0;
        if (calculatedBest && selectedBest === calculatedBest) {
          modelQualityScore = 100;
        } else if (calculatedBest && selectedBest !== calculatedBest) {
          modelQualityScore = 50;
        }

        const flowGradeValues = buildExperimentFlowGradeValues(experiment, {
          allowLegacyV1Write: env.ALLOW_LEGACY_PRODUCTION_PLAN_WRITES,
        });
        const EXP_FLOW_GRADE_COLUMNS = Object.keys(flowGradeValues) as Array<keyof typeof flowGradeValues>;
        const GRADE_COLUMNS = [...EXP_FLOW_GRADE_COLUMNS, "model_quality"];
        const GRADE_UPDATE_SET = GRADE_COLUMNS.map((col) => `${col} = ?`).join(", ");
        const gradeValues = [...EXP_FLOW_GRADE_COLUMNS.map((column) => flowGradeValues[column]), modelQualityScore];

        await connection.query(
          `UPDATE experiment_grade SET ${GRADE_UPDATE_SET} WHERE experiment_id = ? AND student_id = ?`,
          [...gradeValues, experimentId, studentId],
        );

        await connection.query(
          `UPDATE experiment_status
              SET status = 'Completed',
                  completion_time = NOW(),
                  state_version = state_version + 1,
                  last_activity_at = NOW()
            WHERE experiment_id = ? AND student_id = ?`,
          [experimentId, studentId],
        );

        let isUpdate = Boolean(existingReport);
        let reportId = existingReport?.report_id ?? reportResult.insertId;
        if (reportId === 0 && reportResult.affectedRows > 0) {
          const [rows] = await connection.query<RowDataPacket[]>(
            "SELECT report_id FROM experiment_reports WHERE experiment_id = ? AND student_id = ?",
            [experimentId, studentId],
          );
          if (rows.length > 0) {
            reportId = rows[0]!.report_id;
            isUpdate = true;
          }
        }
        if (!Number.isInteger(reportId) || reportId <= 0) {
          throw new Error("报告事务无法确定已写入记录的 report_id");
        }

        const candidate: ReportCommitCandidate = {
          result: {
            data: {
              message: isUpdate ? "报告更新成功" : "报告提交成功",
              report_id: reportId,
              pdf_path: toPublicReportPath(pdfPath) ?? "",
            },
            isUpdate,
          },
          previousPdfPath,
          expectedReportId: reportId,
          pdfPath,
        };

        transactionPhase = "commit-attempted";
        try {
          await connection.commit();
        } catch (error) {
          // The session is no longer trustworthy after a lost commit ACK.
          // Destroying it also forces the server-side transaction to reach a
          // terminal state before the fresh FOR UPDATE probe can proceed.
          disposed = true;
          try {
            connection.destroy();
          } catch {}
          throw new AmbiguousReportCommitError(error, candidate);
        }
        transactionPhase = "committed";
        return {
          result: candidate.result,
          previousPdfPath,
        };
      } catch (error) {
        if (transactionPhase === "before-commit") {
          try {
            await connection.rollback();
          } catch (rollbackError) {
            disposed = true;
            try {
              connection.destroy();
            } catch {}
            throw new Error("报告事务回滚结果未知，拒绝自动重试", {
              cause: { transactionError: error, rollbackError },
            });
          }
        }
        throw error;
      } finally {
        if (!disposed) {
          try {
            connection.release();
          } catch (releaseError) {
            try {
              connection.destroy();
            } catch {}
            logger.warn({ err: releaseError }, "释放报告提交事务连接失败");
          }
        }
      }
    }, {
      maxAttempts: 4,
      retryDelaysMs: [75, 200, 500],
      onRetry: ({ error, attempt, nextAttempt, maxAttempts, delayMs }) => {
        logger.warn(
          { err: error, experimentId, studentId, attempt, nextAttempt, maxAttempts, delayMs },
          "报告提交事务发生锁冲突，准备重试",
        );
      },
    });

    if ("error" in transactionResult.result) {
      await cleanupGeneratedReportPdf(pdfPath, reportsDir);
      return transactionResult.result;
    }

    return finalizeCommittedSubmission(
      transactionResult.result,
      transactionResult.previousPdfPath,
    );
  } catch (error) {
    if (error instanceof AmbiguousReportCommitError) {
      let probeState: ReportCommitProbeState = "unknown";
      try {
        probeState = await probeAmbiguousReportCommit(
          experimentId,
          studentId,
          error.candidate,
        );
      } catch (probeError) {
        logger.warn(
          { err: probeError, experimentId, studentId, pdfPath },
          "报告提交结果未知，保留新 PDF 等待后续核查",
        );
      }

      if (probeState === "committed") {
        return finalizeCommittedSubmission(
          error.candidate.result,
          error.candidate.previousPdfPath,
        );
      }
      if (probeState === "not-committed") {
        await cleanupGeneratedReportPdf(pdfPath, reportsDir);
      } else {
        logger.warn(
          { experimentId, studentId, pdfPath },
          "无法确认报告事务是否提交，保留新 PDF 以避免数据库悬空引用",
        );
      }
      throw error.originalError;
    }
    await cleanupGeneratedReportPdf(pdfPath, reportsDir);
    throw error;
  }
}

export async function getLatestReportStatus(studentId: number) {
  const [[latestExperiment]] = await pool.query<ExperimentRow[]>(
    `SELECT experiment_id, student_id, retry_of_experiment_id, status, current_step, start_time, last_activity_at, completion_time
     FROM experiment_status WHERE student_id = ? ORDER BY experiment_id DESC LIMIT 1`,
    [studentId],
  );

  if (!latestExperiment) {
    return { is_rejected: false, has_report: false, redo_in_progress: false };
  }

  const publicExperiment = {
    ...latestExperiment,
    start_time: databaseDatetimeToIsoString(latestExperiment.start_time, env.DB_TIMEZONE),
    last_activity_at: databaseDatetimeToIsoString(
      latestExperiment.last_activity_at,
      env.DB_TIMEZONE,
    ),
    completion_time: databaseDatetimeToIsoString(
      latestExperiment.completion_time,
      env.DB_TIMEZONE,
    ),
  };

  const [[report]] = await pool.query<ReportRow[]>(
    `SELECT * FROM experiment_reports WHERE experiment_id = ? AND student_id = ?`,
    [latestExperiment.experiment_id, studentId],
  );
  // The immediate rejected parent remains relevant after resubmission: both
  // students and reviewers need to compare the new report against the original
  // correction requirements. It is therefore independent of the child report
  // state; redo_in_progress below still means only "child has no report".
  const rejectedReport = await getRejectedParentReport(
    studentId,
    latestExperiment.retry_of_experiment_id,
  );

  if (!report) {
    return {
      is_rejected: false,
      has_report: false,
      redo_in_progress: rejectedReport !== null,
      experiment: publicExperiment,
      ...(rejectedReport ? { rejected_report: rejectedReport } : {}),
    };
  }

  return {
    is_rejected: report.status === "rejected",
    has_report: true,
    redo_in_progress: false,
    experiment: publicExperiment,
    report: await buildLatestReportRecord(report),
    ...(rejectedReport ? { rejected_report: rejectedReport } : {}),
  };
}

export { HTMLSanitizationError };
