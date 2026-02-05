import path from "path";
import { mkdir } from "node:fs/promises";
import type { RowDataPacket, ResultSetHeader } from "mysql2";
import { marked } from "marked";
import puppeteer from "puppeteer-core";
import { pool } from "../../config";
import { sanitizeHTML, HTMLSanitizationError } from "../../utils/html-sanitizer";
import { CONTENT_LIMITS, PATHS, TIMEOUTS } from "../../utils/constants";

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
  status: string;
  current_step: number;
  selected_best_model: string | null;
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
  status: string;
  feedback: string | null;
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
        .report-content th, .report-content td { border: 1px solid #cbd5f5; padding: 8px 10px; text-align: left; }
        .report-content ul, .report-content ol { padding-left: 1.4rem; }
        .report-content blockquote { border-left: 4px solid #94a3b8; padding-left: 12px; color: #475569; margin: 1rem 0; }
        .report-content hr { border: none; border-top: 1px solid #e2e8f0; margin: 2rem 0; }
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

export async function submitReport(
  experimentId: number,
  studentId: number,
  markdownContent: string,
): Promise<{ error: string; status: 400 | 403 | 404 } | { data: SubmitReportResult; isUpdate: boolean }> {
  // 输入长度限制
  if (markdownContent.length > CONTENT_LIMITS.MARKDOWN_MAX_LENGTH) {
    return {
      error: `报告内容过长，最大允许 ${CONTENT_LIMITS.MARKDOWN_MAX_LENGTH} 字符`,
      status: 400,
    };
  }

  const [[experiment]] = await pool.query<ExperimentRow[]>(
    `SELECT * FROM experiment_status WHERE experiment_id = ? LIMIT 1`,
    [experimentId],
  );

  if (!experiment) {
    return { error: "实验不存在", status: 404 };
  }

  if (experiment.student_id !== studentId) {
    return { error: "权限不足：该实验不属于当前用户", status: 403 };
  }

  const sanitizedReportContent = await markdownToSanitizedHtml(markdownContent);
  const htmlDocument = buildPdfDocument(sanitizedReportContent);

  const reportsDir = path.resolve(import.meta.dir, "../../../uploads", "reports");
  await mkdir(reportsDir, { recursive: true });

  const uniqueFilename = `report-exp${experimentId}-student${studentId}-${Date.now()}.pdf`;
  const pdfPath = path.join(reportsDir, uniqueFilename);

  const browser = await puppeteer.launch({
    executablePath: PATHS.CHROMIUM_PATH,
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  try {
    const page = await browser.newPage();
    await page.setExtraHTTPHeaders({ "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline';" });
    await page.setJavaScriptEnabled(false);
    await page.setContent(htmlDocument, { waitUntil: "domcontentloaded", timeout: 10000 });
    await page.pdf({ path: pdfPath, format: "A4", printBackground: true });
  } finally {
    await browser.close();
  }

  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    const [reportResult] = await connection.query<ResultSetHeader>(
      `INSERT INTO experiment_reports (experiment_id, student_id, report_content, pdf_file_path, status, submitted_at)
       VALUES (?, ?, ?, ?, 'submitted', NOW())
       ON DUPLICATE KEY UPDATE
         report_content = VALUES(report_content),
         pdf_file_path = VALUES(pdf_file_path),
         status = 'submitted',
         submitted_at = NOW()`,
      [experimentId, studentId, sanitizedReportContent, pdfPath],
    );

    const calculatedBest = getBestModel(experiment);
    const selectedBest = experiment.selected_best_model;
    let modelQualityScore = 0;
    if (calculatedBest && selectedBest === calculatedBest) {
      modelQualityScore = 100;
    } else if (calculatedBest && selectedBest !== calculatedBest) {
      modelQualityScore = 50;
    }

    const EXP_FLOW_GRADE_COLUMNS = [
      "exp_flow_demand_data_preparation", "exp_flow_demand_descriptive_stats",
      "exp_flow_demand_model_selection", "exp_flow_demand_generate_results",
      "exp_flow_production_inventory_calc", "exp_flow_production_service_level",
      "exp_flow_production_variable_calc", "exp_flow_production_plan_creation",
      "exp_flow_report_submission",
    ] as const;

    const GRADE_COLUMNS = [...EXP_FLOW_GRADE_COLUMNS, "model_quality"];
    const GRADE_UPDATE_SET = GRADE_COLUMNS.map((col) => `${col} = ?`).join(", ");
    const gradeValues = [...EXP_FLOW_GRADE_COLUMNS.map(() => 100), modelQualityScore];

    await connection.query(
      `UPDATE experiment_grade SET ${GRADE_UPDATE_SET} WHERE experiment_id = ? AND student_id = ?`,
      [...gradeValues, experimentId, studentId],
    );

    await connection.commit();

    let reportId = reportResult.insertId;
    let isUpdate = false;

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

    return { data: { message: isUpdate ? "报告更新成功" : "报告提交成功", report_id: reportId, pdf_path: pdfPath }, isUpdate };
  } catch (error) {
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}

export async function getLatestReportStatus(studentId: number) {
  const [[latestExperiment]] = await pool.query<ExperimentRow[]>(
    `SELECT experiment_id, student_id, status, current_step, start_time, last_activity_at, completion_time
     FROM experiment_status WHERE student_id = ? ORDER BY experiment_id DESC LIMIT 1`,
    [studentId],
  );

  if (!latestExperiment) {
    return { is_rejected: false };
  }

  const [[report]] = await pool.query<ReportRow[]>(
    `SELECT * FROM experiment_reports WHERE experiment_id = ? AND student_id = ?`,
    [latestExperiment.experiment_id, studentId],
  );

  if (report && report.status === "rejected") {
    return { is_rejected: true, experiment: latestExperiment, report };
  }

  return { is_rejected: false };
}

export { HTMLSanitizationError };
