import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { mkdtemp, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  buildCanonicalProductionMps,
  buildProductionCapacityScenarios,
} from "../../../src/modules/experiments/experiments.schema";
import { issueProductionForecastEvidence } from "../../../src/modules/experiments/production-forecast-evidence";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const htmlSanitizerModulePath = path.join(srcRoot, "utils", "html-sanitizer.ts");
const loggerModulePath = path.join(srcRoot, "utils", "logger.ts");
const modelsServiceModulePath = path.join(srcRoot, "modules", "models", "models.service.ts");
const reportsServiceModulePath = path.join(srcRoot, "modules", "reports", "reports.service.ts");
const freshReportsServiceModulePath = `${reportsServiceModulePath}?reports-service-test`;
let reportUploadsDir = "";
const managedReportsDir = path.resolve(srcRoot, "../uploads/reports");
const managedReportFixtures: string[] = [];

const poolQuery = mock(async () => [[]]);
const connectionQuery = mock(async () => [[]]);
const beginTransaction = mock(async () => {});
const commit = mock(async () => {});
const rollback = mock(async () => {});
const release = mock(() => {});
const destroy = mock(() => {});
const mockConnection = {
  query: connectionQuery,
  beginTransaction,
  commit,
  rollback,
  release,
  destroy,
};
const getConnection = mock(async () => mockConnection);

const logger = {
  info: mock(() => {}),
  error: mock(() => {}),
  warn: mock(() => {}),
};

const cleanupExperimentModelArtifacts = mock(() => {});
const pagePdf = mock(async (options: { path: string }) => {
  await Bun.write(options.path, "pdf-fixture");
});
const page = {
  setExtraHTTPHeaders: mock(async () => {}),
  setJavaScriptEnabled: mock(async () => {}),
  setContent: mock(async () => {}),
  pdf: pagePdf,
};
const browserProcessKill = mock((_signal: NodeJS.Signals) => true);
const browserProcess = { kill: browserProcessKill };
const browser = {
  newPage: mock(async () => page),
  close: mock(async () => {}),
  process: mock(() => browserProcess),
};
const launch = mock(async () => browser);

class HTMLSanitizationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HTMLSanitizationError";
  }
}

mock.module(configModulePath, () => ({
  env: {
    DB_HOST: process.env.DB_HOST ?? "127.0.0.1",
    DB_USER: process.env.DB_USER ?? "test",
    DB_PASSWORD: process.env.DB_PASSWORD ?? "test",
    DB_DATABASE: process.env.DB_DATABASE ?? "test",
    DB_TIMEZONE: "Z",
    JWT_SECRET: "test-jwt-secret",
    ALLOW_LEGACY_PRODUCTION_PLAN_WRITES: true,
    PORT: Number(process.env.PORT ?? "3001"),
  },
  connectionLimit: 64,
  pool: {
    query: poolQuery,
    getConnection,
  },
}));

mock.module(htmlSanitizerModulePath, () => ({
  sanitizeHTML: (html: string) => html,
  HTMLSanitizationError,
}));

mock.module(loggerModulePath, () => ({
  default: logger,
}));

mock.module(modelsServiceModulePath, () => ({
  cleanupExperimentModelArtifacts,
}));

mock.module("puppeteer-core", () => ({
  default: { launch },
}));

const {
  submitReport: submitReportService,
  cleanupReplacedReportPdf,
  getLatestReportStatus,
  PdfGenerationTimeoutError,
} = await import(freshReportsServiceModulePath);

const submitReport = (
  experimentId: number,
  studentId: number,
  markdownContent: string,
) => submitReportService(experimentId, studentId, markdownContent, {
  reportsDir: reportUploadsDir,
});

type MockExperimentRow = {
  experiment_id: number;
  student_id: number;
  status: string;
  current_step: number;
  highest_completed_step: number;
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
};

const buildProductionOutputs = () => {
  const forecastPoint = {
    prediction: 100,
    std_dev: 12,
    upper_error_p99: 28,
    uncertainty_source: "model",
    calibration_source: "arima_forecast_distribution",
    calibration_mean_error: 1.5,
    calibration_count: 8,
    bias_diagnostic_source: "outer_selection_assessment",
    interval_lower: 76.48,
    interval_upper: 123.52,
    interval_level: 0.95,
    interval_kind: "model_prediction_interval",
    coverage_claim: "model_nominal",
  };
  const forecasts = Array.from({ length: 6 }, () => ({ ...forecastPoint }));
  const zScore = 1.65;
  const capacity = buildProductionCapacityScenarios(forecasts, zScore).normal;
  const rows = buildCanonicalProductionMps({
    forecasts,
    zScore,
    initialInventory: 0,
    capacity,
  });

  return { forecasts, rows, capacity };
};

const REPORT_RUN_ID = "reports-service-run";
const REPORT_PRODUCTION_REVISION = "reports-service-revision";

const buildExperiment = (
  overrides: Partial<MockExperimentRow> = {},
): MockExperimentRow => {
  const { forecasts, rows, capacity } = buildProductionOutputs();
  const experimentId = overrides.experiment_id ?? 990001;
  const studentId = overrides.student_id ?? 20241002;
  return {
    experiment_id: experimentId,
    student_id: studentId,
    status: "In Progress",
    current_step: 7,
    highest_completed_step: 7,
    selected_industry: "制造业",
    selected_company: "样本企业",
    selected_product: "样本产品",
    selected_best_model: "lstm",
    quiz_about_model_completed: true,
    quiz_about_plan_completed: true,
    production_plan_completed: true,
    data_window_train_start_index: 0,
    data_window_train_end_index: 11,
    data_window_evaluate_start_index: 12,
    data_window_evaluate_end_index: 17,
    production_forecast_periods: 6,
    production_initial_inventory: 0,
    production_target_service_level: 0.95,
    production_safety_stock_z_score: 1.65,
    production_forecast_results: forecasts,
    production_forecast_evidence: issueProductionForecastEvidence({
      experiment_id: experimentId,
      student_id: studentId,
      model_type: "lstm",
      run_id: REPORT_RUN_ID,
      production_revision: REPORT_PRODUCTION_REVISION,
      forecast_steps: forecasts.length,
      forecasts,
    }),
    production_mps_table: rows,
    production_capacity_mode: "scenario",
    production_capacity_scenario: "normal",
    production_capacity: capacity,
    production_custom_capacity: null,
    moving_average_completed: false,
    moving_average_metrics_rmse: null,
    moving_average_metrics_mae: null,
    moving_average_metrics_r2: null,
    exponential_smoothing_completed: false,
    exponential_smoothing_metrics_rmse: null,
    exponential_smoothing_metrics_mae: null,
    exponential_smoothing_metrics_r2: null,
    arima_completed: false,
    arima_metrics_rmse: null,
    arima_metrics_mae: null,
    arima_metrics_r2: null,
    lstm_completed: true,
    lstm_metrics_rmse: 1.2,
    lstm_metrics_mae: 0.8,
    lstm_metrics_r2: 0.91,
    ensemble_weighted_completed: false,
    ensemble_weighted_metrics_rmse: null,
    ensemble_weighted_metrics_mae: null,
    ensemble_weighted_metrics_r2: null,
    ensemble_boosting_completed: false,
    ensemble_boosting_metrics_rmse: null,
    ensemble_boosting_metrics_mae: null,
    ensemble_boosting_metrics_r2: null,
    ensemble_stacking_completed: false,
    ensemble_stacking_metrics_rmse: null,
    ensemble_stacking_metrics_mae: null,
    ensemble_stacking_metrics_r2: null,
    ...overrides,
  };
};

const buildLatestExperimentStatus = (overrides: Record<string, unknown> = {}) => ({
  experiment_id: 990021,
  student_id: 20241002,
  retry_of_experiment_id: null,
  status: "Completed",
  current_step: 8,
  start_time: "2026-03-01T08:00:00.000Z",
  last_activity_at: "2026-03-01T10:00:00.000Z",
  completion_time: "2026-03-01T11:00:00.000Z",
  ...overrides,
});

const buildReportStatusRow = (overrides: Record<string, unknown> = {}) => ({
  report_id: 880021,
  experiment_id: 990021,
  student_id: 20241002,
  report_content: "# report",
  pdf_file_path: "/srv/fz/uploads/reports/student-report.pdf",
  status: "submitted",
  submitted_at: "2026-03-01T11:30:00.000Z",
  grade: null,
  feedback: null,
  graded_by: null,
  ...overrides,
});

const listGeneratedReports = async (experimentId: number, studentId: number) => {
  const entries = await readdir(reportUploadsDir);
  return entries.filter((entry) => entry.startsWith(`report-exp${experimentId}-student${studentId}-`));
};

const buildLostCommitError = () => {
  const error = new Error("commit acknowledgement lost") as Error & {
    code: string;
    errno: number;
    sqlState: string;
  };
  // Deliberately looks retryable. The report service must wrap commit-unknown
  // errors so withDeadlockRetry never replays the write transaction.
  error.code = "ER_LOCK_DEADLOCK";
  error.errno = 1213;
  error.sqlState = "40001";
  return error;
};

describe("reports service", () => {
  beforeEach(async () => {
    reportUploadsDir = await mkdtemp(path.join(tmpdir(), "fz-reports-service-"));
    poolQuery.mockReset();
    connectionQuery.mockReset();
    beginTransaction.mockClear();
    commit.mockReset();
    commit.mockResolvedValue(undefined);
    rollback.mockClear();
    release.mockClear();
    destroy.mockClear();
    getConnection.mockReset();
    getConnection.mockResolvedValue(mockConnection);
    launch.mockReset();
    launch.mockResolvedValue(browser);
    browser.newPage.mockClear();
    browser.close.mockReset();
    browser.close.mockResolvedValue(undefined);
    browser.process.mockClear();
    browserProcessKill.mockClear();
    page.setExtraHTTPHeaders.mockClear();
    page.setJavaScriptEnabled.mockClear();
    page.setContent.mockClear();
    pagePdf.mockReset();
    pagePdf.mockImplementation(async (options: { path: string }) => {
      await Bun.write(options.path, "pdf-fixture");
    });
    logger.error.mockClear();
    logger.warn.mockClear();
    cleanupExperimentModelArtifacts.mockClear();

  });

  afterEach(async () => {
    if (reportUploadsDir) {
      await rm(reportUploadsDir, { recursive: true, force: true });
      reportUploadsDir = "";
    }
    await Promise.all(managedReportFixtures.splice(0).map((fixture) => rm(fixture, { force: true })));
  });

  test("returns no-report status when the student has no experiment", async () => {
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[]];
      throw new Error(`Unexpected SQL in no experiment status test: ${sql}`);
    });

    await expect(getLatestReportStatus(20241002)).resolves.toEqual({
      is_rejected: false,
      has_report: false,
      redo_in_progress: false,
    });
  });

  test("returns latest experiment with no-report status before report submission", async () => {
    const latestExperiment = buildLatestExperimentStatus({ status: "In Progress", current_step: 5 });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[latestExperiment]];
      if (sql.includes("FROM experiment_reports")) return [[]];
      throw new Error(`Unexpected SQL in no report status test: ${sql}`);
    });

    await expect(getLatestReportStatus(20241002)).resolves.toEqual({
      is_rejected: false,
      has_report: false,
      redo_in_progress: false,
      experiment: latestExperiment,
    });
  });

  test("normalizes timezone-free experiment and report timestamps to OpenAPI date-time", async () => {
    const latestExperiment = buildLatestExperimentStatus({
      start_time: "2026-08-04 08:00:00",
      last_activity_at: "2026-08-04 09:00:00",
      completion_time: "2026-08-04 10:00:00",
    });
    const report = buildReportStatusRow({
      submitted_at: "2026-08-04 10:05:00.123456",
    });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[latestExperiment]];
      if (sql.includes("FROM experiment_reports")) return [[report]];
      throw new Error(`Unexpected SQL in timestamp normalization test: ${sql}`);
    });

    await expect(getLatestReportStatus(20241002)).resolves.toMatchObject({
      experiment: {
        start_time: "2026-08-04T08:00:00.000Z",
        last_activity_at: "2026-08-04T09:00:00.000Z",
        completion_time: "2026-08-04T10:00:00.000Z",
      },
      report: {
        submitted_at: "2026-08-04T10:05:00.123Z",
      },
    });
  });

  test("returns parent rejection context while the latest child is being redone", async () => {
    const latestExperiment = buildLatestExperimentStatus({
      experiment_id: 990022,
      retry_of_experiment_id: 990021,
      status: "In Progress",
      current_step: 2,
      completion_time: null,
    });
    const rejectedPdfPath = path.join(
      managedReportsDir,
      `unit-archived-rejected-${process.pid}-${Date.now()}.pdf`,
    );
    managedReportFixtures.push(rejectedPdfPath);
    await Bun.write(rejectedPdfPath, "archived-pdf");
    const rejectedReport = buildReportStatusRow({
      status: "rejected",
      feedback: "请从第一步完整重做。",
      grade: null,
      pdf_file_path: rejectedPdfPath,
    });
    poolQuery.mockImplementation(async (sql: string, values?: unknown[]) => {
      if (sql.includes("FROM experiment_status WHERE student_id")) {
        expect(values).toEqual([20241002]);
        return [[latestExperiment]];
      }
      if (sql.includes("FROM experiment_reports WHERE experiment_id")) {
        expect(values).toEqual([990022, 20241002]);
        return [[]];
      }
      if (sql.includes("FROM experiment_reports er")) {
        expect(sql).toContain("parent.student_id = ?");
        expect(sql).toContain("er.status = 'rejected'");
        expect(values).toEqual([990021, 20241002, 20241002]);
        return [[rejectedReport]];
      }
      throw new Error(`Unexpected SQL in redo status test: ${sql}`);
    });

    await expect(getLatestReportStatus(20241002)).resolves.toEqual({
      is_rejected: false,
      has_report: false,
      redo_in_progress: true,
      experiment: latestExperiment,
      rejected_report: {
        ...rejectedReport,
        pdf_file_path: `reports/${path.basename(rejectedPdfPath)}`,
        pdf_file_available: true,
      },
    });
  });

  test("keeps parent rejection context after the retry report is submitted", async () => {
    const latestExperiment = buildLatestExperimentStatus({
      experiment_id: 990022,
      retry_of_experiment_id: 990021,
    });
    const currentReport = buildReportStatusRow({
      report_id: 880022,
      experiment_id: 990022,
      status: "submitted",
      pdf_file_path: "/missing/current/retry-report.pdf",
    });
    const rejectedPdfPath = path.join(
      managedReportsDir,
      `unit-archived-parent-${process.pid}-${Date.now()}.pdf`,
    );
    managedReportFixtures.push(rejectedPdfPath);
    await Bun.write(rejectedPdfPath, "archived-parent-pdf");
    const rejectedReport = buildReportStatusRow({
      status: "rejected",
      feedback: "请根据本轮意见完整重做。",
      pdf_file_path: rejectedPdfPath,
    });
    poolQuery.mockImplementation(async (sql: string, values?: unknown[]) => {
      if (sql.includes("FROM experiment_status WHERE student_id")) {
        return [[latestExperiment]];
      }
      if (sql.includes("FROM experiment_reports WHERE experiment_id")) {
        expect(values).toEqual([990022, 20241002]);
        return [[currentReport]];
      }
      if (sql.includes("FROM experiment_reports er")) {
        expect(sql).toContain("parent.student_id = ?");
        expect(values).toEqual([990021, 20241002, 20241002]);
        return [[rejectedReport]];
      }
      throw new Error(`Unexpected SQL in submitted retry status test: ${sql}`);
    });

    await expect(getLatestReportStatus(20241002)).resolves.toEqual({
      is_rejected: false,
      has_report: true,
      redo_in_progress: false,
      experiment: latestExperiment,
      report: {
        ...currentReport,
        pdf_file_path: "reports/retry-report.pdf",
        pdf_file_available: false,
      },
      rejected_report: {
        ...rejectedReport,
        pdf_file_path: `reports/${path.basename(rejectedPdfPath)}`,
        pdf_file_available: true,
      },
    });
  });

  test("returns submitted latest report status without marking it rejected", async () => {
    const latestExperiment = buildLatestExperimentStatus();
    const report = buildReportStatusRow({ status: "submitted" });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[latestExperiment]];
      if (sql.includes("FROM experiment_reports")) return [[report]];
      throw new Error(`Unexpected SQL in submitted status test: ${sql}`);
    });

    await expect(getLatestReportStatus(20241002)).resolves.toEqual({
      is_rejected: false,
      has_report: true,
      redo_in_progress: false,
      experiment: latestExperiment,
      report: {
        ...report,
        pdf_file_path: "reports/student-report.pdf",
        pdf_file_available: false,
      },
    });
  });

  test("returns graded latest report status with grade and feedback", async () => {
    const latestExperiment = buildLatestExperimentStatus();
    const report = buildReportStatusRow({
      status: "graded",
      grade: 91,
      feedback: "报告结构完整，误差分析充分。",
      graded_by: 101,
    });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[latestExperiment]];
      if (sql.includes("FROM experiment_reports")) return [[report]];
      throw new Error(`Unexpected SQL in graded status test: ${sql}`);
    });

    await expect(getLatestReportStatus(20241002)).resolves.toEqual({
      is_rejected: false,
      has_report: true,
      redo_in_progress: false,
      experiment: latestExperiment,
      report: {
        ...report,
        pdf_file_path: "reports/student-report.pdf",
        pdf_file_available: false,
      },
    });
  });

  test("keeps rejected latest report status compatible with the old flag", async () => {
    const latestExperiment = buildLatestExperimentStatus();
    const report = buildReportStatusRow({
      status: "rejected",
      feedback: "请补充关键图表。",
      pdf_file_path: null,
    });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[latestExperiment]];
      if (sql.includes("FROM experiment_reports")) return [[report]];
      throw new Error(`Unexpected SQL in rejected status test: ${sql}`);
    });

    await expect(getLatestReportStatus(20241002)).resolves.toEqual({
      is_rejected: true,
      has_report: true,
      redo_in_progress: false,
      experiment: latestExperiment,
      report: {
        ...report,
        pdf_file_path: null,
        pdf_file_available: false,
      },
    });
  });

  test("rejects ineligible experiments before launching Puppeteer", async () => {
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) {
        return [[buildExperiment({ highest_completed_step: 6, current_step: 6, production_plan_completed: false })]];
      }

      throw new Error(`Unexpected SQL in precheck test: ${sql}`);
    });

    const result = await submitReport(990001, 20241002, "# Test");

    expect(result).toEqual({
      error: "请先完成实验主流程后再提交报告。",
      status: 400,
    });
    expect(launch).not.toHaveBeenCalled();
    expect(getConnection).not.toHaveBeenCalled();
    expect(await listGeneratedReports(990001, 20241002)).toEqual([]);
  });

  test("recovers a completed legacy experiment without evidence when no report exists", async () => {
    const currentExperiment = buildExperiment({
      experiment_id: 990011,
      status: "Completed",
    });
    const legacyForecasts = (currentExperiment.production_forecast_results as Array<{
      prediction: number;
      std_dev: number;
    }>).map(({ prediction, std_dev }) => ({ prediction, std_dev }));
    const completedExperiment = buildExperiment({
      ...currentExperiment,
      production_forecast_results: legacyForecasts,
      production_forecast_evidence: null,
      production_mps_table: legacyForecasts.map((_, index) => ({ period: index + 1 })),
    });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) {
        return [[completedExperiment]];
      }
      if (sql.includes("FROM experiment_reports")) return [[]];
      throw new Error(`Unexpected SQL in completed recovery precheck: ${sql}`);
    });

    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") return [[]];
      if (sql.includes("SELECT user_id FROM users")) return [[{ user_id: 20241002 }]];
      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[completedExperiment]];
      }
      if (sql.includes("FROM experiment_reports") && sql.includes("FOR UPDATE")) {
        return [[]];
      }
      if (sql.includes("INSERT INTO experiment_reports")) {
        return [{ insertId: 880011, affectedRows: 1 }];
      }
      if (sql.includes("UPDATE experiment_grade") || sql.includes("UPDATE experiment_status")) {
        return [{ affectedRows: 1 }];
      }
      throw new Error(`Unexpected SQL in completed recovery transaction: ${sql}`);
    });

    await expect(submitReport(990011, 20241002, "# Test")).resolves.toEqual({
      data: {
        message: "报告提交成功",
        report_id: 880011,
        pdf_path: expect.stringMatching(/^reports\/report-exp990011-student20241002-.*\.pdf$/),
      },
      isUpdate: false,
    });
    expect(commit).toHaveBeenCalledTimes(1);
    expect(connectionQuery.mock.calls.some(([sql]) =>
      String(sql).includes("SELECT run_id, production_revision")
    )).toBeFalse();
  });

  test("normalizes Puppeteer timeouts and removes a partial PDF", async () => {
    const experiment = buildExperiment({ experiment_id: 990014 });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) return [[experiment]];
      throw new Error(`Unexpected SQL in PDF timeout test: ${sql}`);
    });
    const puppeteerTimeout = new Error("page.pdf timed out");
    puppeteerTimeout.name = "TimeoutError";
    pagePdf.mockImplementationOnce(async (options: { path: string }) => {
      await Bun.write(options.path, "partial-pdf");
      throw puppeteerTimeout;
    });

    await expect(submitReport(990014, 20241002, "# Test"))
      .rejects.toBeInstanceOf(PdfGenerationTimeoutError);
    expect(getConnection).not.toHaveBeenCalled();
    expect(browser.close).toHaveBeenCalledTimes(1);
    expect(await listGeneratedReports(990014, 20241002)).toEqual([]);
  });

  test("keeps a generated report when browser close rejects and force-terminates the process", async () => {
    const experiment = buildExperiment({ experiment_id: 990013 });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) return [[experiment]];
      throw new Error(`Unexpected SQL in browser close rejection test: ${sql}`);
    });
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") return [[]];
      if (sql.includes("SELECT user_id FROM users")) return [[{ user_id: 20241002 }]];
      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[experiment]];
      }
      if (sql.includes("SELECT run_id, production_revision")) {
        return [[{
          run_id: REPORT_RUN_ID,
          production_revision: REPORT_PRODUCTION_REVISION,
        }]];
      }
      if (sql.includes("FROM experiment_reports") && sql.includes("FOR UPDATE")) return [[]];
      if (sql.includes("INSERT INTO experiment_reports")) {
        return [{ insertId: 880013, affectedRows: 1 }];
      }
      if (sql.includes("UPDATE experiment_grade") || sql.includes("UPDATE experiment_status")) {
        return [{ affectedRows: 1 }];
      }
      throw new Error(`Unexpected SQL in browser close rejection transaction: ${sql}`);
    });
    browser.close.mockRejectedValueOnce(new Error("browser close failed"));
    browserProcessKill.mockImplementationOnce(() => {
      throw new Error("kill acknowledgement failed");
    });

    await expect(submitReport(990013, 20241002, "# Test")).resolves.toMatchObject({
      data: { report_id: 880013 },
      isUpdate: false,
    });
    expect(browserProcessKill).toHaveBeenCalledWith("SIGKILL");
    expect(logger.warn).toHaveBeenCalledTimes(1);
    expect(await listGeneratedReports(990013, 20241002)).toHaveLength(1);
  });

  test("gives same-millisecond report submissions independent PDF ownership", async () => {
    const experiment = buildExperiment({ experiment_id: 990012 });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) return [[experiment]];
      throw new Error(`Unexpected SQL in unique PDF test: ${sql}`);
    });
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") return [[]];
      if (sql.includes("SELECT user_id FROM users")) return [[{ user_id: 20241002 }]];
      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[experiment]];
      }
      if (sql.includes("SELECT run_id, production_revision")) {
        return [[{
          run_id: REPORT_RUN_ID,
          production_revision: REPORT_PRODUCTION_REVISION,
        }]];
      }
      if (sql.includes("FROM experiment_reports") && sql.includes("FOR UPDATE")) return [[]];
      if (sql.includes("INSERT INTO experiment_reports")) {
        return [{ insertId: 880012, affectedRows: 1 }];
      }
      if (sql.includes("UPDATE experiment_grade") || sql.includes("UPDATE experiment_status")) {
        return [{ affectedRows: 1 }];
      }
      throw new Error(`Unexpected SQL in unique PDF transaction: ${sql}`);
    });

    const originalDateNow = Date.now;
    Date.now = () => 1_785_751_200_000;
    try {
      const [first, second] = await Promise.all([
        submitReport(990012, 20241002, "# First"),
        submitReport(990012, 20241002, "# Second"),
      ]);

      expect("data" in first && "data" in second).toBeTrue();
      if (!("data" in first) || !("data" in second)) {
        throw new Error("Both report submissions should reach the mocked success path");
      }
      expect(first.data.pdf_path).not.toBe(second.data.pdf_path);
      expect(await listGeneratedReports(990012, 20241002)).toHaveLength(2);
    } finally {
      Date.now = originalDateNow;
    }
  });

  test("keeps a replaced legacy PDF while another report still references it", async () => {
    const experimentId = 990016;
    const studentId = 20241002;
    const experiment = buildExperiment({ experiment_id: experimentId, student_id: studentId });
    const sharedLegacyPdf = path.join(reportUploadsDir, "shared-legacy.pdf");
    await Bun.write(sharedLegacyPdf, "legacy-pdf");
    let finishReferenceScan: (() => void) | undefined;
    const referenceScanStarted = new Promise<void>((resolve) => {
      finishReferenceScan = resolve;
    });

    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) return [[experiment]];
      if (sql.includes("SELECT pdf_file_path") && sql.includes("WHERE pdf_file_path IS NOT NULL")) {
        finishReferenceScan?.();
        return [[{ pdf_file_path: sharedLegacyPdf }]];
      }
      throw new Error(`Unexpected SQL in shared legacy PDF test: ${sql}`);
    });
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") return [[]];
      if (sql.includes("SELECT user_id FROM users")) return [[{ user_id: 20241002 }]];
      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[experiment]];
      }
      if (sql.includes("SELECT run_id, production_revision")) {
        return [[{ run_id: REPORT_RUN_ID, production_revision: REPORT_PRODUCTION_REVISION }]];
      }
      if (sql.includes("FROM experiment_reports") && sql.includes("FOR UPDATE")) {
        return [[{
          report_id: 880016,
          experiment_id: experimentId,
          student_id: studentId,
          pdf_file_path: sharedLegacyPdf,
          status: "submitted",
        }]];
      }
      if (sql.includes("INSERT INTO experiment_reports")) {
        return [{ insertId: 0, affectedRows: 2 }];
      }
      if (sql.includes("UPDATE experiment_grade") || sql.includes("UPDATE experiment_status")) {
        return [{ affectedRows: 1 }];
      }
      throw new Error(`Unexpected SQL in shared legacy PDF transaction: ${sql}`);
    });

    await expect(submitReport(experimentId, studentId, "# replacement")).resolves.toEqual({
      data: {
        message: "报告更新成功",
        report_id: 880016,
        pdf_path: expect.stringMatching(/^reports\/report-exp990016-student20241002-.*\.pdf$/),
      },
      isUpdate: true,
    });
    await referenceScanStarted;
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(await Bun.file(sharedLegacyPdf).exists()).toBeTrue();
  });

  test("removes a replaced PDF after the reference scan proves it is unowned", async () => {
    const replacedPdf = path.join(reportUploadsDir, "unowned-legacy.pdf");
    await Bun.write(replacedPdf, "legacy-pdf");
    poolQuery.mockResolvedValueOnce([[]]);

    await cleanupReplacedReportPdf(replacedPdf, reportUploadsDir);

    expect(await Bun.file(replacedPdf).exists()).toBeFalse();
  });

  test("returns success and keeps the PDF when a fresh locking probe confirms an ambiguous commit", async () => {
    const experimentId = 990013;
    const studentId = 20241002;
    const experiment = buildExperiment({ experiment_id: experimentId, student_id: studentId });
    let insertedPdfPath = "";
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) return [[experiment]];
      throw new Error(`Unexpected SQL in committed probe precheck: ${sql}`);
    });
    connectionQuery.mockImplementation(async (sql: string, values?: unknown[]) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") return [[]];
      if (sql.includes("SELECT user_id FROM users")) return [[{ user_id: studentId }]];
      if (sql.includes("SELECT experiment_id, student_id, status")) {
        return [[{ experiment_id: experimentId, student_id: studentId, status: "Completed" }]];
      }
      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[experiment]];
      }
      if (sql.includes("SELECT run_id, production_revision")) {
        return [[{ run_id: REPORT_RUN_ID, production_revision: REPORT_PRODUCTION_REVISION }]];
      }
      if (sql.includes("SELECT report_id, experiment_id, student_id, pdf_file_path, status")) {
        return [[{
          report_id: 880013,
          experiment_id: experimentId,
          student_id: studentId,
          pdf_file_path: insertedPdfPath,
          status: "submitted",
        }]];
      }
      if (sql.includes("FROM experiment_reports") && sql.includes("FOR UPDATE")) return [[]];
      if (sql.includes("INSERT INTO experiment_reports")) {
        insertedPdfPath = String(values?.[3]);
        return [{ insertId: 880013, affectedRows: 1 }];
      }
      if (sql.includes("UPDATE experiment_grade") || sql.includes("UPDATE experiment_status")) {
        return [{ affectedRows: 1 }];
      }
      throw new Error(`Unexpected SQL in committed probe transaction: ${sql}`);
    });
    commit.mockRejectedValueOnce(buildLostCommitError());

    await expect(submitReport(experimentId, studentId, "# committed")).resolves.toEqual({
      data: {
        message: "报告提交成功",
        report_id: 880013,
        pdf_path: expect.stringMatching(/^reports\/report-exp990013-student20241002-.*\.pdf$/),
      },
      isUpdate: false,
    });

    expect(getConnection).toHaveBeenCalledTimes(2);
    expect(commit).toHaveBeenCalledTimes(2);
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
    expect(connectionQuery.mock.calls
      .map(([sql]) => String(sql))
      .filter((sql) => sql.includes("FOR UPDATE") && (
        sql.includes("FROM users")
        || sql.includes("FROM experiment_status")
        || sql.includes("FROM experiment_reports")
      ))
      .map((sql) => sql.includes("FROM users")
        ? "user"
        : sql.includes("FROM experiment_status") ? "experiment" : "report"))
      .toEqual(["user", "experiment", "report", "user", "experiment", "report"]);
    expect(await listGeneratedReports(experimentId, studentId)).toHaveLength(1);
  });

  test("trusts the locked committed observation when the read-only probe loses its COMMIT acknowledgement", async () => {
    const experimentId = 990017;
    const studentId = 20241002;
    const experiment = buildExperiment({ experiment_id: experimentId, student_id: studentId });
    let insertedPdfPath = "";
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) return [[experiment]];
      throw new Error(`Unexpected SQL in probe ACK-loss precheck: ${sql}`);
    });
    connectionQuery.mockImplementation(async (sql: string, values?: unknown[]) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") return [[]];
      if (sql.includes("SELECT user_id FROM users")) return [[{ user_id: studentId }]];
      if (sql.includes("SELECT experiment_id, student_id, status")) {
        return [[{ experiment_id: experimentId, student_id: studentId, status: "Completed" }]];
      }
      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[experiment]];
      }
      if (sql.includes("SELECT run_id, production_revision")) {
        return [[{ run_id: REPORT_RUN_ID, production_revision: REPORT_PRODUCTION_REVISION }]];
      }
      if (sql.includes("SELECT report_id, experiment_id, student_id, pdf_file_path, status")) {
        return [[{
          report_id: 880017,
          experiment_id: experimentId,
          student_id: studentId,
          pdf_file_path: insertedPdfPath,
          status: "submitted",
        }]];
      }
      if (sql.includes("FROM experiment_reports") && sql.includes("FOR UPDATE")) return [[]];
      if (sql.includes("INSERT INTO experiment_reports")) {
        insertedPdfPath = String(values?.[3]);
        return [{ insertId: 880017, affectedRows: 1 }];
      }
      if (sql.includes("UPDATE experiment_grade") || sql.includes("UPDATE experiment_status")) {
        return [{ affectedRows: 1 }];
      }
      throw new Error(`Unexpected SQL in probe ACK-loss transaction: ${sql}`);
    });
    commit
      .mockRejectedValueOnce(buildLostCommitError())
      .mockRejectedValueOnce(new Error("read-only probe commit acknowledgement lost"));

    await expect(submitReport(experimentId, studentId, "# committed despite probe ACK loss"))
      .resolves.toEqual({
        data: {
          message: "报告提交成功",
          report_id: 880017,
          pdf_path: expect.stringMatching(/^reports\/report-exp990017-student20241002-.*\.pdf$/),
        },
        isUpdate: false,
      });

    expect(getConnection).toHaveBeenCalledTimes(2);
    expect(commit).toHaveBeenCalledTimes(2);
    // Both the original ambiguous write connection and the probe connection
    // with an uncertain terminal ACK are discarded.
    expect(destroy).toHaveBeenCalledTimes(2);
    expect(release).not.toHaveBeenCalled();
    expect(rollback).not.toHaveBeenCalled();
    expect(await listGeneratedReports(experimentId, studentId)).toHaveLength(1);
  });

  test("deletes the generated PDF and surfaces the original error when the probe proves no commit", async () => {
    const experimentId = 990014;
    const studentId = 20241002;
    const experiment = buildExperiment({ experiment_id: experimentId, student_id: studentId });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) return [[experiment]];
      throw new Error(`Unexpected SQL in not-committed probe precheck: ${sql}`);
    });
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") return [[]];
      if (sql.includes("SELECT user_id FROM users")) return [[{ user_id: studentId }]];
      if (sql.includes("SELECT experiment_id, student_id, status")) {
        return [[{ experiment_id: experimentId, student_id: studentId, status: "In Progress" }]];
      }
      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[experiment]];
      }
      if (sql.includes("SELECT run_id, production_revision")) {
        return [[{ run_id: REPORT_RUN_ID, production_revision: REPORT_PRODUCTION_REVISION }]];
      }
      if (sql.includes("SELECT report_id, experiment_id, student_id, pdf_file_path, status")) {
        return [[]];
      }
      if (sql.includes("FROM experiment_reports") && sql.includes("FOR UPDATE")) return [[]];
      if (sql.includes("INSERT INTO experiment_reports")) {
        return [{ insertId: 880014, affectedRows: 1 }];
      }
      if (sql.includes("UPDATE experiment_grade") || sql.includes("UPDATE experiment_status")) {
        return [{ affectedRows: 1 }];
      }
      throw new Error(`Unexpected SQL in not-committed probe transaction: ${sql}`);
    });
    const commitError = buildLostCommitError();
    commit.mockRejectedValueOnce(commitError);

    await expect(submitReport(experimentId, studentId, "# not committed"))
      .rejects.toBe(commitError);

    expect(getConnection).toHaveBeenCalledTimes(2);
    expect(commit).toHaveBeenCalledTimes(2);
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(await listGeneratedReports(experimentId, studentId)).toEqual([]);
  });

  test("retains the generated PDF without replaying the transaction when the fresh probe fails", async () => {
    const experimentId = 990015;
    const studentId = 20241002;
    const experiment = buildExperiment({ experiment_id: experimentId, student_id: studentId });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) return [[experiment]];
      throw new Error(`Unexpected SQL in failed probe precheck: ${sql}`);
    });
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") return [[]];
      if (sql.includes("SELECT user_id FROM users")) return [[{ user_id: studentId }]];
      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[experiment]];
      }
      if (sql.includes("SELECT run_id, production_revision")) {
        return [[{ run_id: REPORT_RUN_ID, production_revision: REPORT_PRODUCTION_REVISION }]];
      }
      if (sql.includes("FROM experiment_reports") && sql.includes("FOR UPDATE")) return [[]];
      if (sql.includes("INSERT INTO experiment_reports")) {
        return [{ insertId: 880015, affectedRows: 1 }];
      }
      if (sql.includes("UPDATE experiment_grade") || sql.includes("UPDATE experiment_status")) {
        return [{ affectedRows: 1 }];
      }
      throw new Error(`Unexpected SQL in failed probe transaction: ${sql}`);
    });
    const commitError = buildLostCommitError();
    commit.mockRejectedValueOnce(commitError);
    getConnection
      .mockResolvedValueOnce(mockConnection)
      .mockRejectedValueOnce(new Error("fresh probe unavailable"));

    await expect(submitReport(experimentId, studentId, "# probe failed"))
      .rejects.toBe(commitError);

    expect(getConnection).toHaveBeenCalledTimes(2);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(await listGeneratedReports(experimentId, studentId)).toHaveLength(1);
    expect(logger.warn.mock.calls.some(([, message]) => (
      String(message).includes("保留新 PDF")
    ))).toBe(true);
  });

  test("retries the database transaction on report submission deadlock", async () => {
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) {
        return [[buildExperiment({ experiment_id: 990008 })]];
      }

      throw new Error(`Unexpected SQL in retry test: ${sql}`);
    });

    let insertAttempts = 0;
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT user_id FROM users")) return [[{ user_id: 20241002 }]];

      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[buildExperiment({ experiment_id: 990008 })]];
      }

      if (sql.includes("SELECT run_id, production_revision")) {
        return [[{
          run_id: REPORT_RUN_ID,
          production_revision: REPORT_PRODUCTION_REVISION,
        }]];
      }

      if (sql.includes("FROM experiment_reports") && sql.includes("FOR UPDATE")) {
        return [[]];
      }

      if (sql.includes("INSERT INTO experiment_reports")) {
        insertAttempts += 1;
        if (insertAttempts === 1) {
          const error = new Error("Deadlock found when trying to get lock; try restarting transaction") as Error & { code: string; errno: number; sqlState: string };
          error.code = "ER_LOCK_DEADLOCK";
          error.errno = 1213;
          error.sqlState = "40001";
          throw error;
        }
        return [{ insertId: 880008, affectedRows: 1 }];
      }

      if (sql.includes("UPDATE experiment_grade") || sql.includes("UPDATE experiment_status")) {
        return [{ affectedRows: 1 }];
      }

      throw new Error(`Unexpected SQL in retry test: ${sql}`);
    });

    const result = await submitReport(990008, 20241002, "# Test");

    expect(result).toEqual({
      data: {
        message: "报告提交成功",
        report_id: 880008,
        pdf_path: expect.stringMatching(/^reports\/report-exp990008-student20241002-.*\.pdf$/),
      },
      isUpdate: false,
    });
    expect(launch).toHaveBeenCalledTimes(1);
    expect(pagePdf).toHaveBeenCalledTimes(1);
    expect(getConnection).toHaveBeenCalledTimes(2);
    expect(beginTransaction).toHaveBeenCalledTimes(2);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(logger.warn).toHaveBeenCalledTimes(1);
    const statusUpdateCall = connectionQuery.mock.calls.find(([sql]) =>
      String(sql).includes("UPDATE experiment_status")
    );
    expect(statusUpdateCall?.[0]).toContain("state_version = state_version + 1");
    expect(await listGeneratedReports(990008, 20241002)).toHaveLength(1);
  });

  test("removes the generated PDF when the locked experiment becomes ineligible", async () => {
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) {
        return [[buildExperiment({ experiment_id: 990007 })]];
      }

      throw new Error(`Unexpected SQL in locked-ineligible test: ${sql}`);
    });

    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT user_id FROM users")) return [[{ user_id: 20241002 }]];

      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[buildExperiment({
          experiment_id: 990007,
          highest_completed_step: 6,
          current_step: 6,
          production_plan_completed: false,
        })]];
      }

      throw new Error(`Unexpected SQL in locked-ineligible test: ${sql}`);
    });

    const result = await submitReport(990007, 20241002, "# Test");

    expect(result).toEqual({
      error: "请先完成实验主流程后再提交报告。",
      status: 400,
    });
    expect(launch).toHaveBeenCalledTimes(1);
    expect(pagePdf).toHaveBeenCalledWith(expect.objectContaining({
      format: "A4",
      printBackground: true,
      timeout: 30_000,
    }));
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(await listGeneratedReports(990007, 20241002)).toEqual([]);
  });

  test("rejects a saved plan when its production revision changes before report submission", async () => {
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) {
        return [[buildExperiment({ experiment_id: 990010 })]];
      }
      throw new Error(`Unexpected SQL in stale production revision precheck: ${sql}`);
    });

    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") return [[]];
      if (sql.includes("SELECT user_id FROM users")) return [[{ user_id: 20241002 }]];
      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[buildExperiment({ experiment_id: 990010 })]];
      }
      if (sql.includes("SELECT run_id, production_revision")) {
        return [[{
          run_id: REPORT_RUN_ID,
          production_revision: `${REPORT_PRODUCTION_REVISION}-new`,
        }]];
      }
      throw new Error(`Unexpected SQL in stale production revision transaction: ${sql}`);
    });

    const result = await submitReport(990010, 20241002, "# Test");

    expect(result).toEqual({
      error: "生产预测对应的模型版本已失效，请重新准备、预测并保存生产计划后再提交报告。",
      status: 400,
    });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(String(connectionQuery.mock.calls.find(([sql]) =>
      String(sql).includes("SELECT run_id, production_revision")
    )?.[0])).toContain("FOR UPDATE");
    expect(connectionQuery.mock.calls.some(([sql]) =>
      String(sql).includes("INSERT INTO experiment_reports")
    )).toBeFalse();
    expect(await listGeneratedReports(990010, 20241002)).toEqual([]);
  });

  test("adds print styles for report appendix and long tables", async () => {
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT * FROM experiment_status")) {
        return [[buildExperiment({ experiment_id: 990009 })]];
      }

      throw new Error(`Unexpected SQL in report print style test: ${sql}`);
    });

    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql === "SET TRANSACTION ISOLATION LEVEL READ COMMITTED") {
        return [[]];
      }

      if (sql.includes("SELECT user_id FROM users")) return [[{ user_id: 20241002 }]];

      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[buildExperiment({ experiment_id: 990009 })]];
      }

      if (sql.includes("SELECT run_id, production_revision")) {
        return [[{
          run_id: REPORT_RUN_ID,
          production_revision: REPORT_PRODUCTION_REVISION,
        }]];
      }

      if (sql.includes("FROM experiment_reports") && sql.includes("FOR UPDATE")) {
        return [[]];
      }

      if (sql.includes("INSERT INTO experiment_reports")) {
        return [{ insertId: 880009, affectedRows: 1 }];
      }

      if (sql.includes("UPDATE experiment_grade") || sql.includes("UPDATE experiment_status")) {
        return [{ affectedRows: 1 }];
      }

      throw new Error(`Unexpected SQL in report print style test: ${sql}`);
    });

    await submitReport(
      990009,
      20241002,
      `# Test

| 月份 | 销量 |
|---|---|
| 2024-01 | 100 |

<div class="report-appendix-break"></div>

## 附录：原始销量明细`,
    );

    expect(page.setContent).toHaveBeenCalledTimes(1);
    const htmlDocument = page.setContent.mock.calls[0]?.[0] as string;
    expect(htmlDocument).toContain(".report-content thead { display: table-header-group; }");
    expect(htmlDocument).toContain(".report-content tr { break-inside: avoid; page-break-inside: avoid; }");
    expect(htmlDocument).toContain(".report-content .report-appendix-break { break-before: page; page-break-before: always; height: 0; }");
    expect(htmlDocument).toContain('<div class="report-appendix-break"></div>');
  });
});
