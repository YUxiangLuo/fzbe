import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { mkdir, readdir, rm } from "node:fs/promises";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const htmlSanitizerModulePath = path.join(srcRoot, "utils", "html-sanitizer.ts");
const loggerModulePath = path.join(srcRoot, "utils", "logger.ts");
const modelsServiceModulePath = path.join(srcRoot, "modules", "models", "models.service.ts");
const reportsServiceModulePath = path.join(srcRoot, "modules", "reports", "reports.service.ts");
const freshReportsServiceModulePath = `${reportsServiceModulePath}?reports-service-test`;
const reportUploadsDir = path.resolve(srcRoot, "../uploads/reports");

const poolQuery = mock(async () => [[]]);
const connectionQuery = mock(async () => [[]]);
const beginTransaction = mock(async () => {});
const commit = mock(async () => {});
const rollback = mock(async () => {});
const release = mock(() => {});
const getConnection = mock(async () => ({
  query: connectionQuery,
  beginTransaction,
  commit,
  rollback,
  release,
}));

const logger = {
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
const browser = {
  newPage: mock(async () => page),
  close: mock(async () => {}),
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
    JWT_SECRET: "test-jwt-secret",
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

const { submitReport } = await import(freshReportsServiceModulePath);

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

const buildExperiment = (
  overrides: Partial<MockExperimentRow> = {},
): MockExperimentRow => ({
  experiment_id: 990001,
  student_id: 20241002,
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
  production_initial_inventory: 50,
  production_target_service_level: 0.95,
  production_safety_stock_z_score: 1.64,
  production_forecast_results: [{ prediction: 100 }],
  production_mps_table: [{ period: 1 }],
  production_capacity_mode: "auto",
  production_capacity_scenario: "normal",
  production_capacity: 120,
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
});

const listGeneratedReports = async (experimentId: number, studentId: number) => {
  const entries = await readdir(reportUploadsDir);
  return entries.filter((entry) => entry.startsWith(`report-exp${experimentId}-student${studentId}-`));
};

describe("reports service", () => {
  beforeEach(async () => {
    poolQuery.mockReset();
    connectionQuery.mockReset();
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    release.mockClear();
    getConnection.mockClear();
    launch.mockReset();
    launch.mockResolvedValue(browser);
    browser.newPage.mockClear();
    browser.close.mockClear();
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

    await rm(reportUploadsDir, { recursive: true, force: true });
    await mkdir(reportUploadsDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(reportUploadsDir, { recursive: true, force: true });
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

      if (sql.includes("SELECT * FROM experiment_status") && sql.includes("FOR UPDATE")) {
        return [[buildExperiment({ experiment_id: 990008 })]];
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
});
