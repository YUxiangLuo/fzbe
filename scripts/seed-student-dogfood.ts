import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import mysql from "mysql2/promise";
import type { ResultSetHeader, RowDataPacket } from "mysql2/promise";
import {
  buildCanonicalProductionMps,
  buildProductionCapacityScenarios,
} from "../src/modules/experiments/experiments.schema";
import { issueProductionForecastEvidence } from "../src/modules/experiments/production-forecast-evidence";
import {
  ensureDatabaseManagedStorageRoot,
  getDatabaseManagedStorageRoot,
  isDisposableStorageNamespaceEnabled,
  type ManagedStorageEnvironment,
} from "../src/utils/managed-storage";
import { buildStudentDogfoodReportPdf } from "./student-dogfood-report-fixture";

const DATABASE_PREFIX = "fz_student_dogfood_";
const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost", "::1"]);

const STANDARD_SELECTION = {
  industry: "Dogfood标准制造业",
  company: "标准样本企业A",
  product: "季节趋势传感器",
} as const;

const STUDENT_SCENARIOS = [
  {
    username: "20990001",
    fullName: "Dogfood 全流程新生",
    scenario: "fresh",
    description: "无历史实验；用于从登录开始完成真实全流程。",
  },
  {
    username: "20990002",
    fullName: "Dogfood 步骤恢复学生",
    scenario: "resume_selection",
    description: "停留在产品选择前；用于恢复弹窗、路由守卫和前后导航。",
  },
  {
    username: "20990003",
    fullName: "Dogfood 模型阶段学生",
    scenario: "model_ready",
    description: "数据窗口已完成；用于四种基础模型和三种融合模型探索。",
  },
  {
    username: "20990004",
    fullName: "Dogfood 报告待评学生",
    scenario: "report_submitted",
    description: "完整实验且报告待评分；用于状态页和 PDF 下载。",
  },
  {
    username: "20990005",
    fullName: "Dogfood 报告驳回学生",
    scenario: "report_rejected",
    description: "报告已驳回；用于驳回原因、重新实验和重新提交入口。",
  },
  {
    username: "20990006",
    fullName: "Dogfood 遗留损坏状态学生",
    scenario: "legacy_invalid_plan",
    description: "处于生产步骤但持久化预测为遗留不完整格式；用于防御性恢复。",
  },
  {
    username: "20990007",
    fullName: "超长姓名与 Unicode 🧪 Dogfood 学生——用于验证导航栏、个人中心、弹窗以及窄屏布局不会截断关键操作",
    scenario: "long_profile",
    description: "超长 Unicode 资料；用于布局、编辑资料和响应式测试。",
  },
  {
    username: "20990008",
    fullName: "Dogfood 完成无报告学生",
    scenario: "completed_without_report",
    description: "实验完成但没有报告；用于首页决策和历史边界。",
  },
  {
    username: "20990009",
    fullName: "Dogfood 报告已评分学生",
    scenario: "report_graded",
    description: "报告已评分并带反馈；用于成绩、反馈和 PDF 状态。",
  },
] as const;

type CsvRow = Record<string, string>;

interface UserRow extends RowDataPacket {
  user_id: number;
  username: string;
}

interface ClassRow extends RowDataPacket {
  class_id: number;
}

interface PredictionPoint {
  prediction: number;
  std_dev: number;
  upper_error_p99: number;
  uncertainty_source: "empirical";
  uncertainty_reason: string;
  calibration_source: string;
  calibration_mean_error: number;
  calibration_count: number;
  bias_diagnostic_source: null;
  interval_lower: number;
  interval_upper: number;
  interval_level: 0.95;
  interval_kind: "empirical_residual_quantile";
  coverage_claim: "none";
}

const CSV_HEADERS = [
  "行业名称",
  "公司名称",
  "产品名称",
  "年份",
  "月份",
  "销售数量",
  "数量单位",
  "促销投入",
  "价格指数",
  "产能利用率",
] as const;

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function assertSafety() {
  if (!process.argv.includes("--force")) {
    throw new Error("Student dogfood seed is destructive and requires --force.");
  }

  const database = requireEnv("DB_DATABASE");
  const host = requireEnv("DB_HOST");
  requireEnv("DB_USER");

  if (!DATABASE_PREFIX || !database.startsWith(DATABASE_PREFIX)) {
    throw new Error(
      `Refusing to seed database "${database}". It must start with "${DATABASE_PREFIX}".`,
    );
  }
  if (!/^[a-zA-Z0-9_]+$/.test(database)) {
    throw new Error(`Unsafe database name: ${database}`);
  }
  if (!LOCAL_HOSTS.has(host.toLowerCase())) {
    throw new Error(`Refusing to seed non-local host: ${host}`);
  }

  return { database, host };
}

function buildStorageEnvironment(
  database: string,
  host: string,
): ManagedStorageEnvironment {
  const configuredPort = process.env.DB_PORT?.trim();
  return {
    DB_DATABASE: database,
    DB_HOST: host,
    ...(configuredPort ? { DB_PORT: Number(configuredPort) } : {}),
    E2E_ALLOW_DB_RESET: process.env.E2E_ALLOW_DB_RESET,
  };
}

async function removeDisposableNamespaceRoot(
  baseRoot: string,
  namespaceRoot: string,
): Promise<void> {
  if (path.resolve(baseRoot) === path.resolve(namespaceRoot)) {
    throw new Error("Refusing to treat an unscoped dogfood directory as disposable storage.");
  }

  const namespaceParent = path.dirname(namespaceRoot);
  for (const candidate of [baseRoot, namespaceParent]) {
    const stats = await fs.lstat(candidate);
    if (!stats.isDirectory() || stats.isSymbolicLink()) {
      throw new Error(`Refusing to use a symbolic-link or non-directory storage root: ${candidate}`);
    }
  }

  let namespaceStats: Awaited<ReturnType<typeof fs.lstat>>;
  try {
    namespaceStats = await fs.lstat(namespaceRoot);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  if (!namespaceStats.isDirectory() || namespaceStats.isSymbolicLink()) {
    throw new Error(`Refusing to clean a symbolic-link or non-directory namespace: ${namespaceRoot}`);
  }

  const [canonicalWorkspace, canonicalBase, canonicalNamespace] = await Promise.all([
    fs.realpath(process.cwd()),
    fs.realpath(baseRoot),
    fs.realpath(namespaceRoot),
  ]);
  const baseRelative = path.relative(canonicalWorkspace, canonicalBase);
  if (
    !baseRelative
    || baseRelative === ".."
    || baseRelative.startsWith(`..${path.sep}`)
    || path.isAbsolute(baseRelative)
  ) {
    throw new Error(`Refusing to clean a storage root outside the workspace: ${baseRoot}`);
  }
  const relative = path.relative(canonicalBase, canonicalNamespace);
  if (
    !relative
    || relative === ".."
    || relative.startsWith(`..${path.sep}`)
    || path.isAbsolute(relative)
  ) {
    throw new Error(`Refusing to clean a namespace outside its managed root: ${namespaceRoot}`);
  }
  await fs.rm(namespaceRoot, { recursive: true, force: true });
}

async function ensureDogfoodUploadsRoot(uploadsRoot: string): Promise<void> {
  await fs.mkdir(uploadsRoot, { recursive: true });
  const stats = await fs.lstat(uploadsRoot);
  if (!stats.isDirectory() || stats.isSymbolicLink()) {
    throw new Error(`Refusing to use a symbolic-link or non-directory uploads root: ${uploadsRoot}`);
  }
  const [canonicalWorkspace, canonicalUploads] = await Promise.all([
    fs.realpath(process.cwd()),
    fs.realpath(uploadsRoot),
  ]);
  const relative = path.relative(canonicalWorkspace, canonicalUploads);
  if (
    !relative
    || relative === ".."
    || relative.startsWith(`..${path.sep}`)
    || path.isAbsolute(relative)
  ) {
    throw new Error(`Refusing to use an uploads root outside the workspace: ${uploadsRoot}`);
  }
}

function csvEscape(value: string): string {
  return `"${value.replaceAll('"', '""')}"`;
}

function toCsv(rows: CsvRow[]): string {
  const header = CSV_HEADERS.map(csvEscape).join(",");
  const body = rows.map((row) =>
    CSV_HEADERS.map((column) => csvEscape(row[column] ?? "")).join(","),
  );
  return [header, ...body].join("\n");
}

function safeFileName(value: string): string {
  return value
    .replace(/[\/\\?%*:|"<>]/g, "-")
    .replace(/\s+/g, "-")
    .slice(0, 100);
}

function buildRows(input: {
  industry: string;
  company: string;
  product: string;
  months: number;
  startYear: number;
  sales: (index: number, month: number) => number;
  unit?: string;
}): CsvRow[] {
  const rows: CsvRow[] = [];
  for (let index = 0; index < input.months; index += 1) {
    const month = (index % 12) + 1;
    const year = input.startYear + Math.floor(index / 12);
    const promo = 18 + (index % 6) * 7 + (month === 6 || month === 11 ? 25 : 0);
    const price = 98 + index * 0.18 + Math.cos((2 * Math.PI * month) / 6) * 1.4;
    const capacity = 72 + Math.sin((2 * Math.PI * month) / 12) * 11;

    rows.push({
      行业名称: input.industry,
      公司名称: input.company,
      产品名称: input.product,
      年份: String(year),
      月份: String(month),
      销售数量: String(Math.max(0, Math.round(input.sales(index, month)))),
      数量单位: input.unit ?? "件",
      促销投入: promo.toFixed(2),
      价格指数: price.toFixed(2),
      产能利用率: capacity.toFixed(2),
    });
  }
  return rows;
}

function buildDatasetDefinitions() {
  const standardIndustry = STANDARD_SELECTION.industry;
  const standardCompany = STANDARD_SELECTION.company;
  const standardPrimary = buildRows({
    industry: standardIndustry,
    company: standardCompany,
    product: STANDARD_SELECTION.product,
    months: 48,
    startYear: 2022,
    sales: (index, month) =>
      620
      + index * 7.4
      + Math.sin((2 * Math.PI * month) / 12) * 72
      + (month === 6 || month === 11 ? 95 : 0),
  });
  const standardSecondary = buildRows({
    industry: standardIndustry,
    company: standardCompany,
    product: "平稳型执行器",
    months: 48,
    startYear: 2022,
    sales: (index, month) =>
      540
      + Math.sin((2 * Math.PI * month) / 12) * 18
      + ((index % 5) - 2) * 4,
  });

  const boundaryIndustry = "Dogfood边界行业：零销量与极端尖峰";
  const boundaryCompany = "边界压力测试企业";
  const zeroRows = buildRows({
    industry: boundaryIndustry,
    company: boundaryCompany,
    product: "连续零销量产品",
    months: 24,
    startYear: 2024,
    sales: () => 0,
  });
  const spikeRows = buildRows({
    industry: boundaryIndustry,
    company: boundaryCompany,
    product: "尖峰、断崖与恢复产品",
    months: 36,
    startYear: 2023,
    sales: (index) => {
      if (index === 12) return 25_000;
      if (index === 13 || index === 14) return 0;
      if (index === 25) return 48_000;
      return 160 + index * 3 + ((index % 4) - 2) * 22;
    },
  });

  const longIndustry =
    "Dogfood 国际化高端装备与超长行业名称测试专区（含中文、English、数字2026与符号）";
  const longCompany =
    "株式会社・示例企业（亚太）/ Global Operations & Direct Sales";
  const longProduct =
    "超长产品名称——工业级多变量智能预测控制终端 Pro Max 第三代 🧪";
  const longRows = buildRows({
    industry: longIndustry,
    company: longCompany,
    product: longProduct,
    months: 60,
    startYear: 2021,
    sales: (index, month) =>
      1_200
      + index * 11
      + Math.sin((2 * Math.PI * month) / 12) * 140
      + Math.cos((2 * Math.PI * month) / 6) * 65,
    unit: "台（套）",
  });

  return [
    {
      name: "Dogfood 标准季节趋势数据集",
      description: "48个月、两种产品、稳定多变量字段；用于完整模型与生产计划主路径。",
      products: [
        { industry: standardIndustry, company: standardCompany, product: STANDARD_SELECTION.product, rows: standardPrimary },
        { industry: standardIndustry, company: standardCompany, product: "平稳型执行器", rows: standardSecondary },
      ],
    },
    {
      name: "Dogfood 零销量与尖峰边界数据集",
      description: "包含连续零销量、极端尖峰、断崖和恢复；用于图表、统计、验证与错误文案。",
      products: [
        { industry: boundaryIndustry, company: boundaryCompany, product: "连续零销量产品", rows: zeroRows },
        { industry: boundaryIndustry, company: boundaryCompany, product: "尖峰、断崖与恢复产品", rows: spikeRows },
      ],
    },
    {
      name: "Dogfood 长名称与国际化多变量数据集",
      description: "60个月与超长 Unicode 标签；用于溢出、可访问性、选择卡片和多变量 LSTM。",
      products: [
        { industry: longIndustry, company: longCompany, product: longProduct, rows: longRows },
      ],
    },
  ];
}

function buildProductionSnapshot() {
  const predictions = [840, 900, 960, 880, 920, 990];
  const standardDeviations = [45, 50, 55, 48, 52, 58];
  const forecast: PredictionPoint[] = predictions.map((prediction, index) => {
    const spread = Math.round(standardDeviations[index]! * 1.65);
    return {
      prediction,
      std_dev: standardDeviations[index]!,
      upper_error_p99: Math.round(standardDeviations[index]! * 2.33),
      uncertainty_source: "empirical",
      uncertainty_reason: "Dogfood fixture：选择后校准残差",
      calibration_source: "post_selection_calibration",
      calibration_mean_error: 0,
      calibration_count: 12,
      bias_diagnostic_source: null,
      interval_lower: Math.max(0, prediction - spread),
      interval_upper: prediction + spread,
      interval_level: 0.95,
      interval_kind: "empirical_residual_quantile",
      coverage_claim: "none",
    };
  });

  const initialInventory = 0;
  const zScore = 1.65;
  const capacity = buildProductionCapacityScenarios(forecast, zScore).normal;
  const mps = buildCanonicalProductionMps({
    forecasts: forecast,
    initialInventory,
    zScore,
    capacity,
  });

  return { forecast, mps, capacity };
}

async function insertExperiment(
  conn: mysql.Connection,
  studentId: number,
  values: Record<string, unknown>,
): Promise<number> {
  const columns = ["student_id", ...Object.keys(values)];
  const placeholders = columns.map(() => "?").join(", ");
  const [result] = await conn.query<ResultSetHeader>(
    `INSERT INTO experiment_status (${columns.map((column) => `\`${column}\``).join(", ")})
     VALUES (${placeholders})`,
    [studentId, ...Object.values(values)],
  );
  await conn.query(
    "INSERT INTO experiment_grade (experiment_id, student_id) VALUES (?, ?)",
    [result.insertId, studentId],
  );
  return result.insertId;
}

function completedExperimentValues() {
  const production = buildProductionSnapshot();
  return {
    status: "Completed",
    current_step: 7,
    highest_completed_step: 7,
    state_version: 12,
    total_active_duration_seconds: 7_845,
    selected_industry: STANDARD_SELECTION.industry,
    selected_company: STANDARD_SELECTION.company,
    selected_product: STANDARD_SELECTION.product,
    selected_base_models: JSON.stringify(["ma", "exp"]),
    selected_ensemble_models: JSON.stringify(["weighted_avg"]),
    data_window_train_start_index: 0,
    data_window_train_end_index: 35,
    data_window_evaluate_start_index: 36,
    data_window_evaluate_end_index: 47,
    moving_average_completed: 1,
    moving_average_window: 6,
    moving_average_metrics_rmse: 48.2,
    moving_average_metrics_mae: 39.1,
    moving_average_metrics_r2: 0.78,
    exponential_smoothing_completed: 1,
    exponential_smoothing_alpha: 0.3,
    exponential_smoothing_metrics_rmse: 44.8,
    exponential_smoothing_metrics_mae: 36.7,
    exponential_smoothing_metrics_r2: 0.81,
    ensemble_weighted_completed: 1,
    ensemble_weighted_base_models: JSON.stringify(["ma", "exp"]),
    ensemble_weighted_metrics_rmse: 39.5,
    ensemble_weighted_metrics_mae: 31.4,
    ensemble_weighted_metrics_r2: 0.86,
    selected_best_model: "ensemble_weighted",
    quiz_about_model_completed: 1,
    quiz_about_plan_completed: 1,
    production_plan_completed: 1,
    production_forecast_periods: production.forecast.length,
    production_initial_inventory: 0,
    production_target_service_level: 0.95,
    production_safety_stock_z_score: 1.65,
    production_forecast_results: JSON.stringify(production.forecast),
    production_mps_table: JSON.stringify(production.mps),
    production_capacity_mode: "scenario",
    production_capacity_scenario: "normal",
    production_capacity: production.capacity,
    production_custom_capacity: null,
    start_time: new Date("2026-07-20T01:00:00.000Z"),
    last_activity_at: new Date("2026-07-20T03:10:45.000Z"),
    completion_time: new Date("2026-07-20T03:10:45.000Z"),
  };
}

async function seedCompletedRecoveryEvidence(
  conn: mysql.Connection,
  experimentId: number,
  studentId: number,
) {
  const production = buildProductionSnapshot();
  const runId = crypto.randomUUID();
  const productionRevision = crypto.randomUUID();
  const artifactPath = path.resolve(
    process.cwd(),
    "py",
    "trained_models",
    `exp${experimentId}-student${studentId}-dogfood-weighted.pkl`,
  );

  await conn.query(
    `INSERT INTO model_training_runs (
      run_id,
      experiment_id,
      student_id,
      model_type,
      request_hash,
      dataset_revision,
      runtime_revision,
      parameters,
      metrics,
      artifact_path,
      production_artifact_forecast_steps,
      production_revision,
      is_active
    ) VALUES (?, ?, ?, 'weighted_avg', ?, ?, ?, ?, ?, ?, ?, ?, TRUE)`,
    [
      runId,
      experimentId,
      studentId,
      "0".repeat(64),
      "1".repeat(64),
      "2".repeat(64),
      JSON.stringify({ fixture: "student-dogfood-completed-recovery" }),
      JSON.stringify({ rmse: 39.5, mae: 31.4, r2: 0.86 }),
      artifactPath,
      production.forecast.length,
      productionRevision,
    ],
  );

  const evidence = issueProductionForecastEvidence({
    experiment_id: experimentId,
    student_id: studentId,
    model_type: "weighted_avg",
    run_id: runId,
    production_revision: productionRevision,
    forecast_steps: production.forecast.length,
    forecasts: production.forecast,
  });
  await conn.query(
    `UPDATE experiment_status
        SET production_forecast_evidence = ?
      WHERE experiment_id = ? AND student_id = ?`,
    [evidence, experimentId, studentId],
  );
}

async function seedScenarioExperiments(
  conn: mysql.Connection,
  users: Map<string, number>,
  reportRoot: string,
  runId: string,
) {
  const id = (username: string) => {
    const value = users.get(username);
    if (!value) throw new Error(`Missing scenario user: ${username}`);
    return value;
  };

  await insertExperiment(conn, id("20990002"), {
    status: "In Progress",
    current_step: 3,
    highest_completed_step: 2,
    state_version: 3,
    selected_industry: STANDARD_SELECTION.industry,
    selected_company: STANDARD_SELECTION.company,
    start_time: new Date("2026-07-29T02:00:00.000Z"),
    last_activity_at: new Date("2026-07-29T02:18:00.000Z"),
  });

  await insertExperiment(conn, id("20990003"), {
    status: "In Progress",
    current_step: 5,
    highest_completed_step: 4,
    state_version: 7,
    selected_industry: STANDARD_SELECTION.industry,
    selected_company: STANDARD_SELECTION.company,
    selected_product: STANDARD_SELECTION.product,
    selected_base_models: JSON.stringify(["ma", "exp", "arima", "lstm"]),
    data_window_train_start_index: 0,
    data_window_train_end_index: 35,
    data_window_evaluate_start_index: 36,
    data_window_evaluate_end_index: 47,
    start_time: new Date("2026-07-29T03:00:00.000Z"),
    last_activity_at: new Date("2026-07-29T03:42:00.000Z"),
  });

  const submittedExperiment = await insertExperiment(
    conn,
    id("20990004"),
    completedExperimentValues(),
  );
  const rejectedExperiment = await insertExperiment(
    conn,
    id("20990005"),
    completedExperimentValues(),
  );

  await insertExperiment(conn, id("20990006"), {
    status: "In Progress",
    current_step: 7,
    highest_completed_step: 6,
    state_version: 9,
    selected_industry: STANDARD_SELECTION.industry,
    selected_company: STANDARD_SELECTION.company,
    selected_product: STANDARD_SELECTION.product,
    selected_base_models: JSON.stringify(["ma"]),
    data_window_train_start_index: 0,
    data_window_train_end_index: 35,
    data_window_evaluate_start_index: 36,
    data_window_evaluate_end_index: 47,
    moving_average_completed: 1,
    moving_average_window: 6,
    moving_average_metrics_rmse: 48.2,
    moving_average_metrics_mae: 39.1,
    moving_average_metrics_r2: 0.78,
    selected_best_model: "ma",
    quiz_about_model_completed: 1,
    production_plan_completed: 0,
    production_forecast_periods: 6,
    production_forecast_results: JSON.stringify([{ prediction: 840 }]),
    start_time: new Date("2026-07-29T04:00:00.000Z"),
    last_activity_at: new Date("2026-07-29T04:58:00.000Z"),
  });

  await insertExperiment(conn, id("20990007"), {
    status: "In Progress",
    current_step: 1,
    highest_completed_step: 0,
    state_version: 1,
    start_time: new Date("2026-07-29T05:00:00.000Z"),
    last_activity_at: new Date("2026-07-29T05:01:00.000Z"),
  });

  const completedWithoutReportExperiment = await insertExperiment(
    conn,
    id("20990008"),
    completedExperimentValues(),
  );
  await seedCompletedRecoveryEvidence(
    conn,
    completedWithoutReportExperiment,
    id("20990008"),
  );

  const gradedExperiment = await insertExperiment(
    conn,
    id("20990009"),
    completedExperimentValues(),
  );

  const reportFixtures = [
    {
      username: "20990004",
      experimentId: submittedExperiment,
      status: "submitted",
      grade: null,
      feedback: null,
      fileName: `student-dogfood-${runId}-submitted.pdf`,
    },
    {
      username: "20990005",
      experimentId: rejectedExperiment,
      status: "rejected",
      grade: null,
      feedback:
        "请补充模型选择依据，并解释安全库存与产能约束的权衡。驳回说明包含较长文本，用于检查换行、可读性和重新提交入口。",
      fileName: `student-dogfood-${runId}-rejected.pdf`,
    },
    {
      username: "20990009",
      experimentId: gradedExperiment,
      status: "graded",
      grade: 92,
      feedback:
        "模型比较完整，生产计划逻辑清楚。建议进一步讨论尖峰需求下的风险与适用边界。",
      fileName: `student-dogfood-${runId}-graded.pdf`,
    },
  ] as const;

  for (const fixture of reportFixtures) {
    const pdfPath = path.resolve(reportRoot, fixture.fileName);
    await fs.writeFile(
      pdfPath,
      buildStudentDogfoodReportPdf({
        runId,
        username: fixture.username,
        experimentId: fixture.experimentId,
        status: fixture.status,
        grade: fixture.grade,
      }),
      "utf8",
    );
    await conn.query(
      `INSERT INTO experiment_reports
       (experiment_id, student_id, report_content, pdf_file_path, status, submitted_at, grade, feedback)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        fixture.experimentId,
        id(fixture.username),
        `<h1>Dogfood 实验报告</h1><p>场景账号：${fixture.username}</p>`,
        pdfPath,
        fixture.status,
        new Date("2026-07-20T03:10:45.000Z"),
        fixture.grade,
        fixture.feedback,
      ],
    );
  }
}

async function seedDatasets(
  conn: mysql.Connection,
  datasetRoot: string,
) {
  await conn.query("DELETE FROM datasets");
  const manifest: Array<Record<string, unknown>> = [];

  for (const definition of buildDatasetDefinitions()) {
    const [result] = await conn.query<ResultSetHeader>(
      "INSERT INTO datasets (data_name, file_path, description) VALUES (?, ?, ?)",
      [definition.name, "pending", definition.description],
    );
    const datasetId = result.insertId;
    const currentRoot = path.resolve(datasetRoot, String(datasetId));
    const segmentsRoot = path.resolve(currentRoot, "segments");
    await fs.mkdir(segmentsRoot, { recursive: true });

    const allRows = definition.products.flatMap((product) => product.rows);
    const fullPath = path.resolve(currentRoot, "full.csv");
    await fs.writeFile(fullPath, toCsv(allRows), "utf8");
    await conn.query(
      "UPDATE datasets SET file_path = ? WHERE dataset_id = ?",
      [fullPath, datasetId],
    );

    const segments: Array<Record<string, unknown>> = [];
    for (const product of definition.products) {
      const segmentPath = path.resolve(
        segmentsRoot,
        `${safeFileName(product.company)}_${safeFileName(product.product)}.csv`,
      );
      await fs.writeFile(segmentPath, toCsv(product.rows), "utf8");
      await conn.query(
        `INSERT INTO dataset_segments
         (dataset_id, industry_name, company_name, product_name, file_path)
         VALUES (?, ?, ?, ?, ?)`,
        [
          datasetId,
          product.industry,
          product.company,
          product.product,
          segmentPath,
        ],
      );
      segments.push({
        industry: product.industry,
        company: product.company,
        product: product.product,
        rows: product.rows.length,
        filePath: segmentPath,
      });
    }

    manifest.push({
      datasetId,
      name: definition.name,
      description: definition.description,
      rows: allRows.length,
      filePath: fullPath,
      segments,
    });
  }

  return manifest;
}

async function main() {
  const { database, host } = assertSafety();
  const user = requireEnv("DB_USER");
  const password = process.env.DB_PASSWORD ?? "";
  const studentPassword = requireEnv("DOGFOOD_STUDENT_PASSWORD");
  const runId =
    process.env.DOGFOOD_RUN_ID?.trim()
    || database.slice(DATABASE_PREFIX.length);
  if (!/^[a-zA-Z0-9_-]+$/.test(runId)) {
    throw new Error(`DOGFOOD_RUN_ID contains unsafe characters: ${runId}`);
  }
  const legacyArtifactRoot = path.resolve(
    process.cwd(),
    "uploads",
    "dogfood",
    database,
  );
  const storageEnvironment = buildStorageEnvironment(database, host);
  const uploadsRoot = path.resolve(process.cwd(), "uploads");
  const datasetsBaseRoot = path.resolve(uploadsRoot, "datasets");
  const reportsBaseRoot = path.resolve(uploadsRoot, "reports");
  const scopedStorage = isDisposableStorageNamespaceEnabled(storageEnvironment);
  const datasetRoot = scopedStorage
    ? getDatabaseManagedStorageRoot(datasetsBaseRoot, storageEnvironment)
    : path.resolve(legacyArtifactRoot, "datasets");
  // No-opt-in invocations preserve their legacy paths byte-for-byte. Only an
  // explicitly disposable database uses the shared upload roots' DB namespace.
  const artifactRoot = scopedStorage ? datasetRoot : legacyArtifactRoot;
  const reportRoot = getDatabaseManagedStorageRoot(reportsBaseRoot, storageEnvironment);
  const outputRoot = process.env.DOGFOOD_OUTPUT_ROOT
    ? path.resolve(process.env.DOGFOOD_OUTPUT_ROOT)
    : path.resolve(process.cwd(), "..", "dogfood-output", `student-${runId}`);
  const manifestPath = process.env.DOGFOOD_MANIFEST_PATH
    ? path.resolve(process.env.DOGFOOD_MANIFEST_PATH)
    : path.resolve(outputRoot, "database-manifest.json");

  await ensureDogfoodUploadsRoot(uploadsRoot);
  const ensuredReportRoot = await ensureDatabaseManagedStorageRoot(
    reportsBaseRoot,
    storageEnvironment,
  );
  if (path.resolve(ensuredReportRoot) !== path.resolve(reportRoot)) {
    throw new Error("Report storage namespace changed during dogfood initialization");
  }
  if (scopedStorage) {
    const ensuredDatasetRoot = await ensureDatabaseManagedStorageRoot(
      datasetsBaseRoot,
      storageEnvironment,
    );
    if (path.resolve(ensuredDatasetRoot) !== path.resolve(datasetRoot)) {
      throw new Error("Dataset storage namespace changed during dogfood initialization");
    }
    await removeDisposableNamespaceRoot(datasetsBaseRoot, datasetRoot);
    await removeDisposableNamespaceRoot(reportsBaseRoot, reportRoot);
    await ensureDatabaseManagedStorageRoot(datasetsBaseRoot, storageEnvironment);
    await ensureDatabaseManagedStorageRoot(reportsBaseRoot, storageEnvironment);
  } else {
    const legacyBaseRoot = path.resolve(uploadsRoot, "dogfood");
    const ensuredLegacyBaseRoot = await ensureDatabaseManagedStorageRoot(
      legacyBaseRoot,
      storageEnvironment,
    );
    if (path.resolve(ensuredLegacyBaseRoot) !== legacyBaseRoot) {
      throw new Error("Legacy dogfood storage unexpectedly became database-scoped");
    }
    await removeDisposableNamespaceRoot(legacyBaseRoot, artifactRoot);
    await fs.mkdir(artifactRoot);
    await fs.mkdir(datasetRoot);
  }
  await fs.mkdir(path.dirname(manifestPath), { recursive: true });

  const conn = await mysql.createConnection({
    host,
    user,
    password,
    database,
    charset: "utf8mb4",
  });

  try {
    await conn.beginTransaction();

    const usernames = STUDENT_SCENARIOS.map((scenario) => scenario.username);
    const placeholders = usernames.map(() => "?").join(", ");
    await conn.query(
      `DELETE FROM users WHERE username IN (${placeholders})`,
      usernames,
    );

    const [classRows] = await conn.query<ClassRow[]>(
      `SELECT c.class_id
         FROM classes c
         JOIN academic_terms term ON term.term_id = c.term_id
        ORDER BY term.is_active DESC, c.class_id ASC
        LIMIT 1`,
    );
    const classId = classRows[0]?.class_id;
    if (!classId) throw new Error("No class is available for dogfood students.");

    const passwordHash = await Bun.password.hash(studentPassword, {
      algorithm: "bcrypt",
      cost: 10,
    });
    const userIds = new Map<string, number>();

    for (const scenario of STUDENT_SCENARIOS) {
      const [result] = await conn.query<ResultSetHeader>(
        `INSERT INTO users
         (username, password_hash, full_name, email, phone_number, role)
         VALUES (?, ?, ?, ?, ?, 'Student')`,
        [
          scenario.username,
          passwordHash,
          scenario.fullName,
          `${scenario.username}@student-dogfood.local`,
          `188${scenario.username.slice(-8)}`,
        ],
      );
      userIds.set(scenario.username, result.insertId);
      await conn.query(
        "INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)",
        [result.insertId, classId],
      );
    }

    const datasets = await seedDatasets(conn, datasetRoot);
    await seedScenarioExperiments(conn, userIds, reportRoot, runId);
    await conn.commit();

    const [users] = await conn.query<UserRow[]>(
      `SELECT user_id, username
         FROM users
        WHERE username IN (${placeholders})
        ORDER BY username`,
      usernames,
    );

    const manifest = {
      generatedAt: new Date().toISOString(),
      database,
      host,
      runId,
      classId,
      standardSelection: STANDARD_SELECTION,
      scenarios: STUDENT_SCENARIOS.map((scenario) => ({
        ...scenario,
        userId: users.find((row) => row.username === scenario.username)?.user_id,
      })),
      datasets,
      artifactRoot,
      reportRoot,
    };
    await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2), "utf8");

    console.log("Student dogfood fixtures seeded.");
    console.log(`  database: ${database}`);
    console.log(`  students: ${STUDENT_SCENARIOS.length}`);
    console.log(`  datasets: ${datasets.length}`);
    console.log(`  manifest: ${manifestPath}`);
  } catch (error) {
    await conn.rollback().catch(() => {});
    throw error;
  } finally {
    await conn.end();
  }
}

await main();
