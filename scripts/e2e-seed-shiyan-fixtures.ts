import fs from "node:fs/promises";
import path from "node:path";
import mysql from "mysql2/promise";
import type { ResultSetHeader, RowDataPacket } from "mysql2/promise";

type CsvRow = Record<string, string>;

interface ProductConfig {
  name: string;
  base: number;
  trend: number;
  seasonAmp: number;
  promoBase: number;
  priceBase: number;
  capacityBase: number;
}

interface StudentRow extends RowDataPacket {
  user_id: number;
}

interface DatasetCandidateRow extends RowDataPacket {
  dataset_id: number;
  file_path: string | null;
}

interface ReportFileRow extends RowDataPacket {
  pdf_file_path: string | null;
}

const REQUIRED_ENV_KEYS = ["DB_HOST", "DB_USER", "DB_DATABASE"] as const;
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

const DEFAULTS = {
  studentUsername: "20240002",
  datasetName: "E2E-Shiyan-全链路真实训练专用数据集",
  datasetDescription:
    "用于 shiyan 全链路真实训练 e2e；包含两类产品、36个月连续月度样本与可选外生特征。",
  industry: "E2E智能制造业",
  company: "E2E样本企业A",
  unit: "件",
  startYear: 2022,
  monthCount: 36,
  primaryProduct: "智能传感器A型",
  secondaryProduct: "智能传感器B型",
  datasetFileLabel: "e2e-shiyan-fullchain.csv",
} as const;

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing required env var: ${name}`);
  return value;
}

function assertBaseEnv() {
  for (const key of REQUIRED_ENV_KEYS) requireEnv(key);
}

function csvEscape(value: string): string {
  return `"${value.replace(/"/g, '""')}"`;
}

function toCsv(rows: CsvRow[]): string {
  const headerLine = CSV_HEADERS.map((h) => csvEscape(h)).join(",");
  const body = rows.map((row) =>
    CSV_HEADERS.map((header) => csvEscape(row[header] ?? "")).join(","),
  );
  return [headerLine, ...body].join("\n");
}

function safeFileName(value: string): string {
  return value.replace(/[\/\\?%*:|"<>]/g, "-");
}

function buildProductRows(
  industry: string,
  company: string,
  unit: string,
  startYear: number,
  monthCount: number,
  config: ProductConfig,
): CsvRow[] {
  const rows: CsvRow[] = [];

  for (let idx = 0; idx < monthCount; idx += 1) {
    const year = startYear + Math.floor(idx / 12);
    const month = (idx % 12) + 1;
    const season = Math.sin((2 * Math.PI * month) / 12);
    const cycle = Math.cos((2 * Math.PI * month) / 6);
    const promoMonthBoost = month === 3 || month === 6 || month === 11 ? 12 : 0;

    const promoSpend = config.promoBase + (idx % 6) * 4 + promoMonthBoost;
    const priceIndex = config.priceBase + idx * 0.2 + cycle * 0.6;
    const capacityUtil =
      config.capacityBase + season * 6 + ((idx % 3) - 1) * 1.2;

    const sales = Math.round(
      config.base +
        idx * config.trend +
        season * config.seasonAmp +
        promoSpend * 2.1 -
        priceIndex * 1.4 +
        capacityUtil * 2.5,
    );

    rows.push({
      行业名称: industry,
      公司名称: company,
      产品名称: config.name,
      年份: String(year),
      月份: String(month),
      销售数量: String(Math.max(100, sales)),
      数量单位: unit,
      促销投入: promoSpend.toFixed(2),
      价格指数: priceIndex.toFixed(2),
      产能利用率: capacityUtil.toFixed(2),
    });
  }

  return rows;
}

async function cleanupStudentModelFiles(studentId: number) {
  const trainedModelsDir = path.resolve(process.cwd(), "py", "trained_models");
  try {
    const files = await fs.readdir(trainedModelsDir);
    const targets = files.filter((file) => file.startsWith(`${studentId}_`));
    await Promise.all(
      targets.map((file) =>
        fs.rm(path.resolve(trainedModelsDir, file), { force: true }),
      ),
    );
  } catch {
    // Ignore if directory does not exist in current environment.
  }
}

async function rmMany(paths: Array<string | null | undefined>) {
  await Promise.all(
    paths
      .filter((p): p is string => Boolean(p))
      .map((p) => fs.rm(p, { recursive: true, force: true })),
  );
}

async function main() {
  assertBaseEnv();

  const studentUsername =
    process.env.E2E_SHIYAN_STUDENT_USERNAME ?? DEFAULTS.studentUsername;
  const datasetName =
    process.env.E2E_SHIYAN_DATASET_NAME ?? DEFAULTS.datasetName;
  const datasetDescription =
    process.env.E2E_SHIYAN_DATASET_DESCRIPTION ?? DEFAULTS.datasetDescription;
  const industry = process.env.E2E_SHIYAN_INDUSTRY ?? DEFAULTS.industry;
  const company = process.env.E2E_SHIYAN_COMPANY ?? DEFAULTS.company;
  const unit = process.env.E2E_SHIYAN_UNIT ?? DEFAULTS.unit;
  const startYear = Number(process.env.E2E_SHIYAN_START_YEAR ?? DEFAULTS.startYear);
  const monthCount = Number(
    process.env.E2E_SHIYAN_MONTH_COUNT ?? DEFAULTS.monthCount,
  );
  const primaryProduct =
    process.env.E2E_SHIYAN_PRODUCT_PRIMARY ?? DEFAULTS.primaryProduct;
  const secondaryProduct =
    process.env.E2E_SHIYAN_PRODUCT_SECONDARY ?? DEFAULTS.secondaryProduct;

  if (!Number.isInteger(startYear) || startYear < 2000 || startYear > 2100) {
    throw new Error(`Invalid start year: ${startYear}`);
  }
  if (!Number.isInteger(monthCount) || monthCount < 24) {
    throw new Error(`Invalid month count: ${monthCount}. Must be >= 24.`);
  }
  if (primaryProduct === secondaryProduct) {
    throw new Error("Primary and secondary product names must be different.");
  }

  const dbHost = requireEnv("DB_HOST");
  const dbUser = requireEnv("DB_USER");
  const dbPassword = process.env.DB_PASSWORD ?? "";
  const dbDatabase = requireEnv("DB_DATABASE");

  const conn = await mysql.createConnection({
    host: dbHost,
    user: dbUser,
    password: dbPassword,
    database: dbDatabase,
    charset: "utf8mb4",
  });

  let newDatasetFilePath: string | null = null;
  let newSegmentsDir: string | null = null;
  let oldDatasetIds: number[] = [];
  let oldDatasetFiles: string[] = [];
  let oldReportPaths: string[] = [];

  try {
    await conn.beginTransaction();

    const [studentRows] = await conn.query<StudentRow[]>(
      "SELECT user_id FROM users WHERE username = ? LIMIT 1",
      [studentUsername],
    );
    const studentId = studentRows[0]?.user_id;
    if (!studentId) {
      throw new Error(`Cannot find student user: ${studentUsername}`);
    }

    const [reportRows] = await conn.query<ReportFileRow[]>(
      "SELECT pdf_file_path FROM experiment_reports WHERE student_id = ?",
      [studentId],
    );
    oldReportPaths = reportRows
      .map((row) => row.pdf_file_path)
      .filter((p): p is string => Boolean(p));

    await conn.query("DELETE FROM quiz_answers WHERE student_id = ?", [studentId]);
    await conn.query("DELETE FROM experiment_reports WHERE student_id = ?", [studentId]);
    await conn.query("DELETE FROM experiment_grade WHERE student_id = ?", [studentId]);
    await conn.query("DELETE FROM experiment_status WHERE student_id = ?", [studentId]);

    const [existingDatasetRows] = await conn.query<DatasetCandidateRow[]>(
      `SELECT DISTINCT d.dataset_id, d.file_path
         FROM datasets d
         JOIN dataset_segments ds ON ds.dataset_id = d.dataset_id
        WHERE ds.industry_name = ?
          AND ds.company_name = ?
          AND ds.product_name IN (?, ?)`,
      [industry, company, primaryProduct, secondaryProduct],
    );

    oldDatasetIds = existingDatasetRows.map((row) => row.dataset_id);
    oldDatasetFiles = existingDatasetRows
      .map((row) => row.file_path)
      .filter((p): p is string => Boolean(p));

    if (oldDatasetIds.length > 0) {
      const placeholders = oldDatasetIds.map(() => "?").join(", ");
      await conn.query(
        `DELETE FROM datasets WHERE dataset_id IN (${placeholders})`,
        oldDatasetIds,
      );
    }

    const [insertResult] = await conn.query<ResultSetHeader>(
      "INSERT INTO datasets (data_name, file_path, description) VALUES (?, ?, ?)",
      [datasetName, "placeholder", datasetDescription],
    );
    const datasetId = insertResult.insertId;

    const uploadsRoot = path.resolve(process.cwd(), "uploads");
    const datasetsRoot = path.resolve(uploadsRoot, "datasets");
    newDatasetFilePath = path.resolve(
      datasetsRoot,
      `${datasetId}-${DEFAULTS.datasetFileLabel}`,
    );
    newSegmentsDir = path.resolve(datasetsRoot, String(datasetId), "segments");

    await fs.mkdir(datasetsRoot, { recursive: true });
    await fs.mkdir(newSegmentsDir, { recursive: true });

    const primaryRows = buildProductRows(industry, company, unit, startYear, monthCount, {
      name: primaryProduct,
      base: 660,
      trend: 7.2,
      seasonAmp: 48,
      promoBase: 18,
      priceBase: 98,
      capacityBase: 76,
    });
    const secondaryRows = buildProductRows(
      industry,
      company,
      unit,
      startYear,
      monthCount,
      {
        name: secondaryProduct,
        base: 520,
        trend: 5.4,
        seasonAmp: 38,
        promoBase: 14,
        priceBase: 102,
        capacityBase: 72,
      },
    );

    await fs.writeFile(
      newDatasetFilePath,
      toCsv([...primaryRows, ...secondaryRows]),
      "utf8",
    );

    const primarySegmentPath = path.resolve(
      newSegmentsDir,
      `${safeFileName(company)}_${safeFileName(primaryProduct)}.csv`,
    );
    const secondarySegmentPath = path.resolve(
      newSegmentsDir,
      `${safeFileName(company)}_${safeFileName(secondaryProduct)}.csv`,
    );

    await fs.writeFile(primarySegmentPath, toCsv(primaryRows), "utf8");
    await fs.writeFile(secondarySegmentPath, toCsv(secondaryRows), "utf8");

    await conn.query("UPDATE datasets SET file_path = ? WHERE dataset_id = ?", [
      newDatasetFilePath,
      datasetId,
    ]);

    await conn.query(
      `INSERT INTO dataset_segments
      (dataset_id, industry_name, company_name, product_name, file_path)
      VALUES
      (?, ?, ?, ?, ?),
      (?, ?, ?, ?, ?)`,
      [
        datasetId,
        industry,
        company,
        primaryProduct,
        primarySegmentPath,
        datasetId,
        industry,
        company,
        secondaryProduct,
        secondarySegmentPath,
      ],
    );

    await conn.commit();

    const oldDatasetDirs = oldDatasetIds.map((id) =>
      path.resolve(process.cwd(), "uploads", "datasets", String(id)),
    );
    await rmMany([...oldDatasetFiles, ...oldReportPaths, ...oldDatasetDirs]);
    await cleanupStudentModelFiles(studentId);

    console.log("Seeded shiyan e2e dataset fixture.");
    console.log(`  student: ${studentUsername} (id=${studentId})`);
    console.log(`  dataset: ${datasetName} (id=${datasetId})`);
    console.log(`  industry/company: ${industry} / ${company}`);
    console.log(`  products: ${primaryProduct}, ${secondaryProduct}`);
    console.log(`  months each: ${monthCount}`);
  } catch (error) {
    await conn.rollback().catch(() => {});
    await rmMany([newDatasetFilePath, newSegmentsDir]);
    throw error;
  } finally {
    await conn.end();
  }
}

await main();
