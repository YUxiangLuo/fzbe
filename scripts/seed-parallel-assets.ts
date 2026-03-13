import path from "path";
import { mkdir } from "node:fs/promises";
import type { ResultSetHeader, RowDataPacket } from "mysql2/promise";
import {
  assertParallelSeedSafety,
  PARALLEL_DATASET,
  PARALLEL_USER_IDS,
  createDbConnection,
  getDatasetsRoot,
  getManualsRoot,
  rmMany,
  safeFileName,
} from "./parallel-seed.shared";

type CsvRow = Record<string, string>;

interface DatasetCandidateRow extends RowDataPacket {
  dataset_id: number;
  file_path: string | null;
}

interface SegmentFileRow extends RowDataPacket {
  file_path: string | null;
}

interface ManualCandidateRow extends RowDataPacket {
  manual_id: number;
  file_path: string | null;
}

interface ProductConfig {
  name: string;
  baseSales: number;
  trendSlope: number;
  yearlySeasonality: number;
  promoBase: number;
  priceBase: number;
  capacityBase: number;
  inventoryBase: number;
  searchBase: number;
  competitorPriceBase: number;
  holidayScale: number;
  channelSequence: readonly string[];
  regionSequence: readonly string[];
  weatherSequence: readonly string[];
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
  "库存水平",
  "搜索热度",
  "竞争对手价格",
  "节假日强度",
  "渠道类型",
  "区域",
  "天气类型",
] as const;

function csvEscape(value: string): string {
  return `"${value.replace(/"/g, '""')}"`;
}

function toCsv(rows: CsvRow[]): string {
  const headerLine = CSV_HEADERS.map((header) => csvEscape(header)).join(",");
  const bodyLines = rows.map((row) =>
    CSV_HEADERS.map((header) => csvEscape(row[header] ?? "")).join(","),
  );
  return [headerLine, ...bodyLines].join("\n");
}

function buildProductRows(config: ProductConfig): CsvRow[] {
  const rows: CsvRow[] = [];

  for (let index = 0; index < PARALLEL_DATASET.monthCount; index += 1) {
    const year = PARALLEL_DATASET.startYear + Math.floor(index / 12);
    const month = (index % 12) + 1;
    const yearlyWave = Math.sin((2 * Math.PI * month) / 12);
    const halfYearWave = Math.cos((2 * Math.PI * month) / 6);
    const promoPulse = month === 3 || month === 6 || month === 9 || month === 11 ? 1 : 0;
    const holidayStrength = month === 1 || month === 2 || month === 10 ? config.holidayScale : config.holidayScale / 3;

    const promoSpend = config.promoBase + promoPulse * 12 + (index % 4) * 2;
    const priceIndex = config.priceBase + index * 0.12 + halfYearWave * 1.3;
    const capacityUtilization = config.capacityBase + yearlyWave * 5 + ((index % 3) - 1) * 1.1;
    const inventoryLevel = config.inventoryBase + halfYearWave * 18 - promoSpend * 0.6;
    const searchHeat = config.searchBase + yearlyWave * 10 + promoPulse * 14;
    const competitorPrice = config.competitorPriceBase + halfYearWave * 1.8 + (index % 5) * 0.25;
    const channel = config.channelSequence[index % config.channelSequence.length] ?? "直营";
    const region = config.regionSequence[index % config.regionSequence.length] ?? "华东";
    const weather = config.weatherSequence[index % config.weatherSequence.length] ?? "晴朗";

    const channelFactor = channel === "直营" ? 18 : channel === "电商" ? 24 : 10;
    const regionFactor = region === "华东" ? 20 : region === "华南" ? 16 : 11;
    const weatherFactor = weather === "晴朗" ? 8 : weather === "多云" ? 3 : -5;

    const rawSales = Math.round(
      config.baseSales
      + index * config.trendSlope
      + yearlyWave * config.yearlySeasonality
      + promoSpend * 2.2
      - priceIndex * 1.5
      + capacityUtilization * 2.9
      - inventoryLevel * 0.45
      + searchHeat * 1.1
      - competitorPrice * 0.8
      + holidayStrength * 1.6
      + channelFactor
      + regionFactor
      + weatherFactor,
    );

    rows.push({
      行业名称: PARALLEL_DATASET.industry,
      公司名称: PARALLEL_DATASET.company,
      产品名称: config.name,
      年份: String(year),
      月份: String(month),
      销售数量: String(Math.max(120, rawSales)),
      数量单位: PARALLEL_DATASET.unit,
      促销投入: promoSpend.toFixed(2),
      价格指数: priceIndex.toFixed(2),
      产能利用率: capacityUtilization.toFixed(2),
      库存水平: inventoryLevel.toFixed(2),
      搜索热度: searchHeat.toFixed(2),
      竞争对手价格: competitorPrice.toFixed(2),
      节假日强度: holidayStrength.toFixed(2),
      渠道类型: channel,
      区域: region,
      天气类型: weather,
    });
  }

  return rows;
}

function escapePdfText(value: string): string {
  return value
    .replace(/[^\x20-\x7E]/g, "?")
    .replace(/\\/g, "\\\\")
    .replace(/\(/g, "\\(")
    .replace(/\)/g, "\\)");
}

function buildMinimalPdf(lines: string[]): string {
  const content = lines
    .map((line, index) => `BT /F1 12 Tf 50 ${780 - index * 20} Td (${escapePdfText(line)}) Tj ET`)
    .join("\n");
  const contentLength = Buffer.byteLength(content, "utf8");

  const objects = [
    "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj",
    "2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj",
    "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj",
    "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj",
    `5 0 obj\n<< /Length ${contentLength} >>\nstream\n${content}\nendstream\nendobj`,
  ];

  let pdf = "%PDF-1.4\n";
  const offsets = [0];

  for (const object of objects) {
    offsets.push(Buffer.byteLength(pdf, "utf8"));
    pdf += `${object}\n`;
  }

  const xrefOffset = Buffer.byteLength(pdf, "utf8");
  pdf += `xref\n0 ${objects.length + 1}\n`;
  pdf += "0000000000 65535 f \n";
  for (let index = 1; index < offsets.length; index += 1) {
    pdf += `${String(offsets[index]).padStart(10, "0")} 00000 n \n`;
  }
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;

  return pdf;
}

async function main() {
  assertParallelSeedSafety({
    operationName: "Parallel asset seed",
    commandExample: "bun run seed:parallel:assets --force",
  });
  const connection = await createDbConnection();

  let createdManualPath: string | null = null;
  let createdDatasetPath: string | null = null;
  let createdSegmentsDir: string | null = null;

  try {
    await mkdir(getManualsRoot(), { recursive: true });
    await mkdir(getDatasetsRoot(), { recursive: true });

    await connection.beginTransaction();

    const [manualRows] = await connection.query<ManualCandidateRow[]>(
      `SELECT manual_id, file_path
         FROM experiment_manuals
        WHERE file_name = ?`,
      [PARALLEL_DATASET.manualDisplayName],
    );
    const oldManualPaths = manualRows.map((row) => row.file_path);
    await connection.query("UPDATE experiment_manuals SET is_active = false");
    if (manualRows.length > 0) {
      await connection.query(
        `DELETE FROM experiment_manuals WHERE manual_id IN (${manualRows.map(() => "?").join(", ")})`,
        manualRows.map((row) => row.manual_id),
      );
    }

    const [datasetRows] = await connection.query<DatasetCandidateRow[]>(
      `SELECT DISTINCT d.dataset_id, d.file_path
         FROM datasets d
         JOIN dataset_segments ds ON ds.dataset_id = d.dataset_id
        WHERE ds.industry_name = ?
          AND ds.company_name = ?`,
      [PARALLEL_DATASET.industry, PARALLEL_DATASET.company],
    );
    const oldDatasetIds = datasetRows.map((row) => row.dataset_id);
    const oldDatasetPaths = datasetRows.map((row) => row.file_path);
    let oldSegmentPaths: Array<string | null> = [];
    if (oldDatasetIds.length > 0) {
      const [segmentRows] = await connection.query<SegmentFileRow[]>(
        `SELECT file_path
           FROM dataset_segments
          WHERE dataset_id IN (${oldDatasetIds.map(() => "?").join(", ")})`,
        oldDatasetIds,
      );
      oldSegmentPaths = segmentRows.map((row) => row.file_path);
      await connection.query(
        `DELETE FROM datasets WHERE dataset_id IN (${oldDatasetIds.map(() => "?").join(", ")})`,
        oldDatasetIds,
      );
    }

    const [manualInsert] = await connection.query<ResultSetHeader>(
      `INSERT INTO experiment_manuals (file_name, file_path, description, is_active, uploader_id)
       VALUES (?, ?, ?, true, ?)`,
      [
        PARALLEL_DATASET.manualDisplayName,
        "placeholder",
        PARALLEL_DATASET.manualDescription,
        PARALLEL_USER_IDS.admin,
      ],
    );
    const manualId = manualInsert.insertId;
    createdManualPath = path.resolve(
      getManualsRoot(),
      `parallel-shiyan-concurrency-manual-${manualId}.pdf`,
    );
    await Bun.write(
      createdManualPath,
      buildMinimalPdf([
        "Parallel shiyan manual",
        "Purpose: concurrent experiment validation",
        `Dataset file: ${PARALLEL_DATASET.fileLabel}`,
        `Manual file: ${PARALLEL_DATASET.manualFileLabel}`,
        `Products: ${PARALLEL_DATASET.products.length}`,
        `Months per product: ${PARALLEL_DATASET.monthCount}`,
        "Students: 24",
        "Scenarios: start-only, training-burst, mixed-load",
      ]),
    );
    await connection.query(
      "UPDATE experiment_manuals SET file_path = ?, is_active = true WHERE manual_id = ?",
      [createdManualPath, manualId],
    );

    const [datasetInsert] = await connection.query<ResultSetHeader>(
      `INSERT INTO datasets (data_name, file_path, description)
       VALUES (?, ?, ?)`,
      [PARALLEL_DATASET.name, "placeholder", PARALLEL_DATASET.description],
    );
    const datasetId = datasetInsert.insertId;

    createdDatasetPath = path.resolve(
      getDatasetsRoot(),
      `${datasetId}-${PARALLEL_DATASET.fileLabel}`,
    );
    createdSegmentsDir = path.resolve(getDatasetsRoot(), String(datasetId), "segments");
    await mkdir(createdSegmentsDir, { recursive: true });

    const productARows = buildProductRows({
      name: PARALLEL_DATASET.products[0],
      baseSales: 640,
      trendSlope: 4.6,
      yearlySeasonality: 62,
      promoBase: 18,
      priceBase: 98,
      capacityBase: 76,
      inventoryBase: 118,
      searchBase: 70,
      competitorPriceBase: 99,
      holidayScale: 12,
      channelSequence: ["直营", "电商", "分销"],
      regionSequence: ["华东", "华南", "华中"],
      weatherSequence: ["晴朗", "多云", "小雨"],
    });
    const productBRows = buildProductRows({
      name: PARALLEL_DATASET.products[1],
      baseSales: 510,
      trendSlope: 6.2,
      yearlySeasonality: 34,
      promoBase: 24,
      priceBase: 103,
      capacityBase: 73,
      inventoryBase: 104,
      searchBase: 84,
      competitorPriceBase: 106,
      holidayScale: 8,
      channelSequence: ["电商", "直营", "项目制"],
      regionSequence: ["华东", "西南", "华北"],
      weatherSequence: ["晴朗", "阵雨", "多云"],
    });

    await Bun.write(createdDatasetPath, toCsv([...productARows, ...productBRows]));

    const segmentPaths = [
      path.resolve(
        createdSegmentsDir,
        `${safeFileName(PARALLEL_DATASET.company)}_${safeFileName(PARALLEL_DATASET.products[0])}.csv`,
      ),
      path.resolve(
        createdSegmentsDir,
        `${safeFileName(PARALLEL_DATASET.company)}_${safeFileName(PARALLEL_DATASET.products[1])}.csv`,
      ),
    ] as const;
    await Bun.write(segmentPaths[0], toCsv(productARows));
    await Bun.write(segmentPaths[1], toCsv(productBRows));

    await connection.query("UPDATE datasets SET file_path = ? WHERE dataset_id = ?", [
      createdDatasetPath,
      datasetId,
    ]);
    await connection.query(
      `INSERT INTO dataset_segments
        (dataset_id, industry_name, company_name, product_name, file_path)
       VALUES
        (?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?)`,
      [
        datasetId,
        PARALLEL_DATASET.industry,
        PARALLEL_DATASET.company,
        PARALLEL_DATASET.products[0],
        segmentPaths[0],
        datasetId,
        PARALLEL_DATASET.industry,
        PARALLEL_DATASET.company,
        PARALLEL_DATASET.products[1],
        segmentPaths[1],
      ],
    );

    await connection.commit();

    const oldDatasetDirs = oldDatasetIds.map((datasetIdValue) =>
      path.resolve(getDatasetsRoot(), String(datasetIdValue)),
    );
    await rmMany([
      ...oldManualPaths,
      ...oldDatasetPaths,
      ...oldSegmentPaths,
      ...oldDatasetDirs,
    ]);

    console.log("Seeded parallel shiyan assets.");
    console.log(`  manual_id: ${manualId}`);
    console.log(`  dataset_id: ${datasetId}`);
    console.log(`  industry/company: ${PARALLEL_DATASET.industry} / ${PARALLEL_DATASET.company}`);
    console.log(`  products: ${PARALLEL_DATASET.products.join(", ")}`);
    console.log(`  months per product: ${PARALLEL_DATASET.monthCount}`);
  } catch (error) {
    await connection.rollback().catch(() => {});
    await rmMany([createdManualPath, createdDatasetPath, createdSegmentsDir]);
    throw error;
  } finally {
    await connection.end();
  }
}

await main();
