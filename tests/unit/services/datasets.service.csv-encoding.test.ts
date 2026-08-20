import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const datasetsServiceModulePath = path.join(srcRoot, "modules", "datasets", "datasets.service.ts");
const freshDatasetsServiceModulePath = `${datasetsServiceModulePath}?datasets-csv-encoding-test`;

const poolQuery = mock(async () => [[{ dataset_id: 77, file_path: datasetFilePath }]]);

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
  },
}));

const {
  buildDatasetSegmentFilename,
  buildDatasetGenerationPaths,
  getDatasetStructureLimitError,
  getProductFields,
  getProductSales,
  toCsvString,
  uploadDataset,
} = await import(freshDatasetsServiceModulePath);
const { getCsvHeaders, readCsvRows } = await import("../../../src/utils/csv-processor");
const { CSV_LIMITS } = await import("../../../src/utils/constants");

const gbkDatasetCsv = Buffer.from([
  0xd0, 0xd0, 0xd2, 0xb5, 0xc3, 0xfb, 0xb3, 0xc6, 0x2c, 0xb9, 0xab, 0xcb,
  0xbe, 0xc3, 0xfb, 0xb3, 0xc6, 0x2c, 0xb2, 0xfa, 0xc6, 0xb7, 0xc3,
  0xfb, 0xb3, 0xc6, 0x2c, 0xc4, 0xea, 0xb7, 0xdd, 0x2c, 0xd4, 0xc2,
  0xb7, 0xdd, 0x2c, 0xcf, 0xfa, 0xca, 0xdb, 0xca, 0xfd, 0xc1, 0xbf,
  0x2c, 0xca, 0xfd, 0xc1, 0xbf, 0xb5, 0xa5, 0xce, 0xbb, 0x0a, 0xd6,
  0xc6, 0xd4, 0xec, 0xd2, 0xb5, 0x2c, 0xd1, 0xf9, 0xb1, 0xbe, 0xc6,
  0xf3, 0xd2, 0xb5, 0x2c, 0xd1, 0xf9, 0xb1, 0xbe, 0xb2, 0xfa, 0xc6,
  0xb7, 0x2c, 0x32, 0x30, 0x32, 0x34, 0x2c, 0x31, 0x2c, 0x31, 0x30,
  0x30, 0x2c, 0xbc, 0xfe, 0x0a,
]);

let tempDir = "";
let datasetFilePath = "";

describe("datasets service CSV encoding", () => {
  beforeEach(async () => {
    const datasetsRoot = path.resolve(srcRoot, "../uploads/datasets");
    await mkdir(datasetsRoot, { recursive: true });
    tempDir = await mkdtemp(path.join(datasetsRoot, "unit-dataset-csv-test-"));
    datasetFilePath = path.join(tempDir, "dataset.csv");
    await writeFile(datasetFilePath, gbkDatasetCsv);
    poolQuery.mockClear();
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
    tempDir = "";
    datasetFilePath = "";
  });

  test("reads Windows Excel CSV dataset fields and sales data", async () => {
    await expect(getProductFields("制造业", "样本企业", "样本产品")).resolves.toEqual({
      data: ["行业名称", "公司名称", "产品名称", "年份", "月份", "销售数量", "数量单位"],
    });

    await expect(getProductSales("制造业", "样本企业", "样本产品")).resolves.toEqual({
      data: {
        meta: {
          industry: "制造业",
          company: "样本企业",
          product: "样本产品",
          name: "样本产品",
          description: "",
          unit: "件",
        },
        monthlySales: [{ month: "2024-01", sales: 100 }],
        csvData: [
          ["行业名称", "公司名称", "产品名称", "年份", "月份", "销售数量", "数量单位"],
          ["制造业", "样本企业", "样本产品", "2024", "1", "100", "件"],
        ],
      },
    });
  });

  test("writes segment CSV headers and values with RFC-compatible escaping", async () => {
    const csvText = toCsvString([
      {
        "行业名称": "制造业",
        "外部,指标": "12,345",
        "备注\"字段": "包含\"引号\"和值,逗号",
      },
    ]);
    await writeFile(datasetFilePath, csvText);

    await expect(getCsvHeaders(datasetFilePath)).resolves.toEqual([
      "行业名称",
      "外部,指标",
      "备注\"字段",
    ]);
    await expect(readCsvRows(datasetFilePath)).resolves.toEqual([
      {
        "行业名称": "制造业",
        "外部,指标": "12,345",
        "备注\"字段": "包含\"引号\"和值,逗号",
      },
    ]);
  });

  test("rejects a non-contiguous monthly product segment before persistence", async () => {
    const sparseCsv = [
      "行业名称,公司名称,产品名称,年份,月份,销售数量,数量单位",
      "制造业,样本企业,样本产品,2024,1,100,件",
      "制造业,样本企业,样本产品,2024,3,120,件",
    ].join("\n");

    await expect(uploadDataset(
      new File([sparseCsv], "sparse.csv", { type: "text/csv" }),
      "稀疏月度数据",
    )).resolves.toMatchObject({
      error: "时间不连续",
      status: 400,
    });
  });

  test("keeps aggregate uploads and model-readable product segments independently bounded", () => {
    const row = { "年份": "2024", "月份": "1", "销售数量": "100" };
    const baseSegment = {
      industryName: "制造业",
      companyName: "样本企业",
      productName: "样本产品",
      rows: Array(CSV_LIMITS.DATASET_SEGMENT_MAX_ROWS).fill(row),
    };

    expect(getDatasetStructureLimitError(new Map([["valid", baseSegment]]))).toBeNull();
    expect(getDatasetStructureLimitError(new Map([[
      "oversized",
      { ...baseSegment, rows: [...baseSegment.rows, row] },
    ]]))).toContain(`数据行数不能超过 ${CSV_LIMITS.DATASET_SEGMENT_MAX_ROWS}`);

    const tooManySegments = new Map(
      Array.from(
        { length: CSV_LIMITS.DATASET_UPLOAD_MAX_SEGMENTS + 1 },
        (_, index) => [`segment-${index}`, { ...baseSegment, rows: [row] }] as const,
      ),
    );
    expect(getDatasetStructureLimitError(tooManySegments))
      .toContain(`组合数量不能超过 ${CSV_LIMITS.DATASET_UPLOAD_MAX_SEGMENTS}`);
  });

  test("gives distinct bounded names to different tuples and sanitize collisions", () => {
    const baseline = buildDatasetSegmentFilename({
      industryName: "制造业-A",
      companyName: "同名/企业",
      productName: "产品:一",
    });
    const otherIndustry = buildDatasetSegmentFilename({
      industryName: "制造业-B",
      companyName: "同名/企业",
      productName: "产品:一",
    });
    const sanitizeCollision = buildDatasetSegmentFilename({
      industryName: "制造业-A",
      companyName: "同名\\企业",
      productName: "产品?一",
    });

    expect(new Set([baseline, otherIndustry, sanitizeCollision]).size).toBe(3);
    expect(Buffer.byteLength(baseline)).toBeLessThanOrEqual(255);
    expect(baseline).toEndWith(".csv");
  });

  test("keeps long CJK and emoji segment names safely below filesystem byte limits", () => {
    const filename = buildDatasetSegmentFilename({
      industryName: "制造业🚀".repeat(100),
      companyName: "超长企业名称🧪".repeat(100),
      productName: "超长产品名称📦".repeat(100),
    });

    expect(filename).toMatch(/^[a-f0-9]{16}-[A-Za-z0-9._-]+\.csv$/);
    expect(Buffer.byteLength(filename, "utf8")).toBeLessThan(80);
  });

  test("uses a fresh filesystem generation when a dataset ID and input name are reused", () => {
    const first = buildDatasetGenerationPaths("/managed/datasets", 77, "sales.csv");
    const second = buildDatasetGenerationPaths("/managed/datasets", 77, "sales.csv");

    expect(first.generationToken).not.toBe(second.generationToken);
    expect(first.originalPath).not.toBe(second.originalPath);
    expect(first.datasetDirectory).not.toBe(second.datasetDirectory);
    expect(first.segmentsDirectory).not.toBe(second.segmentsDirectory);
    expect(path.basename(first.originalPath)).toMatch(
      /^77-[0-9a-f-]{36}-sales\.csv$/,
    );
  });

  test("bounds the permanent original upload name by filesystem bytes", () => {
    const generation = buildDatasetGenerationPaths(
      "/managed/datasets",
      77,
      `${"超长销售数据".repeat(100)}.csv`,
    );
    const filename = path.basename(generation.originalPath);

    expect(Buffer.byteLength(filename, "utf8")).toBeLessThanOrEqual(255);
    expect(Array.from(generation.originalPath).length).toBeLessThanOrEqual(255);
    expect(filename).toEndWith(".csv");
    expect(filename).not.toContain("�");
  });

  test("bounds long ASCII dataset paths for the datasets VARCHAR(255) column", () => {
    const generation = buildDatasetGenerationPaths(
      "/srv/fangzhen/uploads/datasets/.db/0123456789abcdef",
      77,
      `${"sales".repeat(100)}.csv`,
    );

    expect(Array.from(generation.originalPath).length).toBeLessThanOrEqual(255);
    expect(Buffer.byteLength(path.basename(generation.originalPath), "utf8"))
      .toBeLessThanOrEqual(255);
    expect(generation.originalPath).toEndWith(".csv");
  });

  test("reads legacy grouped sales without changing row cardinality", async () => {
    await writeFile(
      datasetFilePath,
      "行业名称,公司名称,产品名称,年份,月份,销售数量,数量单位\n制造业,样本企业,样本产品,2024,1,\"1,234.50\",件\n",
    );

    const result = await getProductSales("制造业", "样本企业", "样本产品");

    expect(result).toMatchObject({
      data: {
        monthlySales: [{ month: "2024-01", sales: 1234.5 }],
      },
    });
  });

  test("reports invalid legacy rows instead of silently dropping them", async () => {
    await writeFile(
      datasetFilePath,
      "行业名称,公司名称,产品名称,年份,月份,销售数量,数量单位\n制造业,样本企业,样本产品,2024,1,1e309,件\n",
    );

    await expect(getProductSales("制造业", "样本企业", "样本产品")).resolves.toEqual({
      error: "invalid_data",
      details: {
        row: 1,
        field: "销售数量",
        message: "第 1 行销售数量无效",
      },
    });
  });
});
