import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const datasetsServiceModulePath = path.join(srcRoot, "modules", "datasets", "datasets.service.ts");
const freshDatasetsServiceModulePath = `${datasetsServiceModulePath}?datasets-csv-encoding-test`;

const poolQuery = mock(async () => [[{ file_path: datasetFilePath }]]);

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

const { getProductFields, getProductSales, toCsvString } = await import(freshDatasetsServiceModulePath);
const { getCsvHeaders, readCsvRows } = await import("../../../src/utils/csv-processor");

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
    tempDir = await mkdtemp(path.join(os.tmpdir(), "fangzhen-dataset-csv-test-"));
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
});
