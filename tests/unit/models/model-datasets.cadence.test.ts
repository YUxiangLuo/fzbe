import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const datasetsRoot = path.resolve(srcRoot, "../uploads/datasets");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const modelDatasetsModulePath = path.join(srcRoot, "modules", "models", "model-datasets.ts");
let datasetPath = "";

const poolQuery = mock(async () => [[{ file_path: datasetPath }]]);
mock.module(configModulePath, () => ({
  env: {
    DB_HOST: "localhost",
    DB_USER: "test",
    DB_PASSWORD: "test",
    DB_DATABASE: "fangzhen",
    JWT_SECRET: "test-secret",
    PORT: 3001,
  },
  pool: { query: poolQuery },
}));

const { loadDatasetFromDb } = await import(`${modelDatasetsModulePath}?cadence-test`);
let temporaryRoot = "";

beforeEach(async () => {
  await mkdir(datasetsRoot, { recursive: true });
  temporaryRoot = await mkdtemp(path.join(datasetsRoot, "unit-cadence-"));
  datasetPath = path.join(temporaryRoot, "segment.csv");
  poolQuery.mockClear();
});

afterEach(async () => {
  await rm(temporaryRoot, { recursive: true, force: true });
});

describe("model dataset monthly cadence", () => {
  test("accepts a contiguous December to January rollover", async () => {
    await writeFile(datasetPath, "年份,月份,销售数量\n2024,12,100\n2025,1,110\n");

    await expect(loadDatasetFromDb("行业", "企业", "产品")).resolves.toHaveLength(2);
  });

  test("rejects a sparse legacy segment before model execution", async () => {
    await writeFile(datasetPath, "年份,月份,销售数量\n2024,1,100\n2024,3,110\n");

    await expect(loadDatasetFromDb("行业", "企业", "产品")).rejects.toMatchObject({
      message: "数据集第 2 行月度时间点不连续",
      details: { code: "DATASET_MONTHLY_CADENCE_INVALID" },
    });
  });
});
