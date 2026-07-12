import { afterEach, describe, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  CsvEncodingError,
  CsvRowLimitError,
  decodeCsvBuffer,
  getCsvHeaders,
  parseCsvSegments,
  parseCsvFileRows,
  processCsvFile,
} from "../../../src/utils/csv-processor";

const gbkUserCsv = Buffer.from([
  0xd0, 0xd5, 0xc3, 0xfb, 0x2c, 0xca, 0xd6, 0xbb, 0xfa, 0xba, 0xc5, 0x0a,
  0xd5, 0xc5, 0xc8, 0xfd, 0x2c, 0x31, 0x33, 0x38, 0x30, 0x30, 0x31, 0x33,
  0x38, 0x30, 0x30, 0x30, 0x0a,
]);

let tempDirs: string[] = [];

async function writeTempCsv(content: string | Buffer): Promise<string> {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "fangzhen-csv-test-"));
  tempDirs.push(tempDir);
  const filePath = path.join(tempDir, "sample.csv");
  await writeFile(filePath, content);
  return filePath;
}

afterEach(async () => {
  await Promise.all(tempDirs.map((dir) => rm(dir, { recursive: true, force: true })));
  tempDirs = [];
});

describe("csv processor encoding", () => {
  test("decodes UTF-8 CSV with BOM", async () => {
    const filePath = await writeTempCsv("\uFEFF姓名,手机号\n张三,13800138000\n");

    await expect(getCsvHeaders(filePath)).resolves.toEqual(["姓名", "手机号"]);
  });

  test("falls back to GB18030 for Windows Excel CSV", async () => {
    const filePath = await writeTempCsv(gbkUserCsv);

    const result = await processCsvFile(filePath, {
      validator: (row) => ({
        fullName: row["姓名"]?.trim(),
        phone: row["手机号"]?.trim(),
      }),
    });

    expect(result).toEqual({
      validRows: [{ fullName: "张三", phone: "13800138000" }],
      invalidRows: [],
      totalRows: 1,
    });
  });

  test("rejects unsupported byte sequences", () => {
    expect(() => decodeCsvBuffer(Buffer.from([0xff]))).toThrow(CsvEncodingError);
  });

  test("parses GB18030 dataset headers and segments", async () => {
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
    const filePath = await writeTempCsv(gbkDatasetCsv);

    await expect(getCsvHeaders(filePath)).resolves.toEqual([
      "行业名称",
      "公司名称",
      "产品名称",
      "年份",
      "月份",
      "销售数量",
      "数量单位",
    ]);

    const segments = await parseCsvSegments(filePath);
    expect([...segments.keys()]).toEqual(["制造业|样本企业|样本产品"]);
  });

  test("stops parsing as soon as the configured row limit is exceeded", async () => {
    const filePath = await writeTempCsv("value\n1\n2\n3\n");
    const seenRows: string[] = [];

    await expect(parseCsvFileRows(
      filePath,
      (row) => seenRows.push(row.value!),
      { maxRows: 2 },
    )).rejects.toBeInstanceOf(CsvRowLimitError);
    expect(seenRows).toEqual(["1", "2"]);
  });
});
