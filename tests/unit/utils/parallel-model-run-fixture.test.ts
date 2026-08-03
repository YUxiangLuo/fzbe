import { describe, expect, test } from "bun:test";
import { hashDatasetSnapshot } from "../../../src/modules/models/model-runtime-identity";
import {
  normalizeParallelModelDatasetRows,
} from "../../../scripts/parallel-model-run-fixture";

describe("parallel model-run seed fixture", () => {
  test("normalizes dataset rows exactly before calculating the active-run revision", () => {
    const rows = normalizeParallelModelDatasetRows([
      {
        "年份": "2024",
        "月份": "1",
        "销售数量": "1,234.50",
        "渠道": "直营",
      },
      {
        "年份": "",
        "月份": "2024-02",
        "销售数量": "0",
        "渠道": "电商",
      },
    ]);

    expect(rows).toEqual([
      {
        "年份": "2024",
        "月份": "01",
        "销售数量": "1234.50",
        "渠道": "直营",
      },
      {
        "年份": "",
        "月份": "2024-02",
        "销售数量": "0",
        "渠道": "电商",
      },
    ]);
    expect(hashDatasetSnapshot(JSON.stringify(rows))).toMatch(/^[a-f\d]{64}$/);
  });

  test("rejects rows that production dataset loading would reject", () => {
    expect(() => normalizeParallelModelDatasetRows([
      { "年份": "2024", "月份": "13", "销售数量": "10" },
    ])).toThrow("year/month");
    expect(() => normalizeParallelModelDatasetRows([
      { "年份": "2024", "月份": "01", "销售数量": "-1" },
    ])).toThrow("sales value");
  });
});
