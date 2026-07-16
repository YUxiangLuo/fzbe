import { describe, expect, test } from "bun:test";
import {
  formatYearMonth,
  isConsecutiveYearMonth,
  normalizeSalesValue,
  normalizeYearMonth,
  yearMonthOrdinal,
} from "../../../src/modules/datasets/dataset-values";

describe("dataset value normalization", () => {
  test("normalizes finite non-negative sales values", () => {
    expect(normalizeSalesValue(" 1,234.50 ")).toEqual({ canonical: "1234.50", value: 1234.5 });
    expect(normalizeSalesValue("0")).toEqual({ canonical: "0", value: 0 });
    expect(normalizeSalesValue("1e3")).toEqual({ canonical: "1e3", value: 1000 });
  });

  test("rejects empty, negative, and non-finite sales values", () => {
    for (const value of [
      "",
      "   ",
      "-1",
      "NaN",
      "Infinity",
      "1e309",
      "0x10",
      "0b10",
      null,
      undefined,
    ]) {
      expect(normalizeSalesValue(value)).toBeNull();
    }
  });

  test("normalizes strict year and month values", () => {
    expect(normalizeYearMonth("2024", "1")).toEqual({ year: "2024", month: "01" });
    expect(normalizeYearMonth("2024", "12")).toEqual({ year: "2024", month: "12" });
    expect(normalizeYearMonth("1000", "01")).toEqual({ year: "1000", month: "01" });
    expect(normalizeYearMonth("9999", "12")).toEqual({ year: "9999", month: "12" });
    expect(normalizeYearMonth("0999", "1")).toBeNull();
    expect(normalizeYearMonth("0000", "1")).toBeNull();
    expect(normalizeYearMonth("24", "1")).toBeNull();
    expect(normalizeYearMonth("2024", "13")).toBeNull();
  });

  test("compares monthly periods across year boundaries", () => {
    const november = normalizeYearMonth("2024", "11")!;
    const december = normalizeYearMonth("2024", "12")!;
    const january = normalizeYearMonth("2025", "1")!;

    expect(formatYearMonth(january)).toBe("2025-01");
    expect(yearMonthOrdinal(december)).toBe(yearMonthOrdinal(november) + 1);
    expect(isConsecutiveYearMonth(december, january)).toBeTrue();
    expect(isConsecutiveYearMonth(november, january)).toBeFalse();
  });
});
