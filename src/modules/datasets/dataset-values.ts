export interface NormalizedSalesValue {
  canonical: string;
  value: number;
}

export interface NormalizedYearMonth {
  month: string;
  year: string;
}

const DECIMAL_NUMBER_PATTERN = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;

export function normalizeSalesValue(raw: unknown): NormalizedSalesValue | null {
  const canonical = String(raw ?? "").replace(/,/g, "").trim();
  if (!DECIMAL_NUMBER_PATTERN.test(canonical)) return null;

  const value = Number(canonical);
  if (!Number.isFinite(value) || value < 0) return null;

  return { canonical, value };
}

export function normalizeYearMonth(
  yearRaw: unknown,
  monthRaw: unknown,
): NormalizedYearMonth | null {
  const year = String(yearRaw ?? "").trim();
  const rawMonth = String(monthRaw ?? "").trim();
  if (!/^[1-9]\d{3}$/.test(year) || !/^(0?[1-9]|1[0-2])$/.test(rawMonth)) {
    return null;
  }

  return { year, month: rawMonth.padStart(2, "0") };
}

export function yearMonthOrdinal(value: NormalizedYearMonth): number {
  return Number(value.year) * 12 + Number(value.month) - 1;
}

export function isConsecutiveYearMonth(
  previous: NormalizedYearMonth,
  current: NormalizedYearMonth,
): boolean {
  return yearMonthOrdinal(current) === yearMonthOrdinal(previous) + 1;
}

export function formatYearMonth(value: NormalizedYearMonth): string {
  return `${value.year}-${value.month}`;
}
