import type { RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import { CsvEncodingError, CsvRowLimitError, readCsvRows } from "../../utils/csv-processor";
import {
  formatYearMonth,
  isConsecutiveYearMonth,
  normalizeSalesValue,
  normalizeYearMonth,
  yearMonthOrdinal,
  type NormalizedYearMonth,
} from "../datasets/dataset-values";
import { ModelValidationError } from "./model-errors";

interface SegmentRow extends RowDataPacket {
  file_path: string;
}

export async function loadDatasetFromDb(
  industry: string,
  company: string,
  product: string,
): Promise<Record<string, unknown>[]> {
  const [segmentRows] = await pool.query<SegmentRow[]>(
    "SELECT file_path FROM dataset_segments WHERE industry_name = ? AND company_name = ? AND product_name = ? LIMIT 1",
    [industry, company, product],
  );

  if (segmentRows.length === 0) {
    throw new ModelValidationError("未找到数据集分段文件");
  }

  try {
    const rows = await readCsvRows(segmentRows[0]!.file_path);
    let previousTime: NormalizedYearMonth | null = null;
    const seenTimes = new Set<number>();
    return rows.map((row, index) => {
      const separateTime = normalizeYearMonth(row["年份"], row["月份"]);
      const combinedMonthMatch = String(row["月份"] ?? "").trim().match(/^([1-9]\d{3})-(0?[1-9]|1[0-2])$/);
      const combinedTime = combinedMonthMatch
        ? normalizeYearMonth(combinedMonthMatch[1], combinedMonthMatch[2])
        : null;
      const time = separateTime ?? combinedTime;
      if (!time) {
        throw new ModelValidationError(`数据集第 ${index + 1} 行年月格式无效`, {
          row: index + 1,
          field: "年份/月份",
          code: "DATASET_ROW_INVALID",
        });
      }
      const timeOrdinal = yearMonthOrdinal(time);
      if (seenTimes.has(timeOrdinal)) {
        throw new ModelValidationError(`数据集第 ${index + 1} 行年月重复`, {
          row: index + 1,
          field: "年份/月份",
          code: "DATASET_TIME_DUPLICATE",
        });
      }
      seenTimes.add(timeOrdinal);
      if (previousTime && timeOrdinal < yearMonthOrdinal(previousTime)) {
        throw new ModelValidationError(`数据集第 ${index + 1} 行年月顺序颠倒`, {
          row: index + 1,
          field: "年份/月份",
          code: "DATASET_TIME_ORDER",
        });
      }
      if (previousTime && !isConsecutiveYearMonth(previousTime, time)) {
        throw new ModelValidationError(
          `数据集在 ${formatYearMonth(previousTime)} 与 ${formatYearMonth(time)} 之间存在缺失月份`,
          {
            row: index + 1,
            field: "年份/月份",
            code: "DATASET_TIME_GAP",
          },
        );
      }
      previousTime = time;
      const sales = normalizeSalesValue(row["销售数量"]);
      if (!sales) {
        throw new ModelValidationError(`数据集第 ${index + 1} 行销售数量无效`, {
          row: index + 1,
          field: "销售数量",
          code: "DATASET_ROW_INVALID",
        });
      }
      const normalizedRow = {
        ...row,
        "销售数量": sales.canonical,
      };
      return separateTime
        ? { ...normalizedRow, "年份": time.year, "月份": time.month }
        : { ...normalizedRow, "月份": `${time.year}-${time.month}` };
    });
  } catch (error) {
    if (error instanceof CsvEncodingError || error instanceof CsvRowLimitError) {
      throw new ModelValidationError(error.message);
    }
    throw error;
  }
}
