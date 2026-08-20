import path from "node:path";
import type { RowDataPacket } from "mysql2/promise";
import { env, pool } from "../../config";
import { CsvEncodingError, CsvRowLimitError, readCsvRows } from "../../utils/csv-processor";
import {
  getDatabaseManagedStorageRoot,
  type ManagedStorageEnvironment,
} from "../../utils/managed-storage";
import { resolveExistingManagedFile } from "../../utils/managed-file";
import {
  isNextCalendarMonth,
  normalizeSalesValue,
  normalizeYearMonth,
  type NormalizedYearMonth,
} from "../datasets/dataset-values";
import { ModelValidationError } from "./model-errors";
import { queryWithTimeout } from "./model-db-guard";

interface SegmentRow extends RowDataPacket {
  file_path: string;
}

const datasetsManagedRoot = path.resolve(import.meta.dir, "../../../uploads/datasets");

export async function resolveModelDatasetFilePath(
  storedPath: string,
  options: {
    managedRoot?: string;
    environment?: ManagedStorageEnvironment;
  } = {},
): Promise<string> {
  const managedRoot = path.resolve(options.managedRoot ?? datasetsManagedRoot);
  const environment = options.environment ?? env;
  const databaseRoot = getDatabaseManagedStorageRoot(managedRoot, environment);
  const resolved = await resolveExistingManagedFile({
    storedPath,
    managedRoot,
    environment,
    legacyFallbackDirectory: databaseRoot,
    enforceDisposableNamespace: true,
  });
  if (!resolved) {
    throw new Error("数据集分段文件不存在");
  }
  return resolved.absolutePath;
}

export async function loadDatasetFromDb(
  industry: string,
  company: string,
  product: string,
  terminationSignal?: AbortSignal,
): Promise<Record<string, unknown>[]> {
  const throwIfTerminated = () => {
    if (!terminationSignal?.aborted) return;
    throw terminationSignal.reason instanceof Error
      ? terminationSignal.reason
      : new Error("模型数据加载已终止");
  };

  throwIfTerminated();
  const [segmentRows] = await queryWithTimeout<SegmentRow[]>(pool,
    "SELECT file_path FROM dataset_segments WHERE industry_name = ? AND company_name = ? AND product_name = ? LIMIT 1",
    [industry, company, product],
  );

  throwIfTerminated();
  if (segmentRows.length === 0) {
    throw new ModelValidationError("未找到数据集分段文件");
  }

  try {
    const datasetFilePath = await resolveModelDatasetFilePath(segmentRows[0]!.file_path);
    throwIfTerminated();
    const rows = await readCsvRows(datasetFilePath, terminationSignal);
    throwIfTerminated();
    let previousTime: NormalizedYearMonth | null = null;
    return rows.map((row, index) => {
      const separateTime = normalizeYearMonth(row["年份"], row["月份"]);
      const combinedMonthMatch = String(row["月份"] ?? "").trim().match(/^(\d{4})-(0?[1-9]|1[0-2])$/);
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
      if (previousTime && !isNextCalendarMonth(previousTime, time)) {
        throw new ModelValidationError(`数据集第 ${index + 1} 行月度时间点不连续`, {
          row: index + 1,
          field: "年份/月份",
          code: "DATASET_MONTHLY_CADENCE_INVALID",
        });
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
    if (terminationSignal?.aborted) {
      throw terminationSignal.reason instanceof Error
        ? terminationSignal.reason
        : error;
    }
    if (error instanceof ModelValidationError) throw error;
    if (error instanceof CsvEncodingError || error instanceof CsvRowLimitError) {
      throw new ModelValidationError(error.message);
    }
    throw new ModelValidationError("数据集分段文件不可用");
  }
}
