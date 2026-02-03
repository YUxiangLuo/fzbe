/**
 * CSV 处理工具函数
 * 统一 CSV 解析逻辑，添加行数限制保护
 */

import { createReadStream } from "node:fs";
import { mkdir, unlink } from "node:fs/promises";
import path from "path";
import csv from "csv-parser";
import { CSV_LIMITS, FILE_SIZE_LIMITS } from "./constants";
import { validateFile, generateUniqueFilename, FileValidationError } from "./file-validator";
import { logWarning } from "./error-handler";

export { FileValidationError };

export interface CsvValidationResult<T> {
  validRows: T[];
  invalidRows: Array<{ row: number; reason: string }>;
  totalRows: number;
}

export interface CsvProcessorOptions<T> {
  /** 行验证函数，返回 null 表示跳过，返回字符串表示错误原因，返回对象表示有效行 */
  validator: (row: Record<string, string>, rowNumber: number) => T | null | string;
  /** 最大行数限制 */
  maxRows?: number;
  /** 用于检测重复的字段名 */
  duplicateCheckField?: string;
  /** 已存在的值集合 (用于外部去重检查) */
  existingValues?: Set<string>;
}

// Represents a segment of data for a specific product, including its metadata and all its data rows.
export interface SegmentData {
  industryName: string;
  companyName: string;
  productName: string;
  rows: Record<string, string>[]; // All data rows for this segment
}

/**
 * 处理 CSV 文件并进行验证
 * @param filePath CSV 文件路径
 * @param options 处理选项
 * @returns 验证结果
 */
export async function processCsvFile<T>(
  filePath: string,
  options: CsvProcessorOptions<T>,
): Promise<CsvValidationResult<T>> {
  const {
    validator,
    maxRows = CSV_LIMITS.MAX_ROWS,
    duplicateCheckField,
    existingValues,
  } = options;

  const validRows: T[] = [];
  const invalidRows: Array<{ row: number; reason: string }> = [];
  const seenValues = new Set<string>(existingValues || []);
  let rowNumber = 0;

  return new Promise((resolve, reject) => {
    createReadStream(filePath)
      .pipe(csv({ mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim() }))
      .on("data", (row: Record<string, string>) => {
        rowNumber += 1;

        // 检查行数限制
        if (rowNumber > maxRows) {
          invalidRows.push({ row: rowNumber, reason: `超过最大行数限制 ${maxRows}` });
          return;
        }

        // 检查重复值
        if (duplicateCheckField) {
          const value = row[duplicateCheckField]?.trim();
          if (value && seenValues.has(value)) {
            invalidRows.push({ row: rowNumber, reason: `CSV 内存在重复的 ${duplicateCheckField}: ${value}` });
            return;
          }
          if (value) {
            seenValues.add(value);
          }
        }

        // 调用验证函数
        const result = validator(row, rowNumber);

        if (result === null) {
          // 跳过该行
          return;
        }

        if (typeof result === "string") {
          // 验证失败，记录错误原因
          invalidRows.push({ row: rowNumber, reason: result });
          return;
        }

        // 验证通过
        validRows.push(result);
      })
      .on("end", () => {
        resolve({ validRows, invalidRows, totalRows: rowNumber });
      })
      .on("error", reject);
  });
}

/**
 * Parses a CSV file and groups its rows by a composite key (Industry|Company|Product).
 * This function reads the entire file into memory and returns a map where each entry
 * corresponds to a unique product segment.
 *
 * @param filePath The absolute path to the CSV file.
 * @returns A Promise that resolves to a Map<string, SegmentData>.
 */
export function parseCsvSegments(
  filePath: string,
): Promise<Map<string, SegmentData>> {
  return new Promise((resolve, reject) => {
    const segments = new Map<string, SegmentData>();

    createReadStream(filePath)
      .pipe(
        csv({
          mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim(),
        }),
      )
      .on("data", (row: Record<string, string>) => {
        const industryName = (row["行业名称"] ?? "").trim();
        const companyName = (row["公司名称"] ?? "").trim();
        const productName = (row["产品名称"] ?? "").trim();

        if (!industryName || !companyName || !productName) {
          return; // Skip rows that don't have the required identifiers
        }

        const key = `${industryName}|${companyName}|${productName}`;
        let segment = segments.get(key);

        if (!segment) {
          segment = {
            industryName,
            companyName,
            productName,
            rows: [],
          };
          segments.set(key, segment);
        }

        segment.rows.push(row);
      })
      .on("end", () => {
        resolve(segments);
      })
      .on("error", (error) => {
        reject(error);
      });
  });
}

/**
 * Efficiently reads only the headers from a CSV file.
 * @param filePath The absolute path to the CSV file.
 * @returns A Promise that resolves to an array of header strings.
 */
export function getCsvHeaders(filePath: string): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const stream = createReadStream(filePath)
      .pipe(csv({ mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim() }));

    let headersEmitted = false;

    stream.on("headers", (headers: string[]) => {
      headersEmitted = true;
      stream.destroy();
      resolve(headers);
    });

    stream.on("error", (error) => {
      reject(error);
    });

    stream.on("end", () => {
      if (!headersEmitted) {
        resolve([]);
      }
    });
  });
}

/**
 * 上传并处理 CSV 文件的完整流程
 * @param file 上传的文件
 * @param options 处理选项
 * @param tempDirPath 临时目录路径
 * @returns 处理结果
 */
export async function uploadAndProcessCsv<T>(
  file: File,
  options: CsvProcessorOptions<T>,
  tempDirPath: string,
): Promise<
  | { error: string; details?: unknown }
  | { data: CsvValidationResult<T>; tempFilePath: string }
> {
  // 验证文件
  try {
    await validateFile(file, "csv", {
      maxSize: FILE_SIZE_LIMITS.CSV_MAX_SIZE,
      checkMagicBytes: false,
    });
  } catch (error) {
    if (error instanceof FileValidationError) {
      return { error: error.message };
    }
    throw error;
  }

  // 保存到临时目录
  await mkdir(tempDirPath, { recursive: true });
  const uniqueFilename = generateUniqueFilename(file.name);
  const tempFilePath = path.join(tempDirPath, uniqueFilename);
  await Bun.write(tempFilePath, file);

  try {
    const result = await processCsvFile<T>(tempFilePath, options);

    if (result.invalidRows.length > 0) {
      // 有无效行，清理临时文件
      await unlink(tempFilePath).catch((err) =>
        logWarning("Failed to cleanup temp CSV", { error: String(err) }),
      );
      return {
        error: "CSV 文件存在无效记录",
        details: { invalid_rows: result.invalidRows.slice(0, 10) },
      };
    }

    if (result.validRows.length === 0) {
      await unlink(tempFilePath).catch((err) =>
        logWarning("Failed to cleanup temp CSV", { error: String(err) }),
      );
      return { error: "CSV 文件中没有有效记录" };
    }

    return { data: result, tempFilePath };
  } catch (error) {
    await unlink(tempFilePath).catch(() => {});
    throw error;
  }
}

/**
 * 清理临时文件
 */
export async function cleanupTempFile(filePath: string | null): Promise<void> {
  if (filePath) {
    await unlink(filePath).catch((err) =>
      logWarning("Failed to cleanup temp file", { error: String(err) }),
    );
  }
}
