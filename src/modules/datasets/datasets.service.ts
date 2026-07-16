import path from "path";
import { unlink, mkdir, rm } from "node:fs/promises";
import type { RowDataPacket, ResultSetHeader } from "mysql2/promise";
import { pool } from "../../config";
import { CsvEncodingError, CsvRowLimitError, parseCsvSegments, getCsvHeaders, readCsvRows } from "../../utils/csv-processor";
import { validateFile, generateUniqueFilename, FileValidationError } from "../../utils/file-validator";
import { logWarning } from "../../utils/error-handler";
import { FILE_SIZE_LIMITS } from "../../utils/constants";
import type { UpdateDatasetInput } from "./datasets.schema";
import {
  formatYearMonth,
  isConsecutiveYearMonth,
  normalizeSalesValue,
  normalizeYearMonth,
  type NormalizedYearMonth,
} from "./dataset-values";

interface DatasetRow extends RowDataPacket {
  dataset_id: number;
  data_name: string;
  file_path: string;
  description: string | null;
  uploaded_at: Date;
}

interface SegmentRow extends RowDataPacket {
  industry_name?: string;
  company_name?: string;
  product_name?: string;
  file_path?: string;
}

const uploadsRoot = path.resolve(import.meta.dir, "../../../uploads");

function toPublicFilePath(absolutePath: string | null | undefined): string | null {
  if (!absolutePath) return null;
  const relative = path.relative(uploadsRoot, absolutePath);
  if (!relative || relative.startsWith("..")) {
    return `datasets/${path.basename(absolutePath)}`.replace(/\\/g, "/");
  }
  return relative.replace(/\\/g, "/");
}

function normalizeDatasetRow(row: DatasetRow) {
  return { ...row, file_path: toPublicFilePath(row.file_path) };
}

function isDuplicateDatasetSegmentError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const maybeError = error as { code?: string; errno?: number; message?: string };
  return (
    (maybeError.code === "ER_DUP_ENTRY" || maybeError.errno === 1062) &&
    (maybeError.message?.includes("uniq_dataset_segments_product") ?? false)
  );
}

function escapeCsvCell(value: unknown): string {
  const text = String(value ?? "");
  const escaped = text.replace(/"/g, '""');
  return /[",\r\n]/.test(text) ? `"${escaped}"` : escaped;
}

export function toCsvString(data: Record<string, string>[]): string {
  if (data.length === 0) return "";
  const headers = Object.keys(data[0] ?? {});
  const headerRow = headers.map(escapeCsvCell).join(",");
  const bodyRows = data.map((row) => headers.map((header) => escapeCsvCell(row[header])).join(","));
  return [headerRow, ...bodyRows].join("\n");
}

export async function uploadDataset(file: File, dataName: string, description?: string) {
  let tempUploadPath: string | null = null;
  let datasetDirectory: string | null = null;
  let permanentOriginalPath: string | null = null;

  try {
    await validateFile(file, "csv", { maxSize: FILE_SIZE_LIMITS.DATASET_CSV_MAX_SIZE, checkMagicBytes: false });
  } catch (error) {
    if (error instanceof FileValidationError) return { error: error.message, status: 400 as const };
    throw error;
  }

  const tempDir = path.resolve(uploadsRoot, "temp");
  await mkdir(tempDir, { recursive: true });
  const uniqueFilename = generateUniqueFilename(file.name);
  tempUploadPath = path.join(tempDir, uniqueFilename);
  await Bun.write(tempUploadPath, file);

  try {
    const requiredHeaders = ["年份", "月份", "销售数量", "行业名称", "公司名称", "产品名称", "数量单位"];
    const headers = await getCsvHeaders(tempUploadPath);
    const missingHeaders = requiredHeaders.filter((h) => !headers.includes(h));
    if (missingHeaders.length > 0) {
      return { error: "文件格式错误", status: 400 as const, details: { missing_headers: missingHeaders } };
    }

    const segmentsMap = await parseCsvSegments(tempUploadPath);

    for (const [, segment] of segmentsMap.entries()) {
      const timePoints = new Set<string>();
      let previousTime: NormalizedYearMonth | null = null;
      const firstRow = segment.rows[0];
      if (!firstRow) return { error: "数据格式错误", status: 400 as const, details: { message: `产品 "${segment.productName}" 无有效数据` } };

      const expectedUnit = (firstRow["数量单位"] ?? "").trim();

      for (const [i, row] of segment.rows.entries()) {
        const yearRaw = row["年份"]?.trim();
        const monthRaw = row["月份"]?.trim();

        if (!yearRaw || !/^[1-9]\d{3}$/.test(yearRaw)) {
          return { error: "数据格式错误", status: 400 as const, details: { message: `产品 "${segment.productName}" 第 ${i + 1} 行年份格式错误` } };
        }
        if (!monthRaw || !/^(0?[1-9]|1[0-2])$/.test(monthRaw)) {
          return { error: "数据格式错误", status: 400 as const, details: { message: `产品 "${segment.productName}" 第 ${i + 1} 行月份格式错误` } };
        }

        const normalizedTime = normalizeYearMonth(yearRaw, monthRaw)!;
        const currentTime = formatYearMonth(normalizedTime);

        if (timePoints.has(currentTime)) {
          return { error: "数据重复", status: 400 as const, details: { message: `产品 "${segment.productName}" 存在重复的时间点: ${currentTime}` } };
        }
        timePoints.add(currentTime);

        if (previousTime && currentTime < formatYearMonth(previousTime)) {
          return { error: "时间顺序错误", status: 400 as const, details: { message: `产品 "${segment.productName}" 时间顺序颠倒` } };
        }
        if (previousTime && !isConsecutiveYearMonth(previousTime, normalizedTime)) {
          return { error: "时间不连续", status: 400 as const, details: { message: `产品 "${segment.productName}" 在 ${formatYearMonth(previousTime)} 与 ${currentTime} 之间存在缺失月份` } };
        }
        previousTime = normalizedTime;

        const sales = normalizeSalesValue(row["销售数量"]);
        if (!sales) {
          return { error: "数据格式错误", status: 400 as const, details: { message: `产品 "${segment.productName}" 第 ${i + 1} 行销售数量无效` } };
        }

        const unitRaw = (row["数量单位"] ?? "").trim();
        if (unitRaw !== expectedUnit) {
          return { error: "数据不一致", status: 400 as const, details: { message: `产品 "${segment.productName}" 第 ${i + 1} 行数量单位不一致` } };
        }

        // Segment files are the canonical model input. Keep the original upload
        // untouched for audit, but never persist presentation formatting here.
        row["年份"] = normalizedTime.year;
        row["月份"] = normalizedTime.month;
        row["销售数量"] = sales.canonical;
        row["数量单位"] = unitRaw;
      }
    }

    const connection = await pool.getConnection();
    try {
      await connection.beginTransaction();

      if (segmentsMap.size > 0) {
        const duplicateConditions = Array.from({ length: segmentsMap.size }).map(() => "(industry_name = ? AND company_name = ? AND product_name = ?)").join(" OR ");
        const duplicateValues = Array.from(segmentsMap.values()).flatMap((seg) => [seg.industryName, seg.companyName, seg.productName]);
        const [existing] = await connection.query<SegmentRow[]>(`SELECT industry_name, company_name, product_name FROM dataset_segments WHERE ${duplicateConditions}`, duplicateValues);

        if (existing.length > 0) {
          const preview = existing.slice(0, 5).map((row) => `${row.industry_name}/${row.company_name}/${row.product_name}`).join(", ");
          await connection.rollback();
          return { error: `数据库已经存在相同产品的销售数据: ${preview}${existing.length > 5 ? "..." : ""}`, status: 409 as const };
        }
      }

      const [result] = await connection.query<ResultSetHeader>("INSERT INTO datasets (data_name, file_path, description) VALUES (?, ?, ?)", [dataName, "placeholder", description || null]);
      const datasetId = result.insertId;

      const permanentStorageDir = path.resolve(uploadsRoot, "datasets");
      const uniquePermanentFilename = `${datasetId}-${file.name.replace(/[\/\\?%*:|"<>]/g, "-")}`;
      permanentOriginalPath = path.join(permanentStorageDir, uniquePermanentFilename);

      datasetDirectory = path.join(permanentStorageDir, String(datasetId));
      const segmentsDir = path.join(datasetDirectory, "segments");
      await mkdir(segmentsDir, { recursive: true });

      await Bun.write(permanentOriginalPath, Bun.file(tempUploadPath));
      await connection.query("UPDATE datasets SET file_path = ? WHERE dataset_id = ?", [permanentOriginalPath, datasetId]);

      if (segmentsMap.size > 0) {
        const segmentValues: unknown[] = [];
        for (const segment of segmentsMap.values()) {
          const segmentFileName = `${segment.companyName.replace(/[\/\\?%*:|"<>]/g, "-")}_${segment.productName.replace(/[\/\\?%*:|"<>]/g, "-")}.csv`;
          const segmentFilePath = path.join(segmentsDir, segmentFileName);
          await Bun.write(segmentFilePath, toCsvString(segment.rows));
          segmentValues.push(datasetId, segment.industryName, segment.companyName, segment.productName, segmentFilePath);
        }

        const segmentPlaceholders = Array.from({ length: segmentsMap.size }).map(() => "(?, ?, ?, ?, ?)").join(", ");
        await connection.query(`INSERT INTO dataset_segments (dataset_id, industry_name, company_name, product_name, file_path) VALUES ${segmentPlaceholders}`, segmentValues);
      }

      await connection.commit();

      const [rows] = await connection.query<DatasetRow[]>("SELECT * FROM datasets WHERE dataset_id = ?", [datasetId]);
      return { data: normalizeDatasetRow(rows[0]!) };
    } catch (dbError) {
      await connection.rollback().catch(() => {});
      if (datasetDirectory) await rm(datasetDirectory, { recursive: true, force: true }).catch(() => {});
      if (permanentOriginalPath) await rm(permanentOriginalPath, { force: true }).catch(() => {});
      if (isDuplicateDatasetSegmentError(dbError)) {
        return { error: "数据库已经存在相同产品的销售数据", status: 409 as const };
      }
      throw dbError;
    } finally {
      connection.release();
    }
  } catch (error) {
    if (error instanceof CsvEncodingError) {
      return {
        error: "文件格式错误",
        status: 400 as const,
        details: { message: error.message },
      };
    }
    if (error instanceof CsvRowLimitError) {
      return { error: error.message, status: 400 as const };
    }
    throw error;
  } finally {
    if (tempUploadPath) await unlink(tempUploadPath).catch(() => {});
  }
}

export async function getAllDatasets() {
  const [rows] = await pool.query<DatasetRow[]>("SELECT * FROM datasets ORDER BY uploaded_at DESC");
  return rows.map(normalizeDatasetRow);
}

export async function getDatasetById(datasetId: number) {
  const [rows] = await pool.query<DatasetRow[]>("SELECT * FROM datasets WHERE dataset_id = ?", [datasetId]);
  if (rows.length === 0) return null;
  return normalizeDatasetRow(rows[0]!);
}

export async function updateDataset(datasetId: number, input: UpdateDatasetInput) {
  const fields: string[] = [];
  const values: unknown[] = [];
  if (input.data_name) { fields.push("data_name = ?"); values.push(input.data_name); }
  if (input.description !== undefined) { fields.push("description = ?"); values.push(input.description); }
  if (fields.length === 0) return { error: "no_fields" as const };

  values.push(datasetId);
  const [result] = await pool.query<ResultSetHeader>(`UPDATE datasets SET ${fields.join(", ")} WHERE dataset_id = ?`, values);
  if (result.affectedRows === 0) return { error: "not_found" as const };

  const [rows] = await pool.query<DatasetRow[]>("SELECT * FROM datasets WHERE dataset_id = ?", [datasetId]);
  return { data: normalizeDatasetRow(rows[0]!) };
}

export async function deleteDataset(datasetId: number) {
  const [rows] = await pool.query<DatasetRow[]>("SELECT file_path FROM datasets WHERE dataset_id = ?", [datasetId]);
  if (rows.length === 0) return { error: "not_found" as const };

  await rm(rows[0]!.file_path, { force: true }).catch((err) => logWarning("Failed to delete dataset file", { error: String(err) }));
  const datasetSegmentsDirectory = path.resolve(uploadsRoot, "datasets", String(datasetId));
  await rm(datasetSegmentsDirectory, { recursive: true, force: true }).catch((err) => logWarning("Failed to delete segments dir", { error: String(err) }));

  const [result] = await pool.query<ResultSetHeader>("DELETE FROM datasets WHERE dataset_id = ?", [datasetId]);
  if (result.affectedRows === 0) return { error: "not_found" as const };

  return { success: true };
}

export async function getIndustries() {
  const [rows] = await pool.query<SegmentRow[]>("SELECT DISTINCT industry_name FROM dataset_segments ORDER BY industry_name ASC");
  return rows.map((row) => row.industry_name);
}

export async function getCompaniesByIndustry(industry: string) {
  const [rows] = await pool.query<SegmentRow[]>("SELECT DISTINCT company_name FROM dataset_segments WHERE industry_name = ? ORDER BY company_name ASC", [industry]);
  return rows.map((row) => row.company_name);
}

export async function getProductsByCompany(industry: string, company: string) {
  const [rows] = await pool.query<SegmentRow[]>("SELECT DISTINCT product_name FROM dataset_segments WHERE industry_name = ? AND company_name = ? ORDER BY product_name ASC", [industry, company]);
  return rows.map((row) => row.product_name);
}

export async function getProductFields(industry: string, company: string, product: string) {
  const [segments] = await pool.query<SegmentRow[]>("SELECT file_path FROM dataset_segments WHERE industry_name = ? AND company_name = ? AND product_name = ? LIMIT 1", [industry, company, product]);
  if (!segments[0]?.file_path) return { error: "not_found" as const };

  const filePath = segments[0].file_path;
  const fileExists = await Bun.file(filePath).exists();
  if (!fileExists) return { error: "file_missing" as const };

  const fields = (await getCsvHeaders(filePath))
    .map((h) => h.trim())
    .filter((h) => h.length > 0);

  return { data: fields };
}

export async function getProductSales(industry: string, company: string, product: string) {
  const [segments] = await pool.query<SegmentRow[]>("SELECT file_path FROM dataset_segments WHERE industry_name = ? AND company_name = ? AND product_name = ? LIMIT 1", [industry, company, product]);
  if (!segments[0]?.file_path) return { error: "not_found" as const };

  const filePath = segments[0].file_path;
  const fileExists = await Bun.file(filePath).exists();
  if (!fileExists) return { error: "file_missing" as const };

  const monthlySales: Array<{ month: string; sales: number }> = [];
  const csvData: string[][] = [];
  let headers: string[] = [];
  let unit = "";

  const rows = await readCsvRows(filePath);
  for (const [rowIndex, row] of rows.entries()) {
    if (headers.length === 0) { headers = Object.keys(row); csvData.push(headers); }
    csvData.push(headers.map((header) => String(row[header] ?? "").trim()));

    const time = normalizeYearMonth(row["年份"], row["月份"]);
    if (!time) {
      return {
        error: "invalid_data" as const,
        details: { row: rowIndex + 1, field: "年份/月份", message: `第 ${rowIndex + 1} 行年月格式无效` },
      };
    }

    const sales = normalizeSalesValue(row["销售数量"]);
    if (!sales) {
      return {
        error: "invalid_data" as const,
        details: { row: rowIndex + 1, field: "销售数量", message: `第 ${rowIndex + 1} 行销售数量无效` },
      };
    }

    if (!unit && row["数量单位"]) unit = String(row["数量单位"]).trim();
    monthlySales.push({ month: `${time.year}-${time.month}`, sales: sales.value });
  }

  return { data: { meta: { industry, company, product, name: product, description: "", unit }, monthlySales, csvData } };
}

export { FileValidationError };
