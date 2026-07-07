import type { RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import { CsvEncodingError, readCsvRows } from "../../utils/csv-processor";
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
    return await readCsvRows(segmentRows[0]!.file_path);
  } catch (error) {
    if (error instanceof CsvEncodingError) {
      throw new ModelValidationError(error.message);
    }
    throw error;
  }
}

