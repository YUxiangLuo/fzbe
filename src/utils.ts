import fs from "fs";
import csv from "csv-parser";
import { $ } from "bun";
import path from "path";

// Represents a segment of data for a specific product, including its metadata and all its data rows.
export interface SegmentData {
  industryName: string;
  companyName: string;
  productName: string;
  rows: Record<string, string>[]; // All data rows for this segment
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

    fs.createReadStream(filePath)
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

// 输入：时间序列的单个元素
export interface AdfInputPoint {
  /** 月份（或时间标签），字符串格式如 "2021-02" */
  month: string;
  /** 销售额或任意数值型序列 */
  sales: number;
}

// 输出：单个差分阶（d=0,1,2）对应的结果
export interface AdfResultItem {
  /** 差分阶数 */
  diff_order: number;
  /** ADF 统计量 */
  statistic: number | null;
  /** p 值 */
  p_value: number | null;
  /** 是否平稳 */
  stationary: boolean | null;
  /** 临界值 */
  critical_values: {
    "1%": number | null;
    "5%": number | null;
    "10%": number | null;
  } | null;
  /** 若出现错误（例如序列过短） */
  error?: string;
}

// 输出整体数组
export type AdfResult = AdfResultItem[];

/**
 * 调用 Python ADF 工具，执行固定 d=0,1,2 三阶差分检验。
 *
 * @param data 由 {month, sales} 对象组成的数组（推荐长度 ≥ 10）
 * @returns Promise<AdfResult> ADF 检验结果
 */
export async function runAdf(data: AdfInputPoint[]): Promise<AdfResult> {
  // 序列化成 JSON，确保 key 带双引号
  const jsonArg = JSON.stringify(data);

  const root_dir = path.resolve(import.meta.dir, "..");
  console.log(root_dir)

  // 执行 Python 命令（在 venv 中运行）
  // const proc = await $`cd ${root_dir}/py_tools/adf && ./venv/bin/python -m adf_tool '${jsonArg}' --pretty`.quiet();
  const proc = await $`cd ${root_dir}/py_tools/adf && ./venv/bin/python -m adf_tool '${jsonArg}' --pretty`;


  const td = new TextDecoder();
  const text = td.decode(proc.stdout);

  try {
    const parsed = JSON.parse(text);
    return parsed as AdfResult;
  } catch {
    // 如果返回非 JSON（例如 Python 报错）
    throw new Error(`Failed to parse ADF result: ${text}`);
  }
}
