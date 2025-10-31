import fs from "fs";
import csv from "csv-parser";
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

/**
 * Efficiently reads only the headers from a CSV file.
 * @param filePath The absolute path to the CSV file.
 * @returns A Promise that resolves to an array of header strings.
 */
export function getCsvHeaders(filePath: string): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const stream = fs.createReadStream(filePath)
      .pipe(csv({ mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim() }));

    let headersEmitted = false;

    stream.on('headers', (headers: string[]) => {
      headersEmitted = true;
      stream.destroy(); // Stop reading the rest of the file
      resolve(headers);
    });

    stream.on('error', (error) => {
      reject(error);
    });

    // In case the file is empty and 'headers' event is not emitted
    stream.on('end', () => {
      if (!headersEmitted) {
        resolve([]);
      }
    });
  });
}

/**
 * Factory function to create a Hono middleware for validating file uploads.
 *
 * @param options - Configuration for the validator.
 * @param options.allowedMimeTypes - An array of allowed MIME types (e.g., ["text/csv", "application/pdf"]).
 * @param options.maxSizeMb - The maximum allowed file size in megabytes.
 * @returns A Hono middleware function.
 */
export const createUploadValidator = (options: {
  allowedMimeTypes: string[];
  maxSizeMb: number;
}) => {
  const { allowedMimeTypes, maxSizeMb } = options;
  const maxSizeBytes = maxSizeMb * 1024 * 1024;

  return async (c: any, next: any) => {
    try {
      const body = await c.req.parseBody();
      const file = body["file"];

      if (!(file instanceof File)) {
        return c.json({ error: "File is required" }, 400);
      }

      // 1. Validate file type
      if (!allowedMimeTypes.includes(file.type)) {
        return c.json(
          {
            error: "Invalid file type",
            details: `File must be one of type: ${allowedMimeTypes.join(", ")}. Got ${file.type}`,
          },
          400,
        );
      }

      // 2. Validate file size
      if (file.size > maxSizeBytes) {
        return c.json(
          {
            error: "File size exceeds the limit",
            details: `File size must be less than ${maxSizeMb}MB.`,
          },
          413, // Payload Too Large
        );
      }

      // Store the parsed body in the context to avoid re-parsing
      c.set("parsedBody", body);

      await next();
    } catch (error) {
      console.error("Error in upload validation middleware:", error);
      return c.json({ error: "Failed to process file upload" }, 500);
    }
  };
};


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

  const toolRoot = path.resolve(import.meta.dir, "..", "py_tools", "adf");
  const pythonExec =
    process.platform === "win32"
      ? path.join(toolRoot, "venv", "Scripts", "python.exe")
      : path.join(toolRoot, "venv", "bin", "python");

  const proc = Bun.spawn([pythonExec, "-m", "adf_tool", jsonArg, "--pretty"], {
    cwd: toolRoot,
    stdout: "pipe",
    stderr: "pipe",
  });

  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);

  const exitCode = await proc.exited;

  if (exitCode !== 0) {
    const message = stderr.trim() || `Process exited with code ${exitCode}`;
    throw new Error(`Failed to execute ADF tool: ${message}`);
  }

  try {
    const parsed = JSON.parse(stdout);
    return parsed as AdfResult;
  } catch {
    throw new Error(`Failed to parse ADF result: ${stdout}`);
  }
}
