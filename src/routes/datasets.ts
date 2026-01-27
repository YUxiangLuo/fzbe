import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import { unlink, mkdir, rm } from "node:fs/promises";
import { createReadStream } from "node:fs";
import path from "path";
import type { Pool, ResultSetHeader, RowDataPacket } from "mysql2/promise";
import { parseCsvSegments, getCsvHeaders } from "../utils";
import csv from "csv-parser";
import { handleUnexpectedError, logWarning } from "../utils/error-handler";
import { validateFile, generateUniqueFilename, FileValidationError } from "../utils/file-validator";

// Helper to convert an array of objects to a CSV string
const toCsvString = (data: Record<string, string>[]): string => {
  if (data.length === 0) {
    return "";
  }
  const headers = Object.keys(data[0] ?? {});
  const headerRow = headers.join(",");
  const bodyRows = data.map((row) =>
    headers.map((header) => `"${row[header]}"`).join(","),
  );
  return [headerRow, ...bodyRows].join("\n");
};

const industryParamSchema = z.object({
  industry: z.string().min(1),
});

const industryAndCompanyParamSchema = z.object({
  industry: z.string().min(1),
  company: z.string().min(1),
});

const industryCompanyProductParamSchema = z.object({
  industry: z.string().min(1),
  company: z.string().min(1),
  product: z.string().min(1),
});

// Zod Schemas for Validation
const datasetIdParamSchema = z.object({
  datasetId: z
    .string()
    .regex(/^\d+$/, { message: "datasetId 必须是数字" }),
});

const updateDatasetSchema = z.object({
  data_name: z.string().min(1).optional(),
  description: z.string().optional(),
});

type DatasetRow = RowDataPacket & {
  dataset_id: number;
  data_name: string;
  file_path: string;
  description: string | null;
  uploaded_at: Date;
};

type DatasetFileRow = RowDataPacket & {
  file_path: string;
};

type SegmentIdentityRow = RowDataPacket & {
  industry_name: string;
  company_name: string;
  product_name: string;
};

type IndustryRow = RowDataPacket & {
  industry_name: string;
};

type CompanyRow = RowDataPacket & {
  company_name: string;
};

type ProductRow = RowDataPacket & {
  product_name: string;
};

type SegmentDetailRow = RowDataPacket & {
  file_path: string;
};

const isNodeError = (error: unknown): error is NodeJS.ErrnoException => {
  return typeof error === "object" && error !== null && "code" in error;
};

const uploadsRoot = path.resolve(import.meta.dir, "../../uploads");

const toPublicFilePath = (absolutePath: string | null | undefined) => {
  if (!absolutePath) {
    return null;
  }
  const relative = path.relative(uploadsRoot, absolutePath);
  if (!relative || relative.startsWith("..")) {
    return `uploads/${path.basename(absolutePath)}`.replace(/\\/g, "/");
  }
  return `uploads/${relative}`.replace(/\\/g, "/");
};

const normalizeDatasetRow = (row: any) => {
  if (!row) {
    return row;
  }
  return {
    ...row,
    file_path: toPublicFilePath(row.file_path),
  };
};

export default (pool: Pool, JWT_SECRET: string) => {
  const datasetsApi = new Hono();

  // 管理员: 上传新的数据集
  datasetsApi.post("/api/v1/datasets", jwt({ secret: JWT_SECRET }), async (c) => {
    let tempUploadPath: string | null = null;
    let datasetDirectory: string | null = null;
    let uploadSucceeded = false;

    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin") {
        return c.json({ error: "权限不足" }, 403);
      }

      const body = await c.req.parseBody();
      const file = body["file"];
      const dataName = body["data_name"];
      const description = body["description"];

      if (!(file instanceof File)) {
        return c.json({ error: "需要 CSV 文件" }, 400);
      }

      if (!dataName || typeof dataName !== "string") {
        return c.json({ error: "数据集名称不能为空" }, 400);
      }

      // Comprehensive file validation
      try {
        await validateFile(file, "csv", {
          maxSize: 50 * 1024 * 1024, // 50MB
          checkMagicBytes: false, // CSV has no reliable magic bytes
        });
      } catch (error) {
        if (error instanceof FileValidationError) {
          return c.json({ error: error.message }, 400);
        }
        throw error;
      }

      // 1. Save to a temporary location first
      const tempDir = path.resolve(import.meta.dir, "../../uploads", "temp");
      await mkdir(tempDir, { recursive: true });
      const uniqueFilename = generateUniqueFilename(file.name);
      tempUploadPath = path.join(tempDir, uniqueFilename);
      await Bun.write(tempUploadPath, file);

      // 1.5 Validate CSV headers
      const requiredHeaders = ["年份", "月份", "销售数量", "行业名称", "公司名称", "产品名称", "数量单位"];
      const headers = await getCsvHeaders(tempUploadPath);
      const missingHeaders = requiredHeaders.filter(h => !headers.includes(h));

      if (missingHeaders.length > 0) {
        return c.json({
          error: "文件格式错误",
          details: `上传的 CSV 文件缺少必要的列: ${missingHeaders.join(", ")}`,
          missing_headers: missingHeaders,
        }, 400);
      }

      // 2. Parse file in memory to get segments
      const segmentsMap = await parseCsvSegments(tempUploadPath);

      // 2.5 Robustness Check: Validate data consistency for each segment
      for (const [key, segment] of segmentsMap.entries()) {
        const timePoints = new Set<string>();
        let previousTime = "";
        const firstRow = segment.rows[0];

        if (!firstRow) {
          return c.json({
            error: "数据格式错误",
            details: `产品 "${segment.productName}" 无有效数据`,
          }, 400);
        }

        const expectedUnit = (firstRow["数量单位"] ?? "").trim();

        for (const [i, row] of segment.rows.entries()) {
          const yearRaw = row["年份"]?.trim();
          const monthRaw = row["月份"]?.trim();

          // Check 1: Valid numeric year/month
          if (!yearRaw || !/^\d{4}$/.test(yearRaw)) {
            return c.json({
              error: "数据格式错误",
              details: `产品 "${segment.productName}" 第 ${i + 1} 行年份格式错误: "${yearRaw}" (需为4位数字)`,
            }, 400);
          }
          // Simple month validation: allows 1-12, 01-12
          if (!monthRaw || !/^(0?[1-9]|1[0-2])$/.test(monthRaw)) {
            return c.json({
              error: "数据格式错误",
              details: `产品 "${segment.productName}" 第 ${i + 1} 行月份格式错误: "${monthRaw}" (需为1-12)`,
            }, 400);
          }

        const normalizedMonth = monthRaw.padStart(2, "0");
          const currentTime = `${yearRaw}-${normalizedMonth}`;

          // Check 2: Duplicate time points
          if (timePoints.has(currentTime)) {
             return c.json({
              error: "数据重复",
              details: `产品 "${segment.productName}" 存在重复的时间点: ${currentTime}`,
            }, 400);
          }
          timePoints.add(currentTime);

          // Check 3: Time reversal (strictly increasing check, though skipping months is allowed)
          if (previousTime && currentTime < previousTime) {
             return c.json({
              error: "时间顺序错误",
              details: `产品 "${segment.productName}" 时间顺序颠倒: ${currentTime} 出现在 ${previousTime} 之后`,
            }, 400);
          }
          previousTime = currentTime;

          // Check 4: Valid Sales Quantity
          const salesRaw = row["销售数量"];
          // Remove commas for validation (e.g. "1,200" -> "1200")
          const salesCleaned = salesRaw ? salesRaw.replace(/,/g, "").trim() : "";
          const salesNum = Number(salesCleaned);
          
          if (salesCleaned === "" || Number.isNaN(salesNum) || salesNum < 0) {
            return c.json({
              error: "数据格式错误",
              details: `产品 "${segment.productName}" 第 ${i + 1} 行销售数量无效: "${salesRaw}" (需为非负数字)`,
            }, 400);
          }

          // Check 5: Unit Consistency
          const unitRaw = (row["数量单位"] ?? "").trim();
          
          if (unitRaw !== expectedUnit) {
              return c.json({
              error: "数据不一致",
              details: `产品 "${segment.productName}" 第 ${i + 1} 行数量单位 "${unitRaw}" 与首行单位 "${expectedUnit}" 不一致`,
            }, 400);
          }
        }
      }

      const connection = await pool.getConnection();
      try {
        await connection.beginTransaction();

        // 3. Check for duplicates before proceeding
        if (segmentsMap.size > 0) {
          const duplicateConditions = Array.from({ length: segmentsMap.size })
            .map(
              () =>
                "(industry_name = ? AND company_name = ? AND product_name = ?)",
            )
            .join(" OR ");
          const duplicateValues = Array.from(segmentsMap.values()).flatMap(
            (seg) => [seg.industryName, seg.companyName, seg.productName],
          );
          const [existing] = await connection.query<SegmentIdentityRow[]>(
            `SELECT industry_name, company_name, product_name FROM dataset_segments WHERE ${duplicateConditions}`,
            duplicateValues,
          );

          if (existing.length > 0) {
            const preview = existing
              .slice(0, 5)
              .map(
                (row) =>
                  `${row.industry_name}/${row.company_name}/${row.product_name}`,
              )
              .join(", ");
            const suffix = existing.length > 5 ? "..." : "";
            await connection.rollback();
            return c.json(
              {
                error: `数据库已经存在相同产品的销售数据: ${preview}${suffix}`,
              },
              409,
            );
          }
        }

        // 4. Insert into datasets table to get an ID
        const [result] = await connection.query<ResultSetHeader>(
          "INSERT INTO datasets (data_name, file_path, description) VALUES (?, ?, ?)",
          [dataName, "placeholder", description || null], // Use a placeholder path for now
        );
        const datasetId: number = result.insertId;

        // 5. Create dedicated directories for the dataset and its segments
        const permanentStorageDir = path.resolve(
          import.meta.dir,
          "../../uploads",
          "datasets",
        );
        const uniquePermanentFilename = `${datasetId}-${file.name.replace(/[\/\\?%*:|"<>]/g, "-")}`;
        const permanentOriginalPath = path.join(
          permanentStorageDir,
          uniquePermanentFilename,
        );

        datasetDirectory = path.join(permanentStorageDir, String(datasetId));
        const segmentsDir = path.join(datasetDirectory, "segments");
        await mkdir(segmentsDir, { recursive: true });

        // 6. Move original file to its permanent location and update the DB record
        await Bun.write(permanentOriginalPath, Bun.file(tempUploadPath));
        await connection.query(
          "UPDATE datasets SET file_path = ? WHERE dataset_id = ?",
          [permanentOriginalPath, datasetId],
        );

        // 7. Write each segment to its own CSV file and prepare for batch insert
        if (segmentsMap.size > 0) {
          const segmentValues = [];
          for (const segment of segmentsMap.values()) {
            const segmentFileName = `${segment.companyName.replace(/[\/\\?%*:|"<>]/g, "-")}_${segment.productName.replace(/[\/\\?%*:|"<>]/g, "-")}.csv`;
            const segmentFilePath = path.join(segmentsDir, segmentFileName);
            const csvContent = toCsvString(segment.rows);
            await Bun.write(segmentFilePath, csvContent);

            segmentValues.push(
              datasetId,
              segment.industryName,
              segment.companyName,
              segment.productName,
              segmentFilePath, // Path to the new segment file
            );
          }

          const segmentPlaceholders = Array.from({ length: segmentsMap.size })
            .map(() => "(?, ?, ?, ?, ?)")
            .join(", ");

          await connection.query(
            `INSERT INTO dataset_segments (dataset_id, industry_name, company_name, product_name, file_path) VALUES ${segmentPlaceholders}`,
            segmentValues,
          );
        }

        await connection.commit();

        const [rows] = await connection.query<DatasetRow[]>(
          "SELECT * FROM datasets WHERE dataset_id = ?",
          [datasetId],
        );
        uploadSucceeded = true;
        return c.json(normalizeDatasetRow(rows[0]), 201);
      } catch (dbError) {
        try {
          await connection.rollback();
        } catch (rollbackError) {
          logWarning("Rollback failed during dataset upload", {
            error:
              rollbackError instanceof Error
                ? rollbackError.message
                : String(rollbackError),
          });
        }

        if (datasetDirectory) {
          await rm(datasetDirectory, { recursive: true, force: true }).catch(
            (err) =>
              logWarning("Failed to cleanup dataset directory", {
                error: err instanceof Error ? err.message : String(err),
              }),
          );
        }

        return handleUnexpectedError(c, dbError, "上传数据集失败");
      } finally {
        connection.release();
      }
    } catch (error) {
      return handleUnexpectedError(c, error, "上传数据集失败");
    } finally {
      // Always cleanup the temporary file
      if (!uploadSucceeded && datasetDirectory) {
        await rm(datasetDirectory, { recursive: true, force: true }).catch(
          (err) =>
            logWarning("Failed to cleanup dataset directory", {
              error: err instanceof Error ? err.message : String(err),
            }),
        );
      }

      if (tempUploadPath) {
        await unlink(tempUploadPath).catch((err) =>
          logWarning("Failed to cleanup temp file", {
            error: err instanceof Error ? err.message : String(err),
          }),
        );
      }
    }
  });

  // 登录用户: 获取所有数据集列表
  datasetsApi.get("/api/v1/datasets", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const [rows] = await pool.query<DatasetRow[]>(
        "SELECT * FROM datasets ORDER BY uploaded_at DESC",
      );
      const normalized = (rows as any[]).map((row) => normalizeDatasetRow(row));
      return c.json(normalized);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取数据集列表失败");
    }
  });

  // 登录用户: 获取行业/公司/产品层级信息 - needs to be registered before /datasets/:datasetId
  datasetsApi.get("/api/v1/datasets/industries", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const [rows] = await pool.query<IndustryRow[]>(
        "SELECT DISTINCT industry_name FROM dataset_segments ORDER BY industry_name ASC",
      );
      return c.json(rows.map((row) => row.industry_name));
    } catch (error) {
      return handleUnexpectedError(c, error, "获取行业列表失败");
    }
  });

  datasetsApi.get(
    "/api/v1/datasets/industries/:industry/companies",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", industryParamSchema),
    async (c) => {
      try {
        const { industry } = c.req.valid("param");
        const [rows] = await pool.query<CompanyRow[]>(
          "SELECT DISTINCT company_name FROM dataset_segments WHERE industry_name = ? ORDER BY company_name ASC",
          [industry],
        );
        return c.json(rows.map((row) => row.company_name));
      } catch (error) {
        return handleUnexpectedError(c, error, "获取公司列表失败");
      }
    },
  );

  datasetsApi.get(
    "/api/v1/datasets/industries/:industry/companies/:company/products",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", industryAndCompanyParamSchema),
    async (c) => {
      try {
        const { industry, company } = c.req.valid("param");
        const [rows] = await pool.query<ProductRow[]>(
          `SELECT DISTINCT product_name
             FROM dataset_segments
            WHERE industry_name = ? AND company_name = ?
            ORDER BY product_name ASC`,
          [industry, company],
        );
        return c.json(rows.map((row) => row.product_name));
      } catch (error) {
        return handleUnexpectedError(c, error, "获取产品列表失败");
      }
    },
  );

  datasetsApi.get(
    "/api/v1/datasets/industries/:industry/companies/:company/products/:product/fields",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", industryCompanyProductParamSchema),
    async (c) => {
      try {
        const { industry, company, product } = c.req.valid("param");

        const [segments] = await pool.query<SegmentDetailRow[]>(
          `SELECT file_path
             FROM dataset_segments
            WHERE industry_name = ? AND company_name = ? AND product_name = ?
            LIMIT 1`,
          [industry, company, product],
        );

        const segment = segments[0];
        if (!segment) {
          return c.json({ error: "未找到产品分段" }, 404);
        }

        const { file_path: filePath } = segment;
        const fileExists = await Bun.file(filePath).exists();
        if (!fileExists) {
          logWarning("Product segment file missing on disk", { filePath });
          return c.json({ error: "产品分段文件丢失" }, 404);
        }

        const fields = await new Promise<string[]>((resolve, reject) => {
          let settled = false;
          const parser = createReadStream(filePath).pipe(
            csv({
              mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim(),
            }),
          );

          parser
            .once("headers", (headers: string[]) => {
              settled = true;
              // Immediately resolve with trimmed headers; destroy stream to stop further processing
              resolve(
                headers
                  .map((header) => header.trim())
                  .filter((header) => header.length > 0),
              );
              parser.destroy();
            })
            .on("end", () => {
              if (!settled) {
                settled = true;
                resolve([]);
              }
            })
            .on("error", (err) => {
              if (!settled) {
                settled = true;
                reject(err);
              }
            });
        });

        return c.json({ fields });
      } catch (error) {
        return handleUnexpectedError(c, error, "获取产品字段失败");
      }
    },
  );

  datasetsApi.get(
    "/api/v1/datasets/industries/:industry/companies/:company/products/:product/sales",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", industryCompanyProductParamSchema),
    async (c) => {
      try {
        const { industry, company, product } = c.req.valid("param");

        const [segments] = await pool.query<SegmentDetailRow[]>(
          `SELECT file_path
             FROM dataset_segments
            WHERE industry_name = ? AND company_name = ? AND product_name = ?
            LIMIT 1`,
          [industry, company, product],
        );

        const segment = segments[0];
        if (!segment) {
          return c.json({ error: "未找到产品分段" }, 404);
        }

        const { file_path: filePath } = segment;

        // Gracefully handle cases where the file is missing from the filesystem
        const fileExists = await Bun.file(filePath).exists();
        if (!fileExists) {
          logWarning("Product sales data file missing on disk", { filePath });
          return c.json({ error: "产品销售数据文件丢失" }, 404);
        }

        const monthlySales: Array<{ month: string; sales: number }> = [];
        const csvData: string[][] = [];
        let headers: string[] = [];
        let unit = "";

        const normalizeYear = (value: unknown) => {
          const raw = String(value ?? "").trim();
          const match = raw.match(/\d{4}/);
          return match ? match[0] : raw;
        };

        const normalizeMonth = (value: unknown) => {
          const raw = String(value ?? "").trim();
          if (!raw) return raw;
          const match = raw.match(/\d+/);
          if (!match) return raw.padStart(2, "0");
          const monthNumber = Number.parseInt(match[0], 10);
          if (Number.isNaN(monthNumber)) {
            return raw;
          }
          return String(monthNumber).padStart(2, "0");
        };

        const parseSales = (value: unknown) => {
          const raw = String(value ?? "").trim();
          if (!raw) return Number.NaN;
          const cleaned = raw.replace(/,/g, "");
          return Number(cleaned);
        };

        await new Promise<void>((resolve, reject) => {
          const stream = createReadStream(filePath).pipe(
            csv({
              mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim(),
            }),
          );

          let isFirstRow = true;

          stream
            .on("data", (row: Record<string, unknown>) => {
              try {
                // 第一行：获取列名并存储到csvData
                if (isFirstRow) {
                  headers = Object.keys(row);
                  csvData.push(headers);
                  isFirstRow = false;
                }

                // 按照列名顺序提取值，组成数据行
                const dataRow = headers.map((header) => String(row[header] ?? "").trim());
                csvData.push(dataRow);

                // 继续处理月度销量数据
                const year = normalizeYear(row["年份"]);
                const month = normalizeMonth(row["月份"]);

                if (!year || !month) return;

                const salesValue = parseSales(row["销售数量"]);
                if (!Number.isFinite(salesValue)) return;

                if (!unit && row["数量单位"]) {
                  unit = String(row["数量单位"]).trim();
                }

                monthlySales.push({ month: `${year}-${month}`, sales: salesValue });
              } catch (processingError) {
                // Manually destroy the stream on a sync error, which will trigger the 'error' event
                stream.destroy(
                  processingError instanceof Error
                    ? processingError
                    : new Error(String(processingError)),
                );
              }
            })
            .on("end", () => resolve())
            .on("error", reject);
        });

        return c.json({
          meta: {
            industry,
            company,
            product,
            name: product,
            description: "",
            unit,
          },
          monthlySales,
          csvData,
        });
      } catch (error) {
        return handleUnexpectedError(c, error, "获取产品销售数据失败");
      }
    },
  );

  // 登录用户: 获取单个数据集
  datasetsApi.get(
    "/api/v1/datasets/:datasetId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", datasetIdParamSchema),
    async (c) => {
      try {
        const { datasetId } = c.req.valid("param");
        const [rows] = await pool.query<DatasetRow[]>(
          "SELECT * FROM datasets WHERE dataset_id = ?",
          [datasetId],
        );
        if (rows.length === 0) {
          return c.json({ error: "数据集不存在" }, 404);
        }
        return c.json(normalizeDatasetRow(rows[0]));
      } catch (error) {
        return handleUnexpectedError(c, error, "获取数据集失败");
      }
    },
  );

  // 管理员: 更新数据集信息
  datasetsApi.put(
    "/api/v1/datasets/:datasetId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", datasetIdParamSchema),
    zValidator("json", updateDatasetSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin") {
          return c.json({ error: "权限不足" }, 403);
        }

        const { datasetId } = c.req.valid("param");
        const { data_name, description } = c.req.valid("json");

        const fields = [];
        const values = [];
        if (data_name) {
          fields.push("data_name = ?");
          values.push(data_name);
        }
        if (description !== undefined) {
          fields.push("description = ?");
          values.push(description);
        }

        if (fields.length === 0) {
          return c.json({ error: "没有需要更新的字段" }, 400);
        }

        values.push(datasetId);
        const sql = `UPDATE datasets SET ${fields.join(", ")} WHERE dataset_id = ?`;
        const [result] = await pool.query<ResultSetHeader>(sql, values);

        if (result.affectedRows === 0) {
          return c.json({ error: "数据集不存在" }, 404);
        }

        const [rows] = await pool.query<DatasetRow[]>(
          "SELECT * FROM datasets WHERE dataset_id = ?",
          [datasetId],
        );
        return c.json(normalizeDatasetRow(rows[0]));
      } catch (error) {
        return handleUnexpectedError(c, error, "更新数据集失败");
      }
    },
  );

  // 管理员: 删除数据集
  datasetsApi.delete(
    "/api/v1/datasets/:datasetId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", datasetIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin") {
          return c.json({ error: "权限不足" }, 403);
        }

        const { datasetId } = c.req.valid("param");

        const [rows] = await pool.query<DatasetFileRow[]>(
          "SELECT file_path FROM datasets WHERE dataset_id = ?",
          [datasetId],
        );
        const dataset = rows[0];

        if (!dataset) {
          return c.json({ error: "数据集不存在" }, 404);
        }

        // Delete the original file itself
        await rm(dataset.file_path, { force: true, recursive: false }).catch((err) =>
          logWarning("Failed to delete dataset file", {
            filePath: dataset.file_path,
            error: err instanceof Error ? err.message : String(err),
          }),
        );

        // Delete the associated directory which contains segment files
        const datasetSegmentsDirectory = path.resolve(
          import.meta.dir,
          "../../uploads",
          "datasets",
          datasetId,
        );
        await rm(datasetSegmentsDirectory, { recursive: true, force: true }).catch(
          (err) =>
            logWarning("Failed to delete dataset segments directory", {
              directory: datasetSegmentsDirectory,
              error: err instanceof Error ? err.message : String(err),
            }),
        );

        const [result] = await pool.query<ResultSetHeader>(
          "DELETE FROM datasets WHERE dataset_id = ?",
          [datasetId],
        );

        if (result.affectedRows === 0) {
          // This case should ideally not be reached if the directory was found
          return c.json({ error: "数据库中未找到数据集" }, 404);
        }

        return c.json({ message: "数据集及所有相关文件删除成功" });
      } catch (error) {
        return handleUnexpectedError(c, error, "删除数据集失败");
      }
    },
  );

  return datasetsApi;
};
