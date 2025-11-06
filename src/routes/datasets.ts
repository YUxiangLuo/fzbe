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
    .regex(/^\d+$/, { message: "datasetId must be a number" }),
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
        return c.json({ error: "Forbidden" }, 403);
      }

      const body = await c.req.parseBody();
      const file = body["file"];
      const dataName = body["data_name"];
      const description = body["description"];

      if (!(file instanceof File)) {
        return c.json({ error: "CSV file is required" }, 400);
      }

      if (
        file.type &&
        file.type !== "text/csv" &&
        !file.type.includes("csv")
      ) {
        return c.json({ error: "Invalid file type, only CSV is allowed" }, 400);
      }

      const maxSizeBytes = 50 * 1024 * 1024;
      if (file.size > maxSizeBytes) {
        return c.json({ error: "CSV file size cannot exceed 50MB" }, 413);
      }

      if (!dataName || typeof dataName !== "string") {
        return c.json({ error: "data_name is required" }, 400);
      }

      // 1. Save to a temporary location first
      const tempDir = path.resolve(import.meta.dir, "../../uploads", "temp");
      await mkdir(tempDir, { recursive: true });
      const uniqueFilename = `${Date.now()}-${file.name}`;
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
            const segmentFileName = `${segment.productName.replace(/[\/\\?%*:|"<>]/g, "-")}.csv`;
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

        return handleUnexpectedError(c, dbError, "Failed to upload dataset");
      } finally {
        connection.release();
      }
    } catch (error) {
      return handleUnexpectedError(c, error, "Failed to upload dataset");
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
      return handleUnexpectedError(c, error, "Failed to fetch datasets");
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
      return handleUnexpectedError(c, error, "Failed to fetch industries");
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
        return handleUnexpectedError(c, error, "Failed to fetch companies");
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
        return handleUnexpectedError(c, error, "Failed to fetch products");
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
          return c.json({ error: "Product segment not found" }, 404);
        }

        const { file_path: filePath } = segment;
        const fileExists = await Bun.file(filePath).exists();
        if (!fileExists) {
          logWarning("Product segment file missing on disk", { filePath });
          return c.json({ error: "Product segment file is missing" }, 404);
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
        return handleUnexpectedError(c, error, "Failed to fetch product fields");
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
          return c.json({ error: "Product segment not found" }, 404);
        }

        const { file_path: filePath } = segment;

        // Gracefully handle cases where the file is missing from the filesystem
        const fileExists = await Bun.file(filePath).exists();
        if (!fileExists) {
          logWarning("Product sales data file missing on disk", { filePath });
          return c.json({ error: "Product sales data file is missing" }, 404);
        }

        const monthlySales: Array<{ month: string; sales: number }> = [];
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

          stream
            .on("data", (row: Record<string, unknown>) => {
              try {
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
        });
      } catch (error) {
        return handleUnexpectedError(c, error, "Failed to fetch product sales data");
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
          return c.json({ error: "Dataset not found" }, 404);
        }
        return c.json(normalizeDatasetRow(rows[0]));
      } catch (error) {
        return handleUnexpectedError(c, error, "Failed to fetch dataset");
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
          return c.json({ error: "Forbidden" }, 403);
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
          return c.json({ error: "No fields to update" }, 400);
        }

        values.push(datasetId);
        const sql = `UPDATE datasets SET ${fields.join(", ")} WHERE dataset_id = ?`;
        const [result] = await pool.query<ResultSetHeader>(sql, values);

        if (result.affectedRows === 0) {
          return c.json({ error: "Dataset not found" }, 404);
        }

        const [rows] = await pool.query<DatasetRow[]>(
          "SELECT * FROM datasets WHERE dataset_id = ?",
          [datasetId],
        );
        return c.json(normalizeDatasetRow(rows[0]));
      } catch (error) {
        return handleUnexpectedError(c, error, "Failed to update dataset");
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
          return c.json({ error: "Forbidden" }, 403);
        }

        const { datasetId } = c.req.valid("param");

        const [rows] = await pool.query<DatasetFileRow[]>(
          "SELECT file_path FROM datasets WHERE dataset_id = ?",
          [datasetId],
        );
        const dataset = rows[0];

        if (!dataset) {
          return c.json({ error: "Dataset not found" }, 404);
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
          return c.json({ error: "Dataset not found in database" }, 404);
        }

        return c.json({ message: "Dataset and all associated files deleted successfully" });
      } catch (error) {
        return handleUnexpectedError(c, error, "Failed to delete dataset");
      }
    },
  );

  return datasetsApi;
};
