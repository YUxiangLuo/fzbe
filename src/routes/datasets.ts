import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import { unlink } from "node:fs/promises";
import { join } from "node:path";
import path from "path";
import type { Pool } from "mysql2/promise";

// Zod Schemas for Validation
const datasetIdParamSchema = z.object({
  datasetId: z.string().regex(/^\d+$/, { message: "datasetId must be a number" }),
});

const updateDatasetSchema = z.object({
  data_name: z.string().min(1).optional(),
  description: z.string().optional(),
});

export default (pool: Pool, JWT_SECRET: string) => {
  const datasetsApi = new Hono();

  // 管理员: 上传新的数据集
  datasetsApi.post("/datasets", jwt({ secret: JWT_SECRET }), async (c) => {
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
        return c.json({ error: "File is required" }, 400);
      }
      if (!dataName || typeof dataName !== "string") {
        return c.json({ error: "data_name is required" }, 400);
      }

      const uniqueFilename = `${Date.now()}-${file.name}`;
      const uploadPath = path.resolve(
        import.meta.dir,
        "../../uploads",
        "static",
        uniqueFilename,
      );

      await Bun.write(uploadPath, file);

      const [result]: any = await pool.query(
        "INSERT INTO datasets (data_name, file_path, description) VALUES (?, ?, ?)",
        [dataName, uploadPath, description || null],
      );

      const [rows]: any[] = await pool.query(
        "SELECT * FROM datasets WHERE dataset_id = ?",
        [result.insertId],
      );

      // Return the complete object, which now includes `uploaded_at`
      return c.json(rows[0], 201);
    } catch (error) {
      console.error("Error uploading dataset:", error);
      throw error;
    }
  });

  // 登录用户: 获取所有数据集列表
  datasetsApi.get("/datasets", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const [rows] = await pool.query(
        "SELECT * FROM datasets ORDER BY uploaded_at DESC",
      );
      return c.json(rows);
    } catch (error) {
      console.error("Error fetching datasets:", error);
      throw error;
    }
  });

  // 登录用户: 获取单个数据集
  datasetsApi.get(
    "/datasets/:datasetId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", datasetIdParamSchema),
    async (c) => {
      try {
        const { datasetId } = c.req.valid("param");
        const [rows]: any[] = await pool.query(
          "SELECT * FROM datasets WHERE dataset_id = ?",
          [datasetId],
        );
        if (rows.length === 0) {
          return c.json({ error: "Dataset not found" }, 404);
        }
        return c.json(rows[0]);
      } catch (error) {
        console.error("Error fetching dataset:", error);
        throw error;
      }
    },
  );

  // 管理员: 更新数据集信息
  datasetsApi.put(
    "/datasets/:datasetId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", datasetIdParamSchema),
    zValidator("json", updateDatasetSchema),
    async (c) => {
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
      const [result]: any = await pool.query(sql, values);

      if (result.affectedRows === 0) {
        return c.json({ error: "Dataset not found" }, 404);
      }

      const [rows]: any[] = await pool.query(
        "SELECT * FROM datasets WHERE dataset_id = ?",
        [datasetId],
      );
      return c.json(rows[0]);
    },
  );

  // 管理员: 删除数据集
  datasetsApi.delete(
    "/datasets/:datasetId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", datasetIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { datasetId } = c.req.valid("param");

        const [rows]: any[] = await pool.query(
          "SELECT file_path FROM datasets WHERE dataset_id = ?",
          [datasetId],
        );
        const dataset = rows[0];

        if (!dataset) {
          return c.json({ error: "Dataset not found" }, 404);
        }

        try {
          await unlink(dataset.file_path);
        } catch (fsError: any) {
          if (fsError.code !== "ENOENT") {
            throw fsError;
          }
          console.warn(
            `File not found, but proceeding with DB deletion: ${dataset.file_path}`,
          );
        }

        await pool.query("DELETE FROM datasets WHERE dataset_id = ?", [datasetId]);

        return c.json({ message: "Dataset deleted successfully" });
      } catch (error) {
        console.error("Error deleting dataset:", error);
        throw error;
      }
    },
  );

  return datasetsApi;
};
