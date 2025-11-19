import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import { rm } from "node:fs/promises";
import type { Pool } from "mysql2/promise";
import path from "path";
import { handleUnexpectedError } from "../utils/error-handler";
import { validateFile, generateUniqueFilename, FileValidationError } from "../utils/file-validator";

// Zod Schemas for Validation
const manualIdParamSchema = z.object({
  manualId: z.string().regex(/^\d+$/, { message: "manualId 必须是数字" }),
});

const updateManualSchema = z.object({
  file_name: z.string().min(1).optional(),
  description: z.string().optional(),
  is_active: z.boolean().optional(),
});

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

const normalizeManualRow = (row: any) => {
  if (!row) {
    return row;
  }
  return {
    ...row,
    file_path: toPublicFilePath(row.file_path),
  };
};

export default (pool: Pool, JWT_SECRET: string) => {
  const manualsApi = new Hono();

  // 管理员: 上传新的实验手册
  manualsApi.post("/api/v1/manuals", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin") {
        return c.json({ error: "权限不足" }, 403);
      }

      const body = await c.req.parseBody();
      const file = body["file"];
      const fileName = body["file_name"];
      const description = body["description"];
      const uploaderId = payload.sub;

      if (!(file instanceof File)) {
        return c.json({ error: "必须上传文件" }, 400);
      }

      if (!fileName || typeof fileName !== "string") {
        return c.json({ error: "必须提供文件名" }, 400);
      }

      // Comprehensive file validation
      try {
        await validateFile(file, "pdf", {
          maxSize: 50 * 1024 * 1024, // 50MB
          checkMagicBytes: true,
        });
      } catch (error) {
        if (error instanceof FileValidationError) {
          return c.json({ error: error.message }, 400);
        }
        throw error;
      }

      const uniqueFilename = generateUniqueFilename(file.name);
      const uploadPath = path.resolve(
        import.meta.dir,
        "../../uploads",
        "manuals",
        uniqueFilename,
      );

      await Bun.write(uploadPath, file);

      const [result]: any = await pool.query(
        "INSERT INTO experiment_manuals (file_name, file_path, description, uploader_id) VALUES (?, ?, ?, ?)",
        [fileName, uploadPath, description || null, uploaderId],
      );

      const [rows]: any[] = await pool.query(
        `SELECT m.*, u.full_name as uploader_name
           FROM experiment_manuals m
           LEFT JOIN users u ON m.uploader_id = u.user_id
           WHERE m.manual_id = ?`,
        [result.insertId],
      );

      return c.json(normalizeManualRow(rows[0]), 201);
    } catch (error) {
      return handleUnexpectedError(c, error, "上传手册失败");
    }
  });

  // 管理员: 获取所有实验手册列表
  manualsApi.get("/api/v1/manuals", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin") {
        return c.json({ error: "权限不足" }, 403);
      }
      const [rows] = await pool.query(`
          SELECT m.*, u.full_name as uploader_name
          FROM experiment_manuals m
          LEFT JOIN users u ON m.uploader_id = u.user_id
          ORDER BY m.uploaded_at DESC
      `);
      const normalized = (rows as any[]).map((row) => normalizeManualRow(row));
      return c.json(normalized);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取手册列表失败");
    }
  });

  // 任何人: 获取当前激活的实验手册
  manualsApi.get("/api/v1/manuals/active", async (c) => {
    try {
      const [rows]: any[] = await pool.query(`
          SELECT m.*, u.full_name as uploader_name
          FROM experiment_manuals m
          LEFT JOIN users u ON m.uploader_id = u.user_id
          WHERE m.is_active = true
          LIMIT 1
      `);
      const manual = rows[0];
      if (!manual) {
        return c.json({ error: "未找到激活的手册" }, 404);
      }
      return c.json(normalizeManualRow(manual));
    } catch (error) {
      return handleUnexpectedError(c, error, "获取激活手册失败");
    }
  });

  // 管理员: 更新实验手册信息 (名称, 描述, 激活状态)
  manualsApi.put(
    "/api/v1/manuals/:manualId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", manualIdParamSchema),
    zValidator("json", updateManualSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin") {
        return c.json({ error: "权限不足" }, 403);
      }

      const { manualId } = c.req.valid("param");
      const manualIdNumber = parseInt(manualId, 10);
      const { file_name, description, is_active } = c.req.valid("json");

      const connection = await pool.getConnection();
      try {
        await connection.beginTransaction();

        if (is_active === true) {
          await connection.query(
            "UPDATE experiment_manuals SET is_active = false WHERE manual_id != ?",
            [manualIdNumber],
          );
        }

        const fields = [];
        const values = [];
        if (file_name) {
          fields.push("file_name = ?");
          values.push(file_name);
        }
        if (description !== undefined) {
          fields.push("description = ?");
          values.push(description);
        }
        if (is_active !== undefined) {
          fields.push("is_active = ?");
          values.push(is_active);
        }

        if (fields.length === 0) {
          await connection.rollback();
          return c.json({ error: "没有需要更新的字段" }, 400);
        }

        values.push(manualIdNumber);
        const sql = `UPDATE experiment_manuals SET ${fields.join(", ")} WHERE manual_id = ?`;
        const [result]: any = await connection.query(sql, values);

        if (result.affectedRows === 0) {
          await connection.rollback();
          return c.json({ error: "手册不存在" }, 404);
        }

        await connection.commit();

        const [rows]: any[] = await connection.query(
          `SELECT m.*, u.full_name as uploader_name
           FROM experiment_manuals m
           LEFT JOIN users u ON m.uploader_id = u.user_id
           WHERE m.manual_id = ?`,
          [manualIdNumber],
        );
        return c.json(normalizeManualRow(rows[0]));
      } catch (error) {
        await connection.rollback();
        return handleUnexpectedError(c, error, "更新手册失败");
      } finally {
        connection.release();
      }
    },
  );

  // 管理员: 删除实验手册
  manualsApi.delete(
    "/api/v1/manuals/:manualId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", manualIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin") {
          return c.json({ error: "权限不足" }, 403);
        }

        const { manualId } = c.req.valid("param");
        const manualIdNumber = parseInt(manualId, 10);

        const [rows]: any[] = await pool.query(
          "SELECT file_path FROM experiment_manuals WHERE manual_id = ?",
          [manualIdNumber],
        );
        const manual = rows[0];

        if (!manual) {
          return c.json({ error: "手册不存在" }, 404);
        }

        // Use rm with force: true to avoid crashing if the file is already deleted
        await rm(manual.file_path, { force: true });

        await pool.query("DELETE FROM experiment_manuals WHERE manual_id = ?", [
          manualIdNumber,
        ]);

        return c.json({ message: "手册删除成功" });
      } catch (error) {
        return handleUnexpectedError(c, error, "删除手册失败");
      }
    },
  );

  return manualsApi;
};
