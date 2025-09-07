import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import { unlink } from "node:fs/promises";
import { join } from "node:path";
import type { Pool } from "mysql2/promise";

// Zod Schemas for Validation
const paramsSchema = z.object({
  id: z.string().regex(/^\d+$/, { message: "ID must be a number" }),
});

const updateManualSchema = z.object({
  file_name: z.string().min(1).optional(),
  description: z.string().optional(),
  is_active: z.boolean().optional(),
});

export default (pool: Pool, JWT_SECRET: string) => {
  const manualsApi = new Hono();

  // 管理员: 上传新的实验手册
  manualsApi.post("/", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin") {
        return c.json({ error: "Forbidden" }, 403);
      }

      const body = await c.req.parseBody();
      const file = body["file"];
      const fileName = body["file_name"];
      const description = body["description"];
      const uploaderId = payload.sub;

      if (!(file instanceof File)) {
        return c.json({ error: "File is required" }, 400);
      }
      if (!fileName || typeof fileName !== "string") {
        return c.json({ error: "file_name is required" }, 400);
      }

      const uniqueFilename = `${Date.now()}-${file.name}`;
      const uploadPath = join(import.meta.dir, "../../uploads", "manuals", uniqueFilename);
      const dbPath = `/uploads/manuals/${uniqueFilename}`;

      await Bun.write(uploadPath, file);

      const [result]: any = await pool.query(
        "INSERT INTO experiment_manuals (file_name, file_path, description, uploader_id) VALUES (?, ?, ?, ?)",
        [fileName, dbPath, description || null, uploaderId],
      );

      const [rows]: any[] = await pool.query(
          `SELECT m.*, u.full_name as uploader_name 
           FROM experiment_manuals m 
           LEFT JOIN users u ON m.uploader_id = u.user_id 
           WHERE m.manual_id = ?`,
          [result.insertId]
      );

      return c.json(rows[0], 201);
    } catch (error) {
      console.error("Error uploading manual:", error);
      throw error;
    }
  });

  // 管理员: 获取所有实验手册列表
  manualsApi.get("/", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin") {
        return c.json({ error: "Forbidden" }, 403);
      }
      const [rows] = await pool.query(`
          SELECT m.*, u.full_name as uploader_name 
          FROM experiment_manuals m 
          LEFT JOIN users u ON m.uploader_id = u.user_id 
          ORDER BY m.uploaded_at DESC
      `);
      return c.json(rows);
    } catch (error) {
      console.error("Error fetching manuals:", error);
      throw error;
    }
  });

  // 任何人: 获取当前激活的实验手册
  manualsApi.get("/active", async (c) => {
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
        return c.json({ error: "No active manual found" }, 404);
      }
      return c.json(manual);
    } catch (error) {
      console.error("Error fetching active manual:", error);
      throw error;
    }
  });

  // 管理员: 更新实验手册信息 (名称, 描述, 激活状态)
  manualsApi.put(
    "/:id",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", paramsSchema),
    zValidator("json", updateManualSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin") {
        return c.json({ error: "Forbidden" }, 403);
      }

      const { id } = c.req.valid("param");
      const manualId = parseInt(id, 10);
      const { file_name, description, is_active } = c.req.valid("json");

      const connection = await pool.getConnection();
      try {
        await connection.beginTransaction();

        if (is_active === true) {
          await connection.query(
            "UPDATE experiment_manuals SET is_active = false WHERE manual_id != ?",
            [manualId],
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
          return c.json({ error: "No fields to update" }, 400);
        }

        values.push(manualId);
        const sql = `UPDATE experiment_manuals SET ${fields.join(", ")} WHERE manual_id = ?`;
        const [result]: any = await connection.query(sql, values);

        if (result.affectedRows === 0) {
          await connection.rollback();
          return c.json({ error: "Manual not found" }, 404);
        }

        await connection.commit();

        const [rows]: any[] = await connection.query(
          `SELECT m.*, u.full_name as uploader_name 
           FROM experiment_manuals m 
           LEFT JOIN users u ON m.uploader_id = u.user_id 
           WHERE m.manual_id = ?`,
          [manualId],
        );
        return c.json(rows[0]);
      } catch (error) {
        await connection.rollback();
        console.error("Error updating manual:", error);
        throw error;
      } finally {
        connection.release();
      }
    },
  );

  // 管理员: 删除实验手册
  manualsApi.delete(
    "/:id",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", paramsSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { id } = c.req.valid("param");
        const manualId = parseInt(id, 10);

        const [rows]: any[] = await pool.query(
          "SELECT file_path FROM experiment_manuals WHERE manual_id = ?",
          [manualId],
        );
        const manual = rows[0];

        if (!manual) {
          return c.json({ error: "Manual not found" }, 404);
        }

        const filePath = join(import.meta.dir, "../../uploads", manual.file_path.replace('/uploads', ''));
        try {
          await unlink(filePath);
        } catch (fsError: any) {
          if (fsError.code !== 'ENOENT') {
            throw fsError;
          }
          console.warn(`File not found, but proceeding with DB deletion: ${filePath}`);
        }

        await pool.query("DELETE FROM experiment_manuals WHERE manual_id = ?", [
          manualId,
        ]);

        return c.json({ message: "Manual deleted successfully" });
      } catch (error) {
        console.error("Error deleting manual:", error);
        throw error;
      }
    },
  );

  return manualsApi;
}
