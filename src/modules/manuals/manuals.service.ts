import { rm } from "node:fs/promises";
import path from "path";
import type { RowDataPacket, ResultSetHeader } from "mysql2";
import { pool } from "../../config";
import type { ExperimentManual } from "../../types";
import { validateFile, generateUniqueFilename, FileValidationError } from "../../utils/file-validator";
import { FILE_SIZE_LIMITS } from "../../utils/constants";
import type { UpdateManualInput } from "./manuals.schema";

type ManualRow = ExperimentManual & RowDataPacket & { uploader_name?: string };

const uploadsRoot = path.resolve(import.meta.dir, "../../../uploads");

function toPublicFilePath(absolutePath: string | null | undefined): string | null {
  if (!absolutePath) return null;
  const relative = path.relative(uploadsRoot, absolutePath);
  if (!relative || relative.startsWith("..")) {
    return `manuals/${path.basename(absolutePath)}`.replace(/\\/g, "/");
  }
  return relative.replace(/\\/g, "/");
}

function normalizeManualRow<T extends { file_path?: string | null }>(row: T): T & { file_path: string | null } {
  return { ...row, file_path: toPublicFilePath(row.file_path) };
}

export interface UploadManualInput {
  file: File;
  fileName: string;
  description?: string;
  uploaderId: number;
}

export async function uploadManual(input: UploadManualInput) {
  const { file, fileName, description, uploaderId } = input;

  await validateFile(file, "pdf", {
    maxSize: FILE_SIZE_LIMITS.PDF_MAX_SIZE,
    checkMagicBytes: true,
  });

  const uniqueFilename = generateUniqueFilename(file.name);
  const uploadPath = path.resolve(uploadsRoot, "manuals", uniqueFilename);
  await Bun.write(uploadPath, file);

  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    const [result] = await connection.query<ResultSetHeader>(
      "INSERT INTO experiment_manuals (file_name, file_path, description, uploader_id) VALUES (?, ?, ?, ?)",
      [fileName, uploadPath, description || null, uploaderId],
    );

    const [rows] = await connection.query<ManualRow[]>(
      `SELECT m.*, u.full_name as uploader_name
       FROM experiment_manuals m
       LEFT JOIN users u ON m.uploader_id = u.user_id
       WHERE m.manual_id = ?`,
      [result.insertId],
    );

    const manual = normalizeManualRow(rows[0]!);
    await connection.commit();
    return manual;
  } catch (error) {
    await connection.rollback().catch(() => {});
    await rm(uploadPath, { force: true }).catch(() => {});
    throw error;
  } finally {
    connection.release();
  }
}

export async function getAllManuals() {
  const [rows] = await pool.query<ManualRow[]>(`
    SELECT m.*, u.full_name as uploader_name
    FROM experiment_manuals m
    LEFT JOIN users u ON m.uploader_id = u.user_id
    ORDER BY m.uploaded_at DESC
  `);
  return rows.map(normalizeManualRow);
}

export async function getActiveManual() {
  const [rows] = await pool.query<ManualRow[]>(`
    SELECT m.*, u.full_name as uploader_name
    FROM experiment_manuals m
    LEFT JOIN users u ON m.uploader_id = u.user_id
    WHERE m.is_active = true
    LIMIT 1
  `);
  return rows[0] ? normalizeManualRow(rows[0]) : null;
}

export async function updateManual(manualId: number, input: UpdateManualInput) {
  const { file_name, description, is_active } = input;

  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    if (is_active === true) {
      await connection.query(
        "UPDATE experiment_manuals SET is_active = false WHERE manual_id != ?",
        [manualId],
      );
    }

    const fields: string[] = [];
    const values: unknown[] = [];
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
      return { error: "没有需要更新的字段" as const };
    }

    values.push(manualId);
    const [result] = await connection.query<ResultSetHeader>(
      `UPDATE experiment_manuals SET ${fields.join(", ")} WHERE manual_id = ?`,
      values,
    );

    if (result.affectedRows === 0) {
      await connection.rollback();
      return { error: "not_found" as const };
    }

    await connection.commit();

    const [rows] = await connection.query<ManualRow[]>(
      `SELECT m.*, u.full_name as uploader_name
       FROM experiment_manuals m
       LEFT JOIN users u ON m.uploader_id = u.user_id
       WHERE m.manual_id = ?`,
      [manualId],
    );

    return { data: normalizeManualRow(rows[0]!) };
  } catch (error) {
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}

export async function deleteManual(manualId: number) {
  const [rows] = await pool.query<ManualRow[]>(
    "SELECT file_path FROM experiment_manuals WHERE manual_id = ?",
    [manualId],
  );

  if (!rows[0]) {
    return { error: "not_found" as const };
  }

  await rm(rows[0].file_path, { force: true });
  await pool.query("DELETE FROM experiment_manuals WHERE manual_id = ?", [manualId]);

  return { success: true };
}

export { FileValidationError };
