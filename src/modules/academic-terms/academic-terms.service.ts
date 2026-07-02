import type { ResultSetHeader, RowDataPacket, Pool, PoolConnection } from "mysql2/promise";
import { pool } from "../../config";
import type { AcademicTermInput, UpdateAcademicTermInput } from "./academic-terms.schema";

type Queryable = Pool | PoolConnection;

export interface AcademicTermRow extends RowDataPacket {
  term_id: number;
  academic_year: string;
  semester: 1 | 2;
  is_active: number | boolean;
  created_at: Date;
  class_count?: number;
}

export interface AcademicTerm {
  term_id: number;
  academic_year: string;
  semester: 1 | 2;
  term_label: string;
  is_active: boolean;
  created_at: Date;
  class_count?: number;
}

export const formatAcademicTermLabel = (academicYear: string, semester: 1 | 2) =>
  `${academicYear} 第${semester}学期`;

export const mapAcademicTermRow = (row: AcademicTermRow): AcademicTerm => ({
  term_id: row.term_id,
  academic_year: row.academic_year,
  semester: row.semester,
  term_label: formatAcademicTermLabel(row.academic_year, row.semester),
  is_active: Boolean(row.is_active),
  created_at: row.created_at,
  ...(row.class_count !== undefined ? { class_count: Number(row.class_count) } : {}),
});

export async function listAcademicTerms() {
  const [rows] = await pool.query<AcademicTermRow[]>(
    `SELECT at.term_id, at.academic_year, at.semester, at.is_active, at.created_at,
            COUNT(c.class_id) AS class_count
       FROM academic_terms at
       LEFT JOIN classes c ON c.term_id = at.term_id
      GROUP BY at.term_id, at.academic_year, at.semester, at.is_active, at.created_at
      ORDER BY at.academic_year DESC, at.semester DESC, at.term_id DESC`,
  );
  return rows.map(mapAcademicTermRow);
}

export async function getCurrentAcademicTerm(db: Queryable = pool) {
  const [[row]] = await db.query<AcademicTermRow[]>(
    `SELECT term_id, academic_year, semester, is_active, created_at
       FROM academic_terms
      WHERE is_active = TRUE
      ORDER BY term_id DESC
      LIMIT 1`,
  );
  return row ? mapAcademicTermRow(row) : null;
}

export async function getAcademicTermById(termId: number, db: Queryable = pool) {
  const [[row]] = await db.query<AcademicTermRow[]>(
    `SELECT term_id, academic_year, semester, is_active, created_at
       FROM academic_terms
      WHERE term_id = ?
      LIMIT 1`,
    [termId],
  );
  return row ? mapAcademicTermRow(row) : null;
}

export async function resolveAcademicTermId(termId: string | number | undefined | null, db: Queryable = pool) {
  if (termId === undefined || termId === null || termId === "") {
    return { termId: undefined as number | undefined };
  }

  if (termId === "current") {
    const currentTerm = await getCurrentAcademicTerm(db);
    if (!currentTerm) return { error: "no_current_term" as const };
    return { termId: currentTerm.term_id };
  }

  const parsedTermId = typeof termId === "number" ? termId : Number(termId);
  if (!Number.isInteger(parsedTermId) || parsedTermId <= 0) {
    return { error: "invalid_term" as const };
  }

  const term = await getAcademicTermById(parsedTermId, db);
  if (!term) return { error: "invalid_term" as const };
  return { termId: parsedTermId };
}

export async function getDefaultClassTermId(db: Queryable = pool) {
  const currentTerm = await getCurrentAcademicTerm(db);
  if (!currentTerm) return { error: "no_current_term" as const };
  return { termId: currentTerm.term_id };
}

export async function createAcademicTerm(input: AcademicTermInput) {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    const [[countRow]] = await connection.query<RowDataPacket[]>("SELECT COUNT(*) AS total FROM academic_terms");
    const shouldActivate = input.is_active === true || Number(countRow?.total ?? 0) === 0;

    if (shouldActivate) {
      await connection.query("UPDATE academic_terms SET is_active = FALSE WHERE is_active = TRUE");
    }

    const [result] = await connection.query<ResultSetHeader>(
      "INSERT INTO academic_terms (academic_year, semester, is_active) VALUES (?, ?, ?)",
      [input.academic_year.trim(), input.semester, shouldActivate],
    );

    await connection.commit();
    const term = await getAcademicTermById(result.insertId);
    return { data: term! };
  } catch (error: unknown) {
    await connection.rollback();
    const err = error as { code?: string };
    if (err.code === "ER_DUP_ENTRY") return { error: "duplicate" as const };
    throw error;
  } finally {
    connection.release();
  }
}

export async function updateAcademicTerm(termId: number, input: UpdateAcademicTermInput) {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    const existing = await getAcademicTermById(termId, connection);
    if (!existing) {
      await connection.rollback();
      return { error: "not_found" as const };
    }

    if (input.is_active === false && existing.is_active) {
      await connection.rollback();
      return { error: "cannot_deactivate_current" as const };
    }

    if (input.is_active === true) {
      await connection.query("UPDATE academic_terms SET is_active = FALSE WHERE is_active = TRUE");
    }

    const fields: string[] = [];
    const values: unknown[] = [];
    if (input.academic_year !== undefined) {
      fields.push("academic_year = ?");
      values.push(input.academic_year.trim());
    }
    if (input.semester !== undefined) {
      fields.push("semester = ?");
      values.push(input.semester);
    }
    if (input.is_active !== undefined) {
      fields.push("is_active = ?");
      values.push(input.is_active);
    }

    values.push(termId);
    await connection.query(`UPDATE academic_terms SET ${fields.join(", ")} WHERE term_id = ?`, values);

    await connection.commit();
    const term = await getAcademicTermById(termId);
    return { data: term! };
  } catch (error: unknown) {
    await connection.rollback();
    const err = error as { code?: string };
    if (err.code === "ER_DUP_ENTRY") return { error: "duplicate" as const };
    throw error;
  } finally {
    connection.release();
  }
}

export async function deleteAcademicTerm(termId: number) {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    const existing = await getAcademicTermById(termId, connection);
    if (!existing) {
      await connection.rollback();
      return { error: "not_found" as const };
    }
    if (existing.is_active) {
      await connection.rollback();
      return { error: "active_term" as const };
    }

    const [[classCountRow]] = await connection.query<RowDataPacket[]>(
      "SELECT COUNT(*) AS total FROM classes WHERE term_id = ?",
      [termId],
    );
    if (Number(classCountRow?.total ?? 0) > 0) {
      await connection.rollback();
      return { error: "has_classes" as const };
    }

    await connection.query("DELETE FROM academic_terms WHERE term_id = ?", [termId]);
    await connection.commit();
    return { data: { message: "学期删除成功" } };
  } catch (error) {
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}
