import mysql from "mysql2/promise";
import { createMigrationConnection } from "./migration-database";

const backfillAcademicYear = process.env.ACADEMIC_TERM_BACKFILL_YEAR?.trim() || "2025-2026";
const backfillSemester = Number(process.env.ACADEMIC_TERM_BACKFILL_SEMESTER || "2");

if (backfillSemester !== 1 && backfillSemester !== 2) {
  throw new Error("ACADEMIC_TERM_BACKFILL_SEMESTER must be 1 or 2");
}

const connection = await createMigrationConnection({ multipleStatements: true });

async function hasColumn(tableName: string, columnName: string) {
  const [rows] = await connection.query<mysql.RowDataPacket[]>(
    `SELECT 1
       FROM INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = ?
        AND COLUMN_NAME = ?
      LIMIT 1`,
    [tableName, columnName],
  );
  return rows.length > 0;
}

async function hasIndex(tableName: string, indexName: string) {
  const [rows] = await connection.query<mysql.RowDataPacket[]>(
    `SELECT 1
       FROM INFORMATION_SCHEMA.STATISTICS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = ?
        AND INDEX_NAME = ?
      LIMIT 1`,
    [tableName, indexName],
  );
  return rows.length > 0;
}

async function hasForeignKeyOnColumn(tableName: string, columnName: string, referencedTable: string, referencedColumn: string) {
  const [rows] = await connection.query<mysql.RowDataPacket[]>(
    `SELECT 1
       FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = ?
        AND COLUMN_NAME = ?
        AND REFERENCED_TABLE_NAME = ?
        AND REFERENCED_COLUMN_NAME = ?
      LIMIT 1`,
    [tableName, columnName, referencedTable, referencedColumn],
  );
  return rows.length > 0;
}

try {
  await connection.beginTransaction();

  await connection.query(`
    CREATE TABLE IF NOT EXISTS academic_terms (
      term_id INT AUTO_INCREMENT PRIMARY KEY,
      academic_year VARCHAR(20) NOT NULL,
      semester TINYINT UNSIGNED NOT NULL,
      is_active BOOLEAN NOT NULL DEFAULT false,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      UNIQUE KEY academic_year_semester_unique (academic_year, semester),
      INDEX idx_academic_terms_active (is_active),
      CONSTRAINT chk_academic_terms_semester CHECK (semester IN (1, 2))
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  if (!(await hasColumn("classes", "term_id"))) {
    await connection.query("ALTER TABLE classes ADD COLUMN term_id INT NULL AFTER teacher_id");
  }

  const [[activeTerm]] = await connection.query<mysql.RowDataPacket[]>(
    "SELECT term_id FROM academic_terms WHERE is_active = TRUE ORDER BY term_id DESC LIMIT 1",
  );

  let backfillTermId = Number(activeTerm?.term_id ?? 0);
  if (!backfillTermId) {
    await connection.query(
      `INSERT INTO academic_terms (academic_year, semester, is_active)
       VALUES (?, ?, TRUE)
       ON DUPLICATE KEY UPDATE is_active = TRUE`,
      [backfillAcademicYear, backfillSemester],
    );
    await connection.query(
      "UPDATE academic_terms SET is_active = FALSE WHERE NOT (academic_year = ? AND semester = ?)",
      [backfillAcademicYear, backfillSemester],
    );
    const [[createdTerm]] = await connection.query<mysql.RowDataPacket[]>(
      "SELECT term_id FROM academic_terms WHERE academic_year = ? AND semester = ? LIMIT 1",
      [backfillAcademicYear, backfillSemester],
    );
    backfillTermId = Number(createdTerm?.term_id ?? 0);
  }

  if (!backfillTermId) throw new Error("Unable to resolve backfill academic term");

  await connection.query("UPDATE classes SET term_id = ? WHERE term_id IS NULL", [backfillTermId]);
  await connection.query("ALTER TABLE classes MODIFY term_id INT NOT NULL");

  if (!(await hasIndex("classes", "idx_classes_term_id"))) {
    await connection.query("ALTER TABLE classes ADD INDEX idx_classes_term_id (term_id)");
  }
  if (!(await hasIndex("classes", "idx_classes_teacher_term"))) {
    await connection.query("ALTER TABLE classes ADD INDEX idx_classes_teacher_term (teacher_id, term_id)");
  }
  if (!(await hasForeignKeyOnColumn("classes", "term_id", "academic_terms", "term_id"))) {
    await connection.query(
      "ALTER TABLE classes ADD CONSTRAINT classes_term_id_fk FOREIGN KEY (term_id) REFERENCES academic_terms(term_id) ON DELETE RESTRICT",
    );
  }

  await connection.commit();
  console.log(`Academic term migration complete. Backfilled null class terms to term_id=${backfillTermId}.`);
} catch (error) {
  await connection.rollback();
  console.error("Academic term migration failed:", error);
  process.exitCode = 1;
} finally {
  await connection.end();
}
