import mysql from "mysql2/promise";
import { createMigrationConnection } from "./migration-database";

const connection = await createMigrationConnection();

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

try {
  if (!(await hasColumn("question_bank", "answer_explanation"))) {
    await connection.query(
      "ALTER TABLE question_bank ADD COLUMN answer_explanation TEXT NULL AFTER correct_answers",
    );
  }

  if (await hasColumn("quiz_answers", "question_snapshot")) {
    await connection.query(
      `UPDATE quiz_answers qa
          JOIN question_bank qb ON qb.question_id = qa.question_id
           SET qa.question_snapshot = JSON_SET(
                 qa.question_snapshot,
                 '$.answer_explanation',
                 qb.answer_explanation
               )
         WHERE qa.question_snapshot IS NOT NULL
           AND qb.answer_explanation IS NOT NULL
           AND (
             JSON_EXTRACT(qa.question_snapshot, '$.answer_explanation') IS NULL
             OR JSON_TYPE(JSON_EXTRACT(qa.question_snapshot, '$.answer_explanation')) = 'NULL'
           )`,
    );
  }

  console.log("Question answer explanation migration complete.");
} catch (error) {
  console.error("Question answer explanation migration failed:", error);
  process.exitCode = 1;
} finally {
  await connection.end();
}
