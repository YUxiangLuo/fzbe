import type { RowDataPacket } from "mysql2/promise";
import { createMigrationConnection } from "./migration-database";

interface IndexRow extends RowDataPacket {
  column_name: string;
  non_unique: number | string;
  seq_in_index: number;
}

const indexName = "idx_experiment_reports_status_report";
const expectedColumns = ["status", "report_id"];
const connection = await createMigrationConnection();

try {
  const [rows] = await connection.query<IndexRow[]>(
    `SELECT COLUMN_NAME AS column_name,
            NON_UNIQUE AS non_unique,
            SEQ_IN_INDEX AS seq_in_index
       FROM information_schema.STATISTICS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'experiment_reports'
        AND INDEX_NAME = ?
      ORDER BY SEQ_IN_INDEX`,
    [indexName],
  );

  if (rows.length === 0) {
    await connection.query(
      `ALTER TABLE experiment_reports
         ADD INDEX idx_experiment_reports_status_report (status, report_id)`,
    );
    console.log("Added experiment report status/reconciliation index.");
  } else {
    const actualColumns = rows.map((row) => row.column_name);
    const hasExpectedDefinition = rows.every((row) => Number(row.non_unique) === 1)
      && actualColumns.length === expectedColumns.length
      && actualColumns.every((column, index) => column === expectedColumns[index]);
    if (!hasExpectedDefinition) {
      throw new Error(
        `${indexName} exists with an incompatible definition: ${actualColumns.join(", ")}`,
      );
    }
    console.log("Experiment report status/reconciliation index already exists.");
  }
} finally {
  await connection.end();
}
