import { type RowDataPacket } from "mysql2/promise";
import { createMigrationConnection } from "./migration-database";

const connection = await createMigrationConnection();

try {
  const [rows] = await connection.query<RowDataPacket[]>(
    `SELECT 1
       FROM information_schema.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'experiment_status'
        AND COLUMN_NAME = 'production_forecast_evidence'
      LIMIT 1`,
  );
  if (rows.length === 0) {
    await connection.query(
      `ALTER TABLE experiment_status
         ADD COLUMN production_forecast_evidence TEXT DEFAULT NULL
         AFTER production_forecast_results`,
    );
  }
  console.log("Production forecast evidence migration complete.");
} catch (error) {
  console.error("Production forecast evidence migration failed:", error);
  process.exitCode = 1;
} finally {
  await connection.end();
}
