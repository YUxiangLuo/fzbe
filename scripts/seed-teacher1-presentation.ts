import path from "path";
import { mkdir } from "node:fs/promises";
import type { RowDataPacket } from "mysql2/promise";
import { buildMinimalPdf } from "./pdf-fixture";
import {
  assertParallelSeedSafety,
  createDbConnection,
} from "./parallel-seed.shared";

const STUDENT_ID_MIN = 20255001;
const STUDENT_ID_MAX = 20255256;

interface ReportFixtureRow extends RowDataPacket {
  report_id: number;
  experiment_id: number;
  student_id: number;
  username: string;
  full_name: string;
  status: string;
  grade: number | null;
  pdf_file_path: string | null;
}

function getTeacherPresentationSqlPath() {
  return path.resolve(process.cwd(), "sql", "teacher1_presentation_data.sql");
}

function getReportsRoot() {
  return path.resolve(process.cwd(), "uploads", "reports");
}

function getReportFileName(row: ReportFixtureRow) {
  const existingName = row.pdf_file_path ? path.basename(row.pdf_file_path) : "";
  return existingName || `report-exp${row.experiment_id}-student${row.student_id}.pdf`;
}

function buildReportPdf(row: ReportFixtureRow) {
  return buildMinimalPdf([
    "Teacher1 presentation report",
    `Student: ${row.username} (${row.student_id})`,
    `Name: ${row.full_name}`,
    `Experiment: ${row.experiment_id}`,
    `Status: ${row.status}`,
    `Grade: ${row.grade ?? "pending"}`,
    "Generated fixture for teacher portal dogfood",
  ]);
}

async function main() {
  assertParallelSeedSafety({
    operationName: "Teacher1 presentation seed",
    commandExample: "bun run seed:teacher1:presentation --force",
  });

  const connection = await createDbConnection();
  const reportsRoot = getReportsRoot();

  try {
    const sqlPath = getTeacherPresentationSqlPath();
    const sql = await Bun.file(sqlPath).text();
    await connection.query(sql);

    await mkdir(reportsRoot, { recursive: true });

    const [rows] = await connection.query<ReportFixtureRow[]>(
      `SELECT er.report_id,
              er.experiment_id,
              er.student_id,
              u.username,
              u.full_name,
              er.status,
              er.grade,
              er.pdf_file_path
         FROM experiment_reports er
         JOIN users u ON u.user_id = er.student_id
        WHERE er.student_id BETWEEN ? AND ?
          AND er.pdf_file_path IS NOT NULL
        ORDER BY er.report_id ASC`,
      [STUDENT_ID_MIN, STUDENT_ID_MAX],
    );

    for (const row of rows) {
      const fileName = getReportFileName(row);
      const absolutePath = path.resolve(reportsRoot, fileName);
      await Bun.write(absolutePath, buildReportPdf(row));
      await connection.query(
        "UPDATE experiment_reports SET pdf_file_path = ? WHERE report_id = ?",
        [absolutePath, row.report_id],
      );
    }

    console.log(`Seeded teacher1 presentation data and ${rows.length} report PDFs in ${reportsRoot}`);
  } finally {
    await connection.end();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
