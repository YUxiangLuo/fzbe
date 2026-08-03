import type { Connection, ResultSetHeader, RowDataPacket } from "mysql2/promise";
import { createMigrationConnection } from "./migration-database";

interface RejectedReportCandidateRow extends RowDataPacket {
  report_id: number;
  experiment_id: number;
}

const NORMALIZATION_BATCH_SIZE = 500;

export async function normalizeRejectedReportGrades(
  connection: Connection,
  batchSize = NORMALIZATION_BATCH_SIZE,
) {
  if (!Number.isInteger(batchSize) || batchSize <= 0) {
    throw new Error("Rejected report grade normalization batch size must be a positive integer");
  }

  let lastReportId = 0;
  let reportGradesCleared = 0;
  let reportQualityGradesCleared = 0;

  while (true) {
    await connection.beginTransaction();
    try {
      // The (status, report_id) index and monotonic cursor make each pass scan
      // the rejected range once.  FOR UPDATE keeps a batch internally atomic,
      // while committing every batch bounds lock duration on large histories.
      const [candidates] = await connection.query<RejectedReportCandidateRow[]>(
        `SELECT er.report_id, er.experiment_id
           FROM experiment_reports er
           LEFT JOIN experiment_grade eg
             ON eg.experiment_id = er.experiment_id
          WHERE er.status = 'rejected'
            AND er.report_id > ?
            AND (er.grade IS NOT NULL OR eg.report_quality IS NOT NULL)
          ORDER BY er.report_id ASC
          LIMIT ?
          FOR UPDATE`,
        [lastReportId, batchSize],
      );

      if (candidates.length === 0) {
        await connection.commit();
        break;
      }

      const reportIds = candidates.map((candidate) => Number(candidate.report_id));
      const experimentIds = candidates.map((candidate) => Number(candidate.experiment_id));
      const reportPlaceholders = reportIds.map(() => "?").join(", ");
      const experimentPlaceholders = experimentIds.map(() => "?").join(", ");

      const [reportResult] = await connection.query<ResultSetHeader>(
        `UPDATE experiment_reports
            SET grade = NULL
          WHERE report_id IN (${reportPlaceholders})
            AND status = 'rejected'
            AND grade IS NOT NULL`,
        reportIds,
      );
      const [gradeResult] = await connection.query<ResultSetHeader>(
        `UPDATE experiment_grade eg
           INNER JOIN experiment_reports er
             ON er.experiment_id = eg.experiment_id
            SET eg.report_quality = NULL
          WHERE eg.experiment_id IN (${experimentPlaceholders})
            AND er.status = 'rejected'
            AND eg.report_quality IS NOT NULL`,
        experimentIds,
      );

      await connection.commit();
      reportGradesCleared += reportResult.affectedRows;
      reportQualityGradesCleared += gradeResult.affectedRows;
      lastReportId = reportIds.at(-1)!;
    } catch (error) {
      await connection.rollback();
      throw error;
    }
  }

  return { reportGradesCleared, reportQualityGradesCleared };
}

async function main() {
  const connection = await createMigrationConnection();
  try {
    const result = await normalizeRejectedReportGrades(connection);
    console.log(
      "Rejected report grade normalization complete: "
      + `${result.reportGradesCleared} report grade(s), `
      + `${result.reportQualityGradesCleared} report-quality grade(s) cleared.`,
    );
  } finally {
    await connection.end();
  }
}

if (import.meta.main) {
  try {
    await main();
  } catch (error) {
    console.error("Rejected report grade normalization failed:", error);
    process.exitCode = 1;
  }
}
