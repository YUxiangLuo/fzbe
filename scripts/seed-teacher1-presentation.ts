import path from "path";
import type { RowDataPacket } from "mysql2/promise";
import { buildMinimalPdf } from "./pdf-fixture";
import {
  assertParallelSeedSafety,
  createDbConnection,
  ensureReportsRoot,
} from "./parallel-seed.shared";
import {
  buildCanonicalProductionForecasts,
  buildCanonicalProductionMps,
  type ProductionForecastSeed,
} from "./production-plan-fixture";

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

interface ProductionPlanFixtureRow extends RowDataPacket {
  experiment_id: number;
  production_forecast_results: unknown;
  production_initial_inventory: number;
  production_safety_stock_z_score: number;
  production_capacity: number;
}

function getTeacherPresentationSqlPath() {
  return path.resolve(process.cwd(), "sql", "teacher1_presentation_data.sql");
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

const parseForecastSeeds = (
  value: unknown,
  experimentId: number,
): ProductionForecastSeed[] => {
  let parsed = value;
  if (typeof parsed === "string") {
    try {
      parsed = JSON.parse(parsed);
    } catch {
      throw new Error(
        `Teacher presentation experiment ${experimentId} has malformed forecast JSON`,
      );
    }
  }
  if (!Array.isArray(parsed)) {
    throw new Error(
      `Teacher presentation experiment ${experimentId} has invalid forecast rows`,
    );
  }
  return parsed.map((point, index) => {
    if (!point || typeof point !== "object" || Array.isArray(point)) {
      throw new Error(
        `Teacher presentation experiment ${experimentId} forecast ${index + 1} is invalid`,
      );
    }
    const record = point as Record<string, unknown>;
    const prediction = Number(record.prediction);
    const stdDev = Number(record.std_dev);
    if (
      !Number.isFinite(prediction)
      || prediction < 0
      || !Number.isFinite(stdDev)
      || stdDev < 0
    ) {
      throw new Error(
        `Teacher presentation experiment ${experimentId} forecast ${index + 1} is invalid`,
      );
    }
    return { prediction, std_dev: stdDev };
  });
};

async function upgradeTeacherProductionPlans(
  connection: Awaited<ReturnType<typeof createDbConnection>>,
) {
  const [rows] = await connection.query<ProductionPlanFixtureRow[]>(
    `SELECT experiment_id,
            production_forecast_results,
            production_initial_inventory,
            production_safety_stock_z_score,
            production_capacity
       FROM experiment_status
      WHERE student_id BETWEEN ? AND ?
        AND production_plan_completed = true
      ORDER BY experiment_id ASC`,
    [STUDENT_ID_MIN, STUDENT_ID_MAX],
  );

  for (const row of rows) {
    const forecasts = buildCanonicalProductionForecasts(
      parseForecastSeeds(row.production_forecast_results, row.experiment_id),
    );
    const mpsTable = buildCanonicalProductionMps(forecasts, {
      initialInventory: Number(row.production_initial_inventory),
      zScore: Number(row.production_safety_stock_z_score),
      capacity: Number(row.production_capacity),
    });
    await connection.query(
      `UPDATE experiment_status
          SET production_forecast_results = ?,
              production_mps_table = ?
        WHERE experiment_id = ?`,
      [
        JSON.stringify(forecasts),
        JSON.stringify(mpsTable),
        row.experiment_id,
      ],
    );
  }
  return rows.length;
}

async function main() {
  assertParallelSeedSafety({
    operationName: "Teacher1 presentation seed",
    commandExample: "bun run seed:teacher1:presentation --force",
  });

  const reportsRoot = await ensureReportsRoot();
  const connection = await createDbConnection();

  try {
    const sqlPath = getTeacherPresentationSqlPath();
    const sql = await Bun.file(sqlPath).text();
    await connection.query(sql);
    const upgradedPlanCount = await upgradeTeacherProductionPlans(connection);

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

    console.log(
      `Seeded teacher1 presentation data, upgraded ${upgradedPlanCount} production plans`
      + ` and wrote ${rows.length} report PDFs in ${reportsRoot}`,
    );
  } finally {
    await connection.end();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
