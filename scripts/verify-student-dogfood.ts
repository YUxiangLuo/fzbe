import fs from "node:fs/promises";
import mysql from "mysql2/promise";
import type { RowDataPacket } from "mysql2/promise";
import {
  getCompletedProductionPlanValidationError,
  validateCompletedProductionPlan,
} from "../src/modules/experiments/experiments.schema";
import { inspectMinimalPdf } from "./pdf-fixture";

const DATABASE_PREFIX = "fz_student_dogfood_";
const EXPECTED_USERNAMES = Array.from(
  { length: 9 },
  (_, index) => String(20_990_001 + index),
);

interface CountRow extends RowDataPacket {
  count: number;
}

interface ScenarioRow extends RowDataPacket {
  username: string;
  status: string | null;
  current_step: number | null;
  highest_completed_step: number | null;
  report_status: string | null;
}

interface AuthRow extends RowDataPacket {
  username: string;
  password_hash: string;
}

interface ReportFileRow extends RowDataPacket {
  username: string;
  experiment_id: number;
  report_status: "submitted" | "rejected" | "graded";
  grade: number | null;
  pdf_file_path: string | null;
}

const EXPECTED_REPORT_FIXTURES = [
  {
    username: "20990004",
    status: "submitted",
    markers: ["Review state: awaiting teacher grading"],
  },
  {
    username: "20990005",
    status: "rejected",
    markers: [
      "Review state: rejected by teacher",
      "Required action: complete experiment redo",
    ],
  },
  {
    username: "20990009",
    status: "graded",
    markers: ["Review state: graded", "Grade: 92"],
  },
] as const;

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is required`);
  return value;
}

async function main() {
  const database = requireEnv("DB_DATABASE");
  const studentPassword = requireEnv("DOGFOOD_STUDENT_PASSWORD");
  if (!database.startsWith(DATABASE_PREFIX)) {
    throw new Error(
      `Refusing to verify database "${database}". Expected prefix "${DATABASE_PREFIX}".`,
    );
  }

  const conn = await mysql.createConnection({
    host: requireEnv("DB_HOST"),
    user: requireEnv("DB_USER"),
    password: process.env.DB_PASSWORD ?? "",
    database,
    charset: "utf8mb4",
  });

  try {
    const placeholders = EXPECTED_USERNAMES.map(() => "?").join(", ");
    const [[userCount]] = await conn.query<CountRow[]>(
      `SELECT COUNT(*) AS count
         FROM users
        WHERE role = 'Student' AND username IN (${placeholders})`,
      EXPECTED_USERNAMES,
    );
    if (Number(userCount?.count) !== EXPECTED_USERNAMES.length) {
      throw new Error(
        `Expected ${EXPECTED_USERNAMES.length} dogfood students, found ${userCount?.count ?? 0}.`,
      );
    }

    const [authRows] = await conn.query<AuthRow[]>(
      `SELECT username, password_hash
         FROM users
        WHERE username IN (${placeholders})
        ORDER BY username`,
      EXPECTED_USERNAMES,
    );
    for (const row of authRows) {
      if (!(await Bun.password.verify(studentPassword, row.password_hash))) {
        throw new Error(`Password verification failed for ${row.username}.`);
      }
    }

    const [[datasetCount]] = await conn.query<CountRow[]>(
      "SELECT COUNT(*) AS count FROM datasets WHERE data_name LIKE 'Dogfood %'",
    );
    if (Number(datasetCount?.count) !== 3) {
      throw new Error(`Expected 3 dogfood datasets, found ${datasetCount?.count ?? 0}.`);
    }

    const [[segmentCount]] = await conn.query<CountRow[]>(
      `SELECT COUNT(*) AS count
         FROM dataset_segments segment
         JOIN datasets dataset ON dataset.dataset_id = segment.dataset_id
        WHERE dataset.data_name LIKE 'Dogfood %'`,
    );
    if (Number(segmentCount?.count) !== 5) {
      throw new Error(`Expected 5 dogfood segments, found ${segmentCount?.count ?? 0}.`);
    }

    const [missingFiles] = await conn.query<RowDataPacket[]>(
      `SELECT file_path FROM datasets WHERE data_name LIKE 'Dogfood %'
       UNION ALL
       SELECT segment.file_path
         FROM dataset_segments segment
         JOIN datasets dataset ON dataset.dataset_id = segment.dataset_id
        WHERE dataset.data_name LIKE 'Dogfood %'`,
    );
    for (const row of missingFiles) {
      const filePath = String(row.file_path ?? "");
      if (!filePath || !(await fs.stat(filePath).then(() => true).catch(() => false))) {
        throw new Error(`Missing dogfood dataset file: ${filePath || "<empty>"}`);
      }
    }

    const [scenarios] = await conn.query<ScenarioRow[]>(
      `SELECT user.username,
              experiment.status,
              experiment.current_step,
              experiment.highest_completed_step,
              report.status AS report_status
         FROM users user
         LEFT JOIN experiment_status experiment
           ON experiment.student_id = user.user_id
          AND experiment.experiment_id = (
            SELECT MAX(latest.experiment_id)
              FROM experiment_status latest
             WHERE latest.student_id = user.user_id
          )
         LEFT JOIN experiment_reports report
           ON report.experiment_id = experiment.experiment_id
          AND report.student_id = user.user_id
        WHERE user.username IN (${placeholders})
        ORDER BY user.username`,
      EXPECTED_USERNAMES,
    );

    const byUsername = new Map(scenarios.map((row) => [row.username, row]));
    const assertions: Array<[string, (row: ScenarioRow) => boolean, string]> = [
      ["20990001", (row) => row.status === null, "fresh account must have no experiment"],
      ["20990002", (row) => row.status === "In Progress" && row.current_step === 3, "resume account must be at step 3"],
      ["20990003", (row) => row.status === "In Progress" && row.current_step === 5, "model account must be at step 5"],
      ["20990004", (row) => row.status === "Completed" && row.report_status === "submitted", "submitted report state is invalid"],
      ["20990005", (row) => row.status === "Completed" && row.report_status === "rejected", "rejected report state is invalid"],
      ["20990006", (row) => row.status === "In Progress" && row.current_step === 7, "legacy plan account must be at step 7"],
      ["20990007", (row) => row.status === "In Progress" && row.current_step === 1, "long profile account must be at step 1"],
      ["20990008", (row) => row.status === "Completed" && row.report_status === null, "completed-without-report state is invalid"],
      ["20990009", (row) => row.status === "Completed" && row.report_status === "graded", "graded report state is invalid"],
    ];

    for (const [username, predicate, message] of assertions) {
      const row = byUsername.get(username);
      if (!row || !predicate(row)) {
        throw new Error(`${username}: ${message}`);
      }
    }

    const [completedPlans] = await conn.query<RowDataPacket[]>(
      `SELECT experiment.*
         FROM experiment_status experiment
         JOIN users user ON user.user_id = experiment.student_id
        WHERE user.username IN (?, ?, ?, ?)
        ORDER BY user.username`,
      ["20990004", "20990005", "20990008", "20990009"],
    );
    for (const plan of completedPlans) {
      const error = getCompletedProductionPlanValidationError({ ...plan });
      if (error) {
        throw new Error(
          `Completed dogfood production plan ${plan.experiment_id} is invalid: ${error}`,
        );
      }
    }

    const [[recoveryPlan]] = await conn.query<RowDataPacket[]>(
      `SELECT experiment.*,
              run.run_id AS active_run_id,
              run.production_revision AS active_production_revision,
              run.model_type AS active_model_type
         FROM experiment_status experiment
         JOIN users user ON user.user_id = experiment.student_id
         LEFT JOIN model_training_runs run
           ON run.experiment_id = experiment.experiment_id
          AND run.student_id = experiment.student_id
          AND run.is_active = TRUE
          AND run.model_type = 'weighted_avg'
        WHERE user.username = ?
        LIMIT 1`,
      ["20990008"],
    );
    if (!recoveryPlan) {
      throw new Error("Missing completed-without-report recovery plan.");
    }
    const recoveryValidation = validateCompletedProductionPlan({
      ...recoveryPlan,
      status: "In Progress",
    });
    if (!recoveryValidation.ok || !recoveryValidation.evidenceClaims) {
      throw new Error(
        `Completed-without-report fixture lacks current forecast evidence: ${
          recoveryValidation.ok ? "missing claims" : recoveryValidation.message
        }`,
      );
    }
    if (
      recoveryValidation.evidenceClaims.run_id !== recoveryPlan.active_run_id
      || recoveryValidation.evidenceClaims.production_revision
        !== recoveryPlan.active_production_revision
      || recoveryValidation.evidenceClaims.model_type !== recoveryPlan.active_model_type
    ) {
      throw new Error("Completed-without-report fixture evidence does not match its active run.");
    }

    const [reportFiles] = await conn.query<ReportFileRow[]>(
      `SELECT user.username,
              report.experiment_id,
              report.status AS report_status,
              report.grade,
              report.pdf_file_path
         FROM experiment_reports report
         JOIN users user ON user.user_id = report.student_id
        WHERE user.username IN (?, ?, ?)
        ORDER BY user.username`,
      EXPECTED_REPORT_FIXTURES.map((fixture) => fixture.username),
    );
    if (reportFiles.length !== EXPECTED_REPORT_FIXTURES.length) {
      throw new Error(
        `Expected ${EXPECTED_REPORT_FIXTURES.length} dogfood report PDFs, found ${reportFiles.length}.`,
      );
    }
    const reportByUsername = new Map(reportFiles.map((row) => [row.username, row]));
    for (const expected of EXPECTED_REPORT_FIXTURES) {
      const row = reportByUsername.get(expected.username);
      if (!row) {
        throw new Error(`Missing dogfood report row for ${expected.username}.`);
      }
      if (row.report_status !== expected.status) {
        throw new Error(
          `${expected.username}: expected ${expected.status} PDF, found ${row.report_status}.`,
        );
      }
      const expectedGrade = expected.status === "graded" ? 92 : null;
      if (
        (expectedGrade === null && row.grade !== null)
        || (expectedGrade !== null && Number(row.grade) !== expectedGrade)
      ) {
        throw new Error(
          `${expected.username}: expected report grade ${expectedGrade ?? "null"}, found ${row.grade ?? "null"}.`,
        );
      }
      const filePath = String(row.pdf_file_path ?? "");
      let pdfBytes: Uint8Array;
      try {
        pdfBytes = await fs.readFile(filePath);
      } catch {
        throw new Error(`Missing dogfood report PDF: ${filePath || "<empty>"}`);
      }
      const inspection = inspectMinimalPdf(pdfBytes, {
        expectedTextMarkers: [
          "Fangzhen Student Dogfood Report",
          `Account: ${expected.username}`,
          `Experiment: ${row.experiment_id}`,
          `Status: ${expected.status}`,
          ...expected.markers,
        ],
      });
      if (!inspection.ok) {
        throw new Error(
          `Invalid dogfood report PDF for ${expected.username}: ${inspection.error}`,
        );
      }
    }

    console.log("Student dogfood database verification passed.");
    console.table(
      scenarios.map((row) => ({
        username: row.username,
        experiment: row.status ?? "none",
        currentStep: row.current_step ?? "-",
        highestStep: row.highest_completed_step ?? "-",
        report: row.report_status ?? "none",
      })),
    );
  } finally {
    await conn.end();
  }
}

await main();
