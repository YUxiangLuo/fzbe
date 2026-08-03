import { spawn } from "node:child_process";
import process from "node:process";
import type { RowDataPacket } from "mysql2/promise";
import {
  PARALLEL_ACCOUNTS,
  PARALLEL_CLASS,
  PARALLEL_SCENARIOS,
  PARALLEL_STUDENT_USERNAMES,
  assertBaseEnv,
  createDbConnection,
  getCliPositionalArgs,
  type ParallelScenarioName,
} from "./parallel-seed.shared";

interface CountRow extends RowDataPacket {
  count: number;
}

interface TableRow extends RowDataPacket {
  table_name: string;
}

interface ColumnRow extends RowDataPacket {
  table_name: string;
  column_name: string;
}

const REQUIRED_TABLES = [
  "users",
  "classes",
  "enrollments",
  "class_assistants",
  "grade_weights",
  "question_bank",
  "experiment_manuals",
  "datasets",
  "dataset_segments",
  "experiment_status",
  "experiment_grade",
  "experiment_step_events",
  "experiment_reports",
  "model_guided_training_sessions",
  "model_guided_training_step_attempts",
  "model_training_runs",
  "model_training_run_dependencies",
  "model_production_preparation_sessions",
  "model_production_preparation_step_attempts",
] as const;
type RequiredTableName = (typeof REQUIRED_TABLES)[number];

const REQUIRED_COLUMNS = {
  users: ["user_id", "username", "full_name"],
  classes: ["class_id"],
  enrollments: ["student_id", "class_id"],
  class_assistants: ["assistant_id", "class_id"],
  grade_weights: ["class_id"],
  question_bank: ["question_id"],
  experiment_manuals: [
    "manual_id",
    "file_name",
    "file_path",
    "description",
    "is_active",
    "uploader_id",
  ],
  datasets: ["dataset_id", "data_name", "file_path", "description"],
  dataset_segments: [
    "dataset_id",
    "industry_name",
    "company_name",
    "product_name",
    "file_path",
  ],
  experiment_status: [
    "experiment_id",
    "student_id",
    "status",
    "highest_completed_step",
    "current_step",
    "state_version",
    "total_active_duration_seconds",
    "selected_industry",
    "selected_company",
    "selected_product",
    "selected_base_models",
    "selected_ensemble_models",
    "data_window_train_start_index",
    "data_window_train_end_index",
    "data_window_evaluate_start_index",
    "data_window_evaluate_end_index",
    "moving_average_completed",
    "moving_average_window",
    "moving_average_metrics_rmse",
    "moving_average_metrics_mae",
    "moving_average_metrics_r2",
    "exponential_smoothing_completed",
    "exponential_smoothing_alpha",
    "exponential_smoothing_metrics_rmse",
    "exponential_smoothing_metrics_mae",
    "exponential_smoothing_metrics_r2",
    "arima_completed",
    "arima_p",
    "arima_d",
    "arima_q",
    "arima_metrics_rmse",
    "arima_metrics_mae",
    "arima_metrics_r2",
    "arima_adf_stationarity",
    "lstm_completed",
    "lstm_normalization",
    "lstm_features",
    "lstm_target_field",
    "lstm_metrics_rmse",
    "lstm_metrics_mae",
    "lstm_metrics_r2",
    "ensemble_weighted_completed",
    "ensemble_weighted_base_models",
    "ensemble_weighted_metrics_rmse",
    "ensemble_weighted_metrics_mae",
    "ensemble_weighted_metrics_r2",
    "ensemble_boosting_completed",
    "ensemble_boosting_base_models",
    "ensemble_boosting_metrics_rmse",
    "ensemble_boosting_metrics_mae",
    "ensemble_boosting_metrics_r2",
    "ensemble_stacking_completed",
    "ensemble_stacking_base_models",
    "ensemble_stacking_metrics_rmse",
    "ensemble_stacking_metrics_mae",
    "ensemble_stacking_metrics_r2",
    "selected_best_model",
    "quiz_about_model_completed",
    "quiz_about_plan_completed",
    "production_plan_completed",
    "production_forecast_periods",
    "production_initial_inventory",
    "production_target_service_level",
    "production_safety_stock_z_score",
    "production_forecast_results",
    "production_forecast_evidence",
    "production_mps_table",
    "production_capacity_mode",
    "production_capacity_scenario",
    "production_capacity",
    "production_custom_capacity",
    "start_time",
    "last_activity_at",
  ],
  experiment_grade: [
    "experiment_id",
    "student_id",
    "exp_flow_demand_data_preparation",
    "exp_flow_demand_descriptive_stats",
    "exp_flow_demand_model_selection",
    "exp_flow_demand_generate_results",
    "exp_flow_production_inventory_calc",
    "exp_flow_production_service_level",
    "exp_flow_production_variable_calc",
    "exp_flow_production_plan_creation",
    "knowledge_test",
    "model_quality",
  ],
  experiment_step_events: [
    "experiment_id",
    "student_id",
    "step_order",
    "event_type",
    "event_timestamp",
  ],
  experiment_reports: ["experiment_id", "pdf_file_path"],
  model_guided_training_sessions: [
    "session_id",
    "experiment_id",
    "student_id",
    "model_type",
    "request_hash",
    "dataset_revision",
    "runtime_revision",
    "request_payload",
    "status",
    "current_step_id",
    "completed_step_ids",
    "step_outputs",
    "step_plan",
    "result_payload",
    "artifact_dir",
    "protocol_version",
    "committed_state_path",
    "committed_state_sha256",
    "active_attempt_id",
    "error_message",
    "superseded_reason",
    "experiment_state_version",
    "expires_at",
  ],
  model_guided_training_step_attempts: [
    "attempt_id",
    "session_id",
    "step_id",
    "status",
    "input_state_path",
    "input_state_sha256",
    "candidate_state_path",
    "candidate_state_sha256",
    "receipt_path",
    "output_payload",
    "final_result_payload",
    "error_message",
    "run_count",
    "lease_expires_at",
  ],
  model_training_runs: [
    "run_id",
    "session_id",
    "experiment_id",
    "student_id",
    "model_type",
    "request_hash",
    "dataset_revision",
    "runtime_revision",
    "parameters",
    "metrics",
    "artifact_path",
    "artifact_sha256",
    "production_artifact_path",
    "production_artifact_sha256",
    "production_artifact_forecast_steps",
    "production_revision",
    "is_active",
  ],
  model_training_run_dependencies: [
    "run_id",
    "base_model_type",
    "base_run_id",
  ],
  model_production_preparation_sessions: [
    "session_id",
    "experiment_id",
    "student_id",
    "run_id",
    "model_type",
    "request_hash",
    "dataset_revision",
    "runtime_revision",
    "forecast_steps",
    "status",
    "current_step_id",
    "completed_step_ids",
    "step_plan",
    "artifact_dir",
    "candidate_model_path",
    "protocol_version",
    "committed_state_path",
    "committed_state_sha256",
    "active_attempt_id",
    "observed_production_revision",
    "production_revision",
    "error_message",
    "superseded_reason",
    "expires_at",
  ],
  model_production_preparation_step_attempts: [
    "attempt_id",
    "session_id",
    "step_id",
    "status",
    "input_state_path",
    "input_state_sha256",
    "candidate_state_path",
    "candidate_state_sha256",
    "receipt_path",
    "output_payload",
    "error_message",
    "lease_expires_at",
  ],
} as const satisfies Record<RequiredTableName, readonly string[]>;

const REQUIRED_COLUMN_COUNT = REQUIRED_TABLES.reduce(
  (count, tableName) => count + REQUIRED_COLUMNS[tableName].length,
  0,
);
const MIN_QUESTION_BANK_COUNT = 16;

function hasForceFlag() {
  return process.argv.includes("--force");
}

function parseScenario(): ParallelScenarioName | null {
  const rawValue = getCliPositionalArgs()[0]?.trim();
  if (!rawValue) {
    return null;
  }

  if (PARALLEL_SCENARIOS.includes(rawValue as ParallelScenarioName)) {
    return rawValue as ParallelScenarioName;
  }

  throw new Error(
    `Invalid scenario "${rawValue}". Expected one of: ${PARALLEL_SCENARIOS.join(", ")}`,
  );
}

async function runSubcommand(input: { label: string; args: string[] }) {
  console.log(`[parallel-prepare] ${input.label}: bun ${input.args.join(" ")}`);

  const child = spawn("bun", input.args, {
    cwd: process.cwd(),
    env: process.env,
    stdio: "inherit",
  });

  await new Promise<void>((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(
        new Error(
          `${input.label} failed (code=${String(code)}, signal=${String(signal)})`,
        ),
      );
    });
  });
}

async function hasParallelBaseReady() {
  assertBaseEnv();

  const connection = await createDbConnection();
  const requiredUsernames = [
    PARALLEL_ACCOUNTS.adminUsername,
    PARALLEL_ACCOUNTS.teacherUsername,
    PARALLEL_ACCOUNTS.assistantUsername,
    ...PARALLEL_STUDENT_USERNAMES,
  ];

  try {
    const [tableRows] = await connection.query<TableRow[]>(
      `SELECT table_name
         FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND table_name IN (${REQUIRED_TABLES.map(() => "?").join(", ")})`,
      [...REQUIRED_TABLES],
    );
    const existingTables = new Set(tableRows.map((row) => row.table_name));
    const missingTables = REQUIRED_TABLES.filter((tableName) =>
      !existingTables.has(tableName),
    );
    if (missingTables.length > 0) {
      console.log(
        `[parallel-prepare] missing tables: ${missingTables.join(", ")}`,
      );
      return false;
    }

    const [columnRows] = await connection.query<ColumnRow[]>(
      `SELECT table_name, column_name
         FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name IN (${REQUIRED_TABLES.map(() => "?").join(", ")})`,
      [...REQUIRED_TABLES],
    );
    const existingColumns = new Set(
      columnRows.map((row) => `${row.table_name}\0${row.column_name}`),
    );
    const missingColumnsByTable = new Map<RequiredTableName, string[]>();
    for (const tableName of REQUIRED_TABLES) {
      const missingColumns = REQUIRED_COLUMNS[tableName].filter(
        (columnName) =>
          !existingColumns.has(`${tableName}\0${columnName}`),
      );
      if (missingColumns.length > 0) {
        missingColumnsByTable.set(tableName, [...missingColumns]);
      }
    }

    if (missingColumnsByTable.size > 0) {
      const missingColumnCount = [...missingColumnsByTable.values()].reduce(
        (count, columns) => count + columns.length,
        0,
      );
      console.log(
        `[parallel-prepare] missing required columns (${missingColumnCount}):`,
      );
      for (const tableName of REQUIRED_TABLES) {
        const missingColumns = missingColumnsByTable.get(tableName);
        if (missingColumns) {
          console.log(`  - ${tableName}: ${missingColumns.join(", ")}`);
        }
      }
      return false;
    }
    console.log(
      `[parallel-prepare] schema ready: ${REQUIRED_TABLES.length} tables, ${REQUIRED_COLUMN_COUNT} required columns`,
    );

    const [userRows] = await connection.query<CountRow[]>(
      `SELECT COUNT(*) AS count
         FROM users
        WHERE username IN (${requiredUsernames.map(() => "?").join(", ")})`,
      requiredUsernames,
    );
    const userCount = userRows[0]?.count ?? 0;
    if (userCount !== requiredUsernames.length) {
      console.log(
        `[parallel-prepare] expected ${requiredUsernames.length} parallel users, found ${userCount}`,
      );
      return false;
    }

    const [classRows] = await connection.query<CountRow[]>(
      "SELECT COUNT(*) AS count FROM classes WHERE class_id = ?",
      [PARALLEL_CLASS.id],
    );
    if ((classRows[0]?.count ?? 0) !== 1) {
      console.log(
        `[parallel-prepare] missing parallel class ${PARALLEL_CLASS.id}`,
      );
      return false;
    }

    const [enrollmentRows] = await connection.query<CountRow[]>(
      "SELECT COUNT(*) AS count FROM enrollments WHERE class_id = ?",
      [PARALLEL_CLASS.id],
    );
    if ((enrollmentRows[0]?.count ?? 0) !== PARALLEL_STUDENT_USERNAMES.length) {
      console.log(
        `[parallel-prepare] expected ${PARALLEL_STUDENT_USERNAMES.length} enrollments for class ${PARALLEL_CLASS.id}, found ${enrollmentRows[0]?.count ?? 0}`,
      );
      return false;
    }

    const [assistantRows] = await connection.query<CountRow[]>(
      "SELECT COUNT(*) AS count FROM class_assistants WHERE class_id = ? AND assistant_id = ?",
      [PARALLEL_CLASS.id, 900201],
    );
    if ((assistantRows[0]?.count ?? 0) !== 1) {
      console.log(
        `[parallel-prepare] expected assistant 900201 to be assigned to class ${PARALLEL_CLASS.id}`,
      );
      return false;
    }

    const [gradeWeightRows] = await connection.query<CountRow[]>(
      "SELECT COUNT(*) AS count FROM grade_weights WHERE class_id = ?",
      [PARALLEL_CLASS.id],
    );
    if ((gradeWeightRows[0]?.count ?? 0) !== 1) {
      console.log(
        `[parallel-prepare] expected grade_weights for class ${PARALLEL_CLASS.id}`,
      );
      return false;
    }

    const [questionRows] = await connection.query<CountRow[]>(
      "SELECT COUNT(*) AS count FROM question_bank",
    );
    if ((questionRows[0]?.count ?? 0) < MIN_QUESTION_BANK_COUNT) {
      console.log(
        `[parallel-prepare] expected at least ${MIN_QUESTION_BANK_COUNT} question_bank rows, found ${questionRows[0]?.count ?? 0}`,
      );
      return false;
    }

    return true;
  } finally {
    await connection.end();
  }
}

async function main() {
  if (process.argv.includes("--help")) {
    console.log(`Usage: bun run scripts/prepare-parallel.ts [scenario] --force

Scenarios:
  start-only
  training-burst
  mixed-load

Behavior:
  1. Check whether the parallel database baseline exists
  2. Run db:reset:parallel --force if the baseline is missing
  3. Run seed:parallel:assets --force
  4. Optionally run seed:parallel:scenario <scenario> --force
`);
    return;
  }

  if (!hasForceFlag()) {
    throw new Error(
      "Parallel environment preparation requires --force.",
    );
  }

  const scenario = parseScenario();
  const baseReady = await hasParallelBaseReady();

  console.log(`[parallel-prepare] baseReady=${baseReady}`);
  if (!baseReady) {
    await runSubcommand({
      label: "reset-base",
      args: ["run", "db:reset:parallel", "--force"],
    });
  }

  await runSubcommand({
    label: "seed-assets",
    args: ["run", "seed:parallel:assets", "--force"],
  });

  if (scenario) {
    await runSubcommand({
      label: `seed-scenario:${scenario}`,
      args: ["run", "seed:parallel:scenario", scenario, "--force"],
    });
  } else {
    console.log("[parallel-prepare] no scenario requested");
  }
}

await main().catch((error) => {
  console.error("[parallel-prepare] failed", error);
  process.exit(1);
});
