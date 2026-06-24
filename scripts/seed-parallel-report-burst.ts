import type { ResultSetHeader, RowDataPacket } from "mysql2/promise";
import {
  PARALLEL_DATASET,
  PARALLEL_MODEL_SELECTIONS,
  PARALLEL_STUDENT_USERNAMES,
  assertParallelSeedSafety,
  buildModelArtifactPaths,
  cleanupLegacyStudentModelFiles,
  createDbConnection,
  formatSqlTimestamp,
  rmMany,
} from "./parallel-seed.shared";

interface StudentRow extends RowDataPacket {
  user_id: number;
  username: string;
}

interface ExistingExperimentAssetRow extends RowDataPacket {
  experiment_id: number;
  student_id: number;
  pdf_file_path: string | null;
}

interface SeededReportExperiment {
  username: string;
  studentId: number;
  experimentId: number;
}

function parsePositiveIntegerEnv(name: string, fallback: number): number {
  const rawValue = process.env[name]?.trim();
  if (!rawValue) {
    return fallback;
  }

  if (!/^[1-9]\d*$/.test(rawValue)) {
    throw new Error(`${name} must be a positive integer, got: ${rawValue}`);
  }

  const parsed = Number(rawValue);
  if (!Number.isSafeInteger(parsed)) {
    throw new Error(`${name} must be a positive integer, got: ${rawValue}`);
  }
  return parsed;
}

function selectedUsernames() {
  const count = parsePositiveIntegerEnv("E2E_PDF_BURST_STUDENT_COUNT", 4);
  if (count > PARALLEL_STUDENT_USERNAMES.length) {
    throw new Error(
      `E2E_PDF_BURST_STUDENT_COUNT cannot exceed ${PARALLEL_STUDENT_USERNAMES.length}`,
    );
  }
  return PARALLEL_STUDENT_USERNAMES.slice(0, count);
}

function buildStageTimes(index: number) {
  const baseTime = new Date("2026-03-10T13:00:00+08:00");
  baseTime.setMinutes(baseTime.getMinutes() + index * 7);
  const lastActivityAt = new Date(baseTime);
  lastActivityAt.setMinutes(lastActivityAt.getMinutes() + 52);
  return {
    startTime: formatSqlTimestamp(baseTime),
    lastActivityAt: formatSqlTimestamp(lastActivityAt),
  };
}

function buildStepEvents(experimentId: number, studentId: number, startTime: string) {
  const events: unknown[][] = [];
  const cursor = new Date(`${startTime.replace(" ", "T")}+08:00`);

  for (let step = 1; step <= 7; step += 1) {
    events.push([
      experimentId,
      studentId,
      step,
      "STARTED",
      formatSqlTimestamp(cursor),
    ]);
    cursor.setMinutes(cursor.getMinutes() + 4);
    events.push([
      experimentId,
      studentId,
      step,
      "COMPLETED",
      formatSqlTimestamp(cursor),
    ]);
    cursor.setMinutes(cursor.getMinutes() + 3);
  }

  return events;
}

function buildProductionForecastResults(index: number) {
  return [
    { period: 1, prediction: 126 + index, std_dev: 8 + index * 0.1 },
    { period: 2, prediction: 132 + index, std_dev: 9 + index * 0.1 },
  ];
}

function buildProductionMpsTable(index: number) {
  const period1Demand = 126 + index;
  const period2Demand = 132 + index;
  const initialInventory = 120;
  const plannedProduction = 140;
  const safetyStock = 14 + index;
  const period1EndingInventory =
    initialInventory + plannedProduction - period1Demand;
  const period2BeginningInventory = period1EndingInventory;
  const period2EndingInventory =
    period2BeginningInventory + plannedProduction - period2Demand;

  return [
    {
      period: 1,
      period_label: "第1期",
      demand_forecast: period1Demand,
      safety_stock: safetyStock,
      planned_production: plannedProduction,
      beginning_inventory: initialInventory,
      production_output: plannedProduction,
      ending_inventory: period1EndingInventory,
      stockout: 0,
      service_level: 1,
    },
    {
      period: 2,
      period_label: "第2期",
      demand_forecast: period2Demand,
      safety_stock: safetyStock,
      planned_production: plannedProduction,
      beginning_inventory: period2BeginningInventory,
      production_output: plannedProduction,
      ending_inventory: period2EndingInventory,
      stockout: 0,
      service_level: 1,
    },
  ];
}

async function insertReportReadyExperiment(
  connection: Awaited<ReturnType<typeof createDbConnection>>,
  input: {
    studentId: number;
    username: string;
    index: number;
  },
): Promise<SeededReportExperiment> {
  const { startTime, lastActivityAt } = buildStageTimes(input.index);
  const productName =
    PARALLEL_DATASET.products[input.index % PARALLEL_DATASET.products.length]!;

  const [experimentInsert] = await connection.query<ResultSetHeader>(
    `INSERT INTO experiment_status (
      student_id,
      status,
      highest_completed_step,
      current_step,
      total_active_duration_seconds,
      selected_industry,
      selected_company,
      selected_product,
      selected_base_models,
      selected_ensemble_models,
      data_window_train_start_index,
      data_window_train_end_index,
      data_window_evaluate_start_index,
      data_window_evaluate_end_index,
      moving_average_completed,
      moving_average_window,
      moving_average_metrics_rmse,
      moving_average_metrics_mae,
      moving_average_metrics_r2,
      selected_best_model,
      quiz_about_model_completed,
      quiz_about_plan_completed,
      production_plan_completed,
      production_forecast_periods,
      production_initial_inventory,
      production_target_service_level,
      production_safety_stock_z_score,
      production_forecast_results,
      production_mps_table,
      production_capacity_mode,
      production_capacity,
      start_time,
      last_activity_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      input.studentId,
      "In Progress",
      7,
      7,
      3300,
      PARALLEL_DATASET.industry,
      PARALLEL_DATASET.company,
      productName,
      JSON.stringify(PARALLEL_MODEL_SELECTIONS.base),
      JSON.stringify(PARALLEL_MODEL_SELECTIONS.ensemble),
      PARALLEL_DATASET.trainStartIndex,
      PARALLEL_DATASET.trainEndIndex,
      PARALLEL_DATASET.evaluateStartIndex,
      PARALLEL_DATASET.evaluateEndIndex,
      1,
      3,
      25 + input.index,
      18 + input.index,
      0.91,
      "ma",
      1,
      1,
      1,
      6,
      120,
      0.95,
      1.65,
      JSON.stringify(buildProductionForecastResults(input.index)),
      JSON.stringify(buildProductionMpsTable(input.index)),
      "auto",
      140,
      startTime,
      lastActivityAt,
    ],
  );

  const experimentId = experimentInsert.insertId;

  await connection.query(
    `INSERT INTO experiment_grade (
      experiment_id,
      student_id,
      exp_flow_demand_data_preparation,
      exp_flow_demand_descriptive_stats,
      exp_flow_demand_model_selection,
      exp_flow_demand_generate_results,
      exp_flow_production_inventory_calc,
      exp_flow_production_service_level,
      exp_flow_production_variable_calc,
      exp_flow_production_plan_creation,
      knowledge_test,
      model_quality
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      experimentId,
      input.studentId,
      100,
      100,
      100,
      100,
      100,
      100,
      100,
      100,
      80,
      100,
    ],
  );

  await connection.query(
    `INSERT INTO experiment_step_events
      (experiment_id, student_id, step_order, event_type, event_timestamp)
     VALUES ?`,
    [buildStepEvents(experimentId, input.studentId, startTime)],
  );

  return {
    username: input.username,
    studentId: input.studentId,
    experimentId,
  };
}

async function main() {
  assertParallelSeedSafety({
    operationName: "Parallel PDF report burst seed",
    commandExample: "bun run seed:parallel:report-burst --force",
  });

  const usernames = selectedUsernames();
  const connection = await createDbConnection();

  try {
    const [studentRows] = await connection.query<StudentRow[]>(
      `SELECT user_id, username
         FROM users
        WHERE username IN (${usernames.map(() => "?").join(", ")})
        ORDER BY username ASC`,
      usernames,
    );

    if (studentRows.length !== usernames.length) {
      throw new Error(
        `Expected ${usernames.length} students, found ${studentRows.length}. Run \`bun run db:reset:parallel --force\` first.`,
      );
    }

    const studentByUsername = new Map(
      studentRows.map((row) => [row.username, row]),
    );
    const studentIds = studentRows.map((row) => row.user_id);

    const [assetRows] = await connection.query<ExistingExperimentAssetRow[]>(
      `SELECT e.experiment_id, e.student_id, r.pdf_file_path
         FROM experiment_status e
         LEFT JOIN experiment_reports r ON r.experiment_id = e.experiment_id
        WHERE e.student_id IN (${studentIds.map(() => "?").join(", ")})`,
      studentIds,
    );

    await connection.beginTransaction();
    await connection.query(
      `DELETE FROM experiment_status
        WHERE student_id IN (${studentIds.map(() => "?").join(", ")})`,
      studentIds,
    );

    const seeded: SeededReportExperiment[] = [];
    for (const [index, username] of usernames.entries()) {
      const student = studentByUsername.get(username);
      if (!student) {
        throw new Error(`Missing student for username ${username}`);
      }

      seeded.push(
        await insertReportReadyExperiment(connection, {
          studentId: student.user_id,
          username,
          index,
        }),
      );
    }

    await connection.commit();

    const filesToDelete = assetRows.flatMap((row) => [
      row.pdf_file_path,
      ...buildModelArtifactPaths(row.experiment_id, row.student_id),
    ]);
    await rmMany(filesToDelete);
    await Promise.all(studentIds.map((studentId) => cleanupLegacyStudentModelFiles(studentId)));

    console.log(`Seeded PDF report burst scenario for ${seeded.length} students`);
    for (const result of seeded) {
      console.log(
        `  - ${result.username}: student=${result.studentId}, exp=${result.experimentId}`,
      );
    }
  } catch (error) {
    await connection.rollback().catch(() => {});
    throw error;
  } finally {
    await connection.end();
  }
}

await main();
