import type { ResultSetHeader, RowDataPacket } from "mysql2/promise";
import {
  assertParallelSeedSafety,
  PARALLEL_DATASET,
  PARALLEL_MODEL_SELECTIONS,
  PARALLEL_SCENARIOS,
  PARALLEL_STUDENT_GROUPS,
  PARALLEL_STUDENT_USERNAMES,
  cleanupLegacyStudentModelFiles,
  buildModelArtifactPaths,
  createDbConnection,
  formatSqlTimestamp,
  getCliPositionalArgs,
  rmMany,
  type ParallelScenarioName,
} from "./parallel-seed.shared";

interface StudentRow extends RowDataPacket {
  user_id: number;
  username: string;
  full_name: string;
}

interface ExistingExperimentAssetRow extends RowDataPacket {
  experiment_id: number;
  student_id: number;
  pdf_file_path: string | null;
}

interface ScenarioSeedResult {
  username: string;
  experimentId: number;
  stage: "training" | "production";
}

interface ExperimentInsertConfig {
  studentId: number;
  username: string;
  productName: string;
  stage: "training" | "production";
  startTime: string;
  lastActivityAt: string;
}

function parseScenarioName(): ParallelScenarioName {
  const rawValue = getCliPositionalArgs()[0]?.trim();
  if (!rawValue) {
    return "start-only";
  }

  if (PARALLEL_SCENARIOS.includes(rawValue as ParallelScenarioName)) {
    return rawValue as ParallelScenarioName;
  }

  throw new Error(
    `Invalid scenario "${rawValue}". Expected one of: ${PARALLEL_SCENARIOS.join(", ")}`,
  );
}

function buildScenarioGroups(scenario: ParallelScenarioName) {
  switch (scenario) {
    case "start-only":
      return {
        clearedOnly: PARALLEL_STUDENT_GROUPS.all,
        training: [] as string[],
        production: [] as string[],
      };
    case "training-burst":
      return {
        clearedOnly: PARALLEL_STUDENT_GROUPS.all.filter(
          (username) => !PARALLEL_STUDENT_GROUPS.trainingBurst.includes(username),
        ),
        training: [...PARALLEL_STUDENT_GROUPS.trainingBurst],
        production: [] as string[],
      };
    case "mixed-load":
      return {
        clearedOnly: [...PARALLEL_STUDENT_GROUPS.mixedStartOnly],
        training: [...PARALLEL_STUDENT_GROUPS.mixedTraining],
        production: [...PARALLEL_STUDENT_GROUPS.mixedProduction],
      };
  }
}

function buildStageTimes(index: number) {
  const baseTime = new Date("2026-03-10T09:00:00+08:00");
  baseTime.setMinutes(baseTime.getMinutes() + index * 11);
  const startTime = new Date(baseTime);
  const lastActivityTime = new Date(baseTime);
  lastActivityTime.setMinutes(lastActivityTime.getMinutes() + 46);
  return {
    startTime: formatSqlTimestamp(startTime),
    lastActivityAt: formatSqlTimestamp(lastActivityTime),
  };
}

function buildCompletedStepEvents(
  experimentId: number,
  studentId: number,
  startTime: Date,
  completedThroughStep: number,
  activeStep: number,
) {
  const events: unknown[][] = [];
  let cursor = new Date(startTime);

  for (let step = 1; step <= completedThroughStep; step += 1) {
    events.push([
      experimentId,
      studentId,
      step,
      "STARTED",
      formatSqlTimestamp(cursor),
    ]);
    cursor = new Date(cursor.getTime() + 4 * 60 * 1000);
    events.push([
      experimentId,
      studentId,
      step,
      "COMPLETED",
      formatSqlTimestamp(cursor),
    ]);
    cursor = new Date(cursor.getTime() + 3 * 60 * 1000);
  }

  events.push([
    experimentId,
    studentId,
    activeStep,
    "STARTED",
    formatSqlTimestamp(cursor),
  ]);

  return events;
}

async function insertScenarioExperiment(
  connection: Awaited<ReturnType<typeof createDbConnection>>,
  config: ExperimentInsertConfig,
): Promise<ScenarioSeedResult> {
  const isProduction = config.stage === "production";

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
      start_time,
      last_activity_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      config.studentId,
      "In Progress",
      isProduction ? 6 : 4,
      isProduction ? 7 : 5,
      isProduction ? 2800 : 1650,
      PARALLEL_DATASET.industry,
      PARALLEL_DATASET.company,
      config.productName,
      JSON.stringify(PARALLEL_MODEL_SELECTIONS.base),
      JSON.stringify(PARALLEL_MODEL_SELECTIONS.ensemble),
      PARALLEL_DATASET.trainStartIndex,
      PARALLEL_DATASET.trainEndIndex,
      PARALLEL_DATASET.evaluateStartIndex,
      PARALLEL_DATASET.evaluateEndIndex,
      isProduction,
      isProduction ? 3 : null,
      isProduction ? 29.4 : null,
      isProduction ? 22.1 : null,
      isProduction ? 0.882 : null,
      isProduction ? "ma" : null,
      isProduction,
      config.startTime,
      config.lastActivityAt,
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
      knowledge_test,
      model_quality
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      experimentId,
      config.studentId,
      100,
      100,
      isProduction ? 100 : 0,
      isProduction ? 100 : 0,
      isProduction ? 75 : 0,
      isProduction ? 84 : 0,
    ],
  );

  const stepEvents = buildCompletedStepEvents(
    experimentId,
    config.studentId,
    new Date(`${config.startTime.replace(" ", "T")}+08:00`),
    isProduction ? 6 : 4,
    isProduction ? 7 : 5,
  );
  await connection.query(
    `INSERT INTO experiment_step_events
      (experiment_id, student_id, step_order, event_type, event_timestamp)
     VALUES ?`,
    [stepEvents],
  );

  return {
    username: config.username,
    experimentId,
    stage: config.stage,
  };
}

async function main() {
  assertParallelSeedSafety({
    operationName: "Parallel scenario seed",
    commandExample: "bun run seed:parallel:scenario training-burst --force",
  });
  const scenario = parseScenarioName();
  const connection = await createDbConnection();

  try {
    const [studentRows] = await connection.query<StudentRow[]>(
      `SELECT user_id, username, full_name
         FROM users
        WHERE username IN (${PARALLEL_STUDENT_USERNAMES.map(() => "?").join(", ")})
        ORDER BY username ASC`,
      PARALLEL_STUDENT_USERNAMES,
    );

    if (studentRows.length !== PARALLEL_STUDENT_USERNAMES.length) {
      throw new Error(
        `Expected ${PARALLEL_STUDENT_USERNAMES.length} students, found ${studentRows.length}. Run \`bun run db:reset:parallel --force\` first.`,
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
    if (studentIds.length > 0) {
      await connection.query(
        `DELETE FROM experiment_status
          WHERE student_id IN (${studentIds.map(() => "?").join(", ")})`,
        studentIds,
      );
    }

    const scenarioGroups = buildScenarioGroups(scenario);
    const seededResults: ScenarioSeedResult[] = [];

    for (const [index, username] of scenarioGroups.training.entries()) {
      const student = studentByUsername.get(username);
      if (!student) {
        throw new Error(`Missing student for username ${username}`);
      }
      const times = buildStageTimes(index);
      seededResults.push(await insertScenarioExperiment(connection, {
        studentId: student.user_id,
        username,
        productName: PARALLEL_DATASET.products[index % PARALLEL_DATASET.products.length]!,
        stage: "training",
        ...times,
      }));
    }

    for (const [index, username] of scenarioGroups.production.entries()) {
      const student = studentByUsername.get(username);
      if (!student) {
        throw new Error(`Missing student for username ${username}`);
      }
      const times = buildStageTimes(index + scenarioGroups.training.length);
      seededResults.push(await insertScenarioExperiment(connection, {
        studentId: student.user_id,
        username,
        productName: PARALLEL_DATASET.products[index % PARALLEL_DATASET.products.length]!,
        stage: "production",
        ...times,
      }));
    }

    await connection.commit();

    const filesToDelete = assetRows.flatMap((row) => [
      row.pdf_file_path,
      ...buildModelArtifactPaths(row.experiment_id, row.student_id),
    ]);
    await rmMany(filesToDelete);
    await Promise.all(studentIds.map((studentId) => cleanupLegacyStudentModelFiles(studentId)));

    console.log(`Seeded parallel scenario: ${scenario}`);
    console.log(`  fresh-start students: ${scenarioGroups.clearedOnly.length}`);
    console.log(`  training-ready students: ${scenarioGroups.training.length}`);
    console.log(`  production-ready students: ${scenarioGroups.production.length}`);
    if (seededResults.length > 0) {
      console.log("  seeded experiments:");
      for (const result of seededResults) {
        console.log(`    - ${result.username}: exp=${result.experimentId}, stage=${result.stage}`);
      }
    }
  } catch (error) {
    await connection.rollback().catch(() => {});
    throw error;
  } finally {
    await connection.end();
  }
}

await main();
