import assert from "node:assert/strict";
import mysql, { type Pool, type RowDataPacket } from "mysql2/promise";
import { buildMigrationDatabaseOptions } from "../../scripts/migration-database";
import { assertE2eDatabaseResetAllowed } from "../../scripts/test-database-reset-guard";
import { ensureE2eDatabaseExists } from "../../scripts/ensure-e2e-database";

// Real InnoDB regression, deliberately independent of Python and seeded model
// artifacts. It owns only an initially empty, explicitly disposable schema.
assertE2eDatabaseResetAllowed();
assert(["127.0.0.1", "localhost", "::1"].includes(process.env.DB_HOST ?? ""),
  "Concurrency SQL regression requires a local database");
Object.assign(process.env, { DB_CONNECTION_LIMIT: "40", DB_QUEUE_LIMIT: "100" });
await ensureE2eDatabaseExists();
const admin = await mysql.createConnection(buildMigrationDatabaseOptions({ ...process.env }, { multipleStatements: true }));
let pool: Pool | undefined;
let ownsSchema = false;

async function deadline<T>(name: string, work: Promise<T>): Promise<T> {
  const started = performance.now();
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const result = await Promise.race([work, new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${name}: did not finish within 10s`)), 10_000);
    })]);
    console.log(JSON.stringify({ name, elapsedMs: Math.round(performance.now() - started), completed: true }));
    return result;
  } finally {
    clearTimeout(timer);
  }
}

try {
  const [existingTables] = await admin.query<RowDataPacket[]>("SHOW TABLES");
  assert.equal(existingTables.length, 0, "Refusing to replace a non-empty test schema");
  ownsSchema = true;
  await admin.query(await Bun.file(new URL("../../sql/schema.sql", import.meta.url)).text());
  // Exercise RR defaults; fixes must scope RC to their own transaction rather
  // than silently relaxing isolation for unrelated borrowers of the pool.
  const [version] = await admin.query("SELECT VERSION() AS version");
  console.log(JSON.stringify({ bun: Bun.version, database: process.env.DB_DATABASE, version, pool: 40, queue: 100 }));
  for (let id = 1; id <= 151; id++) {
    await admin.query(
      "INSERT INTO users (user_id, username, password_hash, full_name, email, role) VALUES (?, ?, 'unused', ?, ?, 'Student')",
      [id, `concurrency${id}`, `Student ${id}`, `concurrency${id}@example.test`],
    );
  }
  ({ pool } = await import("../../src/config/database"));
  const connections = await Promise.all(Array.from({ length: 40 }, () => pool!.getConnection()));
  await Promise.all(connections.map(connection => connection.query("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")));
  connections.forEach(connection => connection.release());
  const { createExperiment, retryRejectedExperiment } = await import("../../src/modules/experiments/experiments.service");
  const { createExperimentSchema } = await import("../../src/modules/experiments/experiments.schema");
  const createInput = createExperimentSchema.parse({});

  // Round one has an empty experiment index; later cohorts share a populated
  // index. Every actor really executes the creation transaction.
  for (let round = 0; round < 3; round++) {
    const outcomes = await deadline(`create-50-round-${round + 1}`, Promise.all(
      Array.from({ length: 50 }, (_, i) => createExperiment(round * 50 + i + 1, createInput)),
    ));
    assert(outcomes.every(outcome => !("error" in outcome)), "Every new student must create an experiment");
    assert.equal(new Set(outcomes.map(outcome => outcome.data!.experiment_id)).size, 50);
  }
  const duplicates = await deadline("same-student-create-50", Promise.all(
    Array.from({ length: 50 }, () => createExperiment(151, createInput)),
  ));
  assert.equal(duplicates.filter(outcome => !("error" in outcome)).length, 1);
  assert.equal(duplicates.filter(outcome => "error" in outcome && outcome.error === "already_exists").length, 49);
  const [ownership] = await admin.query<RowDataPacket[]>(
    "SELECT student_id, COUNT(*) AS n FROM experiment_status GROUP BY student_id",
  );
  assert.equal(ownership.length, 151);
  assert(ownership.every(row => Number(row.n) === 1));

  // Retry lineage has its own missing-child range. Test different students and
  // idempotent concurrent requests for the same rejected parent.
  await admin.query("UPDATE experiment_status SET status = 'Completed' WHERE student_id <= 50");
  await admin.query(`INSERT INTO experiment_reports (experiment_id, student_id, status)
    SELECT experiment_id, student_id, 'rejected' FROM experiment_status WHERE student_id <= 50`);
  const [reports] = await admin.query<RowDataPacket[]>("SELECT report_id, student_id, experiment_id FROM experiment_reports");
  const retries = await deadline("retry-50-rejected-parents", Promise.all(reports.map(report =>
    retryRejectedExperiment(Number(report.student_id), { rejected_report_id: Number(report.report_id) }),
  )));
  assert(retries.every(outcome => "created" in outcome && outcome.created));
  retries.forEach((outcome, index) => {
    assert.equal(outcome.data!.retry_of_experiment_id, reports[index]!.experiment_id);
    assert.equal(outcome.data!.student_id, reports[index]!.student_id);
  });
  const sameParent = reports[0]!;
  const idempotentRetries = await deadline("same-parent-retry-50", Promise.all(Array.from({ length: 50 }, () =>
    retryRejectedExperiment(Number(sameParent.student_id), { rejected_report_id: Number(sameParent.report_id) }),
  )));
  assert(idempotentRetries.every(outcome => "created" in outcome && !outcome.created));
  assert.equal(new Set(idempotentRetries.map(outcome => outcome.data!.experiment_id)).size, 1);

  const { commitModelTrainingRun, buildExperimentTrainingRevision, buildModelTrainingRequestHash } =
    await import("../../src/modules/models/model-run-store");
  for (let round = 0; round < 3; round++) {
    const inputs: Parameters<typeof commitModelTrainingRun>[0][] = [];
    // These students have never trained, including in earlier rounds.
    for (let studentId = 51 + round * 12; studentId < 63 + round * 12; studentId++) {
      await admin.query(`UPDATE experiment_status SET status = 'In Progress',
        selected_industry = 'test', selected_company = 'test', selected_product = 'test',
        data_window_train_start_index = 0, data_window_train_end_index = 11,
        data_window_evaluate_start_index = 12, data_window_evaluate_end_index = 17,
        moving_average_window = 6 WHERE student_id = ?`, [studentId]);
      const [[experiment]] = await admin.query<RowDataPacket[]>("SELECT * FROM experiment_status WHERE student_id = ?", [studentId]);
      assert(experiment);
      const body = { experiment_id: Number(experiment.experiment_id), moving_average_window: 6 };
      const requestHash = buildModelTrainingRequestHash({
        modelType: "ma", body, datasetRevision: "test", runtimeRevision: "test",
        experimentRevision: buildExperimentTrainingRevision("ma", experiment, body), baseRunFingerprint: {},
      });
      inputs.push({
        runId: crypto.randomUUID(), sessionId: null, experimentId: body.experiment_id, studentId,
        modelType: "ma", requestHash, datasetRevision: "test", runtimeRevision: "test", body,
        result: { status: "success", results: { metrics: { rmse: 1.2, mae: 0.8, r2: 0.91 } } },
        artifactPath: "/sql-regression/no-file-created.pkl", artifactSha256: "a".repeat(64),
      });
    }
    const committed = await deadline(`model-commit-12-round-${round + 1}`, Promise.all(inputs.map(commitModelTrainingRun)));
    assert.equal(new Set(committed.map(outcome => outcome.runId)).size, 12);
    assert(committed.every(outcome => outcome.stateVersion === 1));
  }
  const [runs] = await admin.query<RowDataPacket[]>(
    "SELECT experiment_id, COUNT(*) AS n, SUM(is_active) AS active FROM model_training_runs GROUP BY experiment_id",
  );
  assert.equal(runs.length, 36);
  assert(runs.every(row => Number(row.n) === 1 && Number(row.active) === 1));

  const { createAcademicTerm, updateAcademicTerm } = await import("../../src/modules/academic-terms/academic-terms.service");
  await createAcademicTerm({ academic_year: "2000-2001", semester: 1, is_active: true });
  const terms = await deadline("academic-term-create-50-pool-40", Promise.all(Array.from({ length: 50 }, (_, i) =>
    createAcademicTerm({ academic_year: `${2100 + i}-${2101 + i}`, semester: 1, is_active: false }),
  )));
  assert(terms.every(term => "data" in term && term.data && !term.data.is_active));
  const updated = await deadline("academic-term-update-50-pool-40", Promise.all(terms.map(term => {
    assert("data" in term && term.data);
    return updateAcademicTerm(term.data.term_id, { semester: 2 });
  })));
  assert(updated.every(term => "data" in term && term.data?.semester === 2));

  const finalConnections = await Promise.all(Array.from({ length: 40 }, () => pool!.getConnection()));
  try {
    await Promise.all(finalConnections.map(async connection => {
      // SHOW supports both MySQL 8.4 and MariaDB's older variable name.
      const [rows] = await connection.query<RowDataPacket[]>(
        "SHOW SESSION VARIABLES WHERE Variable_name IN ('transaction_isolation', 'tx_isolation')",
      );
      assert(rows.length > 0);
      assert(rows.every(row => row.Value === "REPEATABLE-READ"), "A scoped RC transaction leaked session isolation");
    }));
  } finally {
    finalConnections.forEach(connection => connection.release());
  }
  console.log("Classroom SQL concurrency and transaction isolation verified.");
} finally {
  await pool?.end();
  if (ownsSchema) {
    const [tables] = await admin.query<RowDataPacket[]>("SHOW TABLES");
    await admin.query("SET FOREIGN_KEY_CHECKS = 0");
    for (const row of tables) {
      const table = String(Object.values(row)[0]).replaceAll("`", "``");
      await admin.query(`DROP TABLE \`${table}\``);
    }
  }
  await admin.end();
}
