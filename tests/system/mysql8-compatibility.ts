import assert from "node:assert/strict";
import path from "node:path";
import mysql, { type Pool, type ResultSetHeader, type RowDataPacket } from "mysql2/promise";
import { buildMigrationDatabaseOptions } from "../../scripts/migration-database";
import { assertE2eDatabaseResetAllowed } from "../../scripts/test-database-reset-guard";
import { ensureE2eDatabaseExists } from "../../scripts/ensure-e2e-database";
import type {
  ProductionPreparationAttemptRow,
  ProductionPreparationSessionRow,
} from "../../src/modules/models/production-preparation-state";

const LOCAL_DATABASE_HOSTS = new Set(["127.0.0.1", "localhost", "::1"]);
const projectRoot = path.resolve(import.meta.dir, "../..");

assertE2eDatabaseResetAllowed();
assert(
  LOCAL_DATABASE_HOSTS.has((process.env.DB_HOST ?? "").trim().toLowerCase()),
  "MySQL 8 compatibility system test requires a local database host",
);
if (!process.env.JWT_SECRET || process.env.JWT_SECRET.trim().length < 32) {
  process.env.JWT_SECRET = "fz_db_compat_test_jwt_secret_at_least_32_bytes_long";
}

const bootstrapOutcome = await ensureE2eDatabaseExists();
const admin = await mysql.createConnection(
  buildMigrationDatabaseOptions({ ...process.env }, { multipleStatements: true }),
);
let pool: Pool | undefined;
let ownsSchema = false;

async function runScript(scriptPath: string): Promise<void> {
  const proc = Bun.spawn([process.execPath, "run", scriptPath], {
    cwd: projectRoot,
    env: process.env,
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  if (exitCode !== 0) {
    throw new Error(
      `Script ${scriptPath} exited with ${exitCode}\nstdout:\n${stdout}\nstderr:\n${stderr}`,
    );
  }
}

async function assertNoDeprecationWarnings(
  connection: mysql.Connection,
  label: string,
): Promise<void> {
  const [warnings] = await connection.query<RowDataPacket[]>("SHOW WARNINGS");
  const deprecations = warnings.filter((row) => Number(row.Code) === 1287);
  assert.equal(
    deprecations.length,
    0,
    `${label} produced MySQL deprecation warnings: ${JSON.stringify(deprecations)}`,
  );
}

try {
  const [existingTables] = await admin.query<RowDataPacket[]>("SHOW TABLES");
  assert.equal(existingTables.length, 0, "Refusing to overwrite a non-empty test schema");
  ownsSchema = true;

  const [[meta]] = await admin.query<RowDataPacket[]>(
    `SELECT VERSION() AS version,
            @@collation_database AS database_collation`,
  );
  console.log(JSON.stringify({
    suite: "mysql8-compatibility",
    database: process.env.DB_DATABASE,
    version: meta?.version,
    collation: meta?.database_collation,
    bootstrapOutcome,
  }));
  if (bootstrapOutcome === "created") {
    assert.equal(
      String(meta?.database_collation ?? "").toLowerCase(),
      "utf8mb4_unicode_ci",
      "Newly created test databases must explicitly use utf8mb4_unicode_ci even on MySQL 8 defaults",
    );
  }

  // 1. Load baseline schema + seed data and run full ledgered migrations (001-013).
  await admin.query(await Bun.file(path.join(projectRoot, "sql/schema.sql")).text());
  await admin.query(await Bun.file(path.join(projectRoot, "sql/seed.sql")).text());
  await runScript("scripts/migrate.ts");

  // 2. P1-1: Simulate legacy ON DELETE CASCADE retry FK and verify both 009 and 011
  // upgrade cleanly without triggering MySQL 8 ER_FK_DUP_NAME (1826).
  for (const scriptPath of [
    "scripts/migrate-experiment-retry-lineage.ts",
    "scripts/migrate-experiment-retry-delete-rule.ts",
  ]) {
    await admin.query(
      "ALTER TABLE experiment_status DROP FOREIGN KEY experiment_status_retry_parent_fk",
    );
    await admin.query(
      `ALTER TABLE experiment_status
         ADD CONSTRAINT experiment_status_retry_parent_fk
         FOREIGN KEY (retry_of_experiment_id)
         REFERENCES experiment_status(experiment_id)
         ON DELETE CASCADE`,
    );
    await runScript(scriptPath);
    const [[fkRow]] = await admin.query<RowDataPacket[]>(
      `SELECT rc.DELETE_RULE AS delete_rule
         FROM information_schema.REFERENTIAL_CONSTRAINTS rc
        WHERE rc.CONSTRAINT_SCHEMA = DATABASE()
          AND rc.TABLE_NAME = 'experiment_status'
          AND rc.CONSTRAINT_NAME = 'experiment_status_retry_parent_fk'`,
    );
    assert.equal(fkRow?.delete_rule, "SET NULL", `${scriptPath} must converge FK delete rule to SET NULL`);
  }

  // 3. P1-2: Verify teacher1_presentation_data.sql imports cleanly under MySQL 8 strict mode.
  await admin.query(
    await Bun.file(path.join(projectRoot, "sql/teacher1_presentation_data.sql")).text(),
  );
  const [[presentationStats]] = await admin.query<RowDataPacket[]>(
    `SELECT COUNT(*) AS total,
            SUM(CASE WHEN start_time IS NOT NULL THEN 1 ELSE 0 END) AS with_start_time
       FROM experiment_status
      WHERE experiment_id BETWEEN 800001 AND 800256`,
  );
  assert(Number(presentationStats?.total) > 0, "teacher1_presentation_data.sql should insert experiments");
  assert(Number(presentationStats?.with_start_time) > 0, "teacher1_presentation_data.sql should populate start_time");

  // 4. P1-3: Verify information_schema explicit aliases return lowercase keys on both engines.
  const [tableRows] = await admin.query<RowDataPacket[]>(
    `SELECT TABLE_NAME AS table_name
       FROM information_schema.tables
      WHERE table_schema = DATABASE()
        AND table_name IN ('users', 'experiment_status', 'model_production_preparation_sessions')`,
  );
  assert.equal(tableRows.length, 3);
  assert(tableRows.every((row) => typeof row.table_name === "string" && row.table_name.length > 0));
  const [columnRows] = await admin.query<RowDataPacket[]>(
    `SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name
       FROM information_schema.columns
      WHERE table_schema = DATABASE()
        AND table_name = 'users'
        AND column_name = 'user_id'`,
  );
  assert.equal(columnRows.length, 1);
  assert.equal(columnRows[0]?.table_name, "users");
  assert.equal(columnRows[0]?.column_name, "user_id");

  // Dynamically load application pool & services now that schema exists.
  const { env } = await import("../../src/config");
  ({ pool } = await import("../../src/config/database"));
  const { probeStartupDatabaseRuntime } = await import("../../src/utils/startup-checks");
  const {
    jsonColumnCasValue,
    jsonColumnEqualsSql,
  } = await import("../../src/modules/models/production-preparation-state");
  const { claimProductionPreparationReceiptReplay } = await import(
    "../../src/modules/models/production-preparation-attempt-store"
  );
  const { updateExperiment } = await import("../../src/modules/experiments/experiments.service");
  const {
    buildCanonicalProductionMps,
    buildProductionCapacityScenarios,
    updateExperimentSchema,
  } = await import("../../src/modules/experiments/experiments.schema");
  const { issueProductionForecastEvidence } = await import(
    "../../src/modules/experiments/production-forecast-evidence"
  );
  const { searchUsers } = await import("../../src/modules/users/users.service");

  // 5. Startup runtime diagnostic probe.
  const startupInfos: string[] = [];
  const startupWarnings: string[] = [];
  await probeStartupDatabaseRuntime(pool, env.DB_TIMEZONE, {
    connectionLimit: 40,
    logInfo: (message) => startupInfos.push(message),
    logWarn: (message) => startupWarnings.push(message),
  });
  assert.equal(startupInfos.length, 1, "startup probe should log database runtime info");
  assert(startupInfos[0]!.includes("Database runtime: version="));
  if (String(meta?.database_collation ?? "").toLowerCase() === "utf8mb4_unicode_ci") {
    assert.equal(
      startupWarnings.filter((warning) => warning.includes("排序规则")).length,
      0,
      "utf8mb4_unicode_ci schema should not emit collation warning",
    );
  }

  // 6. P0-1: Verify all 6 production-preparation JSON CAS SQL shapes against real JSON columns.
  const [studentInsert] = await admin.query<ResultSetHeader>(
    `INSERT INTO users (username, password_hash, full_name, email, phone_number, role)
     VALUES ('dbcompat_stu', 'unused', '兼容测试学生', 'dbcompat@example.test', ' 13800138999 ', 'Student')`,
  );
  const studentId = Number(studentInsert.insertId);
  const [experimentInsert] = await admin.query<ResultSetHeader>(
    `INSERT INTO experiment_status (student_id, status, selected_best_model, arima_completed, state_version)
     VALUES (?, 'In Progress', 'arima', TRUE, 0)`,
    [studentId],
  );
  const experimentId = Number(experimentInsert.insertId);
  const runId = "11111111-1111-4111-8111-111111111111";
  const productionRevision = "22222222-2222-4222-8222-222222222222";
  await admin.query(
    `INSERT INTO model_training_runs (
       run_id, experiment_id, student_id, model_type, request_hash,
       dataset_revision, runtime_revision, parameters, metrics,
       artifact_path, production_artifact_path, production_artifact_forecast_steps,
       production_revision, is_active
     ) VALUES (?, ?, ?, 'arima', ?, ?, ?, '{}', '{}', '/tmp/a.pkl', '/tmp/p.pkl', 2, ?, TRUE)`,
    [
      runId,
      experimentId,
      studentId,
      "a".repeat(64),
      "b".repeat(64),
      "c".repeat(64),
      productionRevision,
    ],
  );

  const stepPlan = [
    { id: "initialize", label: "确认生产准备材料" },
    { id: "fit_model", label: "重拟合生产模型" },
    { id: "finalize", label: "组装并校验生产产物" },
  ];
  const sessionId = "33333333-3333-4333-8333-333333333333";
  const attemptId = "44444444-4444-4444-8444-444444444444";
  await admin.query(
    `INSERT INTO model_production_preparation_sessions (
       session_id, experiment_id, student_id, run_id, model_type,
       request_hash, dataset_revision, runtime_revision, protocol_version,
       forecast_steps, status, current_step_id, completed_step_ids, step_plan,
       artifact_dir, candidate_model_path, expires_at
     ) VALUES (?, ?, ?, ?, 'arima', ?, ?, ?, 1, 2, 'ready', 'initialize', ?, ?, '/tmp/prep', '/tmp/cand.pkl', DATE_ADD(NOW(), INTERVAL 7 DAY))`,
    [
      sessionId,
      experimentId,
      studentId,
      runId,
      "d".repeat(64),
      "b".repeat(64),
      "c".repeat(64),
      JSON.stringify([]),
      JSON.stringify(stepPlan),
    ],
  );

  const loadSession = async (id: string) => {
    const [rows] = await pool!.query<ProductionPreparationSessionRow[]>(
      "SELECT * FROM model_production_preparation_sessions WHERE session_id = ? LIMIT 1",
      [id],
    );
    assert(rows[0], `expected production preparation session ${id}`);
    return rows[0];
  };

  let sessionRow = await loadSession(sessionId);

  // 6a. Negative CAS check: mismatched JSON must match 0 rows.
  const [mismatchedRefresh] = await pool.query<ResultSetHeader>(
    `UPDATE model_production_preparation_sessions
        SET expires_at = DATE_ADD(NOW(), INTERVAL 7 DAY)
      WHERE session_id = ?
        AND ${jsonColumnEqualsSql("completed_step_ids")}
        AND ${jsonColumnEqualsSql("step_plan")}`,
    [sessionId, JSON.stringify(["initialize"]), jsonColumnCasValue(sessionRow.step_plan)],
  );
  assert.equal(mismatchedRefresh.affectedRows, 0, "Mismatched JSON CAS must affect 0 rows");

  // 6b. Lease refresh CAS UPDATE (service.ts L1140).
  const [refreshResult] = await pool.query<ResultSetHeader>(
    `UPDATE model_production_preparation_sessions
        SET expires_at = GREATEST(DATE_ADD(expires_at, INTERVAL 1 SECOND), DATE_ADD(NOW(), INTERVAL 7 DAY))
      WHERE session_id = ?
        AND status = ?
        AND active_attempt_id <=> ?
        AND expires_at <=> ?
        AND ${jsonColumnEqualsSql("completed_step_ids")}
        AND ${jsonColumnEqualsSql("step_plan")}
        AND status IN ('ready', 'failed', 'running')
        AND expires_at > NOW()`,
    [
      sessionId,
      sessionRow.status,
      sessionRow.active_attempt_id,
      sessionRow.expires_at,
      jsonColumnCasValue(sessionRow.completed_step_ids),
      jsonColumnCasValue(sessionRow.step_plan),
    ],
  );
  assert.equal(refreshResult.affectedRows, 1, "Lease refresh JSON CAS must match 1 row");

  // 6c. Real claimProductionPreparationReceiptReplay (attempt-store.ts L148-149).
  await admin.query(
    `INSERT INTO model_production_preparation_step_attempts (
       attempt_id, session_id, step_id, status, candidate_state_path, receipt_path, lease_expires_at
     ) VALUES (?, ?, 'initialize', 'failed', '/tmp/prep/cand.pkl', '/tmp/prep/receipt.json', DATE_SUB(NOW(), INTERVAL 1 MINUTE))`,
    [attemptId, sessionId],
  );
  await admin.query(
    `UPDATE model_production_preparation_sessions
        SET status = 'failed', active_attempt_id = ?
      WHERE session_id = ?`,
    [attemptId, sessionId],
  );
  sessionRow = await loadSession(sessionId);
  const [attemptRows] = await pool.query<ProductionPreparationAttemptRow[]>(
    "SELECT * FROM model_production_preparation_step_attempts WHERE attempt_id = ? LIMIT 1",
    [attemptId],
  );
  const replayed = await claimProductionPreparationReceiptReplay(sessionRow, attemptRows[0]!, 300);
  assert.equal(replayed, true, "claimProductionPreparationReceiptReplay JSON CAS must succeed");

  // Expire attempt lease so supersession CAS is eligible, then verify supersede + claim + delete CAS.
  await admin.query(
    "UPDATE model_production_preparation_step_attempts SET status = 'failed', lease_expires_at = DATE_SUB(NOW(), INTERVAL 1 MINUTE) WHERE attempt_id = ?",
    [attemptId],
  );
  sessionRow = await loadSession(sessionId);
  const [supersededResult] = await pool.query<ResultSetHeader>(
    `UPDATE model_production_preparation_sessions AS session
        SET status = 'superseded',
            current_step_id = NULL,
            active_attempt_id = NULL,
            superseded_reason = 'db-compat-test'
      WHERE session.session_id = ?
        AND session.status = ?
        AND session.active_attempt_id <=> ?
        AND session.expires_at <=> ?
        AND session.current_step_id <=> ?
        AND ${jsonColumnEqualsSql("session.completed_step_ids")}
        AND ${jsonColumnEqualsSql("session.step_plan")}
        AND session.committed_state_path <=> ?
        AND session.committed_state_sha256 <=> ?`,
    [
      sessionRow.session_id,
      sessionRow.status,
      sessionRow.active_attempt_id,
      sessionRow.expires_at,
      sessionRow.current_step_id,
      jsonColumnCasValue(sessionRow.completed_step_ids),
      jsonColumnCasValue(sessionRow.step_plan),
      sessionRow.committed_state_path,
      sessionRow.committed_state_sha256,
    ],
  );
  assert.equal(supersededResult.affectedRows, 1, "Superseded JSON CAS must match 1 row");

  sessionRow = await loadSession(sessionId);
  const [claimSupersededResult] = await pool.query<ResultSetHeader>(
    `UPDATE model_production_preparation_sessions
        SET superseded_reason = 'claim-token'
      WHERE session_id = ?
        AND status = 'superseded'
        AND active_attempt_id <=> ?
        AND ${jsonColumnEqualsSql("completed_step_ids")}
        AND ${jsonColumnEqualsSql("step_plan")}
        AND committed_state_path <=> ?
        AND committed_state_sha256 <=> ?`,
    [
      sessionRow.session_id,
      sessionRow.active_attempt_id,
      jsonColumnCasValue(sessionRow.completed_step_ids),
      jsonColumnCasValue(sessionRow.step_plan),
      sessionRow.committed_state_path,
      sessionRow.committed_state_sha256,
    ],
  );
  assert.equal(claimSupersededResult.affectedRows, 1, "Claim-superseded JSON CAS must match 1 row");

  const [deleteSupersededResult] = await pool.query<ResultSetHeader>(
    `DELETE session
       FROM model_production_preparation_sessions AS session
      WHERE session.session_id = ?
        AND session.status = 'superseded'
        AND session.active_attempt_id IS NULL
        AND ${jsonColumnEqualsSql("session.completed_step_ids")}
        AND ${jsonColumnEqualsSql("session.step_plan")}
        AND session.committed_state_path <=> ?
        AND session.committed_state_sha256 <=> ?`,
    [
      sessionRow.session_id,
      jsonColumnCasValue(sessionRow.completed_step_ids),
      jsonColumnCasValue(sessionRow.step_plan),
      sessionRow.committed_state_path,
      sessionRow.committed_state_sha256,
    ],
  );
  assert.equal(deleteSupersededResult.affectedRows, 1, "Delete-superseded JSON CAS must match 1 row");

  const completedSessionId = "55555555-5555-4555-8555-555555555555";
  await admin.query(
    `INSERT INTO model_production_preparation_sessions (
       session_id, experiment_id, student_id, run_id, model_type,
       request_hash, dataset_revision, runtime_revision, protocol_version,
       forecast_steps, status, current_step_id, completed_step_ids, step_plan,
       artifact_dir, candidate_model_path, production_revision
     ) VALUES (?, ?, ?, ?, 'arima', ?, ?, ?, 1, 2, 'completed', NULL, ?, ?, '/tmp/prep2', '/tmp/cand2.pkl', ?)`,
    [
      completedSessionId,
      experimentId,
      studentId,
      runId,
      "e".repeat(64),
      "b".repeat(64),
      "c".repeat(64),
      JSON.stringify(["initialize", "fit_model", "finalize"]),
      JSON.stringify(stepPlan),
      productionRevision,
    ],
  );
  const completedRow = await loadSession(completedSessionId);
  const [deleteCompletedResult] = await pool.query<ResultSetHeader>(
    `DELETE session
       FROM model_production_preparation_sessions AS session
      WHERE session.session_id = ?
        AND session.status = 'completed'
        AND session.active_attempt_id <=> ?
        AND session.production_revision <=> ?
        AND ${jsonColumnEqualsSql("session.completed_step_ids")}
        AND ${jsonColumnEqualsSql("session.step_plan")}`,
    [
      completedRow.session_id,
      completedRow.active_attempt_id,
      completedRow.production_revision,
      jsonColumnCasValue(completedRow.completed_step_ids),
      jsonColumnCasValue(completedRow.step_plan),
    ],
  );
  assert.equal(deleteCompletedResult.affectedRows, 1, "Delete-completed JSON CAS must match 1 row");

  // 7. P0-2: Real experiments.service.ts updateExperiment roundtrip with unchanged
  // production_forecast_results after MySQL 8 JSON key reordering.
  const forecasts = [
    {
      prediction: 100,
      std_dev: 12,
      upper_error_p99: 28,
      uncertainty_source: "model" as const,
      calibration_source: "arima_forecast_distribution" as const,
      calibration_mean_error: 1.5,
      calibration_count: 8,
      bias_diagnostic_source: "outer_selection_assessment" as const,
      interval_lower: 76.48,
      interval_upper: 123.52,
      interval_level: 0.95,
      interval_kind: "model_prediction_interval" as const,
      coverage_claim: "model_nominal" as const,
    },
    {
      prediction: 110,
      std_dev: 14,
      upper_error_p99: 31,
      uncertainty_source: "model" as const,
      calibration_source: "arima_forecast_distribution" as const,
      calibration_mean_error: 1.2,
      calibration_count: 8,
      bias_diagnostic_source: "outer_selection_assessment" as const,
      interval_lower: 82.56,
      interval_upper: 137.44,
      interval_level: 0.95,
      interval_kind: "model_prediction_interval" as const,
      coverage_claim: "model_nominal" as const,
    },
  ];
  const initialCapacity = buildProductionCapacityScenarios(forecasts, 1.65).normal;
  const evidence = issueProductionForecastEvidence({
    experiment_id: experimentId,
    student_id: studentId,
    model_type: "arima",
    run_id: runId,
    production_revision: productionRevision,
    forecast_steps: forecasts.length,
    forecasts,
  });
  const firstUpdate = await updateExperiment(
    experimentId,
    studentId,
    updateExperimentSchema.parse({
      expected_version: 0,
      production_plan_completed: true,
      production_forecast_periods: forecasts.length,
      production_initial_inventory: 0,
      production_target_service_level: 0.95,
      production_safety_stock_z_score: 1.65,
      production_forecast_results: forecasts,
      production_forecast_evidence: evidence,
      production_mps_table: buildCanonicalProductionMps({
        forecasts,
        zScore: 1.65,
        initialInventory: 0,
        capacity: initialCapacity,
      }),
      production_capacity_mode: "scenario",
      production_capacity_scenario: "normal",
      production_capacity: initialCapacity,
      production_custom_capacity: null,
    }),
  );
  assert(!("error" in firstUpdate), `First updateExperiment failed: ${JSON.stringify(firstUpdate)}`);

  // Second save: resend exact same forecast_results with changed initial_inventory
  // and NO production_forecast_evidence. Must preserve existing evidence on MySQL 8!
  const updatedMps = buildCanonicalProductionMps({
    forecasts,
    zScore: 1.65,
    initialInventory: 25,
    capacity: initialCapacity,
  });
  const secondUpdate = await updateExperiment(
    experimentId,
    studentId,
    updateExperimentSchema.parse({
      expected_version: 1,
      production_plan_completed: true,
      production_forecast_periods: forecasts.length,
      production_initial_inventory: 25,
      production_target_service_level: 0.95,
      production_safety_stock_z_score: 1.65,
      production_forecast_results: forecasts,
      production_mps_table: updatedMps,
      production_capacity_mode: "scenario",
      production_capacity_scenario: "normal",
      production_capacity: initialCapacity,
      production_custom_capacity: null,
    }),
  );
  assert(
    !("error" in secondUpdate),
    `Second updateExperiment must preserve production_forecast_evidence under MySQL 8 JSON key reordering: ${JSON.stringify(secondUpdate)}`,
  );
  const [[persistedExperiment]] = await admin.query<RowDataPacket[]>(
    "SELECT production_forecast_evidence, state_version FROM experiment_status WHERE experiment_id = ?",
    [experimentId],
  );
  assert.equal(persistedExperiment?.production_forecast_evidence, evidence);
  assert.equal(Number(persistedExperiment?.state_version), 2);

  // 8. P2-2: Verify report upsert & phone trim SQL produce zero MySQL 8.0/8.4 Warning 1287.
  for (const content of ["# Report v1", "# Report v2"]) {
    await admin.query(
      `INSERT INTO experiment_reports (experiment_id, student_id, report_content, pdf_file_path, status, submitted_at)
       VALUES (?, ?, ?, ?, 'submitted', NOW())
       ON DUPLICATE KEY UPDATE
         report_content = ?,
         pdf_file_path = ?,
         status = 'submitted',
         submitted_at = NOW()`,
      [experimentId, studentId, content, "/tmp/report.pdf", content, "/tmp/report.pdf"],
    );
    await assertNoDeprecationWarnings(admin, "experiment_reports upsert");
  }
  const [[persistedReport]] = await admin.query<RowDataPacket[]>(
    "SELECT report_content FROM experiment_reports WHERE experiment_id = ? AND student_id = ?",
    [experimentId, studentId],
  );
  assert.equal(persistedReport?.report_content, "# Report v2");

  const [trimResult] = await admin.query<ResultSetHeader>(
    "UPDATE users SET phone_number = TRIM(phone_number) WHERE phone_number IS NOT NULL AND CHAR_LENGTH(phone_number) <> CHAR_LENGTH(TRIM(phone_number))",
  );
  await assertNoDeprecationWarnings(admin, "users phone_number trim");
  assert(trimResult.affectedRows >= 1, "CHAR_LENGTH trim query should normalize whitespace phone");

  // 9. P3: Verify email search with '@' and hyphen does not trigger FULLTEXT 1064 error,
  // and verify non-contiguous auto_increment_increment user reload.
  const emailSearch = await searchUsers({ q: "dbcompat@example.test", page: 1, limit: 10 });
  assert(!("error" in emailSearch), "Email search with '@' must succeed");
  assert.equal(emailSearch.data.searchMethod, "like");
  assert.equal(emailSearch.data.data.length, 1);
  assert.equal(emailSearch.data.data[0]?.username, "dbcompat_stu");

  const hyphenSearch = await searchUsers({ q: "dbcompat-hyphen", page: 1, limit: 10 });
  assert(!("error" in hyphenSearch), "Hyphenated search must succeed");
  assert.equal(hyphenSearch.data.searchMethod, "like");

  const dottedSearch = await searchUsers({ q: "dbcompat_stu", page: 1, limit: 10 });
  assert(!("error" in dottedSearch), "Standard alphanumeric search must succeed");
  assert.equal(dottedSearch.data.searchMethod, "fulltext");
  assert.equal(dottedSearch.data.data.length, 1);

  await admin.query("SET SESSION auto_increment_increment = 2");
  await admin.query(
    `INSERT INTO users (username, password_hash, full_name, email, role)
     VALUES ('step_u1', 'unused', '步长用户1', 'step_u1@example.test', 'Student'),
            ('step_u2', 'unused', '步长用户2', 'step_u2@example.test', 'Student')`,
  );
  const [stepUsers] = await admin.query<RowDataPacket[]>(
    "SELECT user_id, username FROM users WHERE username IN (?, ?) ORDER BY user_id ASC",
    ["step_u1", "step_u2"],
  );
  assert.equal(stepUsers.length, 2);
  assert(
    Number(stepUsers[1]!.user_id) - Number(stepUsers[0]!.user_id) >= 2,
    "Non-contiguous auto_increment_increment should produce spaced IDs",
  );

  console.log("MySQL 8 / MariaDB dual-engine compatibility system test passed.");
} finally {
  if (pool) {
    await pool.end();
  }
  if (ownsSchema) {
    const [tables] = await admin.query<RowDataPacket[]>(
      `SELECT TABLE_NAME AS table_name
         FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND table_type = 'BASE TABLE'`,
    );
    if (tables.length > 0) {
      await admin.query("SET FOREIGN_KEY_CHECKS = 0");
      try {
        await admin.query(
          `DROP TABLE IF EXISTS ${tables.map((row) => `\`${String(row.table_name).replaceAll("`", "``")}\``).join(", ")}`,
        );
      } finally {
        await admin.query("SET FOREIGN_KEY_CHECKS = 1");
      }
    }
  }
  await admin.end();
}
