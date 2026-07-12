import mysql from "mysql2/promise";
import {
  addColumnIfMissing,
  assertTableExists,
  ensureGuidedTrainingIntegrityColumns,
} from "./model-training-migration-columns";

const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE } = process.env;

function requireEnv(name: string, value: string | undefined): string {
  const normalized = value?.trim();
  if (!normalized) throw new Error(`Missing required env var: ${name}`);
  return normalized;
}

const connection = await mysql.createConnection({
  host: requireEnv("DB_HOST", DB_HOST),
  user: requireEnv("DB_USER", DB_USER),
  password: DB_PASSWORD ?? "",
  database: requireEnv("DB_DATABASE", DB_DATABASE),
  multipleStatements: true,
  charset: "utf8mb4",
});

try {
  await assertTableExists(
    connection,
    "model_guided_training_sessions",
    "Run `bun run db:migrate:guided-training` first, or use `bun run db:migrate:model-training`.",
  );
  await addColumnIfMissing(
    connection,
    "experiment_status",
    "state_version",
    "BIGINT UNSIGNED NOT NULL DEFAULT 0 AFTER current_step",
  );
  await ensureGuidedTrainingIntegrityColumns(connection);

  await connection.query(`
    CREATE TABLE IF NOT EXISTS model_training_runs (
      run_id CHAR(36) PRIMARY KEY,
      session_id CHAR(36) DEFAULT NULL,
      experiment_id INT NOT NULL,
      student_id INT NOT NULL,
      model_type ENUM('ma', 'es', 'arima', 'lstm', 'weighted_avg', 'stacking', 'boosting') NOT NULL,
      request_hash CHAR(64) NOT NULL,
      dataset_revision CHAR(64) NOT NULL,
      runtime_revision CHAR(64) NOT NULL,
      parameters JSON NOT NULL,
      metrics JSON NOT NULL,
      artifact_path VARCHAR(1024) NOT NULL,
      production_artifact_path VARCHAR(1024) DEFAULT NULL,
      is_active BOOLEAN NOT NULL DEFAULT true,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      activated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

      FOREIGN KEY (experiment_id) REFERENCES experiment_status(experiment_id) ON DELETE CASCADE,
      FOREIGN KEY (student_id) REFERENCES users(user_id) ON DELETE CASCADE,
      FOREIGN KEY (session_id) REFERENCES model_guided_training_sessions(session_id) ON DELETE SET NULL,

      UNIQUE KEY uniq_model_training_run_session (session_id),
      INDEX idx_model_training_run_active (experiment_id, student_id, model_type, is_active),
      INDEX idx_model_training_run_request (experiment_id, student_id, model_type, request_hash)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await connection.query(`
    CREATE TABLE IF NOT EXISTS model_training_run_dependencies (
      run_id CHAR(36) NOT NULL,
      base_model_type ENUM('ma', 'es', 'arima', 'lstm') NOT NULL,
      base_run_id CHAR(36) DEFAULT NULL,
      PRIMARY KEY (run_id, base_model_type),
      FOREIGN KEY (run_id) REFERENCES model_training_runs(run_id) ON DELETE CASCADE,
      FOREIGN KEY (base_run_id) REFERENCES model_training_runs(run_id) ON DELETE SET NULL,
      INDEX idx_model_training_dependency_base (base_run_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  console.log("Model training integrity migration complete.");
} catch (error) {
  console.error("Model training integrity migration failed:", error);
  process.exitCode = 1;
} finally {
  await connection.end();
}
