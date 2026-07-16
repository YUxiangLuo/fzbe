import mysql from "mysql2/promise";
import {
  addColumnIfMissing,
  ensureGuidedTrainingIntegrityColumns,
  hasColumn,
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
  await connection.query(`
    CREATE TABLE IF NOT EXISTS model_guided_training_sessions (
      session_id CHAR(36) PRIMARY KEY,
      experiment_id INT NOT NULL,
      student_id INT NOT NULL,
      model_type ENUM('ma', 'es', 'arima', 'lstm', 'weighted_avg', 'stacking', 'boosting') NOT NULL,
      request_hash CHAR(64) NOT NULL,
      dataset_revision CHAR(64) NOT NULL,
      runtime_revision CHAR(64) NOT NULL,
      request_payload JSON NOT NULL,
      status ENUM('ready', 'running', 'failed', 'completed', 'superseded') NOT NULL DEFAULT 'ready',
      current_step_id VARCHAR(64) DEFAULT NULL,
      completed_step_ids JSON NOT NULL,
      step_outputs JSON NOT NULL,
      step_plan JSON NOT NULL,
      result_payload JSON DEFAULT NULL,
      artifact_dir VARCHAR(1024) NOT NULL,
      error_message TEXT DEFAULT NULL,
      experiment_state_version BIGINT UNSIGNED DEFAULT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      expires_at TIMESTAMP NULL DEFAULT NULL,

      FOREIGN KEY (experiment_id)
        REFERENCES experiment_status(experiment_id)
        ON DELETE CASCADE,
      FOREIGN KEY (student_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,

      UNIQUE KEY uniq_guided_training_request (experiment_id, student_id, model_type, request_hash),
      INDEX idx_guided_training_student (student_id, experiment_id, model_type, updated_at),
      INDEX idx_guided_training_status (status, expires_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await connection.query(`
    ALTER TABLE model_guided_training_sessions
      MODIFY model_type ENUM('ma', 'es', 'arima', 'lstm', 'weighted_avg', 'stacking', 'boosting') NOT NULL,
      MODIFY status ENUM('ready', 'running', 'failed', 'completed', 'superseded') NOT NULL DEFAULT 'ready'
  `);

  await ensureGuidedTrainingIntegrityColumns(connection);

  if (!await hasColumn(connection, "model_guided_training_sessions", "step_plan")) {
    await connection.query(`
      ALTER TABLE model_guided_training_sessions
        ADD COLUMN step_plan JSON NULL AFTER step_outputs
    `);
  }

  await connection.query(`
    UPDATE model_guided_training_sessions
       SET step_plan = JSON_ARRAY()
     WHERE step_plan IS NULL
  `);

  await connection.query(`
    ALTER TABLE model_guided_training_sessions
      MODIFY step_plan JSON NOT NULL
  `);

  await addColumnIfMissing(
    connection,
    "model_guided_training_sessions",
    "protocol_version",
    "SMALLINT UNSIGNED NULL AFTER runtime_revision",
  );
  await addColumnIfMissing(
    connection,
    "model_guided_training_sessions",
    "committed_state_path",
    "VARCHAR(1024) NULL AFTER artifact_dir",
  );
  await addColumnIfMissing(
    connection,
    "model_guided_training_sessions",
    "committed_state_sha256",
    "CHAR(64) NULL AFTER committed_state_path",
  );
  await addColumnIfMissing(
    connection,
    "model_guided_training_sessions",
    "active_attempt_id",
    "CHAR(36) NULL AFTER committed_state_sha256",
  );
  await addColumnIfMissing(
    connection,
    "model_guided_training_sessions",
    "superseded_reason",
    "TEXT NULL AFTER error_message",
  );

  await connection.query(`
    UPDATE model_guided_training_sessions
       SET protocol_version = 1
     WHERE protocol_version IS NULL
  `);
  await connection.query(`
    UPDATE model_guided_training_sessions
       SET status = 'superseded',
           current_step_id = NULL,
           superseded_reason = 'Guided training checkpoint protocol was upgraded; create a new session.'
     WHERE protocol_version < 4
       AND status IN ('ready', 'running', 'failed')
  `);

  await connection.query(`
    UPDATE model_guided_training_sessions
       SET protocol_version = COALESCE(protocol_version, 1)
  `);
  await connection.query(`
    ALTER TABLE model_guided_training_sessions
      MODIFY protocol_version SMALLINT UNSIGNED NOT NULL DEFAULT 4
  `);

  await connection.query(`
    CREATE TABLE IF NOT EXISTS model_guided_training_step_attempts (
      attempt_id CHAR(36) PRIMARY KEY,
      session_id CHAR(36) NOT NULL,
      step_id VARCHAR(64) NOT NULL,
      status ENUM('running', 'python_completed', 'committed', 'failed') NOT NULL DEFAULT 'running',
      input_state_path VARCHAR(1024) DEFAULT NULL,
      input_state_sha256 CHAR(64) DEFAULT NULL,
      candidate_state_path VARCHAR(1024) NOT NULL,
      candidate_state_sha256 CHAR(64) DEFAULT NULL,
      receipt_path VARCHAR(1024) NOT NULL,
      output_payload JSON DEFAULT NULL,
      final_result_payload JSON DEFAULT NULL,
      error_message TEXT DEFAULT NULL,
      run_count INT UNSIGNED NOT NULL DEFAULT 1,
      lease_expires_at TIMESTAMP NULL DEFAULT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

      FOREIGN KEY (session_id)
        REFERENCES model_guided_training_sessions(session_id)
        ON DELETE CASCADE,
      INDEX idx_guided_attempt_session_step (session_id, step_id, created_at),
      INDEX idx_guided_attempt_lease (status, lease_expires_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  console.log("Guided training session migration complete.");
} catch (error) {
  console.error("Guided training session migration failed:", error);
  process.exitCode = 1;
} finally {
  await connection.end();
}
