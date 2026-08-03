import { type RowDataPacket } from "mysql2/promise";
import { createMigrationConnection } from "./migration-database";

const connection = await createMigrationConnection();

const ensureColumn = async (tableName: string, columnName: string, definition: string) => {
  const [rows] = await connection.query<RowDataPacket[]>(
    `SELECT 1
       FROM information_schema.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = ?
        AND COLUMN_NAME = ?
      LIMIT 1`,
    [tableName, columnName],
  );
  if (rows.length === 0) {
    await connection.query(`ALTER TABLE \`${tableName}\` ADD COLUMN \`${columnName}\` ${definition}`);
  }
};

const ensureIndex = async (tableName: string, indexName: string, definition: string) => {
  const [rows] = await connection.query<RowDataPacket[]>(
    `SELECT 1
       FROM information_schema.STATISTICS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = ?
        AND INDEX_NAME = ?
      LIMIT 1`,
    [tableName, indexName],
  );
  if (rows.length === 0) {
    await connection.query(`ALTER TABLE \`${tableName}\` ADD INDEX \`${indexName}\` (${definition})`);
  }
};

try {
  await connection.query(`
    CREATE TABLE IF NOT EXISTS model_production_preparation_sessions (
      session_id CHAR(36) PRIMARY KEY,
      experiment_id INT NOT NULL,
      student_id INT NOT NULL,
      run_id CHAR(36) NOT NULL,
      model_type ENUM('ma', 'es', 'arima', 'lstm', 'weighted_avg', 'stacking', 'boosting') NOT NULL,
      request_hash CHAR(64) NOT NULL,
      dataset_revision CHAR(64) NOT NULL,
      runtime_revision CHAR(64) NOT NULL,
      protocol_version SMALLINT UNSIGNED NOT NULL DEFAULT 1,
      forecast_steps TINYINT UNSIGNED NOT NULL,
      status ENUM('ready', 'running', 'failed', 'completed', 'superseded') NOT NULL DEFAULT 'ready',
      current_step_id VARCHAR(64) DEFAULT 'initialize',
      completed_step_ids JSON NOT NULL,
      step_plan JSON NOT NULL,
      artifact_dir VARCHAR(1024) NOT NULL,
      candidate_model_path VARCHAR(1024) NOT NULL,
      committed_state_path VARCHAR(1024) DEFAULT NULL,
      committed_state_sha256 CHAR(64) DEFAULT NULL,
      active_attempt_id CHAR(36) DEFAULT NULL,
      observed_production_revision CHAR(36) DEFAULT NULL,
      production_revision CHAR(36) DEFAULT NULL,
      error_message TEXT DEFAULT NULL,
      superseded_reason TEXT DEFAULT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      expires_at TIMESTAMP NULL DEFAULT NULL,
      FOREIGN KEY (experiment_id) REFERENCES experiment_status(experiment_id) ON DELETE CASCADE,
      FOREIGN KEY (student_id) REFERENCES users(user_id) ON DELETE CASCADE,
      UNIQUE KEY uniq_production_preparation_request (experiment_id, student_id, model_type, request_hash),
      INDEX idx_production_preparation_status (status, expires_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);
  await connection.query(`
    CREATE TABLE IF NOT EXISTS model_production_preparation_step_attempts (
      attempt_id CHAR(36) PRIMARY KEY,
      session_id CHAR(36) NOT NULL,
      step_id VARCHAR(64) NOT NULL,
      status ENUM('running', 'committed', 'failed') NOT NULL DEFAULT 'running',
      input_state_path VARCHAR(1024) DEFAULT NULL,
      input_state_sha256 CHAR(64) DEFAULT NULL,
      candidate_state_path VARCHAR(1024) NOT NULL,
      candidate_state_sha256 CHAR(64) DEFAULT NULL,
      receipt_path VARCHAR(1024) NOT NULL,
      output_payload JSON DEFAULT NULL,
      error_message TEXT DEFAULT NULL,
      lease_expires_at TIMESTAMP NULL DEFAULT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      FOREIGN KEY (session_id) REFERENCES model_production_preparation_sessions(session_id) ON DELETE CASCADE,
      INDEX idx_production_attempt_session_step (session_id, step_id, created_at),
      INDEX idx_production_attempt_lease (status, lease_expires_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);
  await ensureColumn(
    "model_production_preparation_sessions",
    "observed_production_revision",
    "CHAR(36) DEFAULT NULL AFTER active_attempt_id",
  );
  await ensureColumn(
    "model_production_preparation_step_attempts",
    "lease_expires_at",
    "TIMESTAMP NULL DEFAULT NULL AFTER error_message",
  );
  await ensureIndex(
    "model_production_preparation_step_attempts",
    "idx_production_attempt_lease",
    "\`status\`, \`lease_expires_at\`",
  );
  console.log("Production preparation session migration complete.");
} finally {
  await connection.end();
}
