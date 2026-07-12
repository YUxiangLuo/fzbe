import mysql from "mysql2/promise";

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
  const [experimentStateVersionColumns] = await connection.query(
    `SELECT COLUMN_NAME
       FROM INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'experiment_status'
        AND COLUMN_NAME = 'state_version'
      LIMIT 1`,
  );
  if (!Array.isArray(experimentStateVersionColumns) || experimentStateVersionColumns.length === 0) {
    await connection.query(`
      ALTER TABLE experiment_status
        ADD COLUMN state_version BIGINT UNSIGNED NOT NULL DEFAULT 0 AFTER student_id
    `);
  }

  const [trainingInputVersionColumns] = await connection.query(
    `SELECT COLUMN_NAME
       FROM INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'experiment_status'
        AND COLUMN_NAME = 'training_input_version'
      LIMIT 1`,
  );
  if (!Array.isArray(trainingInputVersionColumns) || trainingInputVersionColumns.length === 0) {
    await connection.query(`
      ALTER TABLE experiment_status
        ADD COLUMN training_input_version BIGINT UNSIGNED NOT NULL DEFAULT 0 AFTER state_version
    `);
  }

  await connection.query(`
    CREATE TABLE IF NOT EXISTS model_guided_training_sessions (
      session_id CHAR(36) PRIMARY KEY,
      experiment_id INT NOT NULL,
      student_id INT NOT NULL,
      model_type ENUM('ma', 'es', 'arima', 'lstm', 'weighted_avg', 'stacking', 'boosting') NOT NULL,
      request_hash CHAR(64) NOT NULL,
      request_payload JSON NOT NULL,
      status ENUM('ready', 'running', 'failed', 'completed') NOT NULL DEFAULT 'ready',
      current_step_id VARCHAR(64) DEFAULT NULL,
      completed_step_ids JSON NOT NULL,
      step_outputs JSON NOT NULL,
      step_plan JSON NOT NULL,
      result_payload JSON DEFAULT NULL,
      artifact_dir VARCHAR(1024) NOT NULL,
      error_message TEXT DEFAULT NULL,
      training_input_version BIGINT UNSIGNED NOT NULL DEFAULT 0,
      run_token CHAR(36) DEFAULT NULL,
      run_version BIGINT UNSIGNED NOT NULL DEFAULT 0,
      lease_expires_at DATETIME(3) DEFAULT NULL,
      last_heartbeat_at DATETIME(3) DEFAULT NULL,
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
      INDEX idx_guided_training_status (status, expires_at),
      INDEX idx_guided_training_lease (status, lease_expires_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  await connection.query(`
    ALTER TABLE model_guided_training_sessions
      MODIFY model_type ENUM('ma', 'es', 'arima', 'lstm', 'weighted_avg', 'stacking', 'boosting') NOT NULL
  `);

  const [columns] = await connection.query(
    `
      SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
       WHERE TABLE_SCHEMA = DATABASE()
         AND TABLE_NAME = 'model_guided_training_sessions'
         AND COLUMN_NAME = 'step_plan'
      LIMIT 1
    `,
  );

  if (!Array.isArray(columns) || columns.length === 0) {
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

  const ensureColumn = async (columnName: string, definition: string) => {
    const [existingColumns] = await connection.query(
      `SELECT COLUMN_NAME
         FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'model_guided_training_sessions'
          AND COLUMN_NAME = ?
        LIMIT 1`,
      [columnName],
    );
    if (Array.isArray(existingColumns) && existingColumns.length > 0) return;
    await connection.query(
      `ALTER TABLE model_guided_training_sessions ADD COLUMN \`${columnName}\` ${definition}`,
    );
  };

  await ensureColumn("run_token", "CHAR(36) DEFAULT NULL AFTER error_message");
  await ensureColumn(
    "training_input_version",
    "BIGINT UNSIGNED NOT NULL DEFAULT 0 AFTER error_message",
  );
  await ensureColumn("run_version", "BIGINT UNSIGNED NOT NULL DEFAULT 0 AFTER run_token");
  await ensureColumn("lease_expires_at", "DATETIME(3) DEFAULT NULL AFTER run_version");
  await ensureColumn("last_heartbeat_at", "DATETIME(3) DEFAULT NULL AFTER lease_expires_at");

  const [leaseIndexes] = await connection.query(
    `SELECT INDEX_NAME
       FROM INFORMATION_SCHEMA.STATISTICS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'model_guided_training_sessions'
        AND INDEX_NAME = 'idx_guided_training_lease'
      LIMIT 1`,
  );
  if (!Array.isArray(leaseIndexes) || leaseIndexes.length === 0) {
    await connection.query(`
      ALTER TABLE model_guided_training_sessions
        ADD INDEX idx_guided_training_lease (status, lease_expires_at)
    `);
  }

  await connection.query(`
    CREATE TABLE IF NOT EXISTS experiment_model_artifacts (
      experiment_id INT NOT NULL,
      student_id INT NOT NULL,
      model_type ENUM('ma', 'es', 'arima', 'lstm', 'weighted_avg', 'stacking', 'boosting') NOT NULL,
      revision_token CHAR(36) NOT NULL,
      training_input_version BIGINT UNSIGNED NOT NULL,
      source_session_id CHAR(36) DEFAULT NULL,
      backtest_artifact_path VARCHAR(1024) NOT NULL,
      request_payload JSON NOT NULL,
      result_payload JSON NOT NULL,
      production_artifact_path VARCHAR(1024) DEFAULT NULL,
      production_revision_token CHAR(36) DEFAULT NULL,
      activated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
      updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),

      PRIMARY KEY (experiment_id, model_type),
      INDEX idx_experiment_model_artifact_student (student_id, experiment_id),
      INDEX idx_experiment_model_artifact_session (source_session_id),
      FOREIGN KEY (experiment_id) REFERENCES experiment_status(experiment_id) ON DELETE CASCADE,
      FOREIGN KEY (student_id) REFERENCES users(user_id) ON DELETE CASCADE,
      FOREIGN KEY (source_session_id) REFERENCES model_guided_training_sessions(session_id) ON DELETE RESTRICT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);

  const [artifactVersionColumns] = await connection.query(
    `SELECT COLUMN_NAME
       FROM INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'experiment_model_artifacts'
        AND COLUMN_NAME = 'training_input_version'
      LIMIT 1`,
  );
  if (!Array.isArray(artifactVersionColumns) || artifactVersionColumns.length === 0) {
    await connection.query(`
      ALTER TABLE experiment_model_artifacts
        ADD COLUMN training_input_version BIGINT UNSIGNED NOT NULL DEFAULT 0 AFTER revision_token
    `);
  }

  console.log("Guided training session migration complete.");
} catch (error) {
  console.error("Guided training session migration failed:", error);
  process.exitCode = 1;
} finally {
  await connection.end();
}
