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

  console.log("Guided training session migration complete.");
} catch (error) {
  console.error("Guided training session migration failed:", error);
  process.exitCode = 1;
} finally {
  await connection.end();
}
