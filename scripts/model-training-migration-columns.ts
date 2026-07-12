import type { Connection, RowDataPacket } from "mysql2/promise";

export async function hasTable(connection: Connection, tableName: string) {
  const [rows] = await connection.query<RowDataPacket[]>(
    `SELECT 1
       FROM INFORMATION_SCHEMA.TABLES
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = ?
      LIMIT 1`,
    [tableName],
  );
  return rows.length > 0;
}

export async function assertTableExists(
  connection: Connection,
  tableName: string,
  migrationHint: string,
) {
  if (!await hasTable(connection, tableName)) {
    throw new Error(`Missing required table ${tableName}. ${migrationHint}`);
  }
}

export async function hasColumn(
  connection: Connection,
  tableName: string,
  columnName: string,
) {
  const [rows] = await connection.query<RowDataPacket[]>(
    `SELECT 1
       FROM INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = ?
        AND COLUMN_NAME = ?
      LIMIT 1`,
    [tableName, columnName],
  );
  return rows.length > 0;
}

export async function addColumnIfMissing(
  connection: Connection,
  tableName: string,
  columnName: string,
  definition: string,
) {
  if (!await hasColumn(connection, tableName, columnName)) {
    await connection.query(`ALTER TABLE ${tableName} ADD COLUMN ${columnName} ${definition}`);
  }
}

export async function ensureGuidedTrainingIntegrityColumns(connection: Connection) {
  await assertTableExists(
    connection,
    "model_guided_training_sessions",
    "Run `bun run db:migrate:guided-training` first, or use `bun run db:migrate:model-training`.",
  );
  await addColumnIfMissing(
    connection,
    "model_guided_training_sessions",
    "dataset_revision",
    "CHAR(64) NULL AFTER request_hash",
  );
  await addColumnIfMissing(
    connection,
    "model_guided_training_sessions",
    "runtime_revision",
    "CHAR(64) NULL AFTER dataset_revision",
  );
  await addColumnIfMissing(
    connection,
    "model_guided_training_sessions",
    "experiment_state_version",
    "BIGINT UNSIGNED NULL AFTER error_message",
  );

  await connection.query(`
    UPDATE model_guided_training_sessions
       SET dataset_revision = COALESCE(dataset_revision, REPEAT('0', 64)),
           runtime_revision = COALESCE(runtime_revision, REPEAT('0', 64))
  `);
  await connection.query(`
    ALTER TABLE model_guided_training_sessions
      MODIFY dataset_revision CHAR(64) NOT NULL,
      MODIFY runtime_revision CHAR(64) NOT NULL
  `);
}
