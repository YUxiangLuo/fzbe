import mysql from 'mysql2/promise';

const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE } = process.env;

const requireEnv = (name: string, value: string | undefined): string => {
  const normalized = value?.trim();
  if (!normalized) throw new Error(`Missing required env var: ${name}`);
  return normalized;
};

const metricColumns = [
  'moving_average_metrics_mape',
  'exponential_smoothing_metrics_mape',
  'arima_metrics_mape',
  'lstm_metrics_mape',
  'ensemble_weighted_metrics_mape',
  'ensemble_boosting_metrics_mape',
  'ensemble_stacking_metrics_mape',
] as const;

const connection = await mysql.createConnection({
  host: requireEnv('DB_HOST', DB_HOST),
  user: requireEnv('DB_USER', DB_USER),
  password: DB_PASSWORD ?? '',
  database: requireEnv('DB_DATABASE', DB_DATABASE),
  charset: 'utf8mb4',
});

try {
  const [rows] = await connection.query<mysql.RowDataPacket[]>(`
    SELECT COLUMN_NAME
      FROM INFORMATION_SCHEMA.COLUMNS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'experiment_status'
       AND COLUMN_NAME IN (${metricColumns.map(() => '?').join(', ')})
  `, [...metricColumns]);
  const existingColumns = new Set(rows.map(row => String(row.COLUMN_NAME)));

  for (const column of metricColumns) {
    if (existingColumns.has(column)) continue;
    await connection.query(`ALTER TABLE experiment_status ADD COLUMN \`${column}\` FLOAT DEFAULT NULL`);
  }

  console.log('Model MAPE metric migration complete.');
} catch (error) {
  console.error('Model MAPE metric migration failed:', error);
  process.exitCode = 1;
} finally {
  await connection.end();
}
