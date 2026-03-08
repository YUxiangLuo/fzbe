import type { RowDataPacket, ResultSetHeader } from "mysql2/promise";
import { pool } from "../../config";
import type { CreateExperimentInput, UpdateExperimentInput, ExperimentEventInput } from "./experiments.schema";
import { prepareStudentExperimentUpdate } from "./experiment-update-policy";

const BOOLEAN_FIELDS = new Set(["moving_average_completed", "exponential_smoothing_completed", "arima_completed", "lstm_completed", "ensemble_weighted_completed", "ensemble_boosting_completed", "ensemble_stacking_completed", "quiz_about_model_completed", "quiz_about_plan_completed", "production_plan_completed"]);
const JSON_FIELDS = new Set(["arima_adf_stationarity", "lstm_features", "ensemble_weighted_base_models", "ensemble_boosting_base_models", "ensemble_stacking_base_models", "production_mps_table", "production_forecast_results"]);
const INTEGER_FIELDS = new Set(["experiment_id", "student_id", "highest_completed_step", "current_step", "data_window_train_start_index", "data_window_train_end_index", "data_window_evaluate_start_index", "data_window_evaluate_end_index", "moving_average_window", "arima_p", "arima_d", "arima_q", "production_forecast_periods", "production_capacity", "production_custom_capacity", "total_active_duration_seconds"]);
const FLOAT_FIELDS = new Set(["moving_average_metrics_rmse", "moving_average_metrics_mae", "moving_average_metrics_r2", "exponential_smoothing_alpha", "exponential_smoothing_metrics_rmse", "exponential_smoothing_metrics_mae", "exponential_smoothing_metrics_r2", "arima_metrics_rmse", "arima_metrics_mae", "arima_metrics_r2", "lstm_metrics_rmse", "lstm_metrics_mae", "lstm_metrics_r2", "ensemble_weighted_metrics_rmse", "ensemble_weighted_metrics_mae", "ensemble_weighted_metrics_r2", "ensemble_boosting_metrics_rmse", "ensemble_boosting_metrics_mae", "ensemble_boosting_metrics_r2", "ensemble_stacking_metrics_rmse", "ensemble_stacking_metrics_mae", "ensemble_stacking_metrics_r2", "production_initial_inventory", "production_target_service_level", "production_safety_stock_z_score"]);
const datetimeFields = new Set(["start_time", "last_activity_at", "completion_time"]);

function sanitizeForDb(data: Record<string, unknown>) {
  const sanitized: Record<string, unknown> = {};
  for (const [key, rawValue] of Object.entries(data)) {
    if (rawValue === undefined) continue;
    if (rawValue === null) { sanitized[key] = null; continue; }
    if (BOOLEAN_FIELDS.has(key)) { sanitized[key] = rawValue === true || rawValue === 1 ? 1 : 0; continue; }
    if (JSON_FIELDS.has(key)) { sanitized[key] = typeof rawValue === "string" ? rawValue : JSON.stringify(rawValue); continue; }
    if (datetimeFields.has(key)) {
      if (rawValue instanceof Date) { sanitized[key] = rawValue; continue; }
      if (typeof rawValue === "string") {
        const trimmed = rawValue.trim();
        if (!trimmed) { sanitized[key] = null; continue; }
        const parsed = new Date(trimmed);
        sanitized[key] = Number.isNaN(parsed.getTime()) ? trimmed : parsed;
        continue;
      }
    }
    if (rawValue instanceof Date) { sanitized[key] = rawValue; continue; }
    if (typeof rawValue === "string") {
      const trimmed = rawValue.trim();
      sanitized[key] = trimmed.length ? trimmed : null;
      continue;
    }
    sanitized[key] = rawValue;
  }
  return sanitized;
}

function parseJsonField(value: unknown) {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") { try { return JSON.parse(value); } catch { return value; } }
  return value;
}

function normalizeBoolean(value: unknown) {
  if (value === null || value === undefined) return null;
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value === 1;
  if (typeof value === "string") {
    const lowered = value.toLowerCase();
    if (lowered === "1" || lowered === "true") return true;
    if (lowered === "0" || lowered === "false") return false;
  }
  return Boolean(value);
}

function normalizeNumeric(value: unknown) {
  if (value === null || value === undefined) return null;
  const num = Number(value);
  return Number.isNaN(num) ? null : num;
}

export function normalizeExperimentRow(row: RowDataPacket) {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(row)) {
    if (JSON_FIELDS.has(key)) { result[key] = parseJsonField(value); continue; }
    if (BOOLEAN_FIELDS.has(key)) { result[key] = normalizeBoolean(value); continue; }
    if (INTEGER_FIELDS.has(key) || FLOAT_FIELDS.has(key)) { result[key] = normalizeNumeric(value); continue; }
    if (datetimeFields.has(key) && value instanceof Date) { result[key] = value.toISOString(); continue; }
    result[key] = value ?? null;
  }
  return result;
}

async function fetchExperimentById(experimentId: number) {
  const [rows] = await pool.query<RowDataPacket[]>("SELECT * FROM experiment_status WHERE experiment_id = ? LIMIT 1", [experimentId]);
  return rows[0];
}

export async function getActiveExperiment(studentId: number) {
  const [rows] = await pool.query<RowDataPacket[]>("SELECT * FROM experiment_status WHERE student_id = ? ORDER BY experiment_id DESC LIMIT 1", [studentId]);
  if (!rows[0] || rows[0].status === "Completed") return null;
  return normalizeExperimentRow(rows[0]);
}

export async function createExperiment(studentId: number, data: CreateExperimentInput) {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();
    const sanitized = sanitizeForDb({ ...data, student_id: studentId });
    delete sanitized.total_active_duration_seconds;
    const columns = Object.keys(sanitized);
    const placeholders = columns.map(() => "?").join(", ");
    const values = columns.map((key) => sanitized[key]);

    const [insertResult] = await connection.query<ResultSetHeader>(`INSERT INTO experiment_status (${columns.join(", ")}) VALUES (${placeholders})`, values);
    const experimentId = insertResult.insertId;
    await connection.query("INSERT INTO experiment_grade (experiment_id, student_id) VALUES (?, ?)", [experimentId, studentId]);
    await connection.commit();

    const row = await fetchExperimentById(experimentId);
    return { data: normalizeExperimentRow(row!) };
  } catch (error) {
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}

export async function getAllExperiments(studentId: number) {
  const [rows] = await pool.query<RowDataPacket[]>("SELECT * FROM experiment_status WHERE student_id = ? ORDER BY experiment_id DESC", [studentId]);
  return rows.map(normalizeExperimentRow);
}

export async function deleteExperiment(experimentId: number, studentId: number) {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();
    const [[experimentRow]] = await connection.query<RowDataPacket[]>("SELECT status, student_id FROM experiment_status WHERE experiment_id = ? FOR UPDATE", [experimentId]);
    if (!experimentRow || experimentRow.student_id !== studentId) {
      await connection.rollback();
      return { error: "not_found" as const };
    }
    if (experimentRow.status === "Completed") {
      await connection.rollback();
      return { error: "completed" as const };
    }
    await connection.query("DELETE FROM experiment_status WHERE experiment_id = ?", [experimentId]);
    await connection.commit();
    return { success: true };
  } catch (error) {
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}

export async function updateExperiment(experimentId: number, studentId: number, data: UpdateExperimentInput) {
  const row = await fetchExperimentById(experimentId);
  if (!row) return { error: "not_found" as const };
  if (row.student_id !== studentId) return { error: "forbidden" as const };

  const sanitized = sanitizeForDb(data);
  const preparedUpdate = prepareStudentExperimentUpdate(row as typeof row & {
    status: string;
    start_time: string | Date | null;
    last_activity_at: string | Date | null;
    total_active_duration_seconds: number | null;
  }, sanitized);
  if ("error" in preparedUpdate) return preparedUpdate;

  const columns = Object.keys(preparedUpdate.data);
  const assignments = columns.map((col) => `${col} = ?`).join(", ");
  const values = columns.map((key) => preparedUpdate.data[key]);
  values.push(experimentId);

  await pool.query(`UPDATE experiment_status SET ${assignments} WHERE experiment_id = ?`, values);
  const updatedRow = await fetchExperimentById(experimentId);
  return { data: normalizeExperimentRow(updatedRow!) };
}

export async function createExperimentEvent(experimentId: number, studentId: number, input: ExperimentEventInput) {
  const connection = await pool.getConnection();
  try {
    const [experimentRows] = await connection.query<RowDataPacket[]>("SELECT student_id, status FROM experiment_status WHERE experiment_id = ? LIMIT 1", [experimentId]);
    if (experimentRows.length === 0) return { error: "not_found" as const };
    if (experimentRows[0]!.student_id !== studentId) return { error: "forbidden" as const };
    if (experimentRows[0]!.status === "Completed") return { error: "completed" as const };

    let transactionStarted = false;
    if (input.event_type === "COMPLETED") {
      await connection.beginTransaction();
      transactionStarted = true;

      const [latestEventRows] = await connection.query<RowDataPacket[]>(
        "SELECT step_order FROM experiment_step_events WHERE experiment_id = ? ORDER BY event_timestamp DESC LIMIT 1 FOR UPDATE",
        [experimentId],
      );
      const latestStepOrder = (latestEventRows[0] as { step_order?: number } | undefined)?.step_order ?? 0;

      if (latestStepOrder > input.step_order) {
        const syntheticStartedTimestamp = new Date(Date.now() - 10000);
        await connection.query<ResultSetHeader>(
          "INSERT INTO experiment_step_events (experiment_id, student_id, step_order, event_type, event_timestamp) VALUES (?, ?, ?, ?, ?)",
          [experimentId, studentId, input.step_order, "STARTED", syntheticStartedTimestamp],
        );
      }
    }

    const [result] = await connection.query<ResultSetHeader>(
      "INSERT INTO experiment_step_events (experiment_id, student_id, step_order, event_type) VALUES (?, ?, ?, ?)",
      [experimentId, studentId, input.step_order, input.event_type],
    );

    if (transactionStarted) await connection.commit();
    return { data: { event_id: result.insertId } };
  } catch (error) {
    await connection.rollback().catch(() => {});
    throw error;
  } finally {
    connection.release();
  }
}
