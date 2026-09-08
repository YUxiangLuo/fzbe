import type { PoolConnection, RowDataPacket, ResultSetHeader } from "mysql2/promise";
import { env, pool } from "../../config";
import {
  parseCompatibleProductionForecastResults,
  parseProductionMpsTable,
  validateCompletedProductionPlan,
} from "./experiments.schema";
import type {
  CreateExperimentInput,
  ExperimentEventInput,
  RetryExperimentInput,
  UpdateExperimentInput,
} from "./experiments.schema";
import { prepareStudentExperimentUpdate } from "./experiment-update-policy";
import {
  deactivateModelRunsForStudentUpdate,
  getServerInvalidationPatch,
} from "../models/model-run-store";
import { withDeadlockRetry } from "../../utils/db-retry";
import { databaseDatetimeToIsoString } from "../../utils/database-datetime";
import { commitConnectionWithOutcomeGuard } from "../models/model-db-guard";

const BOOLEAN_FIELDS = new Set(["moving_average_completed", "exponential_smoothing_completed", "arima_completed", "lstm_completed", "ensemble_weighted_completed", "ensemble_boosting_completed", "ensemble_stacking_completed", "quiz_about_model_completed", "quiz_about_plan_completed", "production_plan_completed"]);
const JSON_FIELDS = new Set(["selected_base_models", "selected_ensemble_models", "arima_adf_stationarity", "lstm_features", "ensemble_weighted_base_models", "ensemble_boosting_base_models", "ensemble_stacking_base_models", "production_mps_table", "production_forecast_results"]);
const INTEGER_FIELDS = new Set(["experiment_id", "student_id", "retry_of_experiment_id", "state_version", "highest_completed_step", "current_step", "data_window_train_start_index", "data_window_train_end_index", "data_window_evaluate_start_index", "data_window_evaluate_end_index", "moving_average_window", "arima_p", "arima_d", "arima_q", "production_forecast_periods", "production_capacity", "production_custom_capacity", "total_active_duration_seconds"]);
const FLOAT_FIELDS = new Set(["moving_average_metrics_rmse", "moving_average_metrics_mae", "moving_average_metrics_r2", "exponential_smoothing_alpha", "exponential_smoothing_metrics_rmse", "exponential_smoothing_metrics_mae", "exponential_smoothing_metrics_r2", "arima_metrics_rmse", "arima_metrics_mae", "arima_metrics_r2", "lstm_metrics_rmse", "lstm_metrics_mae", "lstm_metrics_r2", "ensemble_weighted_metrics_rmse", "ensemble_weighted_metrics_mae", "ensemble_weighted_metrics_r2", "ensemble_boosting_metrics_rmse", "ensemble_boosting_metrics_mae", "ensemble_boosting_metrics_r2", "ensemble_stacking_metrics_rmse", "ensemble_stacking_metrics_mae", "ensemble_stacking_metrics_r2", "production_initial_inventory", "production_target_service_level", "production_safety_stock_z_score"]);
const datetimeFields = new Set(["start_time", "last_activity_at", "completion_time"]);
const BASE_MODEL_SELECTION_ORDER = ["moving_average", "exponential_smoothing", "arima", "lstm"] as const;
const ENSEMBLE_MODEL_SELECTION_ORDER = ["weighted_ensemble", "boosting_ensemble", "stacking_ensemble"] as const;
const BASE_MODEL_SELECTION_ALIASES: Record<string, (typeof BASE_MODEL_SELECTION_ORDER)[number]> = {
  moving_average: "moving_average",
  ma: "moving_average",
  exponential_smoothing: "exponential_smoothing",
  exp: "exponential_smoothing",
  es: "exponential_smoothing",
  arima: "arima",
  lstm: "lstm",
};
const ENSEMBLE_MODEL_SELECTION_ALIASES: Record<string, (typeof ENSEMBLE_MODEL_SELECTION_ORDER)[number]> = {
  weighted_ensemble: "weighted_ensemble",
  ensemble_weighted: "weighted_ensemble",
  weighted_avg: "weighted_ensemble",
  boosting_ensemble: "boosting_ensemble",
  ensemble_boosting: "boosting_ensemble",
  boosting: "boosting_ensemble",
  stacking_ensemble: "stacking_ensemble",
  ensemble_stacking: "stacking_ensemble",
  stacking: "stacking_ensemble",
};
const BEST_MODEL_COMPLETION_FIELDS: Record<string, string> = {
  ma: "moving_average_completed",
  exp: "exponential_smoothing_completed",
  arima: "arima_completed",
  lstm: "lstm_completed",
  ensemble_weighted: "ensemble_weighted_completed",
  ensemble_boosting: "ensemble_boosting_completed",
  ensemble_stacking: "ensemble_stacking_completed",
};
const MODEL_SELECTION_JSON_FIELDS = new Set([
  "selected_base_models",
  "selected_ensemble_models",
  "ensemble_weighted_base_models",
  "ensemble_boosting_base_models",
  "ensemble_stacking_base_models",
]);

function normalizeSelectionArray<TSelection extends string>(
  value: unknown,
  allowedOrder: readonly TSelection[],
  aliases: Record<string, TSelection>,
) {
  let items: unknown[];
  if (Array.isArray(value)) {
    items = value;
  } else if (typeof value === "string") {
    items = value
      .split(/[\s,]+/)
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
  } else {
    return [];
  }

  const selected = new Set<TSelection>();
  for (const item of items) {
    if (typeof item !== "string") {
      continue;
    }

    const normalizedItem = aliases[item];
    if (normalizedItem) {
      selected.add(normalizedItem);
    }
  }

  return allowedOrder.filter((item) => selected.has(item));
}

function normalizeModelSelectionField(key: string, value: unknown) {
  if (key === "selected_ensemble_models") {
    return normalizeSelectionArray(
      value,
      ENSEMBLE_MODEL_SELECTION_ORDER,
      ENSEMBLE_MODEL_SELECTION_ALIASES,
    );
  }

  return normalizeSelectionArray(
    value,
    BASE_MODEL_SELECTION_ORDER,
    BASE_MODEL_SELECTION_ALIASES,
  );
}

function prepareJsonFieldForDb(key: string, rawValue: unknown) {
  if (key === "production_forecast_results") {
    const parsed = parseCompatibleProductionForecastResults(rawValue);
    if (parsed === null) {
      throw new TypeError(
        "production_forecast_results 必须包含完整且版本一致的生产预测点",
      );
    }
    return JSON.stringify(parsed.data);
  }

  if (key === "production_mps_table") {
    const parsed = parseProductionMpsTable(rawValue);
    if (parsed === null) {
      throw new TypeError("production_mps_table 必须包含完整有效的 MPS 行");
    }
    return JSON.stringify(parsed);
  }

  if (MODEL_SELECTION_JSON_FIELDS.has(key)) {
    let parsedValue = rawValue;

    if (typeof rawValue === "string") {
      try {
        parsedValue = JSON.parse(rawValue);
      } catch {
        parsedValue = rawValue;
      }
    }

    return JSON.stringify(normalizeModelSelectionField(key, parsedValue));
  }

  return typeof rawValue === "string" ? rawValue : JSON.stringify(rawValue);
}

function sanitizeForDb(data: Record<string, unknown>) {
  const sanitized: Record<string, unknown> = {};
  for (const [key, rawValue] of Object.entries(data)) {
    if (rawValue === undefined) continue;
    if (rawValue === null) { sanitized[key] = null; continue; }
    if (BOOLEAN_FIELDS.has(key)) { sanitized[key] = rawValue === true || rawValue === 1 ? 1 : 0; continue; }
    if (JSON_FIELDS.has(key)) { sanitized[key] = prepareJsonFieldForDb(key, rawValue); continue; }
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
    if (JSON_FIELDS.has(key)) {
      if (key === "production_forecast_results") {
        const forecasts = parseCompatibleProductionForecastResults(value);
        // During the bounded rollout bridge the previous browser must receive
        // its own v1 snapshot back after saving it. Strict mode hides active v1
        // data so current clients regenerate instead; completed history remains
        // readable independently of the rollout flag.
        result[key] = forecasts?.version === "v1"
          && row.status !== "Completed"
          && !env.ALLOW_LEGACY_PRODUCTION_PLAN_WRITES
          ? null
          : forecasts?.data ?? null;
        continue;
      }
      if (key === "production_mps_table") {
        const forecasts = parseCompatibleProductionForecastResults(
          row.production_forecast_results,
        );
        // The original MPS payload had no row-level contract.  Keep that
        // payload available alongside a completed v1 forecast, while v2 (and
        // every active draft) continues to use the strict current schema.
        result[key] = forecasts?.version === "v1" && row.status === "Completed"
          ? parseJsonField(value)
          : parseProductionMpsTable(value);
        continue;
      }
      const parsedValue = parseJsonField(value);
      result[key] = MODEL_SELECTION_JSON_FIELDS.has(key)
        ? normalizeModelSelectionField(key, parsedValue)
        : parsedValue;
      continue;
    }
    if (BOOLEAN_FIELDS.has(key)) { result[key] = normalizeBoolean(value); continue; }
    if (INTEGER_FIELDS.has(key) || FLOAT_FIELDS.has(key)) { result[key] = normalizeNumeric(value); continue; }
    if (datetimeFields.has(key)) {
      result[key] = databaseDatetimeToIsoString(value, env.DB_TIMEZONE);
      continue;
    }
    result[key] = value ?? null;
  }
  return result;
}

async function lockStudent(connection: PoolConnection, studentId: number) {
  const [[student]] = await connection.query<RowDataPacket[]>(
    "SELECT user_id FROM users WHERE user_id = ? LIMIT 1 FOR UPDATE",
    [studentId],
  );
  return student;
}

async function insertBlankExperiment(
  connection: PoolConnection,
  studentId: number,
  retryOfExperimentId?: number,
) {
  const columns = retryOfExperimentId === undefined
    ? ["student_id"]
    : ["student_id", "retry_of_experiment_id"];
  const values = retryOfExperimentId === undefined
    ? [studentId]
    : [studentId, retryOfExperimentId];
  const placeholders = columns.map(() => "?").join(", ");
  const [insertResult] = await connection.query<ResultSetHeader>(
    `INSERT INTO experiment_status (${columns.join(", ")}) VALUES (${placeholders})`,
    values,
  );
  await connection.query(
    "INSERT INTO experiment_grade (experiment_id, student_id) VALUES (?, ?)",
    [insertResult.insertId, studentId],
  );
  return insertResult.insertId;
}

export async function getActiveExperiment(studentId: number) {
  const [rows] = await pool.query<RowDataPacket[]>("SELECT * FROM experiment_status WHERE student_id = ? ORDER BY experiment_id DESC LIMIT 1", [studentId]);
  const latestExperiment = rows[0];
  if (!latestExperiment) return null;

  if (latestExperiment.status === "Completed") {
    const [reportRows] = await pool.query<RowDataPacket[]>(
      `SELECT report_id
         FROM experiment_reports
        WHERE experiment_id = ? AND student_id = ?
        LIMIT 1`,
      [latestExperiment.experiment_id, studentId],
    );
    if (reportRows[0]) return null;
  }

  return normalizeExperimentRow(latestExperiment);
}

export async function createExperiment(studentId: number, data: CreateExperimentInput) {
  return withDeadlockRetry(() => createExperimentAttempt(studentId, data));
}

async function createExperimentAttempt(studentId: number, data: CreateExperimentInput) {
  const connection = await pool.getConnection();
  let commitAttempted = false;
  let disposed = false;
  try {
    // The users row serializes one student's creation. Avoid locking empty
    // experiment index gaps shared by other students starting at the same time.
    await connection.query("SET TRANSACTION ISOLATION LEVEL READ COMMITTED");
    await connection.beginTransaction();
    const student = await lockStudent(connection, studentId);
    if (!student) {
      await connection.rollback();
      return { error: "student_not_found" as const };
    }

    const [[existingExperiment]] = await connection.query<RowDataPacket[]>(
      `SELECT *
         FROM experiment_status
        WHERE student_id = ?
        ORDER BY experiment_id DESC
        LIMIT 1
        FOR UPDATE`,
      [studentId],
    );
    if (existingExperiment) {
      await connection.rollback();
      return {
        error: "already_exists" as const,
        data: normalizeExperimentRow(existingExperiment),
      };
    }

    const sanitized = sanitizeForDb({ ...data, student_id: studentId });
    delete sanitized.total_active_duration_seconds;
    const columns = Object.keys(sanitized);
    const placeholders = columns.map(() => "?").join(", ");
    const values = columns.map((key) => sanitized[key]);

    const [insertResult] = await connection.query<ResultSetHeader>(`INSERT INTO experiment_status (${columns.join(", ")}) VALUES (${placeholders})`, values);
    const experimentId = insertResult.insertId;
    await connection.query("INSERT INTO experiment_grade (experiment_id, student_id) VALUES (?, ?)", [experimentId, studentId]);
    const [[row]] = await connection.query<RowDataPacket[]>(
      "SELECT * FROM experiment_status WHERE experiment_id = ? LIMIT 1",
      [experimentId],
    );
    if (!row) {
      throw new Error("Created experiment could not be reloaded");
    }
    const normalized = normalizeExperimentRow(row);
    commitAttempted = true;
    await commitConnectionWithOutcomeGuard(connection);
    return { data: normalized };
  } catch (error) {
    if (commitAttempted) {
      // The commit guard destroys an ambiguous connection and returns an
      // outcome-unknown error, which must never enter transaction retry.
      disposed = true;
    } else {
      try {
        await connection.rollback();
      } catch (rollbackError) {
        disposed = true;
        connection.destroy();
        throw new Error("实验创建事务回滚失败", { cause: rollbackError });
      }
    }
    throw error;
  } finally {
    if (!disposed) connection.release();
  }
}

export async function retryRejectedExperiment(
  studentId: number,
  input: RetryExperimentInput,
) {
  return withDeadlockRetry(() => retryRejectedExperimentAttempt(studentId, input));
}

async function retryRejectedExperimentAttempt(studentId: number, input: RetryExperimentInput) {
  const connection = await pool.getConnection();
  let commitAttempted = false;
  let disposed = false;
  try {
    await connection.query("SET TRANSACTION ISOLATION LEVEL READ COMMITTED");
    await connection.beginTransaction();
    const student = await lockStudent(connection, studentId);
    if (!student) {
      await connection.rollback();
      return { error: "student_not_found" as const };
    }

    const [[rejectedReport]] = await connection.query<RowDataPacket[]>(
      `SELECT er.report_id,
              er.experiment_id,
              er.student_id AS report_student_id,
              er.status AS report_status,
              es.student_id AS experiment_student_id,
              es.status AS experiment_status
         FROM experiment_reports er
         JOIN experiment_status es ON es.experiment_id = er.experiment_id
        WHERE er.report_id = ?
        LIMIT 1
        FOR UPDATE`,
      [input.rejected_report_id],
    );
    if (
      !rejectedReport
      || Number(rejectedReport.report_student_id) !== studentId
      || Number(rejectedReport.experiment_student_id) !== studentId
    ) {
      await connection.rollback();
      return { error: "report_not_found" as const };
    }
    if (rejectedReport.report_status !== "rejected") {
      await connection.rollback();
      return { error: "report_not_rejected" as const };
    }
    if (rejectedReport.experiment_status !== "Completed") {
      await connection.rollback();
      return { error: "parent_not_completed" as const };
    }

    const parentExperimentId = Number(rejectedReport.experiment_id);
    const [[latestExperiment]] = await connection.query<RowDataPacket[]>(
      `SELECT *
         FROM experiment_status
        WHERE student_id = ?
        ORDER BY experiment_id DESC
        LIMIT 1
        FOR UPDATE`,
      [studentId],
    );

    const [[existingRetry]] = await connection.query<RowDataPacket[]>(
      `SELECT *
         FROM experiment_status
        WHERE retry_of_experiment_id = ?
        LIMIT 1
        FOR UPDATE`,
      [parentExperimentId],
    );
    if (existingRetry) {
      if (Number(existingRetry.student_id) !== studentId) {
        await connection.rollback();
        return { error: "invalid_retry_lineage" as const };
      }
      // Repeating the creation request is idempotent only while that child is
      // still the student's official latest attempt and has not produced a
      // report. A Completed child without a report remains eligible here so
      // an interrupted report submission can recover through the existing
      // Completed-without-report bridge.
      if (Number(latestExperiment?.experiment_id) !== Number(existingRetry.experiment_id)) {
        await connection.rollback();
        return {
          error: "report_not_latest" as const,
          data: latestExperiment ? normalizeExperimentRow(latestExperiment) : null,
        };
      }

      const [[existingRetryReport]] = await connection.query<RowDataPacket[]>(
        `SELECT report_id
           FROM experiment_reports
          WHERE experiment_id = ?
            AND student_id = ?
          LIMIT 1
          FOR UPDATE`,
        [existingRetry.experiment_id, studentId],
      );
      if (existingRetryReport) {
        await connection.rollback();
        return {
          error: "report_not_latest" as const,
          data: normalizeExperimentRow(existingRetry),
        };
      }
      const normalized = normalizeExperimentRow(existingRetry);
      commitAttempted = true;
      await commitConnectionWithOutcomeGuard(connection);
      return { data: normalized, created: false };
    }

    if (Number(latestExperiment?.experiment_id) !== parentExperimentId) {
      await connection.rollback();
      return {
        error: "report_not_latest" as const,
        data: latestExperiment ? normalizeExperimentRow(latestExperiment) : null,
      };
    }

    const experimentId = await insertBlankExperiment(
      connection,
      studentId,
      parentExperimentId,
    );
    const [[row]] = await connection.query<RowDataPacket[]>(
      "SELECT * FROM experiment_status WHERE experiment_id = ? LIMIT 1",
      [experimentId],
    );
    if (!row) {
      throw new Error("Created retry experiment could not be reloaded");
    }

    const normalized = normalizeExperimentRow(row);
    commitAttempted = true;
    await commitConnectionWithOutcomeGuard(connection);
    return { data: normalized, created: true };
  } catch (error) {
    if (commitAttempted) {
      disposed = true;
    } else {
      try {
        await connection.rollback();
      } catch (rollbackError) {
        disposed = true;
        connection.destroy();
        throw new Error("重做实验事务回滚失败", { cause: rollbackError });
      }
    }
    throw error;
  } finally {
    if (!disposed) connection.release();
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
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();

    const [[row]] = await connection.query<RowDataPacket[]>(
      "SELECT * FROM experiment_status WHERE experiment_id = ? LIMIT 1 FOR UPDATE",
      [experimentId],
    );

    if (!row) {
      await connection.rollback();
      return { error: "not_found" as const };
    }

    if (row.student_id !== studentId) {
      await connection.rollback();
      return { error: "forbidden" as const };
    }

    if (row.status === "Completed") {
      await connection.rollback();
      return { error: "completed" as const };
    }

    const expectedVersion = data.expected_version;
    if (Number(row.state_version ?? 0) !== expectedVersion) {
      await connection.rollback();
      return {
        error: "conflict" as const,
        data: normalizeExperimentRow(row),
      };
    }

    const sanitized = sanitizeForDb(data);
    delete sanitized.expected_version;
    const preparedUpdate = prepareStudentExperimentUpdate(row as typeof row & {
      status: string;
      start_time: string | Date | null;
      last_activity_at: string | Date | null;
      total_active_duration_seconds: number | null;
    }, sanitized, new Date(), env.DB_TIMEZONE);
    if ("error" in preparedUpdate) {
      await connection.rollback();
      return preparedUpdate;
    }

    const valuesEqual = (left: unknown, right: unknown) => {
      if (left === right) return true;
      if (left === null || left === undefined || right === null || right === undefined) return false;
      if (typeof left === "object" || typeof right === "object") {
        try {
          const parse = (value: unknown) => typeof value === "string" ? JSON.parse(value) : value;
          return JSON.stringify(parse(left)) === JSON.stringify(parse(right));
        } catch {
          return false;
        }
      }
      return String(left) === String(right);
    };
    const changedFields = new Set(
      Object.keys(sanitized).filter((field) => !valuesEqual(sanitized[field], row[field])),
    );
    const invalidation = getServerInvalidationPatch(changedFields);
    const requestedBestModel = sanitized.selected_best_model;
    if (typeof requestedBestModel === "string") {
      const completionField = BEST_MODEL_COMPLETION_FIELDS[requestedBestModel];
      if (!completionField) {
        await connection.rollback();
        return { error: "invalid_best_model" as const };
      }
      const effectiveCompletion = invalidation.patch[completionField] ?? row[completionField];
      if (effectiveCompletion !== true && Number(effectiveCompletion) !== 1) {
        await connection.rollback();
        return { error: "invalid_best_model" as const };
      }
    }
    await deactivateModelRunsForStudentUpdate(
      connection,
      experimentId,
      studentId,
      invalidation.modelsToDeactivate,
    );

    // Server-side invalidation is authoritative. A client patch must never be
    // able to restore dependent results in the same request that invalidates
    // their model/configuration inputs.
    //
    // This spread order is load-bearing together with `getServerInvalidationPatch`:
    // the patch wins over the client here, and in exchange that function omits
    // the configuration fields the request itself changed. Reversing either one
    // alone breaks the pair — reverse this and a client could resurrect a stale
    // production plan; drop the omission there and the student's new model
    // configuration would be discarded the moment they submit it.
    const updateData = {
      ...preparedUpdate.data,
      ...invalidation.patch,
    };
    if (
      changedFields.has("production_forecast_results")
      && !Object.prototype.hasOwnProperty.call(sanitized, "production_forecast_evidence")
    ) {
      updateData.production_forecast_evidence = null;
    }

    const productionPlanValidation = validateCompletedProductionPlan({
      ...row,
      ...updateData,
    }, {
      allowLegacyV1Write: env.ALLOW_LEGACY_PRODUCTION_PLAN_WRITES,
    });
    if (!productionPlanValidation.ok) {
      await connection.rollback();
      return {
        error: "invalid_production_plan" as const,
        code: productionPlanValidation.code,
        message: productionPlanValidation.message,
      };
    }
    if (productionPlanValidation.evidenceClaims) {
      const claims = productionPlanValidation.evidenceClaims;
      const [activeRunRows] = await connection.query<RowDataPacket[]>(
        `SELECT run_id, production_revision
           FROM model_training_runs
          WHERE experiment_id = ?
            AND student_id = ?
            AND model_type = ?
            AND is_active = TRUE
          ORDER BY activated_at DESC
          LIMIT 1
          FOR UPDATE`,
        [experimentId, studentId, claims.model_type],
      );
      const activeRun = activeRunRows[0];
      if (
        !activeRun
        || activeRun.run_id !== claims.run_id
        || activeRun.production_revision !== claims.production_revision
      ) {
        await connection.rollback();
        return {
          error: "invalid_production_plan" as const,
          code: "PRODUCTION_FORECAST_EVIDENCE_STALE" as const,
          message: "生产预测凭证对应的模型版本已失效，请重新准备并生成预测",
        };
      }
    }
    const columns = Object.keys(updateData);
    const assignments = columns.map((col) => `${col} = ?`).join(", ");
    const values = columns.map((key) => updateData[key]);
    values.push(experimentId);

    await connection.query(
      `UPDATE experiment_status SET ${assignments}, state_version = state_version + 1 WHERE experiment_id = ?`,
      values,
    );

    const [[updatedRow]] = await connection.query<RowDataPacket[]>(
      "SELECT * FROM experiment_status WHERE experiment_id = ? LIMIT 1",
      [experimentId],
    );

    await connection.commit();
    return { data: normalizeExperimentRow(updatedRow!) };
  } catch (error) {
    await connection.rollback().catch(() => {});
    throw error;
  } finally {
    connection.release();
  }
}

export async function createExperimentEvent(experimentId: number, studentId: number, input: ExperimentEventInput) {
  return await withDeadlockRetry(async () => {
    const connection = await pool.getConnection();
    try {
      await connection.beginTransaction();

      // Keep the same users -> experiment_status lock order as report
      // submission, grading, retry creation, model-run commit, and account
      // deletion. The bounded transaction retry remains defense in depth for
      // database paths outside this lock graph.
      const [studentRows] = await connection.query<RowDataPacket[]>(
        "SELECT user_id FROM users WHERE user_id = ? LIMIT 1 FOR UPDATE",
        [studentId],
      );
      if (studentRows.length === 0) {
        await connection.rollback();
        return { error: "not_found" as const };
      }

      const [experimentRows] = await connection.query<RowDataPacket[]>(
        "SELECT student_id, status FROM experiment_status WHERE experiment_id = ? LIMIT 1 FOR UPDATE",
        [experimentId],
      );
      if (experimentRows.length === 0) {
        await connection.rollback();
        return { error: "not_found" as const };
      }
      if (experimentRows[0]!.student_id !== studentId) {
        await connection.rollback();
        return { error: "forbidden" as const };
      }
      if (experimentRows[0]!.status === "Completed") {
        await connection.rollback();
        return { error: "completed" as const };
      }

      if (input.event_type === "COMPLETED") {
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

      await connection.commit();
      return { data: { event_id: result.insertId } };
    } catch (error) {
      await connection.rollback().catch(() => {});
      throw error;
    } finally {
      connection.release();
    }
  });
}
