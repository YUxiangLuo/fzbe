import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool, ResultSetHeader, RowDataPacket } from "mysql2/promise";
import { handleUnexpectedError } from "../utils/error-handler";

const STATUS_VALUES = ["Not Started", "In Progress", "Completed"] as const;
const BEST_MODEL_VALUES = [
  "ma",
  "exp",
  "arima",
  "lstm",
  "ensemble_weighted",
  "ensemble_boosting",
  "ensemble_stacking",
] as const;
const NORMALIZATION_VALUES = ["minmax", "zscore"] as const;
const CAPACITY_MODE_VALUES = ["scenario", "auto", "custom"] as const;
const CAPACITY_SCENARIO_VALUES = ["tight", "normal", "abundant"] as const;

const STEP_COUNT = 7;

const EXP_FLOW_GRADE_COLUMNS = [
  "exp_flow_demand_data_preparation",
  "exp_flow_demand_descriptive_stats",
  "exp_flow_demand_model_selection",
  "exp_flow_demand_generate_results",
  "exp_flow_production_inventory_calc",
  "exp_flow_production_service_level",
  "exp_flow_production_variable_calc",
  "exp_flow_production_plan_creation",
  "exp_flow_report_submission",
] as const;

const EXP_FLOW_GRADE_UPDATE_SET = EXP_FLOW_GRADE_COLUMNS.map(
  (col) => `${col} = ?`,
).join(", ");

const BOOLEAN_FIELDS = new Set<string>([
  "moving_average_completed",
  "exponential_smoothing_completed",
  "arima_completed",
  "lstm_completed",
  "ensemble_weighted_completed",
  "ensemble_boosting_completed",
  "ensemble_stacking_completed",
  "quiz_about_model_completed",
  "quiz_about_plan_completed",
  "production_plan_completed",
]);

const JSON_FIELDS = new Set<string>([
  "arima_adf_stationarity",
  "lstm_features",
  "ensemble_weighted_base_models",
  "ensemble_boosting_base_models",
  "ensemble_stacking_base_models",
  "production_mps_table",
  "production_forecast_results",
]);

const INTEGER_FIELDS = new Set<string>([
  "experiment_id",
  "student_id",
  "highest_completed_step",
  "current_step",
  "data_window_train_start_index",
  "data_window_train_end_index",
  "data_window_evaluate_start_index",
  "data_window_evaluate_end_index",
  "moving_average_window",
  "arima_p",
  "arima_d",
  "arima_q",
  "production_forecast_periods",
  "production_capacity",
  "production_custom_capacity",
  "total_active_duration_seconds",
]);

const FLOAT_FIELDS = new Set<string>([
  "moving_average_metrics_rmse",
  "moving_average_metrics_mae",
  "moving_average_metrics_r2",
  "exponential_smoothing_alpha",
  "exponential_smoothing_metrics_rmse",
  "exponential_smoothing_metrics_mae",
  "exponential_smoothing_metrics_r2",
  "arima_metrics_rmse",
  "arima_metrics_mae",
  "arima_metrics_r2",
  "lstm_metrics_rmse",
  "lstm_metrics_mae",
  "lstm_metrics_r2",
  "ensemble_weighted_metrics_rmse",
  "ensemble_weighted_metrics_mae",
  "ensemble_weighted_metrics_r2",
  "ensemble_boosting_metrics_rmse",
  "ensemble_boosting_metrics_mae",
  "ensemble_boosting_metrics_r2",
  "ensemble_stacking_metrics_rmse",
  "ensemble_stacking_metrics_mae",
  "ensemble_stacking_metrics_r2",
  "production_initial_inventory",
  "production_target_service_level",
  "production_safety_stock_z_score",
]);

const datetimeFields = new Set<string>([
  "start_time",
  "last_activity_at",
  "completion_time",
]);

const experimentIdParamSchema = z.object({
  experimentId: z.string().regex(/^\d+$/, {
    message: "experimentId must be a number",
  }),
});

const numberPreprocess = (value: unknown) => {
  if (value === undefined || value === null) {
    return value;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const parsed = Number(trimmed);
    return Number.isNaN(parsed) ? value : parsed;
  }
  return value;
};

const booleanPreprocess = (value: unknown) => {
  if (value === undefined || value === null) {
    return value;
  }
  if (typeof value === "string") {
    const trimmed = value.trim().toLowerCase();
    if (!trimmed) {
      return null;
    }
    if (trimmed === "true" || trimmed === "1") {
      return true;
    }
    if (trimmed === "false" || trimmed === "0") {
      return false;
    }
  }
  if (typeof value === "number") {
    if (value === 1) return true;
    if (value === 0) return false;
  }
  return value;
};

const stringPreprocess = (value: unknown) => {
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length ? trimmed : null;
  }
  return value;
};

const jsonPreprocess = (value: unknown) => {
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    try {
      return JSON.parse(trimmed);
    } catch {
      return trimmed;
    }
  }
  return value;
};

const buildIntSchema = (min?: number, max?: number) => {
  let schema = z.number().int();
  if (min !== undefined) {
    schema = schema.min(min);
  }
  if (max !== undefined) {
    schema = schema.max(max);
  }
  return schema;
};

const buildFloatSchema = (min?: number) => {
  let schema = z.number();
  if (min !== undefined) {
    schema = schema.min(min);
  }
  return schema;
};

const optionalInt = (min?: number, max?: number) =>
  z
    .preprocess(numberPreprocess, buildIntSchema(min, max))
    .nullable()
    .optional();

const optionalFloat = (min?: number) =>
  z
    .preprocess(numberPreprocess, buildFloatSchema(min))
    .nullable()
    .optional();

const optionalBoolean = z
  .preprocess(booleanPreprocess, z.boolean())
  .nullable()
  .optional();

const optionalString = z
  .preprocess(stringPreprocess, z.string())
  .nullable()
  .optional();

const optionalJson = z
  .preprocess(
    jsonPreprocess,
    z.union([z.array(z.any()), z.record(z.string(), z.any()), z.string()]),
  )
  .nullable()
  .optional();

const optionalFields: Record<string, z.ZodTypeAny> = {
  highest_completed_step: optionalInt(0, 7),
  selected_industry: optionalString,
  selected_company: optionalString,
  selected_product: optionalString,
  data_window_train_start_index: optionalInt(0),
  data_window_train_end_index: optionalInt(0),
  data_window_evaluate_start_index: optionalInt(0),
  data_window_evaluate_end_index: optionalInt(0),
  moving_average_completed: optionalBoolean,
  moving_average_window: optionalInt(1),
  moving_average_metrics_rmse: optionalFloat(0),
  moving_average_metrics_mae: optionalFloat(0),
  moving_average_metrics_r2: optionalFloat(),
  exponential_smoothing_completed: optionalBoolean,
  exponential_smoothing_alpha: optionalFloat(0),
  exponential_smoothing_metrics_rmse: optionalFloat(0),
  exponential_smoothing_metrics_mae: optionalFloat(0),
  exponential_smoothing_metrics_r2: optionalFloat(),
  arima_completed: optionalBoolean,
  arima_p: optionalInt(0),
  arima_d: optionalInt(0),
  arima_q: optionalInt(0),
  arima_metrics_rmse: optionalFloat(0),
  arima_metrics_mae: optionalFloat(0),
  arima_metrics_r2: optionalFloat(),
  arima_adf_stationarity: optionalJson,
  lstm_completed: optionalBoolean,
  lstm_normalization: z
    .preprocess(stringPreprocess, z.enum(NORMALIZATION_VALUES))
    .nullable()
    .optional(),
  lstm_features: optionalJson,
  lstm_target_field: optionalString,
  lstm_metrics_rmse: optionalFloat(0),
  lstm_metrics_mae: optionalFloat(0),
  lstm_metrics_r2: optionalFloat(),
  ensemble_weighted_completed: optionalBoolean,
  ensemble_weighted_base_models: optionalJson,
  ensemble_weighted_metrics_rmse: optionalFloat(0),
  ensemble_weighted_metrics_mae: optionalFloat(0),
  ensemble_weighted_metrics_r2: optionalFloat(),
  ensemble_boosting_completed: optionalBoolean,
  ensemble_boosting_base_models: optionalJson,
  ensemble_boosting_metrics_rmse: optionalFloat(0),
  ensemble_boosting_metrics_mae: optionalFloat(0),
  ensemble_boosting_metrics_r2: optionalFloat(),
  ensemble_stacking_completed: optionalBoolean,
  ensemble_stacking_base_models: optionalJson,
  ensemble_stacking_metrics_rmse: optionalFloat(0),
  ensemble_stacking_metrics_mae: optionalFloat(0),
  ensemble_stacking_metrics_r2: optionalFloat(),
  selected_best_model: z
    .preprocess(stringPreprocess, z.enum(BEST_MODEL_VALUES))
    .nullable()
    .optional(),
  quiz_about_model_completed: optionalBoolean,
  quiz_about_plan_completed: optionalBoolean,
  production_plan_completed: optionalBoolean,
  production_forecast_periods: optionalInt(1),
  production_initial_inventory: optionalFloat(0),
  production_target_service_level: optionalFloat(0),
  production_safety_stock_z_score: optionalFloat(),
  production_mps_table: optionalJson,
  production_forecast_results: optionalJson,
  production_capacity_mode: z
    .preprocess(stringPreprocess, z.enum(CAPACITY_MODE_VALUES))
    .nullable()
    .optional(),
  production_capacity_scenario: z
    .preprocess(stringPreprocess, z.enum(CAPACITY_SCENARIO_VALUES))
    .nullable()
    .optional(),
  production_capacity: optionalInt(0),
  production_custom_capacity: optionalInt(0),
  start_time: optionalString,
  last_activity_at: optionalString,
  completion_time: optionalString,
  total_active_duration_seconds: optionalInt(0),
};

const createExperimentSchema = z.object({
  status: z.enum(STATUS_VALUES).optional(),
  current_step: optionalInt(1, 7),
  ...optionalFields,
});

const updateExperimentSchema = z
  .object({
    status: z.enum(STATUS_VALUES).optional(),
    current_step: optionalInt(1, 7),
    ...optionalFields,
  })
  .refine(
    (data) => Object.keys(data).length > 0,
    "At least one field must be provided",
  );

const sanitizeForDb = (data: Record<string, unknown>) => {
  const sanitized: Record<string, unknown> = {};

  const toMySqlDateTime = (date: Date) =>
    date.toISOString().slice(0, 19).replace("T", " ");

  for (const [key, rawValue] of Object.entries(data)) {
    if (rawValue === undefined) {
      continue;
    }

    if (rawValue === null) {
      sanitized[key] = null;
      continue;
    }

    if (BOOLEAN_FIELDS.has(key)) {
      sanitized[key] = rawValue === true || rawValue === 1 ? 1 : 0;
      continue;
    }

    if (JSON_FIELDS.has(key)) {
      if (typeof rawValue === "string") {
        sanitized[key] = rawValue;
      } else {
        sanitized[key] = JSON.stringify(rawValue);
      }
      continue;
    }

    if (datetimeFields.has(key)) {
      if (rawValue instanceof Date) {
        sanitized[key] = rawValue; // Pass Date object directly
        continue;
      }

      if (typeof rawValue === "string") {
        const trimmed = rawValue.trim();
        if (!trimmed) {
          sanitized[key] = null;
          continue;
        }

        const parsed = new Date(trimmed);
        // Pass Date object directly if valid, otherwise pass original string
        sanitized[key] = Number.isNaN(parsed.getTime()) ? trimmed : parsed;
        continue;
      }
    }

    if (rawValue instanceof Date) {
      sanitized[key] = rawValue; // Pass Date object directly
      continue;
    }

    if (typeof rawValue === "string") {
      const trimmed = rawValue.trim();
      sanitized[key] = trimmed.length ? trimmed : null;
      continue;
    }

    sanitized[key] = rawValue;
  }

  return sanitized;
};

const parseJsonField = (value: unknown) => {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "string") {
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  }
  return value;
};

const normalizeBoolean = (value: unknown) => {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    return value === 1;
  }
  if (typeof value === "string") {
    const lowered = value.toLowerCase();
    if (lowered === "1" || lowered === "true") {
      return true;
    }
    if (lowered === "0" || lowered === "false") {
      return false;
    }
  }
  return Boolean(value);
};

const normalizeNumeric = (value: unknown) => {
  if (value === null || value === undefined) {
    return null;
  }
  const num = Number(value);
  return Number.isNaN(num) ? null : num;
};

const normalizeExperimentRow = (row: RowDataPacket) => {
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(row)) {
    if (JSON_FIELDS.has(key)) {
      result[key] = parseJsonField(value);
      continue;
    }

    if (BOOLEAN_FIELDS.has(key)) {
      result[key] = normalizeBoolean(value);
      continue;
    }

    if (INTEGER_FIELDS.has(key) || FLOAT_FIELDS.has(key)) {
      result[key] = normalizeNumeric(value);
      continue;
    }

    if (datetimeFields.has(key) && value instanceof Date) {
      result[key] = value.toISOString();
      continue;
    }

    result[key] = value ?? null;
  }

  return result;
};

const fetchExperimentById = async (pool: Pool, experimentId: number) => {
  const [rows] = await pool.query<RowDataPacket[]>(
    "SELECT * FROM experiment_status WHERE experiment_id = ? LIMIT 1",
    [experimentId],
  );
  return rows[0];
};

const fetchLatestNotCompletedExperimentForStudent = async (pool: Pool, studentId: number) => {
  // Optimized: Filter in database instead of application layer
  // Uses idx_student_status_id index for better performance
  const [rows] = await pool.query<RowDataPacket[]>(
    `SELECT *
       FROM experiment_status
      WHERE student_id = ?
        AND status != 'Completed'
      ORDER BY experiment_id DESC
      LIMIT 1`,
    [studentId],
  );
  return rows[0];
};

export default (pool: Pool, JWT_SECRET: string) => {
  const experimentsApi = new Hono();

  // GET /api/v1/students/me/experiment-runs/active - 查询当前学生正在进行的实验状态
  experimentsApi.get("/api/v1/students/me/experiment-runs/active", jwt({ secret: JWT_SECRET }), async (c) => {
    const payload = c.get("jwtPayload");
    if (payload.role !== "Student") {
      return c.json({ error: "Forbidden" }, 403);
    }
    const studentId = payload.sub;

    try {
      const row = await fetchLatestNotCompletedExperimentForStudent(pool, studentId);
      if (!row) {
        return c.json({ error: "Experiment status not found" }, 404);
      }
      return c.json(normalizeExperimentRow(row));
    } catch (error) {
      return handleUnexpectedError(c, error, "Failed to fetch experiment status");
    }
  });

  // 创建新的实验状态记录
  experimentsApi.post(
    "/api/v1/students/me/experiment-runs",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", createExperimentSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Forbidden" }, 403);
      }
      const studentId = Number.parseInt(payload.sub, 10);
      const data = c.req.valid("json");

      const connection = await pool.getConnection();
      let transactionStarted = false;
      try {
        await connection.beginTransaction();
        transactionStarted = true;

        const sanitized = sanitizeForDb({ ...data, student_id: studentId });
        delete sanitized.total_active_duration_seconds;
        const columns = Object.keys(sanitized);

        const placeholders = columns.map(() => "?").join(", ");
        const values = columns.map((key) => sanitized[key]);

        const [insertResult] = await connection.query<ResultSetHeader>(
          `INSERT INTO experiment_status (${columns.join(", ")}) VALUES (${placeholders})`,
          values,
        );

        const experimentId = insertResult.insertId;

        const gradeSql = `INSERT INTO experiment_grade (experiment_id, student_id) VALUES (?, ?)`;

        await connection.query(gradeSql, [experimentId, studentId]);

        await connection.commit();

        const row = await fetchExperimentById(pool, experimentId);
        return c.json(normalizeExperimentRow(row!), 201);
      } catch (error) {
        if (transactionStarted) {
          await connection.rollback();
        }
        return handleUnexpectedError(c, error, "Failed to create experiment status");
      } finally {
        connection.release();
      }
    },
  );

  // GET /api/v1/students/me/experiment-runs - 获取当前学生的所有实验记录
  experimentsApi.get(
    "/api/v1/students/me/experiment-runs",
    jwt({ secret: JWT_SECRET }),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Forbidden" }, 403);
      }
      const studentId = Number.parseInt(payload.sub, 10);

      try {
        const [rows] = await pool.query<RowDataPacket[]>(
          `SELECT *
             FROM experiment_status
            WHERE student_id = ?
            ORDER BY experiment_id DESC`,
          [studentId],
        );

        const normalized = (rows as RowDataPacket[]).map((row) =>
          normalizeExperimentRow(row),
        );
        return c.json(normalized);
      } catch (error) {
        return handleUnexpectedError(c, error, "Failed to load experiments");
      }
    },
  );

  // DELETE /api/v1/students/me/experiment-runs/:experimentId - 删除未完成的实验
  experimentsApi.delete(
    "/api/v1/students/me/experiment-runs/:experimentId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", experimentIdParamSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Forbidden" }, 403);
      }
      const studentId = Number.parseInt(payload.sub, 10);
      const { experimentId } = c.req.valid("param");
      const experimentIdNumber = Number.parseInt(experimentId, 10);

      try {
        const [[experimentRow]]: any[] = await pool.query(
          `SELECT status, student_id
             FROM experiment_status
            WHERE experiment_id = ?`,
          [experimentIdNumber],
        );

        if (!experimentRow || Number(experimentRow.student_id) !== studentId) {
          return c.json({ error: "Experiment not found" }, 404);
        }

        if (experimentRow.status === "Completed") {
          return c.json({ error: "Completed experiments cannot be deleted" }, 400);
        }

        await pool.query(
          "DELETE FROM experiment_status WHERE experiment_id = ?",
          [experimentIdNumber],
        );

        return c.json({ message: "Experiment deleted successfully" });
      } catch (error) {
        return handleUnexpectedError(c, error, "Failed to delete experiment");
      }
    },
  );

  // 更新指定实验状态
  experimentsApi.put(
    "/api/v1/experiment-runs/:experimentId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", experimentIdParamSchema),
    zValidator("json", updateExperimentSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Forbidden" }, 403);
      }

      const { experimentId } = c.req.valid("param");
      const experimentIdNumber = Number.parseInt(experimentId, 10);
      const studentId = payload.sub;
      const body = c.req.valid("json");

      const sanitized = sanitizeForDb(body);
      delete sanitized.total_active_duration_seconds;

      if (Object.keys(sanitized).length === 0) {
        return c.json({ error: "No fields provided to update" }, 400);
      }

      if (sanitized["current_step"] === null) {
        return c.json({ error: "current_step cannot be null" }, 400);
      }
      if (sanitized["highest_completed_step"] === null) {
        return c.json({ error: "highest_completed_step cannot be null" }, 400);
      }

      try {
        const row = await fetchExperimentById(pool, experimentIdNumber);
        if (!row) {
          return c.json({ error: "Experiment status not found" }, 404);
        }

        if (row.student_id !== studentId) {
          return c.json({ error: "Forbidden" }, 403);
        }

        if (Object.prototype.hasOwnProperty.call(sanitized, "last_activity_at")) {
          const newValue = sanitized["last_activity_at"];
          let newTimestamp: Date | null = null;

          if (newValue instanceof Date) {
            newTimestamp = newValue;
          } else if (typeof newValue === "string") {
            const parsed = new Date(newValue);
            if (!Number.isNaN(parsed.getTime())) {
              newTimestamp = parsed;
              sanitized["last_activity_at"] = parsed;
            }
          }

          const previousLastValue = row.last_activity_at ?? row.start_time ?? null;
          const previousLast =
            previousLastValue instanceof Date
              ? previousLastValue
              : previousLastValue
                ? new Date(previousLastValue as string)
                : null;

          if (newTimestamp && previousLast) {
            const diffMs = newTimestamp.getTime() - previousLast.getTime();
            const maxGapMs = 6 * 60 * 60 * 1000;
            if (diffMs > 0 && diffMs <= maxGapMs) {
              const prevTotal = Number(row.total_active_duration_seconds ?? 0);
              const additionalSeconds = Math.max(1, Math.floor(diffMs / 1000));
              const updatedTotal = prevTotal + additionalSeconds;
              sanitized.total_active_duration_seconds = updatedTotal;
            }
          }
        }

        const columns = Object.keys(sanitized);

        const assignments = columns.map((col) => `${col} = ?`).join(", ");
        const values = columns.map((key) => sanitized[key]);
        values.push(experimentIdNumber);

        await pool.query(
          `UPDATE experiment_status SET ${assignments} WHERE experiment_id = ?`,
          values,
        );

        const updatedRow = await fetchExperimentById(pool, experimentIdNumber);
        return c.json(normalizeExperimentRow(updatedRow!));
      } catch (error) {
        return handleUnexpectedError(c, error, "Failed to update experiment status");
      }
    },
  );

  const experimentEventSchema = z.object({
    step_order: z.number().int().min(1).max(7),
    event_type: z.enum(["STARTED", "COMPLETED"]),
  });

  // 为实验步骤创建开始/结束事件
  experimentsApi.post(
    "/api/v1/experiment-runs/:experimentId/events",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", experimentIdParamSchema),
    zValidator("json", experimentEventSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Forbidden" }, 403);
      }
      const studentId = Number.parseInt(payload.sub, 10);
      const { experimentId } = c.req.valid("param");
      const experiment_id = Number.parseInt(experimentId, 10);
      const { step_order, event_type } = c.req.valid("json");

      const connection = await pool.getConnection();
      let transactionStarted = false;
      let resetExpFlow = false;
      let markExpFlowComplete = false;
      try {
        // 验证该实验是否属于当前学生
        const [experimentRows] = await connection.query<RowDataPacket[]>(
          "SELECT student_id FROM experiment_status WHERE experiment_id = ? LIMIT 1",
          [experiment_id]
        );

        if (experimentRows.length === 0) {
          return c.json({ error: "Experiment not found" }, 404);
        }
        if ((experimentRows[0] as { student_id: number }).student_id !== studentId) {
          return c.json({ error: "Forbidden: You do not own this experiment" }, 403);
        }

        // --- 新的优化逻辑开始 ---
        if (event_type === "COMPLETED") {
          await connection.beginTransaction();
          transactionStarted = true;

          // 1. 获取当前实验最新的事件，并锁定该行以待更新
          const [latestEventRows] = await connection.query<RowDataPacket[]>(
            `SELECT step_order
               FROM experiment_step_events
              WHERE experiment_id = ?
              ORDER BY event_timestamp DESC
              LIMIT 1
              FOR UPDATE`,
            [experiment_id]
          );

          const latestStepOrderRow = latestEventRows[0] as { step_order?: number } | undefined;
          const latestStepOrder = latestStepOrderRow?.step_order ?? 0;

          // 2. 如果最新步骤比当前要完成的步骤还要靠后，说明发生了“跳回”完成
          if (latestStepOrder > step_order) {
            resetExpFlow = true;
            // 3. 自动插入一个10秒前的 "STARTED" 事件
            const syntheticStartedTimestamp = new Date(Date.now() - 10000);
            await connection.query<ResultSetHeader>(
              `INSERT INTO experiment_step_events (experiment_id, student_id, step_order, event_type, event_timestamp)
               VALUES (?, ?, ?, ?, ?)`,
              [experiment_id, studentId, step_order, "STARTED", syntheticStartedTimestamp]
            );
          }

          if (step_order === STEP_COUNT) {
            markExpFlowComplete = true;
          }
        }
        // --- 新的优化逻辑结束 ---

        // 插入用户请求的原始事件
        const [result] = await connection.query<ResultSetHeader>(
          `INSERT INTO experiment_step_events (experiment_id, student_id, step_order, event_type)
           VALUES (?, ?, ?, ?)`,
          [experiment_id, studentId, step_order, event_type]
        );

        // 如果在事务中，则提交
        if (transactionStarted) {
          if (resetExpFlow) {
            const zeroValues = EXP_FLOW_GRADE_COLUMNS.map(() => 0);
            await connection.query(
              `UPDATE experiment_grade SET ${EXP_FLOW_GRADE_UPDATE_SET} WHERE experiment_id = ? AND student_id = ?`,
              [...zeroValues, experiment_id, studentId],
            );
          } else if (markExpFlowComplete) {
            const fullValues = EXP_FLOW_GRADE_COLUMNS.map(() => 100);
            await connection.query(
              `UPDATE experiment_grade SET ${EXP_FLOW_GRADE_UPDATE_SET} WHERE experiment_id = ? AND student_id = ?`,
              [...fullValues, experiment_id, studentId],
            );
          }
          await connection.commit();
        }

        return c.json({
          message: "Event created successfully",
          event_id: result.insertId,
        }, 201);

      } catch (error) {
        if (transactionStarted) {
          await connection.rollback();
        }
        return handleUnexpectedError(c, error, "Failed to create experiment step event");
      } finally {
        connection.release();
      }
    }
  );

  return experimentsApi;
};
