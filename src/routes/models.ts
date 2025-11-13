import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import path from "path";
import fs from "fs";
import type { Pool } from "mysql2/promise";
import csv from "csv-parser";
import { handleUnexpectedError } from "../utils/error-handler";
import logger from "../logger";

const TRAINING_TIMEOUT = 30000;

// ========== Common Schemas ==========
const baseDataSchema = z.object({
  selected_industry: z.string().min(1),
  selected_company: z.string().min(1),
  selected_product: z.string().min(1),
});

const trainRangeSchema = z.object({
  data_window_train_start_index: z.coerce.number().int().min(0),
  data_window_train_end_index: z.coerce.number().int().min(0),
});

const evalRangeSchema = z.object({
  data_window_evaluate_start_index: z.coerce.number().int().min(0),
  data_window_evaluate_end_index: z.coerce.number().int().min(0),
});

// ========== Base Model Training Schemas ==========
const trainMovingAverageSchema = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .extend({
    moving_average_window: z.coerce.number().int().min(1),
  })
  .refine(
    (data) =>
      data.data_window_train_start_index <= data.data_window_train_end_index,
    {
      message: "train_start_index must be <= train_end_index",
      path: ["data_window_train_end_index"],
    },
  )
  .refine(
    (data) =>
      data.data_window_evaluate_start_index <=
      data.data_window_evaluate_end_index,
    {
      message: "evaluate_start_index must be <= evaluate_end_index",
      path: ["data_window_evaluate_end_index"],
    },
  );

const trainExponentialSmoothingSchema = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .extend({
    exponential_smoothing_alpha: z.coerce.number().min(0).max(1),
  })
  .refine(
    (data) =>
      data.data_window_train_start_index <= data.data_window_train_end_index,
    {
      message: "train_start_index must be <= train_end_index",
      path: ["data_window_train_end_index"],
    },
  )
  .refine(
    (data) =>
      data.data_window_evaluate_start_index <=
      data.data_window_evaluate_end_index,
    {
      message: "evaluate_start_index must be <= evaluate_end_index",
      path: ["data_window_evaluate_end_index"],
    },
  );

const trainArimaSchema = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .extend({
    arima_d: z.coerce.number().int().min(0),
  })
  .refine(
    (data) =>
      data.data_window_train_start_index <= data.data_window_train_end_index,
    {
      message: "train_start_index must be <= train_end_index",
      path: ["data_window_train_end_index"],
    },
  )
  .refine(
    (data) =>
      data.data_window_evaluate_start_index <=
      data.data_window_evaluate_end_index,
    {
      message: "evaluate_start_index must be <= evaluate_end_index",
      path: ["data_window_evaluate_end_index"],
    },
  );

const trainLstmSchema = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .extend({
    lstm_features: z
      .array(z.string())
      .min(1, "At least one feature is required"),
    lstm_target_field: z.string().min(1, "Target field is required"),
    lstm_normalization: z.enum(["zscore", "minmax"]).default("zscore"),
  })
  .refine(
    (data) =>
      data.data_window_train_start_index <= data.data_window_train_end_index,
    {
      message: "train_start_index must be <= train_end_index",
      path: ["data_window_train_end_index"],
    },
  )
  .refine(
    (data) =>
      data.data_window_evaluate_start_index <=
      data.data_window_evaluate_end_index,
    {
      message: "evaluate_start_index must be <= evaluate_end_index",
      path: ["data_window_evaluate_end_index"],
    },
  );

// ========== Ensemble Model Training Schemas ==========
const trainWeightedAverageSchema = baseDataSchema
  .merge(evalRangeSchema)
  .extend({
    ensemble_weighted_base_models: z
      .array(z.enum(["ma", "es", "arima", "lstm"]))
      .min(2),
    weight_strategy: z
      .literal("inverse_variance")
      .optional()
      .default("inverse_variance"),
  })
  .refine(
    (data) =>
      data.data_window_evaluate_start_index <=
      data.data_window_evaluate_end_index,
    {
      message: "evaluate_start_index must be <= evaluate_end_index",
      path: ["data_window_evaluate_end_index"],
    },
  );

const trainBoostingSchema = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .extend({
    ensemble_boosting_base_models: z
      .array(z.enum(["ma", "es", "arima", "lstm"]))
      .min(1),
    arima_d: z.coerce.number().int().min(0).optional(),
    moving_average_window: z.coerce.number().int().min(1).optional(),
    exponential_smoothing_alpha: z.coerce.number().min(0).max(1).optional(),
    lstm_features: z.array(z.string()).optional(),
    lstm_target_field: z.string().optional(),
    lstm_normalization: z.enum(["zscore", "minmax"]).optional(),
    learning_rate: z.coerce.number().positive().optional(),
  })
  .refine(
    (data) =>
      data.data_window_train_start_index <= data.data_window_train_end_index,
    {
      message: "train_start_index must be <= train_end_index",
      path: ["data_window_train_end_index"],
    },
  )
  .refine(
    (data) =>
      data.data_window_evaluate_start_index <=
      data.data_window_evaluate_end_index,
    {
      message: "evaluate_start_index must be <= evaluate_end_index",
      path: ["data_window_evaluate_end_index"],
    },
  );

const trainStackingSchema = baseDataSchema
  .merge(evalRangeSchema)
  .extend({
    ensemble_stacking_base_models: z
      .array(z.enum(["ma", "es", "arima", "lstm"]))
      .min(2),
    meta_l2: z.coerce.number().min(0).optional(),
  })
  .refine(
    (data) =>
      data.data_window_evaluate_start_index <=
      data.data_window_evaluate_end_index,
    {
      message: "evaluate_start_index must be <= evaluate_end_index",
      path: ["data_window_evaluate_end_index"],
    },
  );

// ========== Prediction Schema ==========
const predictSchema = z.object({
  model_type: z.enum([
    "ma",
    "es",
    "arima",
    "lstm",
    "weighted_average",
    "boosting",
    "stacking",
  ]),
  forecast_steps: z.coerce.number().int().min(1),
});

// ========== Helper Functions ==========
async function loadDatasetFromDb(
  pool: Pool,
  industry: string,
  company: string,
  product: string,
): Promise<Record<string, any>[]> {
  const [segmentRows]: any[] = await pool.query(
    "SELECT file_path FROM dataset_segments WHERE industry_name = ? AND company_name = ? AND product_name = ?",
    [industry, company, product],
  );

  if (segmentRows.length === 0) {
    throw new Error("Dataset segment file not found");
  }

  const allData: Record<string, any>[] = [];
  await new Promise<void>((resolve, reject) => {
    fs.createReadStream(segmentRows[0].file_path)
      .pipe(
        csv({
          mapHeaders: ({ header }: { header: string }) =>
            header.replace(/^\uFEFF/, "").trim(),
        }),
      )
      .on("data", (row: Record<string, any>) => allData.push(row))
      .on("end", resolve)
      .on("error", reject);
  });

  return allData;
}

function prepareSimpleData(data: Record<string, any>[]): string {
  // Simple format: [{"month": "2021-01", "sales": 37200}, ...]
  // Used by MA/ES/ARIMA models
  return JSON.stringify(
    data.map((row) => {
      const year = row["年份"] || row.year || "0";
      const month = row["月份"] || row.month || "1";
      const sales = parseFloat(
        row["销售数量"] || row["销售总额"] || row.sales || row.y || "0",
      );

      const ts = `${year}-${String(month).padStart(2, "0")}`;

      return {
        month: ts,
        sales: sales,
      };
    }),
  );
}

function prepareFullData(
  data: Record<string, any>[],
  lstmFeatures?: string[],
): {
  dataFull: string;
  featureTypes: Record<string, "numeric" | "binary" | "categorical">;
  invalidFeatures: string[];
} {
  const excludeFields = [
    "年份",
    "月份",
    "销售数量",
    "销售总额",
    "year",
    "month",
    "sales",
    "y",
    "行业名称",
    "公司代码",
    "公司名称",
    "产品编码",
    "产品名称",
  ];

  let featuresToUse: string[];
  if (lstmFeatures && lstmFeatures.length > 0) {
    featuresToUse = lstmFeatures;
  } else {
    const allColumns = Object.keys(data[0] || {});
    featuresToUse = allColumns.filter((col) => !excludeFields.includes(col));
  }

  const featureTypes: Record<string, "numeric" | "binary" | "categorical"> = {};
  const invalidFeatures: string[] = [];

  for (const feature of featuresToUse) {
    const values = data
      .map((row) => row[feature])
      .filter((val) => val !== null && val !== undefined && val !== "");

    if (values.length === 0) {
      featureTypes[feature] = "numeric"; // Assume numeric if no data
      continue;
    }

    const uniqueValues = new Set(
      values.map((v) => String(v).trim().toLowerCase()),
    );

    if (uniqueValues.size <= 2) {
      featureTypes[feature] = "binary";
      continue;
    }

    let numericCount = 0;
    for (const val of values) {
      const num = parseFloat(String(val));
      if (!isNaN(num) && isFinite(num)) {
        numericCount++;
      }
    }

    if (numericCount / values.length > 0.8) {
      featureTypes[feature] = "numeric";
    } else if (uniqueValues.size <= 10) { // Heuristic for categorical
      featureTypes[feature] = "categorical";
    } else {
      // This is an unsupported categorical feature
      invalidFeatures.push(feature);
    }
  }

  const dataArray = data.map((row) => {
    const year = row["年份"] || row.year || "0";
    const month = row["月份"] || row.month || "1";
    const sales = parseFloat(
      row["销售数量"] || row["销售总额"] || row.sales || row.y || "0",
    );
    const ts = `${year}-${String(month).padStart(2, "0")}`;

    const features: Record<string, any> = {};
    for (const feature of featuresToUse) {
      if (row[feature] !== undefined) {
        const numValue = parseFloat(row[feature]);
        features[feature] = isNaN(numValue) ? row[feature] : numValue;
      }
    }

    return { ts: ts, y: sales, x: features };
  });

  return {
    dataFull: JSON.stringify(dataArray),
    featureTypes: featureTypes,
    invalidFeatures: invalidFeatures,
  };
}

async function runPythonMain(args: string[], timeoutMs?: number): Promise<any> {
  const pyDir = path.resolve(import.meta.dir, "../../py");
  const pythonExec =
    process.platform === "win32"
      ? path.join(pyDir, "venv", "Scripts", "python.exe")
      : path.join(pyDir, "venv", "bin", "python");

  const proc = Bun.spawn([pythonExec, "src/main.py", ...args], {
    cwd: pyDir,
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, PATH: `${pyDir}/venv/bin:${process.env.PATH}` },
  });

  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  const executionPromise = new Promise<any>(async (resolve, reject) => {
    try {
      const [stdout, stderr] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
      ]);

      await proc.exited;

      if (proc.exitCode !== 0) {
        logger.error({ stderr }, "Python execution failed");
        const pythonErrorMessage = stderr.trim();
        if (pythonErrorMessage.startsWith("[VALIDATION]")) {
          reject(new Error(`[PYTHON_VALIDATION] ${pythonErrorMessage.substring(13).trim()}`));
        } else if (pythonErrorMessage.startsWith("[DATA]")) {
          reject(new Error(`[PYTHON_DATA] ${pythonErrorMessage.substring(7).trim()}`));
        } else {
          reject(new Error(`Python execution failed: ${stderr}`));
        }
      } else {
        try {
          resolve(JSON.parse(stdout));
        } catch (e) {
          logger.error({ stdout }, "Failed to parse python output");
          reject(new Error("Invalid output from python"));
        }
      }
    } catch (err) {
      reject(err);
    } finally {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    }
  });

  if (!timeoutMs) {
    return executionPromise;
  }

  const timeoutPromise = new Promise<any>((_, reject) => {
    timeoutId = setTimeout(() => {
      proc.kill();
      reject(new Error(`Python execution timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });

  return Promise.race([executionPromise, timeoutPromise]);
}

export default (pool: Pool, JWT_SECRET: string) => {
  const modelsApi = new Hono();

  // ========== Base Model Training Routes ==========

  // MA Training
  modelsApi.post(
    "/api/v1/models/ma/training",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", trainMovingAverageSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Only students can train models" }, 403);
      }
      const studentId = payload.sub;
      const body = c.req.valid("json");

      const trainDataLength = body.data_window_train_end_index - body.data_window_train_start_index + 1;
      if (body.moving_average_window > trainDataLength) {
        return c.json(
          {
            error: `Moving average window (${body.moving_average_window}) cannot be greater than the training data length (${trainDataLength}).`,
          },
          400,
        );
      }

      try {
        const allData = await loadDatasetFromDb(
          pool,
          body.selected_industry,
          body.selected_company,
          body.selected_product,
        );

        if (
          body.data_window_train_end_index >= allData.length ||
          body.data_window_evaluate_end_index >= allData.length
        ) {
          return c.json(
            {
              error: "Data window indices out of bounds",
              dataset_size: allData.length,
            },
            400,
          );
        }

        const dataSimple = prepareSimpleData(allData);
        const modelSaveLoc = path.resolve(
          import.meta.dir,
          `../../py/trained_models/${studentId}_ma.pkl`,
        );

        const result = await runPythonMain(
          [
            "--method",
            "ma",
            "--data",
            dataSimple,
            "--train_start",
            String(body.data_window_train_start_index),
            "--train_end",
            String(body.data_window_train_end_index),
            "--eval_start",
            String(body.data_window_evaluate_start_index),
            "--eval_end",
            String(body.data_window_evaluate_end_index),
            "--window",
            String(body.moving_average_window),
            "--save_model",
            modelSaveLoc,
            "--pretty",
          ],
          TRAINING_TIMEOUT,
        ); // 30-second timeout

        return c.json({ status: "success", results: result });
      } catch (error: any) {
        if (error.message.includes("timed out")) {
          return c.json({ error: "Training timed out after 30 seconds" }, 504);
        }
        if (error.message.startsWith("[PYTHON_VALIDATION]")) {
          return c.json({ error: error.message.substring(20) }, 400);
        }
        if (error.message.startsWith("[PYTHON_DATA]")) {
          return c.json({ error: error.message.substring(14) }, 400);
        }
        const message =
          error instanceof Error ? error.message : "MA training failed";
        return handleUnexpectedError(c, error, message);
      }
    },
  );

  // ES Training
  modelsApi.post(
    "/api/v1/models/es/training",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", trainExponentialSmoothingSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Only students can train models" }, 403);
      }
      const studentId = payload.sub;
      const body = c.req.valid("json");

      try {
        const allData = await loadDatasetFromDb(
          pool,
          body.selected_industry,
          body.selected_company,
          body.selected_product,
        );

        if (
          body.data_window_train_end_index >= allData.length ||
          body.data_window_evaluate_end_index >= allData.length
        ) {
          return c.json(
            {
              error: "Data window indices out of bounds",
              dataset_size: allData.length,
            },
            400,
          );
        }

        const dataSimple = prepareSimpleData(allData);
        const modelSaveLoc = path.resolve(
          import.meta.dir,
          `../../py/trained_models/${studentId}_es.pkl`,
        );

        const result = await runPythonMain(
          [
            "--method",
            "es",
            "--data",
            dataSimple,
            "--train_start",
            String(body.data_window_train_start_index),
            "--train_end",
            String(body.data_window_train_end_index),
            "--eval_start",
            String(body.data_window_evaluate_start_index),
            "--eval_end",
            String(body.data_window_evaluate_end_index),
            "--alpha",
            String(body.exponential_smoothing_alpha),
            "--save_model",
            modelSaveLoc,
            "--pretty",
          ],
          TRAINING_TIMEOUT,
        ); // 30-second timeout

        return c.json({ status: "success", results: result });
      } catch (error: any) {
        if (error.message.includes("timed out")) {
          return c.json({ error: "Training timed out after 30 seconds" }, 504);
        }
        if (error.message.startsWith("[PYTHON_VALIDATION]")) {
          return c.json({ error: error.message.substring(20) }, 400);
        }
        if (error.message.startsWith("[PYTHON_DATA]")) {
          return c.json({ error: error.message.substring(14) }, 400);
        }
        const message =
          error instanceof Error ? error.message : "ES training failed";
        return handleUnexpectedError(c, error, message);
      }
        return handleUnexpectedError(c, error, message);
      }
    },
  );

  // ARIMA Training
  modelsApi.post(
    "/api/v1/models/arima/training",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", trainArimaSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Only students can train models" }, 403);
      }
      const studentId = payload.sub;
      const body = c.req.valid("json");

      try {
        const allData = await loadDatasetFromDb(
          pool,
          body.selected_industry,
          body.selected_company,
          body.selected_product,
        );

        if (
          body.data_window_train_end_index >= allData.length ||
          body.data_window_evaluate_end_index >= allData.length
        ) {
          return c.json(
            {
              error: "Data window indices out of bounds",
              dataset_size: allData.length,
            },
            400,
          );
        }

        const dataSimple = prepareSimpleData(allData);
        const modelSaveLoc = path.resolve(
          import.meta.dir,
          `../../py/trained_models/${studentId}_arima.pkl`,
        );

        const result = await runPythonMain(
          [
            "--method",
            "arima",
            "--data",
            dataSimple,
            "--train_start",
            String(body.data_window_train_start_index),
            "--train_end",
            String(body.data_window_train_end_index),
            "--eval_start",
            String(body.data_window_evaluate_start_index),
            "--eval_end",
            String(body.data_window_evaluate_end_index),
            "--d",
            String(body.arima_d),
            "--save_model",
            modelSaveLoc,
            "--pretty",
          ],
          TRAINING_TIMEOUT,
        ); // 30-second timeout

        return c.json({ status: "success", results: result });
      } catch (error: any) {
        if (error.message.includes("timed out")) {
          return c.json({ error: "Training timed out after 30 seconds" }, 504);
        }
        if (error.message.startsWith("[PYTHON_VALIDATION]")) {
          return c.json({ error: error.message.substring(20) }, 400);
        }
        if (error.message.startsWith("[PYTHON_DATA]")) {
          return c.json({ error: error.message.substring(14) }, 400);
        }
        const message =
          error instanceof Error ? error.message : "ARIMA training failed";
        return handleUnexpectedError(c, error, message);
      }
    },
  );

  // LSTM Training
  modelsApi.post(
    "/api/v1/models/lstm/training",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", trainLstmSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Only students can train models" }, 403);
      }
      const studentId = payload.sub;
      const body = c.req.valid("json");

      try {
        const allData = await loadDatasetFromDb(
          pool,
          body.selected_industry,
          body.selected_company,
          body.selected_product,
        );

        if (
          body.data_window_train_end_index >= allData.length ||
          body.data_window_evaluate_end_index >= allData.length
        ) {
          return c.json(
            {
              error: "Data window indices out of bounds",
              dataset_size: allData.length,
            },
            400,
          );
        }

        const { dataFull, featureTypes, invalidFeatures } = prepareFullData(
          allData,
          body.lstm_features,
        );

        // Validate that all selected features are usable
        if (invalidFeatures.length > 0) {
          return c.json(
            {
              error: "Invalid feature(s) provided for LSTM training.",
              details: `The following features could not be classified as numeric or binary: ${invalidFeatures.join(", ")}`,
              invalid_features: invalidFeatures,
            },
            400,
          );
        }

        const modelSaveLoc = path.resolve(
          import.meta.dir,
          `../../py/trained_models/${studentId}_lstm.pt`,
        );

        const result = await runPythonMain(
          [
            "--method",
            "lstm",
            "--data",
            dataFull,
            "--feature_types",
            JSON.stringify(featureTypes),
            "--train_start",
            String(body.data_window_train_start_index),
            "--train_end",
            String(body.data_window_train_end_index),
            "--eval_start",
            String(body.data_window_evaluate_start_index),
            "--eval_end",
            String(body.data_window_evaluate_end_index),
            "--norm",
            body.lstm_normalization,
            "--save_model",
            modelSaveLoc,
            "--pretty",
          ],
          TRAINING_TIMEOUT,
        ); // 30-second timeout

        return c.json({
          status: "success",
          results: result,
          inferred_feature_types: featureTypes,
        });
      } catch (error: any) {
        if (error.message.includes("timed out")) {
          return c.json({ error: "Training timed out after 30 seconds" }, 504);
        }
        if (error.message.startsWith("[PYTHON_VALIDATION]")) {
          return c.json({ error: error.message.substring(20) }, 400);
        }
        if (error.message.startsWith("[PYTHON_DATA]")) {
          return c.json({ error: error.message.substring(14) }, 400);
        }
        const message =
          error instanceof Error ? error.message : "LSTM training failed";
        return handleUnexpectedError(c, error, message);
      }
    },
  );

  // ========== Ensemble Model Training Routes ==========

  // Weighted Average Training
  modelsApi.post(
    "/api/v1/models/weighted-average/training",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", trainWeightedAverageSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Only students can train models" }, 403);
      }
      const studentId = payload.sub;
      const body = c.req.valid("json");

      try {
        const allData = await loadDatasetFromDb(
          pool,
          body.selected_industry,
          body.selected_company,
          body.selected_product,
        );

        if (body.data_window_evaluate_end_index >= allData.length) {
          return c.json(
            {
              error: "Data window indices out of bounds",
              dataset_size: allData.length,
            },
            400,
          );
        }

        const { dataFull } = prepareFullData(allData);
        const modelPaths = body.ensemble_weighted_base_models.map((m) =>
          path.resolve(
            import.meta.dir,
            `../../py/trained_models/${studentId}_${m}.${m === "lstm" ? "pt" : "pkl"}`,
          ),
        );
        const ensembleSaveLoc = path.resolve(
          import.meta.dir,
          `../../py/trained_models/${studentId}_weighted_average_config.pkl`,
        );

        const result = await runPythonMain(
          [
            "--method",
            "weighted_average",
            "--data",
            dataFull,
            "--eval_start",
            String(body.data_window_evaluate_start_index),
            "--eval_end",
            String(body.data_window_evaluate_end_index),
            "--weight_strategy",
            "inverse_variance",
            "--save_ensemble",
            ensembleSaveLoc,
            "--models",
            ...modelPaths,
            "--pretty",
          ],
          TRAINING_TIMEOUT,
        ); // 10-second timeout

        return c.json({
          status: "success",
          results: result,
        });
      } catch (error: any) {
        if (error.message.includes("timed out")) {
          return c.json({ error: "Training timed out after 30 seconds" }, 504);
        }
        if (error.message.includes("An unexpected error occurred")) {
          // 意外退出，返回Python原始错误
          return c.json({ error: "内部系统错误: " + error.message }, 500);
        }
        if (
          error.message.includes("Error: Ensemble requires at least 2 models")
        ) {
          return c.json(
            {
              error: "加权平均集成需要至少2个基础模型，请提供更多有效的模型。",
            },
            500,
          );
        }
        const message =
          error instanceof Error
            ? error.message
            : "Weighted average training failed";
        return handleUnexpectedError(c, error, message);
      }
    },
  );

  // Boosting Training
  modelsApi.post(
    "/api/v1/models/boosting/training",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", trainBoostingSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Only students can train models" }, 403);
      }
      const studentId = payload.sub;
      const body = c.req.valid("json");

      try {
        const allData = await loadDatasetFromDb(
          pool,
          body.selected_industry,
          body.selected_company,
          body.selected_product,
        );

        if (
          body.data_window_train_end_index >= allData.length ||
          body.data_window_evaluate_end_index >= allData.length
        ) {
          return c.json(
            {
              error: "Data window indices out of bounds",
              dataset_size: allData.length,
            },
            400,
          );
        }

        // Validate base model parameters
        if (body.ensemble_boosting_base_models.includes("ma")) {
          if (body.moving_average_window === undefined) {
            return c.json(
              {
                error:
                  "moving_average_window is required when MA is included in base models",
              },
              400,
            );
          }
        }

        if (body.ensemble_boosting_base_models.includes("es")) {
          if (body.exponential_smoothing_alpha === undefined) {
            return c.json(
              {
                error:
                  "exponential_smoothing_alpha is required when ES is included in base models",
              },
              400,
            );
          }
        }

        if (body.ensemble_boosting_base_models.includes("arima")) {
          if (body.arima_d === undefined) {
            return c.json(
              {
                error:
                  "arima_d is required when ARIMA is included in base models",
              },
              400,
            );
          }
        }

        if (body.ensemble_boosting_base_models.includes("lstm")) {
          if (!body.lstm_features || body.lstm_features.length === 0) {
            return c.json(
              {
                error:
                  "lstm_features is required when LSTM is included in base models",
              },
              400,
            );
          }
          if (!body.lstm_target_field) {
            return c.json(
              {
                error:
                  "lstm_target_field is required when LSTM is included in base models",
              },
              400,
            );
          }
          if (!body.lstm_normalization) {
            return c.json(
              {
                error:
                  "lstm_normalization is required when LSTM is included in base models",
              },
              400,
            );
          }
        }

        const { dataFull, featureTypes } = prepareFullData(
          allData,
          body.lstm_features,
        );
        const modelPaths = body.ensemble_boosting_base_models.map((m) =>
          path.resolve(
            import.meta.dir,
            `../../py/trained_models/${studentId}_${m}.${m === "lstm" ? "pt" : "pkl"}`,
          ),
        );
        const ensembleSaveLoc = path.resolve(
          import.meta.dir,
          `../../py/trained_models/${studentId}_boosting_config.pkl`,
        );

        const result = await runPythonMain(
          [
            "--method",
            "boosting",
            "--data",
            dataFull,
            "--train_start",
            String(body.data_window_train_start_index),
            "--train_end",
            String(body.data_window_train_end_index),
            "--eval_start",
            String(body.data_window_evaluate_start_index),
            "--eval_end",
            String(body.data_window_evaluate_end_index),
            "--d",
            String(body.arima_d ?? 0),
            "--window",
            String(body.moving_average_window ?? 1),
            "--alpha",
            String(body.exponential_smoothing_alpha ?? 0.5),
            "--lookback",
            "4",
            "--norm",
            body.lstm_normalization ?? "zscore",
            "--feature_types",
            JSON.stringify(featureTypes),
            "--learning_rate",
            String(body.learning_rate ?? 1),
            "--save_ensemble",
            ensembleSaveLoc,
            "--models",
            ...modelPaths,
            "--pretty",
          ],
          TRAINING_TIMEOUT,
        ); // 10-second timeout

        return c.json({
          status: "success",
          results: result,
          ...(Object.keys(featureTypes).length > 0 && {
            inferred_feature_types: featureTypes,
          }),
        });
      } catch (error: any) {
        if (error.message.includes("timed out")) {
          return c.json({ error: "Training timed out after 30 seconds" }, 504);
        }
        if (error.message.includes("An unexpected error occurred")) {
          // 意外退出，返回Python原始错误
          return c.json({ error: "内部系统错误: " + error.message }, 500);
        }
        if (
          error.message.includes(
            "Error: AdaBoost.R2 training failed to produce any valid models",
          )
        ) {
          return c.json(
            {
              error:
                "无法训练有效的AdaBoost集成模型，请确保至少一个基础模型训练成功且数据质量良好。",
            },
            500,
          );
        }
        if (
          error.message.includes(
            "Error: AdaBoost.R2 failed to get predictions from any valid models",
          )
        ) {
          return c.json(
            {
              error:
                "AdaBoost模型预测失败，无法生成有效预测结果，请检查模型文件完整性。",
            },
            500,
          );
        }
        const message =
          error instanceof Error ? error.message : "Boosting training failed";
        return handleUnexpectedError(c, error, message);
      }
    },
  );

  // Stacking Training
  modelsApi.post(
    "/api/v1/models/stacking/training",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", trainStackingSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Only students can train models" }, 403);
      }
      const studentId = payload.sub;
      const body = c.req.valid("json");

      try {
        const allData = await loadDatasetFromDb(
          pool,
          body.selected_industry,
          body.selected_company,
          body.selected_product,
        );

        if (body.data_window_evaluate_end_index >= allData.length) {
          return c.json(
            {
              error: "Data window indices out of bounds",
              dataset_size: allData.length,
            },
            400,
          );
        }

        const { dataFull } = prepareFullData(allData);
        const modelPaths = body.ensemble_stacking_base_models.map((m) =>
          path.resolve(
            import.meta.dir,
            `../../py/trained_models/${studentId}_${m}.${m === "lstm" ? "pt" : "pkl"}`,
          ),
        );
        const ensembleSaveLoc = path.resolve(
          import.meta.dir,
          `../../py/trained_models/${studentId}_stacking_config.pkl`,
        );

        let dataToSend = "";
        let featureTypes: Record<string, "numeric" | "binary" | "categorical"> = {};
        const args = [
          "--method",
          "stackingv2",
          "--eval_start",
          String(body.data_window_evaluate_start_index),
          "--eval_end",
          String(body.data_window_evaluate_end_index),
          "--base_models",
          ...body.ensemble_stacking_base_models,
          "--meta_l2",
          String(body.meta_l2 ?? 0.1),
          "--pretty",
        ];

        if (body.ensemble_stacking_base_models.includes("lstm")) {
          const { dataFull, featureTypes: ft, invalidFeatures } = prepareFullData(allData, body.lstm_features);
          if (invalidFeatures.length > 0) {
            return c.json({ error: `Invalid features for LSTM: ${invalidFeatures.join(", ")}` }, 400);
          }
          dataToSend = dataFull;
          featureTypes = ft;
          args.push("--feature_types", JSON.stringify(featureTypes));
        } else {
          dataToSend = prepareSimpleData(allData);
        }
        args.push("--data", dataToSend);

        const result = await runPythonMain(args, TRAINING_TIMEOUT); // 10-second timeout

        return c.json({ status: "success", results: result });
      } catch (error: any) {
        if (error.message.includes("timed out")) {
          return c.json({ error: "Training timed out after 30 seconds" }, 504);
        }
        const message =
          error instanceof Error ? error.message : "Stacking training failed";
        return handleUnexpectedError(c, error, message);
      }
    },
  );

  // ========== Prediction Routes ==========

  // Universal Prediction Endpoint
  modelsApi.post(
    "/api/v1/models/predictions",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", predictSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Only students can predict" }, 403);
      }
      const studentId = payload.sub;
      const body = c.req.valid("json");

      try {
        const modelType = body.model_type;
        const isEnsemble = [
          "weighted_average",
          "boosting",
          "stacking",
        ].includes(modelType);

        let modelPath: string;
        if (isEnsemble) {
          modelPath = path.resolve(
            import.meta.dir,
            `../../py/trained_models/${studentId}_${modelType}_config.pkl`,
          );
        } else {
          const ext = modelType === "lstm" ? "pt" : "pkl";
          modelPath = path.resolve(
            import.meta.dir,
            `../../py/trained_models/${studentId}_${modelType}.${ext}`,
          );
        }

        // Check if model exists
        if (!fs.existsSync(modelPath)) {
          return c.json(
            {
              error: `Model not found. Please train the ${modelType} model first.`,
            },
            404,
          );
        }

        const pythonArgs = ["--method", modelType];
        if (isEnsemble) {
          pythonArgs.push("--load_ensemble", modelPath);
        } else {
          pythonArgs.push("--load_model", modelPath);
        }
        pythonArgs.push(
          "--forecast_steps",
          String(body.forecast_steps),
          "--pretty",
        );

        const result = await runPythonMain(pythonArgs);

        return c.json({ status: "success", results: result });
      } catch (error: any) {
        const message =
          error instanceof Error ? error.message : "Prediction failed";
        return handleUnexpectedError(c, error, message);
      }
    },
  );

  return modelsApi;
};
