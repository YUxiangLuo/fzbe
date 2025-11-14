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

// ========== Prediction Schema ==========
const predictSchema = z.object({
  model_type: z.enum([
    "ma",
    "es",
    "arima",
    "lstm",
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

  return modelsApi;
};
