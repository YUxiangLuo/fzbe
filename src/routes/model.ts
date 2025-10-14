import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import path from "path";
import fs from "fs";
import type { Pool } from "mysql2/promise";
import csv from "csv-parser";

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
  .refine((data) => data.data_window_train_start_index <= data.data_window_train_end_index, {
    message: "train_start_index must be <= train_end_index",
    path: ["data_window_train_end_index"],
  })
  .refine((data) => data.data_window_evaluate_start_index <= data.data_window_evaluate_end_index, {
    message: "evaluate_start_index must be <= evaluate_end_index",
    path: ["data_window_evaluate_end_index"],
  });

const trainExponentialSmoothingSchema = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .extend({
    exponential_smoothing_alpha: z.coerce.number().min(0).max(1),
  })
  .refine((data) => data.data_window_train_start_index <= data.data_window_train_end_index, {
    message: "train_start_index must be <= train_end_index",
    path: ["data_window_train_end_index"],
  })
  .refine((data) => data.data_window_evaluate_start_index <= data.data_window_evaluate_end_index, {
    message: "evaluate_start_index must be <= evaluate_end_index",
    path: ["data_window_evaluate_end_index"],
  });

const trainArimaSchema = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .extend({
    arima_d: z.coerce.number().int().min(0),
  })
  .refine((data) => data.data_window_train_start_index <= data.data_window_train_end_index, {
    message: "train_start_index must be <= train_end_index",
    path: ["data_window_train_end_index"],
  })
  .refine((data) => data.data_window_evaluate_start_index <= data.data_window_evaluate_end_index, {
    message: "evaluate_start_index must be <= evaluate_end_index",
    path: ["data_window_evaluate_end_index"],
  });

const trainLstmSchema = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .extend({
    lstm_features: z.array(z.string()).min(1, "At least one feature is required"),
    lstm_target_field: z.string().min(1, "Target field is required"),
    lstm_normalization: z.enum(["zscore", "minmax"]).default("zscore"),
  })
  .refine((data) => data.data_window_train_start_index <= data.data_window_train_end_index, {
    message: "train_start_index must be <= train_end_index",
    path: ["data_window_train_end_index"],
  })
  .refine((data) => data.data_window_evaluate_start_index <= data.data_window_evaluate_end_index, {
    message: "evaluate_start_index must be <= evaluate_end_index",
    path: ["data_window_evaluate_end_index"],
  });

// ========== Ensemble Model Training Schemas ==========
const trainWeightedAverageSchema = baseDataSchema
  .merge(evalRangeSchema)
  .extend({
    ensemble_weighted_base_models: z.array(z.enum(["ma", "es", "arima", "lstm"])).min(2),
    weight_strategy: z.enum(["equal", "inverse_rmse", "grid_search", "linear_regression"]).default("inverse_rmse"),
  })
  .refine((data) => data.data_window_evaluate_start_index <= data.data_window_evaluate_end_index, {
    message: "evaluate_start_index must be <= evaluate_end_index",
    path: ["data_window_evaluate_end_index"],
  });

const trainBoostingSchema = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .extend({
    ensemble_boosting_base_models: z.array(z.enum(["ma", "es", "arima", "lstm"])).min(1),
    arima_d: z.coerce.number().int().min(0).optional(),
    moving_average_window: z.coerce.number().int().min(1).optional(),
    exponential_smoothing_alpha: z.coerce.number().min(0).max(1).optional(),
    lstm_features: z.array(z.string()).optional(),
    lstm_target_field: z.string().optional(),
    lstm_normalization: z.enum(["zscore", "minmax"]).optional(),
  })
  .refine((data) => data.data_window_train_start_index <= data.data_window_train_end_index, {
    message: "train_start_index must be <= train_end_index",
    path: ["data_window_train_end_index"],
  })
  .refine((data) => data.data_window_evaluate_start_index <= data.data_window_evaluate_end_index, {
    message: "evaluate_start_index must be <= evaluate_end_index",
    path: ["data_window_evaluate_end_index"],
  });

const trainStackingSchema = baseDataSchema
  .merge(evalRangeSchema)
  .extend({
    ensemble_stacking_base_models: z.array(z.enum(["ma", "es", "arima", "lstm"])).min(2),
  })
  .refine((data) => data.data_window_evaluate_start_index <= data.data_window_evaluate_end_index, {
    message: "evaluate_start_index must be <= evaluate_end_index",
    path: ["data_window_evaluate_end_index"],
  });

// ========== Prediction Schema ==========
const predictSchema = z.object({
  model_type: z.enum(["ma", "es", "arima", "lstm", "weighted_average", "boosting", "stacking"]),
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
      .pipe(csv({ mapHeaders: ({ header }: { header: string }) => header.replace(/^\uFEFF/, "").trim() }))
      .on("data", (row: Record<string, any>) => allData.push(row))
      .on("end", resolve)
      .on("error", reject);
  });

  return allData;
}

function prepareSimpleData(data: Record<string, any>[]): string {
  // Simple format: [{"month": "2021-01", "sales": 37200}, ...]
  // Used by MA/ES/ARIMA models
  return JSON.stringify(data.map(row => {
    const year = row['年份'] || row.year || "0";
    const month = row['月份'] || row.month || "1";
    const sales = parseFloat(row['销售数量'] || row['销售总额'] || row.sales || row.y || "0");

    const ts = `${year}-${String(month).padStart(2, '0')}`;

    return {
      month: ts,
      sales: sales
    };
  }));
}

function prepareFullData(
  data: Record<string, any>[],
  lstmFeatures?: string[]
): {
  dataFull: string;
  featureTypes: Record<string, "numeric" | "binary">;
  invalidFeatures: string[];
} {
  const excludeFields = ['年份', '月份', '销售数量', '销售总额', 'year', 'month', 'sales', 'y',
                         '行业名称', '公司代码', '公司名称', '产品编码', '产品名称'];

  let featuresToUse: string[];
  if (lstmFeatures && lstmFeatures.length > 0) {
    featuresToUse = lstmFeatures;
  } else {
    const allColumns = Object.keys(data[0] || {});
    featuresToUse = allColumns.filter(col => !excludeFields.includes(col));
  }

  const featureTypes: Record<string, "numeric" | "binary"> = {};
  const invalidFeatures: string[] = [];

  for (const feature of featuresToUse) {
    const values = data
      .map(row => row[feature])
      .filter(val => val !== null && val !== undefined && val !== "");

    if (values.length === 0) {
      featureTypes[feature] = "numeric"; // Assume numeric if no data
      continue;
    }

    const uniqueValues = new Set(values.map(v => String(v).trim().toLowerCase()));

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
    } else {
      // This is an unsupported categorical feature
      invalidFeatures.push(feature);
    }
  }

  const dataArray = data.map(row => {
    const year = row['年份'] || row.year || "0";
    const month = row['月份'] || row.month || "1";
    const sales = parseFloat(row['销售数量'] || row['销售总额'] || row.sales || row.y || "0");
    const ts = `${year}-${String(month).padStart(2, '0')}`;

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


async function runShellScript(
  scriptName: string,
  args: string[],
): Promise<any> {
  const pyDir = path.resolve(import.meta.dir, "../../py");
  const scriptPath = path.join(pyDir, scriptName);

  const proc = Bun.spawn(["bash", scriptPath, ...args], {
    cwd: pyDir,
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, PATH: `${pyDir}/venv/bin:${process.env.PATH}` },
  });

  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);

  await proc.exited;

  if (proc.exitCode !== 0) {
    console.error("Script error:", stderr);
    throw new Error(`Script execution failed: ${stderr}`);
  }

  try {
    return JSON.parse(stdout);
  } catch (e) {
    console.error("Failed to parse script output:", stdout);
    throw new Error("Invalid output from script");
  }
}

export default (pool: Pool, JWT_SECRET: string) => {
  const modelApi = new Hono();

  modelApi.use("*", jwt({ secret: JWT_SECRET }));

  // ========== Base Model Training Routes ==========

  // MA Training
  modelApi.post("/model/ma/train", zValidator("json", trainMovingAverageSchema), async (c) => {
    const payload = c.get("jwtPayload");
    if (payload.role !== "Student") {
      return c.json({ error: "Only students can train models" }, 403);
    }
    const studentId = payload.sub;
    const body = c.req.valid("json");

    try {
      const allData = await loadDatasetFromDb(pool, body.selected_industry, body.selected_company, body.selected_product);

      if (body.data_window_train_end_index >= allData.length || body.data_window_evaluate_end_index >= allData.length) {
        return c.json({ error: "Data window indices out of bounds", dataset_size: allData.length }, 400);
      }

      const dataSimple = prepareSimpleData(allData);
      const modelSaveLoc = path.resolve(import.meta.dir, `../../py/trained_models/${studentId}_ma.pkl`);

      const result = await runShellScript("train_ma.sh", [
        dataSimple,
        String(body.data_window_train_start_index),
        String(body.data_window_train_end_index),
        String(body.data_window_evaluate_start_index),
        String(body.data_window_evaluate_end_index),
        String(body.moving_average_window),
        modelSaveLoc,
      ]);

      return c.json({ status: "success", results: result });
    } catch (error: any) {
      console.error("MA training error:", error);
      return c.json({ error: error.message }, 500);
    }
  });

  // ES Training
  modelApi.post("/model/es/train", zValidator("json", trainExponentialSmoothingSchema), async (c) => {
    const payload = c.get("jwtPayload");
    if (payload.role !== "Student") {
      return c.json({ error: "Only students can train models" }, 403);
    }
    const studentId = payload.sub;
    const body = c.req.valid("json");

    try {
      const allData = await loadDatasetFromDb(pool, body.selected_industry, body.selected_company, body.selected_product);

      if (body.data_window_train_end_index >= allData.length || body.data_window_evaluate_end_index >= allData.length) {
        return c.json({ error: "Data window indices out of bounds", dataset_size: allData.length }, 400);
      }

      const dataSimple = prepareSimpleData(allData);
      const modelSaveLoc = path.resolve(import.meta.dir, `../../py/trained_models/${studentId}_es.pkl`);

      const result = await runShellScript("train_es.sh", [
        dataSimple,
        String(body.data_window_train_start_index),
        String(body.data_window_train_end_index),
        String(body.data_window_evaluate_start_index),
        String(body.data_window_evaluate_end_index),
        String(body.exponential_smoothing_alpha),
        modelSaveLoc,
      ]);

      return c.json({ status: "success", results: result });
    } catch (error: any) {
      console.error("ES training error:", error);
      return c.json({ error: error.message }, 500);
    }
  });

  // ARIMA Training
  modelApi.post("/model/arima/train", zValidator("json", trainArimaSchema), async (c) => {
    const payload = c.get("jwtPayload");
    if (payload.role !== "Student") {
      return c.json({ error: "Only students can train models" }, 403);
    }
    const studentId = payload.sub;
    const body = c.req.valid("json");

    try {
      const allData = await loadDatasetFromDb(pool, body.selected_industry, body.selected_company, body.selected_product);

      if (body.data_window_train_end_index >= allData.length || body.data_window_evaluate_end_index >= allData.length) {
        return c.json({ error: "Data window indices out of bounds", dataset_size: allData.length }, 400);
      }

      const dataSimple = prepareSimpleData(allData);
      const modelSaveLoc = path.resolve(import.meta.dir, `../../py/trained_models/${studentId}_arima.pkl`);

      const result = await runShellScript("train_arima.sh", [
        dataSimple,
        String(body.data_window_train_start_index),
        String(body.data_window_train_end_index),
        String(body.data_window_evaluate_start_index),
        String(body.data_window_evaluate_end_index),
        String(body.arima_d),
        modelSaveLoc,
      ]);

      return c.json({ status: "success", results: result });
    } catch (error: any) {
      console.error("ARIMA training error:", error);
      return c.json({ error: error.message }, 500);
    }
  });

  // LSTM Training
  modelApi.post("/model/lstm/train", zValidator("json", trainLstmSchema), async (c) => {
    const payload = c.get("jwtPayload");
    if (payload.role !== "Student") {
      return c.json({ error: "Only students can train models" }, 403);
    }
    const studentId = payload.sub;
    const body = c.req.valid("json");

    try {
      const allData = await loadDatasetFromDb(pool, body.selected_industry, body.selected_company, body.selected_product);

      if (body.data_window_train_end_index >= allData.length || body.data_window_evaluate_end_index >= allData.length) {
        return c.json({ error: "Data window indices out of bounds", dataset_size: allData.length }, 400);
      }

      const { dataFull, featureTypes, invalidFeatures } = prepareFullData(allData, body.lstm_features);

      // Validate that all selected features are usable
      if (invalidFeatures.length > 0) {
        return c.json({
          error: "Invalid feature(s) provided for LSTM training.",
          details: `The following features could not be classified as numeric or binary: ${invalidFeatures.join(", ")}`,
          invalid_features: invalidFeatures,
        }, 400);
      }

      const modelSaveLoc = path.resolve(import.meta.dir, `../../py/trained_models/${studentId}_lstm.pt`);

      const result = await runShellScript("train_lstm.sh", [
        dataFull,
        JSON.stringify(featureTypes),
        String(body.data_window_train_start_index),
        String(body.data_window_train_end_index),
        String(body.data_window_evaluate_start_index),
        String(body.data_window_evaluate_end_index),
        body.lstm_normalization,
        modelSaveLoc,
      ]);

      return c.json({
        status: "success",
        results: result,
        inferred_feature_types: featureTypes,
      });
    } catch (error: any) {
      console.error("LSTM training error:", error);
      return c.json({ error: error.message }, 500);
    }
  });

  // ========== Ensemble Model Training Routes ==========

  // Weighted Average Training
  modelApi.post("/model/weighted-average/train", zValidator("json", trainWeightedAverageSchema), async (c) => {
    const payload = c.get("jwtPayload");
    if (payload.role !== "Student") {
      return c.json({ error: "Only students can train models" }, 403);
    }
    const studentId = payload.sub;
    const body = c.req.valid("json");

    try {
      const allData = await loadDatasetFromDb(pool, body.selected_industry, body.selected_company, body.selected_product);

      if (body.data_window_evaluate_end_index >= allData.length) {
        return c.json({ error: "Data window indices out of bounds", dataset_size: allData.length }, 400);
      }

      const { dataFull } = prepareFullData(allData);
      const modelPaths = body.ensemble_weighted_base_models.map(m =>
        path.resolve(import.meta.dir, `../../py/trained_models/${studentId}_${m}.${m === "lstm" ? "pt" : "pkl"}`)
      );
      const ensembleSaveLoc = path.resolve(import.meta.dir, `../../py/trained_models/${studentId}_weighted_average_config.pkl`);

      const result = await runShellScript("train_weighted_average.sh", [
        dataFull,
        String(body.data_window_evaluate_start_index),
        String(body.data_window_evaluate_end_index),
        body.weight_strategy,
        ensembleSaveLoc,
        ...modelPaths,
      ]);

      return c.json({ status: "success", results: result });
    } catch (error: any) {
      console.error("Weighted Average training error:", error);
      return c.json({ error: error.message }, 500);
    }
  });

  // Boosting Training
  modelApi.post("/model/boosting/train", zValidator("json", trainBoostingSchema), async (c) => {
    const payload = c.get("jwtPayload");
    if (payload.role !== "Student") {
      return c.json({ error: "Only students can train models" }, 403);
    }
    const studentId = payload.sub;
    const body = c.req.valid("json");

    try {
      const allData = await loadDatasetFromDb(pool, body.selected_industry, body.selected_company, body.selected_product);

      if (body.data_window_train_end_index >= allData.length || body.data_window_evaluate_end_index >= allData.length) {
        return c.json({ error: "Data window indices out of bounds", dataset_size: allData.length }, 400);
      }

      // Validate base model parameters
      if (body.ensemble_boosting_base_models.includes("ma")) {
        if (body.moving_average_window === undefined) {
          return c.json({
            error: "moving_average_window is required when MA is included in base models"
          }, 400);
        }
      }

      if (body.ensemble_boosting_base_models.includes("es")) {
        if (body.exponential_smoothing_alpha === undefined) {
          return c.json({
            error: "exponential_smoothing_alpha is required when ES is included in base models"
          }, 400);
        }
      }

      if (body.ensemble_boosting_base_models.includes("arima")) {
        if (body.arima_d === undefined) {
          return c.json({
            error: "arima_d is required when ARIMA is included in base models"
          }, 400);
        }
      }

      if (body.ensemble_boosting_base_models.includes("lstm")) {
        if (!body.lstm_features || body.lstm_features.length === 0) {
          return c.json({
            error: "lstm_features is required when LSTM is included in base models"
          }, 400);
        }
        if (!body.lstm_target_field) {
          return c.json({
            error: "lstm_target_field is required when LSTM is included in base models"
          }, 400);
        }
        if (!body.lstm_normalization) {
          return c.json({
            error: "lstm_normalization is required when LSTM is included in base models"
          }, 400);
        }
      }

      const { dataFull, featureTypes } = prepareFullData(allData, body.lstm_features);
      const modelPaths = body.ensemble_boosting_base_models.map(m =>
        path.resolve(import.meta.dir, `../../py/trained_models/${studentId}_${m}.${m === "lstm" ? "pt" : "pkl"}`)
      );
      const ensembleSaveLoc = path.resolve(import.meta.dir, `../../py/trained_models/${studentId}_boosting_config.pkl`);

      const result = await runShellScript("train_boosting.sh", [
        dataFull,
        String(body.data_window_train_start_index),
        String(body.data_window_train_end_index),
        String(body.data_window_evaluate_start_index),
        String(body.data_window_evaluate_end_index),
        String(body.arima_d ?? 0),
        String(body.moving_average_window ?? 1),
        String(body.exponential_smoothing_alpha ?? 0.5),
        "4", // lstm_lookback is fixed at 4
        body.lstm_normalization ?? 'zscore',
        JSON.stringify(featureTypes),
        ensembleSaveLoc,
        ...modelPaths,
      ]);

      return c.json({
        status: "success",
        results: result,
        ...(Object.keys(featureTypes).length > 0 && { inferred_feature_types: featureTypes })
      });
    } catch (error: any) {
      console.error("Boosting training error:", error);
      return c.json({ error: error.message }, 500);
    }
  });

  // Stacking Training
  modelApi.post("/model/stacking/train", zValidator("json", trainStackingSchema), async (c) => {
    const payload = c.get("jwtPayload");
    if (payload.role !== "Student") {
      return c.json({ error: "Only students can train models" }, 403);
    }
    const studentId = payload.sub;
    const body = c.req.valid("json");

    try {
      const allData = await loadDatasetFromDb(pool, body.selected_industry, body.selected_company, body.selected_product);

      if (body.data_window_evaluate_end_index >= allData.length) {
        return c.json({ error: "Data window indices out of bounds", dataset_size: allData.length }, 400);
      }

      const { dataFull } = prepareFullData(allData);
      const modelPaths = body.ensemble_stacking_base_models.map(m =>
        path.resolve(import.meta.dir, `../../py/trained_models/${studentId}_${m}.${m === "lstm" ? "pt" : "pkl"}`)
      );
      const ensembleSaveLoc = path.resolve(import.meta.dir, `../../py/trained_models/${studentId}_stacking_config.pkl`);

      const result = await runShellScript("train_stacking.sh", [
        dataFull,
        String(body.data_window_evaluate_start_index),
        String(body.data_window_evaluate_end_index),
        ensembleSaveLoc,
        ...modelPaths,
      ]);

      return c.json({ status: "success", results: result });
    } catch (error: any) {
      console.error("Stacking training error:", error);
      return c.json({ error: error.message }, 500);
    }
  });

  // ========== Prediction Routes ==========

  // Universal Prediction Endpoint
  modelApi.post("/model/predict", zValidator("json", predictSchema), async (c) => {
    const payload = c.get("jwtPayload");
    if (payload.role !== "Student") {
      return c.json({ error: "Only students can predict" }, 403);
    }
    const studentId = payload.sub;
    const body = c.req.valid("json");

    try {
      const modelType = body.model_type;
      const isEnsemble = ["weighted_average", "boosting", "stacking"].includes(modelType);

      let modelPath: string;
      if (isEnsemble) {
        modelPath = path.resolve(import.meta.dir, `../../py/trained_models/${studentId}_${modelType}_config.pkl`);
      } else {
        const ext = modelType === "lstm" ? "pt" : "pkl";
        modelPath = path.resolve(import.meta.dir, `../../py/trained_models/${studentId}_${modelType}.${ext}`);
      }

      // Check if model exists
      if (!fs.existsSync(modelPath)) {
        return c.json({ error: `Model not found. Please train the ${modelType} model first.` }, 404);
      }

      const scriptName = `predict_${modelType}.sh`;
      const result = await runShellScript(scriptName, [
        modelPath,
        String(body.forecast_steps),
      ]);

      return c.json({ status: "success", results: result });
    } catch (error: any) {
      console.error("Prediction error:", error);
      return c.json({ error: error.message }, 500);
    }
  });

  return modelApi;
};
