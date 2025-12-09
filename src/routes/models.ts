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

const TRAINING_TIMEOUT = 300000; // 5 minutes for all models
const ENSEMBLE_TRAINING_TIMEOUT = 300000; // 5 minutes for all models

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

// ========== Model Parameter Schemas ==========
const modelParamsSchema = z.object({
  // MA
  moving_average_window: z.coerce.number().int().min(2).optional(),
  // ES
  exponential_smoothing_alpha: z.coerce.number().min(0).max(1).optional(),
  // ARIMA (for auto-search)
  arimaD: z.coerce.number().int().min(0).optional(),

  // lstm
  lstmFeatures: z.string().min(1).optional(),
  lstmTargetFeature: z.string().min(1).optional(),
  lstmNormalization: z.enum(['minmax', 'zscore']).optional(),

  // Ensemble Models
  models: z.string().min(1).optional(),
});

// ========== Unified Training Schema ==========
const trainSchemaBase = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .merge(modelParamsSchema);

const trainSchema = trainSchemaBase
  .refine((data: z.infer<typeof trainSchemaBase>) => data.data_window_train_start_index <= data.data_window_train_end_index, {
    message: "训练起始索引必须小于等于训练结束索引",
    path: ["data_window_train_end_index"],
  })
  .refine((data: z.infer<typeof trainSchemaBase>) => data.data_window_evaluate_start_index <= data.data_window_evaluate_end_index, {
    message: "评估起始索引必须小于等于评估结束索引",
    path: ["data_window_evaluate_end_index"],
  });

// ========== Unified Prediction Schema ==========
const predictSchema = z.object({
  forecast_steps: z.coerce.number().int().min(1),
});

// ========== Model Definitions ==========
const ALL_MODELS = [
  "ma", "es", "arima", "lstm",
  "stacking", "weighted_avg", "boosting"
] as const;
type ModelType = (typeof ALL_MODELS)[number];

const modelParameterMappings: Record<ModelType, Record<string, string>> = {
  ma: { moving_average_window: "window" },
  es: { exponential_smoothing_alpha: "alpha" },
  arima: { arimaD: "d" }, // arimaOrder for ensembles, arimaD for auto_arima
  lstm: {
    lstmLookBack: "look-back",
    lstmEpochs: "epochs",
    lstmFeatures: "lstm-features",
    lstmTargetFeature: "lstm-target-feature",
    lstmFeatureTypes: "feature-types",
    lstmNormalization: "normalization"
  },
  stacking: { models: "models" },
  weighted_avg: { models: "models" },
  boosting: { models: "models" }
};

// Define all possible parameters that any model might use
const allPossiblePythonParams: Record<string, string> = {
  // Common
  data: "data",
  trainStart: "train-start",
  trainEnd: "train-end",
  evalStart: "eval-start",
  evalEnd: "eval-end",
  saveModel: "save-model",
  loadModel: "load-model",
  forecastSteps: "forecast-steps",
  pretty: "pretty",
  // MA
  moving_average_window: "window",
  // ES
  exponential_smoothing_alpha: "alpha",
  // ARIMA
  arimaD: "d",
  arimaMaxP: "max-p", // For auto_arima
  arimaMaxQ: "max-q", // For auto_arima
  // LSTM
  lstmLookBack: "look-back",
  lstmEpochs: "epochs",
  lstmFeatures: "lstm-features",
  lstmTargetFeature: "lstm-target-feature",
  lstmFeatureTypes: "feature-types",
  lstmNormalization: "normalization",
  // Ensemble Models
  models: "models", // For stacking, weighted_avg, and boosting
};

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
    throw new Error("未找到数据集分段文件");
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

async function runPythonModel(
  modelType: string,
  params: Record<string, any>,
  timeoutMs?: number,
): Promise<any> {
  const pyDir = path.resolve(import.meta.dir, "../../py");
  const pythonExec =
    process.platform === "win32"
      ? path.join(pyDir, "venv", "Scripts", "python.exe")
      : path.join(pyDir, "venv", "bin", "python");

  // Use the unified entry script
  const scriptPath = path.join(pyDir, "run_model.py");

  if (!fs.existsSync(scriptPath)) {
    throw new Error(`未找到 Python 入口脚本: ${scriptPath}`);
  }

  // Convert camelCase params to kebab-case CLI arguments
  const args = Object.entries(params).flatMap(([key, value]) => {
    // Use the predefined mapping for script keys
    const scriptKey = allPossiblePythonParams[key] || key; // Fallback to original if not mapped
    const cliArg = `--${scriptKey}`;
    if (typeof value === 'boolean') {
      return value ? [cliArg] : [];
    }
    return [cliArg, String(value)];
  });

  // Prepend model_type to args
  const fullArgs = [modelType, ...args];

  const proc = Bun.spawn([pythonExec, scriptPath, ...fullArgs], {
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
        logger.error({ stderr, modelType, args: fullArgs }, "Python 执行失败");
        const pythonErrorMessage = stderr.trim();
        if (pythonErrorMessage.startsWith("[VALIDATION]")) {
          reject(new Error(`[PYTHON_VALIDATION] ${pythonErrorMessage.substring(13).trim()}`));
        } else if (pythonErrorMessage.startsWith("[DATA]")) {
          reject(new Error(`[PYTHON_DATA] ${pythonErrorMessage.substring(7).trim()}`));
        } else {
          reject(new Error(`Python 执行失败: ${stderr}`));
        }
      } else {
        try {
          resolve(JSON.parse(stdout));
        } catch (e) {
          logger.error({ stdout, modelType, args: fullArgs }, "解析 Python 输出失败");
          reject(new Error("Python 输出无效"));
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
      reject(new Error(`Python 执行超时 (${timeoutMs}ms)`));
    }, timeoutMs);
  });

  return Promise.race([executionPromise, timeoutPromise]);
}

// --- Result Transformers for Frontend Compatibility ---
const resultTransformers: Record<ModelType, (result: any) => any> = {
  arima: (result) => ({
    best_order: result.best_order,
    search_summary: result.search_summary, // Add search_summary here
    // Keep diagnostics for ARIMA as it contains conf_int
    diagnostics: result.diagnostics ? { ...result.diagnostics, y_pred_eval: result.predictions?.map((p: any) => p.prediction) || [] } : undefined,
  }),
  es: (result) => ({}), // No specific extra fields for ES
  ma: (result) => ({}), // No specific extra fields for MA
  lstm: (result) => ({
    feature_types: result.feature_types,
    diagnostics: result.diagnostics ? { ...result.diagnostics, y_pred_eval: result.predictions?.map((p: any) => p.prediction) || [] } : undefined,
  }),
  stacking: (result) => ({
    model_names: result.models,
  }),
  weighted_avg: (result) => ({
    model_names: result.models,
    weights: result.models?.map((name: string) => result.weights[name]),
  }),
  boosting: (result) => ({
    model_names: result.model_chain,
    search_summary: result.search_summary,
  }),
};

// ========== API Endpoints ==========
export default (pool: Pool, JWT_SECRET: string) => {
  const modelsApi = new Hono();

  // ========== Unified Model Training Route ==========
  modelsApi.post(
    "/api/v1/models/:model_type/training",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", trainSchema),
    async (c) => {
      const modelType = c.req.param("model_type") as ModelType;
      if (!ALL_MODELS.includes(modelType)) {
        return c.json({ error: "指定的模型类型无效" }, 400);
      }

      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "只有学生可以训练模型" }, 403);
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
            { error: "数据窗口索引超出范围", dataset_size: allData.length },
            400,
          );
        }

        const modelSaveLoc = path.resolve(
          import.meta.dir,
          `../../py/trained_models/${studentId}_${modelType}.pkl`,
        );

        const pythonParams: Record<string, any> = {
          data: JSON.stringify(allData),
          trainStart: body.data_window_train_start_index,
          trainEnd: body.data_window_train_end_index,
          evalStart: body.data_window_evaluate_start_index,
          evalEnd: body.data_window_evaluate_end_index,
          saveModel: modelSaveLoc,
          pretty: true,
        };

        // Dynamically add all relevant parameters from the body to pythonParams
        for (const [bodyKey, scriptKey] of Object.entries(allPossiblePythonParams)) {
          if (body[bodyKey as keyof typeof body] !== undefined) {
            pythonParams[scriptKey] = body[bodyKey as keyof typeof body];
          }
        }

        // Use longer timeout for ensemble models
        const isEnsembleModel = ["stacking", "weighted_avg", "boosting"].includes(modelType);
        const timeout = isEnsembleModel ? ENSEMBLE_TRAINING_TIMEOUT : TRAINING_TIMEOUT;
        const result = await runPythonModel(modelType, pythonParams, timeout);

        // --- Data Transformation for Frontend Compatibility ---
        const baseResults = {
          metrics: result.metrics,
          eval_y_true: result.eval_y_true,
          eval_predictions: result.predictions?.map((p: any) => p.prediction) || [],
          notes: result.notes || [],
          saved_model: result.saved_model,
          logs: result.logs, // <-- Add logs here
        };

        const modelSpecificTransformedResult = resultTransformers[modelType]?.(result) || {};
        const finalTransformedResult = { ...baseResults, ...modelSpecificTransformedResult };

        const finalResponse: { status: string; results: any; inferred_feature_types?: any } = {
          status: "success",
          results: finalTransformedResult,
        };

        // Special handling for LSTM inferred feature types (if present)
        if (modelType === 'lstm' && result.feature_types) {
          finalResponse.inferred_feature_types = result.feature_types;
        }

        return c.json(finalResponse);
      } catch (error: any) {
        if (error.message.includes("timed out")) {
          const isEnsembleModel = ["stacking", "weighted_avg", "boosting"].includes(modelType);
          const timeoutSeconds = isEnsembleModel ? 120 : 30;
          return c.json({ error: `训练超时（${timeoutSeconds}秒）` }, 504);
        }
        if (error.message.startsWith("[PYTHON_VALIDATION]")) {
          return c.json({ error: error.message.substring(20) }, 400);
        }
        if (error.message.startsWith("[PYTHON_DATA]")) {
          return c.json({ error: error.message.substring(14) }, 400);
        }
        const message = error instanceof Error ? error.message : `${modelType.toUpperCase()} 训练失败`;
        return handleUnexpectedError(c, error, message);
      }
    },
  );

  // ========== Unified Model Prediction Route ==========
  modelsApi.post(
    "/api/v1/models/:model_type/predict",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", predictSchema),
    async (c) => {
      const modelType = c.req.param("model_type") as ModelType;
      if (!ALL_MODELS.includes(modelType)) {
        return c.json({ error: "指定的模型类型无效" }, 400);
      }

      const payload = c.get("jwtPayload");
      const studentId = payload.sub;
      const body = c.req.valid("json");

      try {
        const modelLoadPath = path.resolve(
          import.meta.dir,
          `../../py/trained_models/${studentId}_${modelType}.pkl`,
        );

        if (!fs.existsSync(modelLoadPath)) {
          return c.json({ error: `未找到该用户的已训练 ${modelType.toUpperCase()} 模型` }, 404);
        }

        const pythonParams = {
          loadModel: modelLoadPath,
          forecastSteps: body.forecast_steps,
          pretty: true,
        };

        const result = await runPythonModel(modelType, pythonParams);

        // Return predictions with std_dev
        return c.json({
          status: "success",
          results: {
            predictions: result.predictions || []
          }
        });
      } catch (error: any) {
        const message = error instanceof Error ? error.message : `${modelType.toUpperCase()} 预测失败`;
        return handleUnexpectedError(c, error, message);
      }
    },
  );

  return modelsApi;
};