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

// ========== Model Parameter Schemas ==========
const modelParamsSchema = z.object({
  // MA
  maWindow: z.coerce.number().int().min(1).optional(),
  // ES
  esAlpha: z.coerce.number().min(0).max(1).optional(),
  // ARIMA (for auto-search)
  arimaD: z.coerce.number().int().min(0).optional(),
  // ARIMA (for ensembles)
  arimaOrder: z.string().regex(/^\d+,\d+,\d+$/).optional(),
  // LSTM
  lstmLookBack: z.coerce.number().int().min(1).optional(),
  lstmEpochs: z.coerce.number().int().min(1).optional(),
  lstmFeatureTypes: z.string().optional(), // JSON string
  // Ensemble Models
  models: z.string().min(1).optional(),
  modelChain: z.string().min(1).optional(),
});

// ========== Unified Training Schema ==========
const trainSchemaBase = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .merge(modelParamsSchema);

const trainSchema = trainSchemaBase
  .refine((data: z.infer<typeof trainSchemaBase>) => data.data_window_train_start_index <= data.data_window_train_end_index, {
    message: "train_start_index must be <= train_end_index",
    path: ["data_window_train_end_index"],
  })
  .refine((data: z.infer<typeof trainSchemaBase>) => data.data_window_evaluate_start_index <= data.data_window_evaluate_end_index, {
    message: "evaluate_start_index must be <= evaluate_end_index",
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
  ma: { maWindow: "window" },
  es: { esAlpha: "alpha" },
  arima: { arimaD: "d" },
  lstm: { 
    lstmLookBack: "lookBack", 
    lstmEpochs: "epochs", 
    lstmFeatureTypes: "featureTypes" 
  },
  stacking: { models: "models" },
  weighted_avg: { models: "models" },
  boosting: { modelChain: "modelChain" },
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

  const scriptName = `run_${modelType}.py`;
  const scriptPath = path.join(pyDir, scriptName);

  if (!fs.existsSync(scriptPath)) {
    throw new Error(`Python script not found for model type: ${modelType}`);
  }

  // Convert camelCase params to kebab-case CLI arguments
  const args = Object.entries(params).flatMap(([key, value]) => {
    const kebabKey = key.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
    const cliArg = `--${kebabKey}`;
    if (typeof value === 'boolean') {
      return value ? [cliArg] : [];
    }
    return [cliArg, String(value)];
  });

  const proc = Bun.spawn([pythonExec, scriptPath, ...args], {
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
        logger.error({ stderr, modelType, args }, "Python execution failed");
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
          logger.error({ stdout, modelType, args }, "Failed to parse python output");
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
        return c.json({ error: "Invalid model type specified" }, 400);
      }

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
            { error: "Data window indices out of bounds", dataset_size: allData.length },
            400,
          );
        }

        const modelSaveLoc = path.resolve(
          import.meta.dir,
          `../../py/trained_models/${studentId}_${modelType}.pkl`,
        );

        // Dynamically build params for the python script
        const pythonParams: Record<string, any> = {
          data: JSON.stringify(allData),
          trainStart: body.data_window_train_start_index,
          trainEnd: body.data_window_train_end_index,
          evalStart: body.data_window_evaluate_start_index,
          evalEnd: body.data_window_evaluate_end_index,
          saveModel: modelSaveLoc,
          pretty: true,
        };

        // Add model-specific params
        const paramMapping = modelParameterMappings[modelType];
        for (const [bodyKey, scriptKey] of Object.entries(paramMapping)) {
          if (body[bodyKey as keyof typeof body] !== undefined) {
            pythonParams[scriptKey] = body[bodyKey as keyof typeof body];
          }
        }
        
        // Add shared ensemble params (if any)
        if (body.models) pythonParams.models = body.models;
        if (body.modelChain) pythonParams.modelChain = body.modelChain;

        // Add all base model params for ensembles
        if (["stacking", "weighted_avg", "boosting"].includes(modelType)) {
            if (body.maWindow) pythonParams.maWindow = body.maWindow;
            if (body.esAlpha) pythonParams.esAlpha = body.esAlpha;
            if (body.arimaOrder) pythonParams.arimaOrder = body.arimaOrder;
            if (body.lstmLookBack) pythonParams.lstmLookBack = body.lstmLookBack;
            if (body.lstmEpochs) pythonParams.lstmEpochs = body.lstmEpochs;
            if (body.lstmFeatureTypes) pythonParams.lstmFeatureTypes = body.lstmFeatureTypes;
        }

        const result = await runPythonModel(modelType, pythonParams, TRAINING_TIMEOUT);

        // --- Data Transformation for Frontend Compatibility (Simplified) ---
        const transformedResult: any = {
          metrics: result.metrics,
          eval_y_true: result.eval_y_true,
          eval_predictions: result.predictions 
            ? result.predictions.map((p: { prediction: number }) => p.prediction)
            : [],
          notes: [], // Always include empty notes
          saved_model: result.saved_model || null,
        };

        if (modelType === 'arima') {
          transformedResult.best_order = result.best_order;
          // Keep diagnostics for ARIMA as it contains conf_int
          if (result.diagnostics) {
            transformedResult.diagnostics = result.diagnostics;
            transformedResult.diagnostics.y_pred_eval = transformedResult.eval_predictions;
          }
        }

        if (modelType === 'weighted_avg') {
          if (result.models && result.weights) {
            const modelList = result.models;
            transformedResult.model_names = modelList;
            transformedResult.weights = modelList.map((name: string) => result.weights[name]);
          }
        }
        
        // For other ensembles, if model_names is desired for consistency
        if (modelType === 'stacking' && result.models) {
            transformedResult.model_names = result.models;
        }
        if (modelType === 'boosting' && result.model_chain) {
            transformedResult.model_names = result.model_chain;
        }

        const finalResponse: { status: string; results: any; inferred_feature_types?: any } = {
          status: "success",
          results: transformedResult,
        };

        if (modelType === 'lstm' && result.feature_types) {
          finalResponse.inferred_feature_types = result.feature_types;
        }

        return c.json(finalResponse);
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
        const message = error instanceof Error ? error.message : `${modelType.toUpperCase()} training failed`;
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
        return c.json({ error: "Invalid model type specified" }, 400);
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
          return c.json({ error: `No trained ${modelType.toUpperCase()} model found for this user.` }, 404);
        }

        const pythonParams = {
          loadModel: modelLoadPath,
          forecastSteps: body.forecast_steps,
          pretty: true,
        };

        const result = await runPythonModel(modelType, pythonParams);

        // --- Data Transformation for Frontend Compatibility ---
        const transformedResult = { ...result };
        if (transformedResult.predictions && Array.isArray(transformedResult.predictions)) {
          transformedResult.forecasts = transformedResult.predictions.map(
            (p: { prediction: number }) => p.prediction
          );
          delete transformedResult.predictions;
        }

        return c.json({ status: "success", results: transformedResult });
      } catch (error: any) {
        const message = error instanceof Error ? error.message : `${modelType.toUpperCase()} prediction failed`;
        return handleUnexpectedError(c, error, message);
      }
    },
  );

  return modelsApi;
};
