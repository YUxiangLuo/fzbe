import path from "path";
import fs from "fs";
import type { RowDataPacket } from "mysql2";
import csv from "csv-parser";
import { pool } from "../../config";
import logger from "../../logger";
import type { ModelType, TrainInput } from "./models.schema";
import { TIMEOUTS, CONCURRENCY_LIMITS } from "../../utils/constants";

const TRAINING_TIMEOUT = TIMEOUTS.MODEL_TRAINING;
const MAX_CONCURRENT_MODEL_JOBS = CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS;
let activeModelJobs = 0;

export const tryAcquireModelSlot = (): boolean => {
  if (activeModelJobs >= MAX_CONCURRENT_MODEL_JOBS) return false;
  activeModelJobs += 1;
  return true;
};

export const releaseModelSlot = () => {
  if (activeModelJobs > 0) activeModelJobs -= 1;
};

interface SegmentRow extends RowDataPacket {
  file_path: string;
}

const allPossiblePythonParams: Record<string, string> = {
  data: "data",
  trainStart: "train-start",
  trainEnd: "train-end",
  evalStart: "eval-start",
  evalEnd: "eval-end",
  saveModel: "save-model",
  loadModel: "load-model",
  forecastSteps: "forecast-steps",
  pretty: "pretty",
  moving_average_window: "window",
  exponential_smoothing_alpha: "alpha",
  arimaD: "d",
  arimaMaxP: "max-p",
  arimaMaxQ: "max-q",
  lstmLookBack: "look-back",
  lstmEpochs: "epochs",
  lstmFeatures: "lstm-features",
  lstmTargetFeature: "lstm-target-feature",
  lstmFeatureTypes: "feature-types",
  lstmNormalization: "normalization",
  models: "models",
};

export async function loadDatasetFromDb(
  industry: string,
  company: string,
  product: string,
): Promise<Record<string, unknown>[]> {
  const [segmentRows] = await pool.query<SegmentRow[]>(
    "SELECT file_path FROM dataset_segments WHERE industry_name = ? AND company_name = ? AND product_name = ?",
    [industry, company, product],
  );

  if (segmentRows.length === 0) {
    throw new Error("未找到数据集分段文件");
  }

  const allData: Record<string, unknown>[] = [];
  await new Promise<void>((resolve, reject) => {
    fs.createReadStream(segmentRows[0]!.file_path)
      .pipe(csv({ mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim() }))
      .on("data", (row: Record<string, unknown>) => allData.push(row))
      .on("end", resolve)
      .on("error", reject);
  });

  return allData;
}

export async function runPythonModel(
  modelType: string,
  params: Record<string, unknown>,
  timeoutMs?: number,
): Promise<unknown> {
  const pyDir = path.resolve(import.meta.dir, "../../../py");
  const pythonExec = process.platform === "win32"
    ? path.join(pyDir, "venv", "Scripts", "python.exe")
    : path.join(pyDir, "venv", "bin", "python");

  const scriptPath = path.join(pyDir, "run_model.py");
  if (!fs.existsSync(scriptPath)) {
    throw new Error(`未找到 Python 入口脚本: ${scriptPath}`);
  }

  const args = Object.entries(params).flatMap(([key, value]) => {
    const scriptKey = allPossiblePythonParams[key] || key;
    const cliArg = `--${scriptKey}`;
    if (typeof value === "boolean") return value ? [cliArg] : [];
    return [cliArg, String(value)];
  });

  const fullArgs = [modelType, ...args];

  const proc = Bun.spawn([pythonExec, scriptPath, ...fullArgs], {
    cwd: pyDir,
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, PATH: `${pyDir}/venv/bin:${process.env.PATH}` },
  });

  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  const executionPromise = new Promise<unknown>(async (resolve, reject) => {
    try {
      const [stdout, stderr] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
      ]);

      await proc.exited;

      if (proc.exitCode !== 0) {
        // 截断日志，避免泄露完整数据
        const truncatedArgs = fullArgs.map((arg) =>
          typeof arg === "string" && arg.length > 200 ? `${arg.slice(0, 200)}...[truncated]` : arg,
        );
        logger.error({ stderr: stderr.slice(0, 2000), modelType, args: truncatedArgs }, "Python 执行失败");
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
        } catch {
          const truncatedArgs = fullArgs.map((arg) =>
            typeof arg === "string" && arg.length > 200 ? `${arg.slice(0, 200)}...[truncated]` : arg,
          );
          logger.error({ stdout: stdout.slice(0, 2000), modelType, args: truncatedArgs }, "解析 Python 输出失败");
          reject(new Error("Python 输出无效"));
        }
      }
    } catch (err) {
      reject(err);
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
    }
  });

  if (!timeoutMs) return executionPromise;

  const timeoutPromise = new Promise<unknown>((_, reject) => {
    timeoutId = setTimeout(() => {
      proc.kill();
      reject(new Error(`Python 执行超时 (${timeoutMs}ms)`));
    }, timeoutMs);
  });

  return Promise.race([executionPromise, timeoutPromise]);
}

const resultTransformers: Record<ModelType, (result: Record<string, unknown>) => Record<string, unknown>> = {
  arima: (result) => ({
    best_order: result.best_order,
    search_summary: result.search_summary,
    diagnostics: result.diagnostics
      ? { ...(result.diagnostics as object), y_pred_eval: (result.predictions as { prediction: number }[] | undefined)?.map((p) => p.prediction) || [] }
      : undefined,
  }),
  es: () => ({}),
  ma: () => ({}),
  lstm: (result) => ({
    feature_types: result.feature_types,
    diagnostics: result.diagnostics
      ? { ...(result.diagnostics as object), y_pred_eval: (result.predictions as { prediction: number }[] | undefined)?.map((p) => p.prediction) || [] }
      : undefined,
  }),
  stacking: (result) => ({ model_names: result.models }),
  weighted_avg: (result) => ({
    model_names: result.models,
    weights: (result.models as string[] | undefined)?.map((name) => (result.weights as Record<string, number>)?.[name]),
  }),
  boosting: (result) => ({
    model_names: result.model_chain,
    search_summary: result.search_summary,
  }),
};

export async function trainModel(modelType: ModelType, studentId: number, body: TrainInput) {
  const allData = await loadDatasetFromDb(body.selected_industry, body.selected_company, body.selected_product);

  if (body.data_window_train_end_index >= allData.length || body.data_window_evaluate_end_index >= allData.length) {
    return { error: "数据窗口索引超出范围", dataset_size: allData.length };
  }

  const modelSaveLoc = path.resolve(import.meta.dir, `../../../py/trained_models/${studentId}_${modelType}.pkl`);

  const pythonParams: Record<string, unknown> = {
    data: JSON.stringify(allData),
    trainStart: body.data_window_train_start_index,
    trainEnd: body.data_window_train_end_index,
    evalStart: body.data_window_evaluate_start_index,
    evalEnd: body.data_window_evaluate_end_index,
    saveModel: modelSaveLoc,
    pretty: true,
  };

  for (const [bodyKey, scriptKey] of Object.entries(allPossiblePythonParams)) {
    const value = body[bodyKey as keyof TrainInput];
    if (value !== undefined) {
      pythonParams[scriptKey] = value;
    }
  }

  const isEnsembleModel = ["stacking", "weighted_avg", "boosting"].includes(modelType);
  const timeout = TRAINING_TIMEOUT;
  const result = (await runPythonModel(modelType, pythonParams, timeout)) as Record<string, unknown>;

  const baseResults = {
    metrics: result.metrics,
    eval_y_true: result.eval_y_true,
    eval_predictions: (result.predictions as { prediction: number }[] | undefined)?.map((p) => p.prediction) || [],
    notes: result.notes || [],
    saved_model: result.saved_model,
    logs: result.logs,
  };

  const modelSpecificTransformedResult = resultTransformers[modelType]?.(result) || {};
  const finalTransformedResult = { ...baseResults, ...modelSpecificTransformedResult };

  const finalResponse: { status: string; results: unknown; inferred_feature_types?: unknown } = {
    status: "success",
    results: finalTransformedResult,
  };

  if (modelType === "lstm" && result.feature_types) {
    finalResponse.inferred_feature_types = result.feature_types;
  }

  return finalResponse;
}

export async function predictModel(modelType: ModelType, studentId: number, forecastSteps: number) {
  const modelLoadPath = path.resolve(import.meta.dir, `../../../py/trained_models/${studentId}_${modelType}.pkl`);

  if (!fs.existsSync(modelLoadPath)) {
    return { error: `未找到该用户的已训练 ${modelType.toUpperCase()} 模型` };
  }

  const pythonParams = {
    loadModel: modelLoadPath,
    forecastSteps,
    pretty: true,
  };

  const result = (await runPythonModel(modelType, pythonParams)) as Record<string, unknown>;

  return {
    status: "success",
    results: { predictions: result.predictions || [] },
  };
}
