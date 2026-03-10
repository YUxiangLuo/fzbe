import path from "path";
import fs from "fs";
import type { PoolConnection, RowDataPacket } from "mysql2/promise";
import csv from "csv-parser";
import { pool } from "../../config";
import logger from "../../utils/logger";
import type { ModelType, TrainInput } from "./models.schema";
import { TIMEOUTS, CONCURRENCY_LIMITS } from "../../utils/constants";
import { PY_DIR, resolvePythonExecutable } from "../../utils/python";
import {
  buildExperimentScopedModelPath,
  buildModelExecutionLockName,
  buildModelSlotLockName,
} from "./model-runtime";

const TRAINING_TIMEOUT = TIMEOUTS.MODEL_TRAINING;
const MAX_CONCURRENT_MODEL_JOBS = CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS;


interface AdvisoryLockRow extends RowDataPacket {
  acquired: 0 | 1 | null;
}

interface ExperimentOwnershipRow extends RowDataPacket {
  student_id: number;
  status: string;
}

interface ExperimentModelContextRow extends ExperimentOwnershipRow {
  selected_industry: string | null;
  selected_company: string | null;
  selected_product: string | null;
  data_window_train_start_index: number | null;
  data_window_train_end_index: number | null;
  data_window_evaluate_start_index: number | null;
  data_window_evaluate_end_index: number | null;
}

interface ExperimentTrainingConfig {
  selected_industry: string;
  selected_company: string;
  selected_product: string;
  data_window_train_start_index: number;
  data_window_train_end_index: number;
  data_window_evaluate_start_index: number;
  data_window_evaluate_end_index: number;
}

export interface AdvisoryLockHandle {
  connection: PoolConnection;
  name: string;
}

async function acquireAdvisoryLock(name: string): Promise<AdvisoryLockHandle | null> {
  const connection = await pool.getConnection();

  try {
    const [rows] = await connection.query<AdvisoryLockRow[]>(
      "SELECT GET_LOCK(?, 0) AS acquired",
      [name],
    );

    if (rows[0]?.acquired === 1) {
      return { connection, name };
    }

    try { connection.release(); } catch { /* connection may already be destroyed */ }
    return null;
  } catch (error) {
    try { connection.release(); } catch { /* connection may already be destroyed */ }
    throw error;
  }
}

async function releaseAdvisoryLock(handle: AdvisoryLockHandle | null | undefined) {
  if (!handle) {
    return;
  }

  try {
    await handle.connection.query("SELECT RELEASE_LOCK(?)", [handle.name]);
  } catch (error) {
    logger.warn({ error, lock: handle.name }, "释放数据库 advisory lock 失败");
  } finally {
    try {
      handle.connection.release();
    } catch {
      // Connection may already be released or destroyed
    }
  }
}

export const acquireModelSlot = async (): Promise<AdvisoryLockHandle | null> => {
  const connection = await pool.getConnection();

  try {
    for (let slot = 1; slot <= MAX_CONCURRENT_MODEL_JOBS; slot += 1) {
      const name = buildModelSlotLockName(slot);
      const [rows] = await connection.query<AdvisoryLockRow[]>(
        "SELECT GET_LOCK(?, 0) AS acquired",
        [name],
      );

      if (rows[0]?.acquired === 1) {
        return { connection, name };
      }
    }

    try { connection.release(); } catch { /* connection may already be destroyed */ }
    return null;
  } catch (error) {
    try { connection.release(); } catch { /* connection may already be destroyed */ }
    throw error;
  }
};

export const releaseModelSlot = async (handle: AdvisoryLockHandle | null | undefined) => {
  await releaseAdvisoryLock(handle);
};

export const acquireModelExecutionLock = async (
  experimentId: number,
  modelType: ModelType,
): Promise<AdvisoryLockHandle | null> => {
  return await acquireAdvisoryLock(buildModelExecutionLockName(experimentId, modelType));
};

export const releaseModelExecutionLock = async (handle: AdvisoryLockHandle | null | undefined) => {
  await releaseAdvisoryLock(handle);
};

interface SegmentRow extends RowDataPacket {
  file_path: string;
}

export class ModelValidationError extends Error {
  constructor(message: string, public details?: Record<string, unknown>) {
    super(message);
  }
}

export class ModelNotFoundError extends Error {}
export class ModelAccessError extends Error {
  constructor(message: string, public status: 400 | 403 | 404) {
    super(message);
  }
}


/** TS 参数名 → Python CLI 参数名 的映射 */
const allPossiblePythonParams: Record<string, string> = {
  // 通用参数（由 service 层手动设置）
  data: "data",
  trainStart: "train-start",
  trainEnd: "train-end",
  evalStart: "eval-start",
  evalEnd: "eval-end",
  saveModel: "save-model",
  loadModel: "load-model",
  forecastSteps: "forecast-steps",
  pretty: "pretty",
  // 模型参数（从前端 body 透传）
  moving_average_window: "window",
  exponential_smoothing_alpha: "alpha",
  arimaD: "d",
  lstmFeatures: "lstm-features",
  lstmTargetFeature: "lstm-target-feature",
  lstmNormalization: "normalization",
  models: "models",
};

const SETSID_PATH = process.platform === "win32" ? undefined : Bun.which("setsid");

const extractPythonTaggedError = (stderr: string): Error | null => {
  const markers = [
    { marker: "[VALIDATION]", prefix: "[PYTHON_VALIDATION]" },
    { marker: "[DATA]", prefix: "[PYTHON_DATA]" },
  ] as const;

  const lines = stderr
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index]!;
    for (const { marker, prefix } of markers) {
      const markerIndex = line.indexOf(marker);
      if (markerIndex === -1) {
        continue;
      }

      const detail = line.slice(markerIndex + marker.length).trim();
      return new Error(`${prefix} ${detail}`.trim());
    }
  }

  return null;
};

export async function loadDatasetFromDb(
  industry: string,
  company: string,
  product: string,
): Promise<Record<string, unknown>[]> {
  const [segmentRows] = await pool.query<SegmentRow[]>(
    "SELECT file_path FROM dataset_segments WHERE industry_name = ? AND company_name = ? AND product_name = ? LIMIT 1",
    [industry, company, product],
  );

  if (segmentRows.length === 0) {
    throw new ModelValidationError("未找到数据集分段文件");
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

async function assertExperimentModelAccess(experimentId: number, studentId: number) {
  const [rows] = await pool.query<ExperimentModelContextRow[]>(
    `SELECT student_id,
            status,
            selected_industry,
            selected_company,
            selected_product,
            data_window_train_start_index,
            data_window_train_end_index,
            data_window_evaluate_start_index,
            data_window_evaluate_end_index
       FROM experiment_status
      WHERE experiment_id = ?
      LIMIT 1`,
    [experimentId],
  );

  const experiment = rows[0];
  if (!experiment) {
    throw new ModelAccessError("实验不存在", 404);
  }

  if (experiment.student_id !== studentId) {
    throw new ModelAccessError("权限不足：该实验不属于当前用户", 403);
  }

  if (experiment.status === "Completed") {
    throw new ModelAccessError("已完成的实验不允许继续训练或预测模型", 400);
  }

  return experiment;
}

const hasDefinedValue = <T>(value: T | null | undefined): value is T => value !== null && value !== undefined;

const assertTrainingRequestMatchesExperiment = (
  experiment: ExperimentModelContextRow,
  body: TrainInput,
) => {
  const comparisons = [
    ["selected_industry", "行业", body.selected_industry, experiment.selected_industry],
    ["selected_company", "企业", body.selected_company, experiment.selected_company],
    ["selected_product", "产品", body.selected_product, experiment.selected_product],
    [
      "data_window_train_start_index",
      "训练起始索引",
      body.data_window_train_start_index,
      experiment.data_window_train_start_index,
    ],
    [
      "data_window_train_end_index",
      "训练结束索引",
      body.data_window_train_end_index,
      experiment.data_window_train_end_index,
    ],
    [
      "data_window_evaluate_start_index",
      "评估起始索引",
      body.data_window_evaluate_start_index,
      experiment.data_window_evaluate_start_index,
    ],
    [
      "data_window_evaluate_end_index",
      "评估结束索引",
      body.data_window_evaluate_end_index,
      experiment.data_window_evaluate_end_index,
    ],
  ] as const;

  for (const [field, label, requestValue, persistedValue] of comparisons) {
    if (requestValue !== undefined && requestValue !== persistedValue) {
      throw new ModelValidationError(`请求中的${label}与实验记录不一致，请刷新页面后重试`, {
        field,
        request_value: requestValue,
        persisted_value: persistedValue,
      });
    }
  }
};

const getExperimentTrainingConfig = (
  experiment: ExperimentModelContextRow,
): ExperimentTrainingConfig => {
  if (
    !hasDefinedValue(experiment.selected_industry)
    || !hasDefinedValue(experiment.selected_company)
    || !hasDefinedValue(experiment.selected_product)
  ) {
    throw new ModelValidationError("实验尚未完成数据集选择，无法训练模型");
  }

  if (
    !hasDefinedValue(experiment.data_window_train_start_index)
    || !hasDefinedValue(experiment.data_window_train_end_index)
    || !hasDefinedValue(experiment.data_window_evaluate_start_index)
    || !hasDefinedValue(experiment.data_window_evaluate_end_index)
  ) {
    throw new ModelValidationError("实验尚未完成数据窗口配置，无法训练模型");
  }

  if (experiment.data_window_train_start_index > experiment.data_window_train_end_index) {
    throw new ModelValidationError("实验记录中的训练数据窗口无效，请重新选择数据窗口后重试");
  }

  if (experiment.data_window_evaluate_start_index > experiment.data_window_evaluate_end_index) {
    throw new ModelValidationError("实验记录中的评估数据窗口无效，请重新选择数据窗口后重试");
  }

  return {
    selected_industry: experiment.selected_industry,
    selected_company: experiment.selected_company,
    selected_product: experiment.selected_product,
    data_window_train_start_index: experiment.data_window_train_start_index,
    data_window_train_end_index: experiment.data_window_train_end_index,
    data_window_evaluate_start_index: experiment.data_window_evaluate_start_index,
    data_window_evaluate_end_index: experiment.data_window_evaluate_end_index,
  };
}

export async function runPythonModel(
  modelType: string,
  params: Record<string, unknown>,
  timeoutMs?: number,
): Promise<unknown> {
  const pythonExec = resolvePythonExecutable();

  const scriptPath = path.join(PY_DIR, "run_model.py");
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
  const command = process.platform !== "win32" && SETSID_PATH
    ? [SETSID_PATH, pythonExec, scriptPath, ...fullArgs]
    : [pythonExec, scriptPath, ...fullArgs];

  const proc = Bun.spawn(command, {
    cwd: PY_DIR,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      PATH: `${path.dirname(pythonExec)}${path.delimiter}${process.env.PATH ?? ""}`,
    },
  });

  const killPythonProcessTree = () => {
    if (proc.exitCode !== null || proc.killed) return;

    if (process.platform !== "win32" && SETSID_PATH) {
      try {
        process.kill(-proc.pid, "SIGKILL");
        return;
      } catch (error) {
        logger.warn(
          { error, pid: proc.pid, modelType },
          "终止 Python 进程组失败，回退为终止主进程",
        );
      }
    }

    try {
      proc.kill("SIGKILL");
    } catch (error) {
      logger.warn({ error, pid: proc.pid, modelType }, "终止 Python 主进程失败");
    }
  };

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
        const taggedError = extractPythonTaggedError(stderr);
        if (taggedError) {
          reject(taggedError);
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
      killPythonProcessTree();
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
  const experiment = await assertExperimentModelAccess(body.experiment_id, studentId);
  assertTrainingRequestMatchesExperiment(experiment, body);
  const trainingConfig = getExperimentTrainingConfig(experiment);

  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
  );

  if (
    trainingConfig.data_window_train_end_index >= allData.length
    || trainingConfig.data_window_evaluate_end_index >= allData.length
  ) {
    throw new ModelValidationError("数据窗口索引超出范围", { dataset_size: allData.length });
  }

  const modelSaveLoc = buildExperimentScopedModelPath(body.experiment_id, studentId, modelType);

  const pythonParams: Record<string, unknown> = {
    data: JSON.stringify(allData),
    trainStart: trainingConfig.data_window_train_start_index,
    trainEnd: trainingConfig.data_window_train_end_index,
    evalStart: trainingConfig.data_window_evaluate_start_index,
    evalEnd: trainingConfig.data_window_evaluate_end_index,
    saveModel: modelSaveLoc,
    pretty: true,
  };

  for (const bodyKey of Object.keys(allPossiblePythonParams)) {
    const value = body[bodyKey as keyof TrainInput];
    if (value !== undefined) {
      pythonParams[bodyKey] = value;
    }
  }

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

export async function predictExperimentModel(
  modelType: ModelType,
  studentId: number,
  experimentId: number,
  forecastSteps: number,
) {
  await assertExperimentModelAccess(experimentId, studentId);

  const modelLoadPath = buildExperimentScopedModelPath(experimentId, studentId, modelType);

  if (!fs.existsSync(modelLoadPath)) {
    throw new ModelNotFoundError(`未找到该实验的已训练 ${modelType.toUpperCase()} 模型`);
  }

  const pythonParams = {
    loadModel: modelLoadPath,
    forecastSteps,
    pretty: true,
  };

  const result = (await runPythonModel(modelType, pythonParams, TIMEOUTS.MODEL_PREDICTION)) as Record<string, unknown>;

  return {
    status: "success",
    results: { predictions: result.predictions || [] },
  };
}
