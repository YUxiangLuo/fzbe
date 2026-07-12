import path from "path";
import fs from "fs";
import logger from "../../utils/logger";
import { PYTHON_RUNTIME_CONFIG, TIMEOUTS } from "../../utils/constants";
import { PY_DIR, resolvePythonExecutable } from "../../utils/python";

const TRAINING_TIMEOUT = TIMEOUTS.MODEL_TRAINING;
const GUIDED_ENSEMBLE_MODEL_TYPES = new Set(["weighted_avg", "boosting", "stacking"]);

export function buildPythonModelProcessEnv(): Record<string, string> {
  const modelThreads = String(PYTHON_RUNTIME_CONFIG.MODEL_PROCESS_THREADS);
  const tfInterOpThreads = String(PYTHON_RUNTIME_CONFIG.TF_INTER_OP_THREADS);

  return {
    OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? modelThreads,
    OPENBLAS_NUM_THREADS: process.env.OPENBLAS_NUM_THREADS ?? modelThreads,
    MKL_NUM_THREADS: process.env.MKL_NUM_THREADS ?? modelThreads,
    VECLIB_MAXIMUM_THREADS: process.env.VECLIB_MAXIMUM_THREADS ?? modelThreads,
    NUMEXPR_NUM_THREADS: process.env.NUMEXPR_NUM_THREADS ?? modelThreads,
    TF_NUM_INTRAOP_THREADS: process.env.TF_NUM_INTRAOP_THREADS ?? modelThreads,
    TF_NUM_INTEROP_THREADS: process.env.TF_NUM_INTEROP_THREADS ?? tfInterOpThreads,
    TF_CPP_MIN_LOG_LEVEL: process.env.TF_CPP_MIN_LOG_LEVEL ?? "3",
  };
}

/** TS 参数名 → Python CLI 参数名 的映射 */
export const allPossiblePythonParams: Record<string, string> = {
  // 通用参数（由 service 层手动设置）
  data: "data",
  dataFile: "data-file",
  trainStart: "train-start",
  trainEnd: "train-end",
  evalStart: "eval-start",
  evalEnd: "eval-end",
  saveModel: "save-model",
  loadModel: "load-model",
  forecastSteps: "forecast-steps",
  fitOnly: "fit-only",
  pretty: "pretty",
  baseModelFiles: "base-model-files",
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

export interface ModelProgressEvent {
  type: "phase" | "log" | "progress";
  message: string;
  phase?: string;
  percent?: number;
  source?: "backend" | "python";
  timestamp?: string;
}

export type ModelProgressReporter = (event: ModelProgressEvent) => void;

export const reportModelProgress = (
  reporter: ModelProgressReporter | undefined,
  event: ModelProgressEvent,
) => {
  reporter?.({
    ...event,
    timestamp: event.timestamp ?? new Date().toISOString(),
  });
};

const SUPPRESSED_PYTHON_PROGRESS_PREFIXES = [
  "Skipping non-numeric feature for subprocess model:",
];

const PYTHON_PROGRESS_LOG_LINE_PATTERN = /^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+-\s+(DEBUG|INFO|WARNING|ERROR)\s+-\s+(.*)$/;

const UNSAFE_PYTHON_PROGRESS_PATTERNS = [
  /^Traceback \(most recent call last\):/,
  /^File \".*\", line \d+/,
  /^\^+$/,
  /^(?:[A-Za-z_][\w.]*)(?:Error|Exception|Warning):/,
  /\b(?:args|vars\(args\))\s*[:=]/i,
  /\b--data\b/i,
  /\bJSONDecodeError\b/i,
  /\bsite-packages\b/i,
  /\bvenv\b/i,
  /\/(?:opt|root|home|usr|tmp)\//i,
  /[A-Za-z]:\\/,
];

const SAFE_PYTHON_PROGRESS_PATTERNS = [
  /^\[[^\]]+\]/,
  /^✓\s/,
  /^(?:数据划分|权重训练集|权重验证集|评估集|误差计算|权重归一化)[:：]/,
  /^(?:训练|重新训练|复用|生成|计算|评估|预测|扩展horizon后生成|稳定性回退)\b/,
  /^Round \d+ candidate /,
  /^Final RMSE:/,
  /^Model chain:/,
  /^Entering training mode\.$/,
  /^Starting (?:MA|ES|ARIMA|LSTM|model training|prediction|.*train_and_evaluate)/,
  /^Data loaded successfully\./,
  /^Validation passed:/,
  /^Auto-selected /,
  /^Using /,
  /^Running auto_arima /,
  /^Selected /,
  /^Final selected order:/,
  /^Model complexity:/,
  /^Model converged successfully$/,
  /^Model did not converge properly$/,
  /^Forecasting \d+ steps/,
  /^Fitting /,
  /^Generated \d+ /,
  /^Calculated residuals:/,
  /^Evaluation metrics computed:/,
  /^Training finished\./,
  /^Prediction finished successfully$/,
  /^Run finished successfully\.$/,
  /^Training dataset created:/,
  /^Creating autoregressive /,
  /^Training (?:direct|autoregressive) LSTM model /,
  /^Initial learning rate:/,
  /^Model training completed$/,
  /^Fitting preprocessor /,
  /^No feature_types provided/,
  /^Using provided feature_types/,
  /^Resolved feature_types:/,
  /^After d=/,
  /^(?:Very small|Small) effective sample size /,
  /^High parameter-to-sample ratio /,
  /^Empty evaluation range/,
];

const sanitizePythonProgressLine = (line: string): string | null => {
  const trimmed = line.trim();
  if (!trimmed) return null;

  const logMatch = trimmed.match(PYTHON_PROGRESS_LOG_LINE_PATTERN);
  const level = logMatch?.[1];
  const message = logMatch?.[2]?.trim() ?? trimmed;
  if (!message) return null;

  if (level === "DEBUG" || level === "ERROR") {
    return null;
  }

  if (SUPPRESSED_PYTHON_PROGRESS_PREFIXES.some((prefix) => message.startsWith(prefix))) {
    return null;
  }

  if (UNSAFE_PYTHON_PROGRESS_PATTERNS.some((pattern) => pattern.test(message))) {
    return null;
  }

  if (!SAFE_PYTHON_PROGRESS_PATTERNS.some((pattern) => pattern.test(message))) {
    return null;
  }

  return message.length > 320
    ? `${message.slice(0, 320)}...`
    : message;
};

const readStreamText = async (
  stream: ReadableStream<Uint8Array> | null | undefined,
  onLine?: (line: string) => void,
): Promise<string> => {
  if (!stream) return "";

  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let text = "";
  let lineBuffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    text += chunk;

    if (onLine) {
      lineBuffer += chunk;
      const lines = lineBuffer.split(/\r?\n/);
      lineBuffer = lines.pop() ?? "";
      for (const line of lines) onLine(line);
    }
  }

  const finalChunk = decoder.decode();
  if (finalChunk) {
    text += finalChunk;
    if (onLine) lineBuffer += finalChunk;
  }

  if (onLine && lineBuffer.trim()) {
    onLine(lineBuffer);
  }

  return text;
};

export async function runPythonModel(
  modelType: string,
  params: Record<string, unknown>,
  timeoutMs?: number,
  progressReporter?: ModelProgressReporter,
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
    if (Array.isArray(value)) return [cliArg, JSON.stringify(value)];
    return [cliArg, String(value)];
  });

  const fullArgs = [modelType, ...args];
  const command = process.platform !== "win32" && SETSID_PATH
    ? [SETSID_PATH, pythonExec, scriptPath, ...fullArgs]
    : [pythonExec, scriptPath, ...fullArgs];

  reportModelProgress(progressReporter, {
    type: "phase",
    phase: "python_start",
    percent: 24,
    source: "backend",
    message: `已启动 ${modelType.toUpperCase()} Python 训练进程`,
  });

  const proc = Bun.spawn(command, {
    cwd: PY_DIR,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      ...buildPythonModelProcessEnv(),
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
      logger.warn({ err: error, pid: proc.pid, modelType }, "终止 Python 主进程失败");
    }
  };

  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  const executionPromise = new Promise<unknown>(async (resolve, reject) => {
    try {
      let pythonLogProgress = 26;
      const [stdout, stderr] = await Promise.all([
        readStreamText(proc.stdout),
        readStreamText(proc.stderr, (line) => {
          const message = sanitizePythonProgressLine(line);
          if (!message) return;

          pythonLogProgress = Math.min(86, pythonLogProgress + 2);
          reportModelProgress(progressReporter, {
            type: "log",
            phase: "python_training",
            percent: pythonLogProgress,
            source: "python",
            message,
          });
        }),
      ]);

      await proc.exited;

      reportModelProgress(progressReporter, {
        type: "phase",
        phase: "python_finished",
        percent: 88,
        source: "backend",
        message: `${modelType.toUpperCase()} Python 训练进程已结束，正在解析结果`,
      });

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
          reject(new Error("Python 执行失败"));
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

  // 超时后 executionPromise 仍会 reject（进程被杀后 exitCode !== 0），
  // 若不捕获会触发 unhandledRejection。用 .catch 吞掉滞后的 rejection。
  executionPromise.catch((err) => {
    logger.debug({ err, modelType }, "超时后 executionPromise 滞后 rejection（已忽略）");
  });

  return Promise.race([executionPromise, timeoutPromise]);
}

export interface GuidedPythonStepResult {
  step_id: string;
  output?: unknown;
  final_result?: Record<string, unknown>;
}

export async function runPythonGuidedTrainingStep(
  modelType: string,
  stepId: string,
  params: Record<string, unknown>,
  stateFilePath: string,
  artifactDir: string,
  timeoutMs = TRAINING_TIMEOUT,
): Promise<GuidedPythonStepResult> {
  const pythonExec = resolvePythonExecutable();
  const scriptPath = path.join(PY_DIR, GUIDED_ENSEMBLE_MODEL_TYPES.has(modelType) ? "guided_ensemble_model.py" : "guided_model.py");
  if (!fs.existsSync(scriptPath)) {
    throw new Error(`未找到 Python guided 训练入口脚本: ${scriptPath}`);
  }

  fs.mkdirSync(artifactDir, { recursive: true });
  const args = Object.entries(params).flatMap(([key, value]) => {
    const scriptKey = allPossiblePythonParams[key] || key;
    const cliArg = `--${scriptKey}`;
    if (typeof value === "boolean") return value ? [cliArg] : [];
    if (Array.isArray(value)) return [cliArg, JSON.stringify(value)];
    return [cliArg, String(value)];
  });

  const fullArgs = [
    modelType,
    stepId,
    "--state-file",
    stateFilePath,
    "--artifact-dir",
    artifactDir,
    ...args,
  ];
  const command = process.platform !== "win32" && SETSID_PATH
    ? [SETSID_PATH, pythonExec, scriptPath, ...fullArgs]
    : [pythonExec, scriptPath, ...fullArgs];

  const proc = Bun.spawn(command, {
    cwd: PY_DIR,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      ...buildPythonModelProcessEnv(),
      PATH: `${path.dirname(pythonExec)}${path.delimiter}${process.env.PATH ?? ""}`,
    },
  });

  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  const killPythonProcessTree = () => {
    if (proc.exitCode !== null || proc.killed) return;

    if (process.platform !== "win32" && SETSID_PATH) {
      try {
        process.kill(-proc.pid, "SIGKILL");
        return;
      } catch (error) {
        logger.warn({ error, pid: proc.pid, modelType, stepId }, "终止 guided Python 进程组失败");
      }
    }

    try {
      proc.kill("SIGKILL");
    } catch (error) {
      logger.warn({ err: error, pid: proc.pid, modelType, stepId }, "终止 guided Python 主进程失败");
    }
  };

  const executionPromise = new Promise<GuidedPythonStepResult>(async (resolve, reject) => {
    try {
      const [stdout, stderr] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
      ]);
      await proc.exited;

      if (proc.exitCode !== 0) {
        logger.error(
          { stderr: stderr.slice(0, 2000), modelType, stepId },
          "guided Python 阶段执行失败",
        );
        const taggedError = extractPythonTaggedError(stderr);
        reject(taggedError ?? new Error("Python 执行失败"));
        return;
      }

      try {
        resolve(JSON.parse(stdout) as GuidedPythonStepResult);
      } catch {
        logger.error(
          { stdout: stdout.slice(0, 2000), modelType, stepId },
          "解析 guided Python 输出失败",
        );
        reject(new Error("Python 输出无效"));
      }
    } catch (err) {
      reject(err);
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
    }
  });

  const timeoutPromise = new Promise<GuidedPythonStepResult>((_, reject) => {
    timeoutId = setTimeout(() => {
      killPythonProcessTree();
      reject(new Error(`Python 执行超时 (${timeoutMs}ms)`));
    }, timeoutMs);
  });

  executionPromise.catch((err) => {
    logger.debug({ err, modelType, stepId }, "guided Python 超时后滞后 rejection（已忽略）");
  });

  return Promise.race([executionPromise, timeoutPromise]);
}
