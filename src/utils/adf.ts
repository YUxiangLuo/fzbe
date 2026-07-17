import { PY_DIR, resolvePythonExecutable } from "./python";
import { TIMEOUTS } from "./constants";
import path from "node:path";
import fs from "node:fs";
import crypto from "node:crypto";
import os from "node:os";
import { terminateProcessTree } from "./process-tree";
import { ModelTimeoutError } from "../modules/models/model-errors";

export interface AdfInputPoint {
  month: string;
  sales: number;
}

export interface AdfResultItem {
  diff_order: number;
  statistic: number;
  p_value: number;
  used_lags: number;
  n_obs: number;
  /** Interpretation of the ADF unit-root null; `stationary` is retained for compatibility. */
  unit_root_conclusion: "reject_unit_root" | "fail_to_reject_unit_root";
  stationary: boolean;
  critical_values: {
    "1%": number;
    "5%": number;
    "10%": number;
  };
}

export type AdfResult = AdfResultItem[];

export class AdfValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AdfValidationError";
  }
}

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

const isNonNegativeSafeInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value >= 0;

const isPositiveSafeInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value >= 1;

const isUnitRootConclusion = (
  value: unknown,
): value is AdfResultItem["unit_root_conclusion"] =>
  value === "reject_unit_root" || value === "fail_to_reject_unit_root";

export const parseAdfResultPayload = (value: unknown): AdfResult => {
  if (!Array.isArray(value)) {
    throw new Error("[PYTHON_DATA] ADF result must be an array");
  }
  if (value.length > 3) {
    throw new Error("[PYTHON_DATA] ADF result cannot contain more than three rows");
  }

  const validResults: AdfResult = [];
  const seenDiffOrders = new Set<number>();
  value.forEach((candidate, index) => {
    if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
      throw new Error(`[PYTHON_DATA] ADF result ${index} must be an object`);
    }
    const item = candidate as Record<string, unknown>;
    if (typeof item.error === "string" && item.error.length > 0) {
      return;
    }
    const criticalValues = item.critical_values;
    if (!criticalValues || typeof criticalValues !== "object" || Array.isArray(criticalValues)) {
      throw new Error(`[PYTHON_DATA] ADF result ${index} has invalid critical values`);
    }
    const critical = criticalValues as Record<string, unknown>;
    if (
      !isNonNegativeSafeInteger(item.diff_order)
      || item.diff_order > 2
      || !isFiniteNumber(item.statistic)
      || !isFiniteNumber(item.p_value)
      || item.p_value < 0
      || item.p_value > 1
      || !isNonNegativeSafeInteger(item.used_lags)
      || !isPositiveSafeInteger(item.n_obs)
      || typeof item.stationary !== "boolean"
      || !isUnitRootConclusion(item.unit_root_conclusion)
      || item.unit_root_conclusion
        !== (item.stationary ? "reject_unit_root" : "fail_to_reject_unit_root")
      || !isFiniteNumber(critical["1%"])
      || !isFiniteNumber(critical["5%"])
      || !isFiniteNumber(critical["10%"])
    ) {
      throw new Error(`[PYTHON_DATA] ADF result ${index} has invalid fields`);
    }
    if (seenDiffOrders.has(item.diff_order)) {
      throw new Error(`[PYTHON_DATA] ADF result ${index} repeats a differencing order`);
    }
    seenDiffOrders.add(item.diff_order);
    validResults.push({
      diff_order: item.diff_order,
      statistic: item.statistic,
      p_value: item.p_value,
      used_lags: item.used_lags,
      n_obs: item.n_obs,
      stationary: item.stationary,
      unit_root_conclusion: item.unit_root_conclusion,
      critical_values: {
        "1%": critical["1%"],
        "5%": critical["5%"],
        "10%": critical["10%"],
      },
    });
  });

  if (validResults.length === 0) {
    throw new AdfValidationError("数据量不足，无法进行 ADF 检测");
  }
  return validResults;
};

const getTerminationError = (signal: AbortSignal) => signal.reason instanceof Error
  ? signal.reason
  : new Error("ADF 任务已终止");
const SETSID_PATH = process.platform === "win32" ? undefined : Bun.which("setsid");

/**
 * 调用 Python ADF 工具，执行固定 d=0,1,2 三阶差分检验。
 *
 * @param data 由 {month, sales} 对象组成的数组（推荐长度 ≥ 10）
 * @returns Promise<AdfResult> ADF 检验结果
 */
export async function runAdf(
  data: AdfInputPoint[],
  terminationSignal?: AbortSignal,
): Promise<AdfResult> {
  if (terminationSignal?.aborted) throw getTerminationError(terminationSignal);
  const inputPath = path.join(os.tmpdir(), `fz-adf-${crypto.randomUUID()}.json`);
  try {
    await Bun.write(inputPath, JSON.stringify(data));
    if (terminationSignal?.aborted) throw getTerminationError(terminationSignal);
    const pythonExec = resolvePythonExecutable();
    const command = SETSID_PATH
      ? [SETSID_PATH, pythonExec, "-m", "src.adf", inputPath, "--pretty"]
      : [pythonExec, "-m", "src.adf", inputPath, "--pretty"];
    const proc = Bun.spawn(command, {
      cwd: PY_DIR,
      stdout: "pipe",
      stderr: "pipe",
    });

    let forcedTerminationError: Error | null = null;
    let terminationPromise: Promise<void> | null = null;
    const terminate = (error: Error) => {
      forcedTerminationError ??= error;
      if (!terminationPromise) {
        terminationPromise = terminateProcessTree(proc, {
          graceMs: TIMEOUTS.MODEL_PROCESS_TERMINATION_GRACE,
          processGroup: Boolean(SETSID_PATH),
          label: "ADF Python",
        });
      }
      return terminationPromise;
    };
    const abortHandler = terminationSignal
      ? () => terminate(getTerminationError(terminationSignal))
      : undefined;
    if (terminationSignal && abortHandler) {
      terminationSignal.addEventListener("abort", abortHandler, { once: true });
      if (terminationSignal.aborted) abortHandler();
    }
    const timer = setTimeout(() => {
      void terminate(new ModelTimeoutError(
        `ADF 执行超时 (${TIMEOUTS.ADF}ms)`,
        TIMEOUTS.ADF,
        "python_subprocess",
      ));
    }, TIMEOUTS.ADF);
    timer.unref?.();

    try {
      let stdout = "";
      let stderr = "";
      let streamError: unknown;
      try {
        [stdout, stderr] = await Promise.all([
          new Response(proc.stdout).text(),
          new Response(proc.stderr).text(),
        ]);
      } catch (error) {
        streamError = error;
      }
      // Killing a subprocess can either close or error its pipe streams. Always
      // wait for the process itself before releasing the model-operation lease.
      const exitCode = await proc.exited;
      if (terminationPromise) await terminationPromise;
      if (forcedTerminationError) throw forcedTerminationError;
      if (streamError !== undefined) throw streamError;
      if (exitCode !== 0) {
        const message = stderr.trim() || `Process exited with code ${exitCode}`;
        throw new Error(`ADF 执行失败: ${message}`);
      }

      return parseAdfResultPayload(JSON.parse(stdout) as unknown);
    } catch (error) {
      if (error instanceof Error) throw error;
      throw new Error("Failed to parse ADF result");
    } finally {
      clearTimeout(timer);
      if (terminationSignal && abortHandler) {
        terminationSignal.removeEventListener("abort", abortHandler);
      }
    }
  } finally {
    try { fs.unlinkSync(inputPath); } catch {}
  }
}
