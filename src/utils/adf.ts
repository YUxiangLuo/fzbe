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
  statistic: number | null;
  p_value: number | null;
  stationary: boolean | null;
  critical_values: {
    "1%": number | null;
    "5%": number | null;
    "10%": number | null;
  } | null;
  error?: string;
}

export type AdfResult = AdfResultItem[];

export class AdfValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AdfValidationError";
  }
}

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

      const parsed = JSON.parse(stdout) as AdfResult;
      const validResults = parsed.filter((item) => !item.error);
      if (validResults.length === 0) {
        throw new AdfValidationError("数据量不足，无法进行 ADF 检测");
      }
      return validResults;
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
