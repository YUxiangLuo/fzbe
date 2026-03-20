import { PY_DIR, resolvePythonExecutable } from "./python";
import { TIMEOUTS } from "./constants";

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

/**
 * 调用 Python ADF 工具，执行固定 d=0,1,2 三阶差分检验。
 *
 * @param data 由 {month, sales} 对象组成的数组（推荐长度 ≥ 10）
 * @returns Promise<AdfResult> ADF 检验结果
 */
export async function runAdf(data: AdfInputPoint[]): Promise<AdfResult> {
  const jsonArg = JSON.stringify(data);
  const pythonExec = resolvePythonExecutable();

  const proc = Bun.spawn([pythonExec, "-m", "src.adf", jsonArg, "--pretty"], {
    cwd: PY_DIR,
    stdout: "pipe",
    stderr: "pipe",
  });

  const timeoutMs = TIMEOUTS.ADF;
  const timer = setTimeout(() => {
    try { proc.kill("SIGKILL"); } catch {}
  }, timeoutMs);

  try {
    const [stdout, stderr] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);

    const exitCode = await proc.exited;

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
    throw new Error(`Failed to parse ADF result`);
  } finally {
    clearTimeout(timer);
  }
}
