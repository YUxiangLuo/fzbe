import path from "path";
import { existsSync } from "node:fs";

// 输入：时间序列的单个元素
export interface AdfInputPoint {
  /** 月份（或时间标签），字符串格式如 "2021-02" */
  month: string;
  /** 销售额或任意数值型序列 */
  sales: number;
}

// 输出：单个差分阶（d=0,1,2）对应的结果
export interface AdfResultItem {
  /** 差分阶数 */
  diff_order: number;
  /** ADF 统计量 */
  statistic: number | null;
  /** p 值 */
  p_value: number | null;
  /** 是否平稳 */
  stationary: boolean | null;
  /** 临界值 */
  critical_values: {
    "1%": number | null;
    "5%": number | null;
    "10%": number | null;
  } | null;
  /** 若出现错误（例如序列过短） */
  error?: string;
}

// 输出整体数组
export type AdfResult = AdfResultItem[];

function resolvePythonExecutable(toolRoot: string): string {
  const candidates = process.platform === "win32"
    ? [
      path.join(toolRoot, "venv", "Scripts", "python.exe"),
      path.join(toolRoot, ".venv", "Scripts", "python.exe"),
    ]
    : [
      path.join(toolRoot, "venv", "bin", "python"),
      path.join(toolRoot, ".venv", "bin", "python"),
    ];

  const executable = candidates.find((candidate) => existsSync(candidate));
  if (!executable) {
    throw new Error(
      "ADF Python virtual environment not found. Please run py/adf/bootstrap.sh",
    );
  }
  return executable;
}

/**
 * 调用 Python ADF 工具，执行固定 d=0,1,2 三阶差分检验。
 *
 * @param data 由 {month, sales} 对象组成的数组（推荐长度 ≥ 10）
 * @returns Promise<AdfResult> ADF 检验结果
 */
export async function runAdf(data: AdfInputPoint[]): Promise<AdfResult> {
  // 序列化成 JSON，确保 key 带双引号
  const jsonArg = JSON.stringify(data);

  const toolRoot = path.resolve(import.meta.dir, "..", "..", "py", "adf");
  const pythonExec = resolvePythonExecutable(toolRoot);

  const proc = Bun.spawn([pythonExec, "-m", "adf_tool", jsonArg, "--pretty"], {
    cwd: toolRoot,
    stdout: "pipe",
    stderr: "pipe",
  });

  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);

  const exitCode = await proc.exited;

  if (exitCode !== 0) {
    const message = stderr.trim() || `Process exited with code ${exitCode}`;
    throw new Error(`Failed to execute ADF tool: ${message}`);
  }

  try {
    const parsed = JSON.parse(stdout) as AdfResult;
    // 过滤掉包含 error 字段的结果项，避免前端访问 null 值崩溃
    const validResults = parsed.filter((item) => !item.error);

    // 如果所有结果都被过滤掉了（数据量太少），抛出错误
    if (validResults.length === 0) {
      throw new Error("数据量不足，无法进行 ADF 检测");
    }

    return validResults;
  } catch (error) {
    // 如果是我们抛出的错误，直接传递
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(`Failed to parse ADF result: ${stdout}`);
  }
}
