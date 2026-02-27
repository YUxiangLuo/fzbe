import path from "path";
import { existsSync } from "node:fs";

/** py/ 目录的绝对路径 */
export const PY_DIR = path.resolve(import.meta.dir, "..", "..", "py");

/** 解析 py/venv 中的 Python 可执行文件路径 */
export function resolvePythonExecutable(): string {
  const candidates = process.platform === "win32"
    ? [
      path.join(PY_DIR, "venv", "Scripts", "python.exe"),
      path.join(PY_DIR, ".venv", "Scripts", "python.exe"),
    ]
    : [
      path.join(PY_DIR, "venv", "bin", "python"),
      path.join(PY_DIR, ".venv", "bin", "python"),
    ];

  const executable = candidates.find((candidate) => existsSync(candidate));
  if (!executable) {
    throw new Error("Python virtual environment not found. Please run py/install.sh");
  }
  return executable;
}
