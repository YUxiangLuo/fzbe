import { accessSync, constants as fsConstants, existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { pool } from "../config";
import { PATHS } from "./constants";
import logger from "./logger";
import { PY_DIR, resolvePythonExecutable } from "./python";

const DEFAULT_COMMAND_TIMEOUT_MS = 30_000;

export interface StartupCheckDependencies {
  ensureExecutableFile: (filePath: string) => void;
  ensureWritableDirectory: (dirPath: string) => void;
  fileExists: (filePath: string) => boolean;
  getChromiumPath: () => string;
  getPyDir: () => string;
  logInfo: (message: string) => void;
  logWarn: (message: string) => void;
  queryDatabase: () => Promise<unknown>;
  resolvePythonExecutable: () => string;
  runCommand: (input: { command: string[]; cwd?: string; timeoutMs?: number }) => Promise<void>;
}

function ensureWritableDirectory(dirPath: string) {
  mkdirSync(dirPath, { recursive: true });
  accessSync(dirPath, fsConstants.R_OK | fsConstants.W_OK);
}

function ensureExecutableFile(filePath: string) {
  accessSync(filePath, fsConstants.R_OK | fsConstants.X_OK);
}

async function runCommand({
  command,
  cwd,
  timeoutMs = DEFAULT_COMMAND_TIMEOUT_MS,
}: {
  command: string[];
  cwd?: string;
  timeoutMs?: number;
}): Promise<void> {
  const proc = Bun.spawn(command, {
    cwd,
    stdout: "pipe",
    stderr: "pipe",
  });

  const timer = setTimeout(() => {
    try {
      proc.kill("SIGKILL");
    } catch {
      // Process may already be gone.
    }
  }, timeoutMs);

  try {
    const [stdout, stderr] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ]);

    if (proc.exitCode !== 0) {
      const detail = stderr.trim() || stdout.trim() || `exit code ${proc.exitCode}`;
      throw new Error(detail);
    }
  } finally {
    clearTimeout(timer);
  }
}

const defaultDependencies: StartupCheckDependencies = {
  ensureExecutableFile,
  ensureWritableDirectory,
  fileExists: existsSync,
  getChromiumPath: () => PATHS.CHROMIUM_PATH,
  getPyDir: () => PY_DIR,
  logInfo: (message) => logger.info(message),
  logWarn: (message) => logger.warn({ check: "startup-environment" }, message),
  queryDatabase: async () => {
    await pool.query("SELECT 1");
  },
  resolvePythonExecutable,
  runCommand,
};

export async function runStartupEnvironmentChecks(
  deps: StartupCheckDependencies = defaultDependencies,
): Promise<void> {
  const failures: string[] = [];

  try {
    await deps.queryDatabase();
  } catch (error) {
    failures.push(`数据库连接失败: ${error instanceof Error ? error.message : String(error)}`);
  }

  const pyDir = deps.getPyDir();
  let pythonExecutable: string | null = null;

  try {
    pythonExecutable = deps.resolvePythonExecutable();
    deps.ensureExecutableFile(pythonExecutable);
  } catch (error) {
    failures.push(`Python 运行环境不可用: ${error instanceof Error ? error.message : String(error)}`);
  }

  const requiredFiles = [
    { label: "Python 模型入口脚本", filePath: path.join(pyDir, "run_model.py") },
    { label: "ADF 模块入口", filePath: path.join(pyDir, "src", "adf", "__main__.py") },
  ];

  for (const { label, filePath } of requiredFiles) {
    if (!deps.fileExists(filePath)) {
      failures.push(`${label}不存在: ${filePath}`);
    }
  }

  if (pythonExecutable) {
    const pythonChecks = [
      {
        label: "Python ADF 自检",
        command: [pythonExecutable, "-m", "src.adf", "--help"],
        cwd: pyDir,
      },
      {
        label: "Python 模型运行时自检",
        command: [pythonExecutable, "run_model.py", "--help"],
        cwd: pyDir,
      },
    ];

    for (const check of pythonChecks) {
      try {
        await deps.runCommand(check);
      } catch (error) {
        failures.push(`${check.label}失败: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  }

  const chromiumPath = deps.getChromiumPath();
  let chromiumExecutableReady = true;
  try {
    deps.ensureExecutableFile(chromiumPath);
  } catch (error) {
    chromiumExecutableReady = false;
    failures.push(`Chromium 不可用 (${chromiumPath}): ${error instanceof Error ? error.message : String(error)}`);
  }

  if (chromiumExecutableReady) {
    try {
      await deps.runCommand({
        command: [chromiumPath, "--version"],
        timeoutMs: 10_000,
      });
    } catch (error) {
      failures.push(`Chromium 运行时自检失败 (${chromiumPath}): ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  const requiredDirectories = [
    path.resolve(import.meta.dir, "../../uploads"),
    path.resolve(import.meta.dir, "../../uploads/manuals"),
    path.resolve(import.meta.dir, "../../uploads/datasets"),
    path.resolve(import.meta.dir, "../../uploads/reports"),
    path.resolve(import.meta.dir, "../../uploads/exports"),
    path.resolve(import.meta.dir, "../../uploads/temp"),
    path.join(pyDir, "trained_models"),
  ];

  for (const dirPath of requiredDirectories) {
    try {
      deps.ensureWritableDirectory(dirPath);
    } catch (error) {
      failures.push(`目录不可读写: ${dirPath} (${error instanceof Error ? error.message : String(error)})`);
    }
  }

  if (failures.length > 0) {
    for (const failure of failures) {
      deps.logWarn(failure);
    }
    throw new Error(`启动环境检查失败，共 ${failures.length} 项`);
  }

  deps.logInfo("Startup environment checks passed.");
}
