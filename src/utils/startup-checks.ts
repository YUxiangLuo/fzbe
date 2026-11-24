import { accessSync, constants as fsConstants, existsSync, lstatSync, mkdirSync } from "node:fs";
import path from "node:path";
import type { Pool, RowDataPacket } from "mysql2/promise";
import puppeteer from "puppeteer-core";
import { env, pool } from "../config";
import { DB_POOL_CONFIG, PATHS } from "./constants";
import logger from "./logger";
import { PY_DIR, resolvePythonExecutable } from "./python";
import { ensureDatabaseManagedStorageRoot } from "./managed-storage";
import { resolveDatabaseSessionTimezone } from "../config/database-session";

const DEFAULT_COMMAND_TIMEOUT_MS = 30_000;
const EXPECTED_DATABASE_COLLATION = "utf8mb4_unicode_ci";

interface StartupHealthCheckRow extends RowDataPacket {
  ok?: number;
  session_time_zone?: string;
  utc_offset_seconds?: number | string;
  database_version?: string;
  database_collation?: string;
  max_connections?: number | string;
}

export interface DatabaseRuntimeProbeOptions {
  connectionLimit?: number;
  logInfo?: (message: string) => void;
  logWarn?: (message: string) => void;
}

export interface StartupCheckDependencies {
  ensureExecutableFile: (filePath: string) => void;
  ensureManagedStorageRoot: (managedRoot: string) => Promise<string>;
  ensureWritableDirectory: (dirPath: string) => void;
  fileExists: (filePath: string) => boolean;
  getChromiumPath: () => string;
  getPyDir: () => string;
  logInfo: (message: string) => void;
  logWarn: (message: string) => void;
  probeChromiumRuntime: (input: { chromiumPath: string; timeoutMs?: number }) => Promise<void>;
  queryDatabase: () => Promise<unknown>;
  resolvePythonExecutable: () => string;
  runCommand: (input: { command: string[]; cwd?: string; timeoutMs?: number }) => Promise<void>;
}

function ensureWritableDirectory(dirPath: string) {
  mkdirSync(dirPath, { recursive: true });
  const stats = lstatSync(dirPath);
  if (!stats.isDirectory() || stats.isSymbolicLink()) {
    throw new Error("目标路径不是可用的真实目录");
  }
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

async function probeChromiumRuntime({
  chromiumPath,
  timeoutMs = 10_000,
}: {
  chromiumPath: string;
  timeoutMs?: number;
}): Promise<void> {
  const browser = await puppeteer.launch({
    executablePath: chromiumPath,
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-gpu",
      "--disable-dev-shm-usage",
    ],
    timeout: timeoutMs,
  });

  try {
    const page = await browser.newPage();
    await page.setExtraHTTPHeaders({
      "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline';",
    });
    await page.setJavaScriptEnabled(false);
    await page.setContent("<html><body>startup-check</body></html>", {
      waitUntil: "domcontentloaded",
      timeout: timeoutMs,
    });
    await page.pdf({ format: "A4", printBackground: true, timeout: timeoutMs });
  } finally {
    await browser.close().catch(() => {
      // Ignore teardown failures during health probing.
    });
  }
}

export async function probeStartupDatabaseRuntime(
  dbPool: Pick<Pool, "getConnection"> = pool,
  databaseTimezone: string = env.DB_TIMEZONE,
  options: DatabaseRuntimeProbeOptions = {},
): Promise<void> {
  const connection = await dbPool.getConnection();
  let reusable = false;

  try {
    const [probeRows] = await connection.query<StartupHealthCheckRow[]>(
      `SELECT 1 AS ok,
              @@session.time_zone AS session_time_zone,
              TIMESTAMPDIFF(SECOND, UTC_TIMESTAMP(6), NOW(6)) AS utc_offset_seconds,
              VERSION() AS database_version,
              @@collation_database AS database_collation,
              @@max_connections AS max_connections`,
    );
    const probe = probeRows[0];
    if (probe?.ok !== 1) {
      throw new Error("SELECT 1 探针返回异常结果");
    }

    const expectedTimezone = resolveDatabaseSessionTimezone(databaseTimezone);
    if (
      probe.session_time_zone !== expectedTimezone.mysqlValue
      || Number(probe.utc_offset_seconds) !== expectedTimezone.offsetSeconds
    ) {
      throw new Error(
        `数据库会话时区不一致: expected=${expectedTimezone.mysqlValue}, actual=${probe.session_time_zone ?? "unknown"}, offset_seconds=${String(probe.utc_offset_seconds ?? "unknown")}`,
      );
    }

    const logInfo = options.logInfo ?? ((message: string) => logger.info(message));
    const logWarn = options.logWarn ?? ((message: string) => logger.warn({ check: "startup-database-runtime" }, message));
    const configuredConnectionLimit = options.connectionLimit ?? DB_POOL_CONFIG.CONNECTION_LIMIT;

    const databaseVersion = typeof probe.database_version === "string"
      ? probe.database_version.trim()
      : "";
    const databaseCollation = typeof probe.database_collation === "string"
      ? probe.database_collation.trim()
      : "";
    const maxConnections = probe.max_connections !== undefined && probe.max_connections !== null
      ? Number(probe.max_connections)
      : Number.NaN;

    if (databaseVersion) {
      logInfo(
        `Database runtime: version=${databaseVersion}, collation=${databaseCollation || "unknown"}, max_connections=${Number.isFinite(maxConnections) ? String(maxConnections) : "unknown"}`,
      );
    }
    if (
      databaseCollation
      && databaseCollation.toLowerCase() !== EXPECTED_DATABASE_COLLATION
    ) {
      logWarn(
        `数据库默认排序规则为 ${databaseCollation}，与建表基线 ${EXPECTED_DATABASE_COLLATION} 不一致，建议在维护窗口执行 ALTER DATABASE 统一排序规则以避免联表冲突。`,
      );
    }
    if (
      Number.isFinite(maxConnections)
      && maxConnections > 0
      && maxConnections < configuredConnectionLimit
    ) {
      logWarn(
        `数据库 max_connections (${maxConnections}) 小于当前连接池上限 (${configuredConnectionLimit})，高并发场景可能触发连接耗尽。`,
      );
    }

    reusable = true;
  } finally {
    if (reusable) {
      try {
        connection.release();
      } catch (error) {
        connection.destroy();
        throw error;
      }
    } else {
      connection.destroy();
    }
  }
}

const defaultDependencies: StartupCheckDependencies = {
  ensureExecutableFile,
  ensureManagedStorageRoot: (managedRoot) => ensureDatabaseManagedStorageRoot(managedRoot, env),
  ensureWritableDirectory,
  fileExists: existsSync,
  getChromiumPath: () => PATHS.CHROMIUM_PATH,
  getPyDir: () => PY_DIR,
  logInfo: (message) => logger.info(message),
  logWarn: (message) => logger.warn({ check: "startup-environment" }, message),
  probeChromiumRuntime,
  queryDatabase: probeStartupDatabaseRuntime,
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
    failures.push(`数据库运行时检查失败: ${error instanceof Error ? error.message : String(error)}`);
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
    const chromiumCheck = {
      label: "Chromium/Puppeteer 自检",
      chromiumPath,
      timeoutMs: 10_000,
    };

    try {
      await deps.probeChromiumRuntime(chromiumCheck);
    } catch (error) {
      failures.push(
        `${chromiumCheck.label}失败 (${chromiumPath}): ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  const uploadsRoot = path.resolve(import.meta.dir, "../../uploads");
  const requiredDirectories = [
    uploadsRoot,
    path.join(pyDir, "trained_models"),
  ];

  for (const dirPath of requiredDirectories) {
    try {
      deps.ensureWritableDirectory(dirPath);
    } catch (error) {
      failures.push(`目录不可读写: ${dirPath} (${error instanceof Error ? error.message : String(error)})`);
    }
  }

  for (const category of ["manuals", "datasets", "reports", "exports", "temp"]) {
    const categoryRoot = path.join(uploadsRoot, category);
    try {
      const databaseRoot = await deps.ensureManagedStorageRoot(categoryRoot);
      deps.ensureWritableDirectory(databaseRoot);
    } catch (error) {
      failures.push(`目录不可读写: ${categoryRoot} (${error instanceof Error ? error.message : String(error)})`);
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
