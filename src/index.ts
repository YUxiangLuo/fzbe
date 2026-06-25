import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { cors } from "hono/cors";
import path from "path";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { Readable } from "node:stream";
import type { RowDataPacket } from "mysql2/promise";
import { env, pool, connectionLimit } from "./config";
import { jwtAuth } from "./middleware";
import logger from "./utils/logger";
import { CONCURRENCY_LIMITS, DB_POOL_CONFIG, PYTHON_RUNTIME_CONFIG, TIMEOUTS } from "./utils/constants";
import { runStartupEnvironmentChecks } from "./utils/startup-checks";
import type { JwtPayload, UserRole } from "./types";
// Import new modular controllers
import { authController } from "./modules/auth";
import { toolsController } from "./modules/tools";
import { manualsController } from "./modules/manuals";
import { questionBankController } from "./modules/question-bank";
import { quizzesController } from "./modules/quizzes";
import { reportsController } from "./modules/reports";
import { getPdfQueueStats } from "./modules/reports/reports.service";
import { modelsController } from "./modules/models";
import { classesController } from "./modules/classes";
import { classMembersController } from "./modules/class-members";
import { datasetsController } from "./modules/datasets";
import { experimentsController } from "./modules/experiments";
import { classGradesController } from "./modules/class-grades";
import { usersController } from "./modules/users";

// --- Hono App Initialization ---
const app = new Hono();
const uploadsRoot = path.resolve(import.meta.dir, "../uploads");
const reportsRoot = path.resolve(uploadsRoot, "reports");
const exportsRoot = path.resolve(uploadsRoot, "exports");

interface StudentOwnerRow extends RowDataPacket {
  student_id: number;
}

interface ClassOwnerRow extends RowDataPacket {
  teacher_id: number;
}

interface AssignmentRow extends RowDataPacket {
  assignment_id?: number;
  enrollment_id?: number;
}

interface ManualAccessRow extends RowDataPacket {
  manual_id: number;
  is_active: number | boolean;
}

interface ServerVariableRow extends RowDataPacket {
  Variable_name: string;
  Value: string;
}

type FileDisposition = "inline" | "attachment";

interface LocalFileResponseOptions {
  absolutePath: string;
  filename: string;
  disposition: FileDisposition;
  fallbackContentType: string;
  rangeHeader?: string;
}

interface ByteRange {
  start: number;
  end: number;
}

type ParsedByteRange = ByteRange | "invalid" | "unsupported" | null;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSafeFilename(filename: string): boolean {
  return filename === path.basename(filename) && !filename.includes("\0");
}

function toAsciiHeaderFilename(filename: string): string {
  const basename = path.basename(filename).replace(/[\r\n]/g, "_").replace(/["\\]/g, "_");
  const extension = path.extname(basename);
  const stem = extension ? basename.slice(0, -extension.length) : basename;

  const safeStem = stem
    .replace(/[^A-Za-z0-9_-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");

  const normalizedStem = safeStem.length > 0 ? safeStem : "download";

  const safeExtensionBody = extension
    .replace(/^\./, "")
    .replace(/[^A-Za-z0-9]/g, "");
  const normalizedExtension = safeExtensionBody ? `.${safeExtensionBody}` : "";

  return `${normalizedStem}${normalizedExtension}`;
}

function encodeRFC5987Value(value: string): string {
  return encodeURIComponent(value).replace(/[!'()*]/g, (char) =>
    `%${char.charCodeAt(0).toString(16).toUpperCase()}`,
  );
}

function buildContentDisposition(type: "inline" | "attachment", filename: string): string {
  const basename = path.basename(filename);
  const asciiFilename = toAsciiHeaderFilename(basename);
  const encodedFilename = encodeRFC5987Value(basename);
  return `${type}; filename="${asciiFilename}"; filename*=UTF-8''${encodedFilename}`;
}

function parseSingleByteRange(rangeHeader: string | undefined, fileSize: number): ParsedByteRange {
  if (!rangeHeader) return null;
  if (fileSize <= 0) return "invalid";

  const trimmedRange = rangeHeader.trim();
  if (!trimmedRange.toLowerCase().startsWith("bytes=")) return "unsupported";

  const rawRangeSet = trimmedRange.slice("bytes=".length);
  if (rawRangeSet.includes(",")) return "unsupported";

  const match = rawRangeSet.match(/^(\d*)-(\d*)$/);
  if (!match) return "invalid";

  const rawStart = match[1] ?? "";
  const rawEnd = match[2] ?? "";
  if (!rawStart && !rawEnd) return "invalid";

  const parseRangeNumber = (value: string): number | null => {
    if (!/^\d+$/.test(value)) return null;
    const parsed = Number(value);
    return Number.isSafeInteger(parsed) ? parsed : null;
  };

  if (!rawStart) {
    const suffixLength = parseRangeNumber(rawEnd);
    if (!suffixLength || suffixLength <= 0) return "invalid";
    return {
      start: Math.max(fileSize - suffixLength, 0),
      end: fileSize - 1,
    };
  }

  const start = parseRangeNumber(rawStart);
  if (start === null || start >= fileSize) return "invalid";

  const end = rawEnd ? parseRangeNumber(rawEnd) : fileSize - 1;
  if (end === null || start > end) return "invalid";

  return {
    start,
    end: Math.min(end, fileSize - 1),
  };
}

async function buildLocalFileResponse(options: LocalFileResponseOptions): Promise<Response | null> {
  const fileStats = await stat(options.absolutePath).catch(() => null);
  if (!fileStats?.isFile()) return null;

  const file = Bun.file(options.absolutePath);
  const contentType = file.type || options.fallbackContentType;
  const etag = `"${fileStats.size.toString(16)}-${Math.trunc(fileStats.mtimeMs).toString(16)}"`;
  const lastModified = fileStats.mtime.toUTCString();
  const baseHeaders = {
    "Content-Type": contentType,
    "Content-Disposition": buildContentDisposition(options.disposition, options.filename),
    "Cache-Control": "private, max-age=60",
    "Accept-Ranges": "bytes",
    "ETag": etag,
    "Last-Modified": lastModified,
    "X-Content-Type-Options": "nosniff",
  };

  const range = parseSingleByteRange(options.rangeHeader, fileStats.size);
  if (range === "invalid") {
    return new Response(null, {
      status: 416,
      headers: {
        ...baseHeaders,
        "Content-Range": `bytes */${fileStats.size}`,
      },
    });
  }

  if (range && range !== "unsupported") {
    const contentLength = range.end - range.start + 1;
    const body = Readable.toWeb(createReadStream(options.absolutePath, {
      start: range.start,
      end: range.end,
    })) as ReadableStream;
    return new Response(body, {
      status: 206,
      headers: {
        ...baseHeaders,
        "Content-Length": String(contentLength),
        "Content-Range": `bytes ${range.start}-${range.end}/${fileStats.size}`,
      },
    });
  }

  return new Response(file, {
    headers: {
      ...baseHeaders,
      "Content-Length": String(fileStats.size),
    },
  });
}

async function isTeacherOfStudent(studentId: number, teacherId: number): Promise<boolean> {
  const [[row]] = await pool.query<AssignmentRow[]>(
    `SELECT 1
       FROM enrollments e
       JOIN classes c ON e.class_id = c.class_id
      WHERE e.student_id = ? AND c.teacher_id = ?
      LIMIT 1`,
    [studentId, teacherId],
  );
  return Boolean(row);
}

async function isAssistantOfStudent(studentId: number, assistantId: number): Promise<boolean> {
  const [[row]] = await pool.query<AssignmentRow[]>(
    `SELECT 1
       FROM enrollments e
       JOIN class_assistants ca ON e.class_id = ca.class_id
      WHERE e.student_id = ? AND ca.assistant_id = ?
      LIMIT 1`,
    [studentId, assistantId],
  );
  return Boolean(row);
}

async function canAccessReportFile(filename: string, userId: number, role: UserRole): Promise<boolean> {
  if (role === "Admin") return true;

  const [rows] = await pool.query<StudentOwnerRow[]>(
    `SELECT student_id
       FROM experiment_reports
      WHERE pdf_file_path LIKE ?
      ORDER BY report_id DESC
      LIMIT 1`,
    [`%/${filename}`],
  );
  const ownerStudentId = rows[0]?.student_id;
  if (!ownerStudentId) return false;

  if (role === "Student") return ownerStudentId === userId;
  if (role === "Teacher") return isTeacherOfStudent(ownerStudentId, userId);
  if (role === "Assistant") return isAssistantOfStudent(ownerStudentId, userId);

  return false;
}

function parseClassIdFromExportFilename(filename: string): number | null {
  const match = filename.match(/^class-(\d+)-[a-z-]+-\d+\.(zip|csv)$/);
  if (!match?.[1]) return null;
  const classId = Number(match[1]);
  return Number.isInteger(classId) ? classId : null;
}

async function canAccessExportFile(filename: string, userId: number, role: UserRole): Promise<boolean> {
  if (role === "Admin") return true;

  const classId = parseClassIdFromExportFilename(filename);
  if (!classId) return false;

  if (role === "Teacher") {
    const [rows] = await pool.query<ClassOwnerRow[]>(
      "SELECT teacher_id FROM classes WHERE class_id = ? LIMIT 1",
      [classId],
    );
    return rows[0]?.teacher_id === userId;
  }

  if (role === "Assistant") {
    const [[row]] = await pool.query<AssignmentRow[]>(
      "SELECT assignment_id FROM class_assistants WHERE class_id = ? AND assistant_id = ? LIMIT 1",
      [classId, userId],
    );
    return Boolean(row);
  }

  return false;
}

async function checkManualFileAccess(filename: string, role: UserRole): Promise<"allowed" | "forbidden" | "not_found"> {
  const [rows] = await pool.query<ManualAccessRow[]>(
    `SELECT manual_id, is_active
       FROM experiment_manuals
      WHERE file_path LIKE ?
      ORDER BY manual_id DESC
      LIMIT 1`,
    [`%/${filename}`],
  );

  const manual = rows[0];
  if (!manual) return "not_found";

  if (role === "Admin") return "allowed";
  return manual.is_active ? "allowed" : "forbidden";
}

// --- Global Middleware ---
// app.use(timestampedLogger);
app.use(cors());
app.use("*", async (c, next) => {
  await next();

  const currentResponse = c.res;
  if (currentResponse.status < 200 || currentResponse.status >= 300) {
    return;
  }

  const contentType = currentResponse.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    return;
  }

  let payload: unknown;
  try {
    payload = await currentResponse.clone().json();
  } catch {
    return;
  }

  if (isPlainObject(payload) && ("data" in payload || "error" in payload)) {
    return;
  }

  const wrappedHeaders = new Headers(currentResponse.headers);
  wrappedHeaders.set("content-type", "application/json; charset=UTF-8");

  c.res = new Response(
    JSON.stringify({ data: payload }),
    {
      status: currentResponse.status,
      headers: wrappedHeaders,
    },
  );
});
app.get("/api/v1/runtime-info", (c) =>
  c.json({
    app: "fangzhen-be",
    database: env.DB_DATABASE,
    port: env.PORT,
    concurrency: {
      systemParallelism: CONCURRENCY_LIMITS.SYSTEM_PARALLELISM,
      maxConcurrentModelJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS,
      maxConcurrentPdfJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_PDF_JOBS,
      dbConnectionLimit: connectionLimit,
      dbLockConnections: DB_POOL_CONFIG.LOCK_CONNECTIONS,
      dbQueryHeadroom: DB_POOL_CONFIG.QUERY_HEADROOM,
      pythonModelProcessThreads: PYTHON_RUNTIME_CONFIG.MODEL_PROCESS_THREADS,
      pythonTensorflowInterOpThreads: PYTHON_RUNTIME_CONFIG.TF_INTER_OP_THREADS,
    },
    timeouts: {
      pdfQueueMs: TIMEOUTS.PDF_QUEUE,
      pdfGenerationMs: TIMEOUTS.PDF_GENERATION,
    },
    pdfQueue: getPdfQueueStats(),
  }));
app.get("/manuals/:filename", jwtAuth, async (c) => {
  const payload = c.get("jwtPayload") as JwtPayload;
  const filename = c.req.param("filename");

  if (!isSafeFilename(filename)) return c.json({ error: "无效的文件名" }, 400);

  const access = await checkManualFileAccess(filename, payload.role);
  if (access === "not_found") return c.json({ error: "文件不存在" }, 404);
  if (access === "forbidden") return c.json({ error: "权限不足" }, 403);

  const response = await buildLocalFileResponse({
    absolutePath: path.resolve(uploadsRoot, "manuals", filename),
    filename,
    disposition: "inline",
    fallbackContentType: "application/pdf",
    rangeHeader: c.req.header("range"),
  });
  if (!response) return c.json({ error: "文件不存在" }, 404);

  return response;
});

app.get("/datasets/:filename", jwtAuth, async (c) => {
  const payload = c.get("jwtPayload") as JwtPayload;
  if (payload.role !== "Admin") return c.json({ error: "权限不足" }, 403);

  const filename = c.req.param("filename");
  if (!isSafeFilename(filename)) return c.json({ error: "无效的文件名" }, 400);

  const response = await buildLocalFileResponse({
    absolutePath: path.resolve(uploadsRoot, "datasets", filename),
    filename,
    disposition: "attachment",
    fallbackContentType: "text/csv; charset=UTF-8",
    rangeHeader: c.req.header("range"),
  });
  if (!response) return c.json({ error: "文件不存在" }, 404);

  return response;
});
app.get("/reports/:filename", jwtAuth, async (c) => {
  const payload = c.get("jwtPayload") as JwtPayload;
  const filename = c.req.param("filename");

  if (!isSafeFilename(filename)) return c.json({ error: "无效的文件名" }, 400);

  const allowed = await canAccessReportFile(filename, payload.sub, payload.role);
  if (!allowed) return c.json({ error: "权限不足" }, 403);

  const response = await buildLocalFileResponse({
    absolutePath: path.resolve(reportsRoot, filename),
    filename,
    disposition: "inline",
    fallbackContentType: "application/pdf",
    rangeHeader: c.req.header("range"),
  });
  if (!response) return c.json({ error: "文件不存在" }, 404);

  return response;
});

app.get("/exports/:filename", jwtAuth, async (c) => {
  const payload = c.get("jwtPayload") as JwtPayload;
  const filename = c.req.param("filename");

  if (!isSafeFilename(filename)) return c.json({ error: "无效的文件名" }, 400);

  const allowed = await canAccessExportFile(filename, payload.sub, payload.role);
  if (!allowed) return c.json({ error: "权限不足" }, 403);

  const response = await buildLocalFileResponse({
    absolutePath: path.resolve(exportsRoot, filename),
    filename,
    disposition: "attachment",
    fallbackContentType: "application/octet-stream",
    rangeHeader: c.req.header("range"),
  });
  if (!response) return c.json({ error: "文件不存在" }, 404);

  return response;
});
// --- Mount Routes ---
// New modular controllers (no pool/secret injection needed)
app.route("/", authController);
app.route("/", toolsController);
app.route("/", manualsController);
app.route("/", questionBankController);
app.route("/", quizzesController);
app.route("/", reportsController);
app.route("/", modelsController);
app.route("/", classesController);
app.route("/", classMembersController);
app.route("/", datasetsController);
app.route("/", experimentsController);
app.route("/", classGradesController);
app.route("/", usersController);

// --- 404 & Error Handling ---
app.notFound((c) => c.json({ error: "接口不存在" }, 404));

app.onError((err, c) => {
  if (err instanceof HTTPException) {
    if (err.status >= 500) {
      logger.error(
        {
          err: { message: err.message, stack: err.stack },
          path: c.req.path,
          method: c.req.method,
          status: err.status,
        },
        "HTTP exception",
      );
    } else if (err.status !== 401 && err.status !== 403) {
      logger.warn(
        {
          path: c.req.path,
          method: c.req.method,
          status: err.status,
          message: err.message,
        },
        "Handled client error",
      );
    }
    return c.json({ error: err.message }, err.status);
  }
  logger.error(
    {
      err: err instanceof Error ? { message: err.message, stack: err.stack } : err,
      path: c.req.path,
      method: c.req.method,
    },
    "Unhandled error",
  );
  return c.json({ error: "服务器内部错误" }, 500);
});

export const serverConfig = {
  port: env.PORT,
  fetch: app.fetch,
  error(error: Error) {
    logger.error(
      { err: { message: error.message, stack: error.stack } },
      "Bun.serve unhandled error",
    );
    return new Response("Internal Server Error", { status: 500 });
  },
};

async function readDatabaseMaxConnections(): Promise<number | null> {
  const [rows] = await pool.query<ServerVariableRow[]>("SHOW VARIABLES LIKE 'max_connections'");
  const rawValue = rows[0]?.Value;
  if (!rawValue) return null;

  const parsed = Number.parseInt(rawValue, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

export async function runServerStartupPreflight() {
  // ---- 启动校验：连接池容量 vs. 并发模型任务 ----
  const minRequired = DB_POOL_CONFIG.LOCK_CONNECTIONS + DB_POOL_CONFIG.QUERY_HEADROOM;
  if (connectionLimit < minRequired) {
    logger.fatal(
      {
        connectionLimit,
        maxConcurrentModelJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS,
        lockConnections: DB_POOL_CONFIG.LOCK_CONNECTIONS,
        queryHeadroom: DB_POOL_CONFIG.QUERY_HEADROOM,
        minRequired,
      },
      "数据库连接池容量不足：connectionLimit 应至少为 2 × MAX_CONCURRENT_MODEL_JOBS + DB_QUERY_HEADROOM，否则并发训练时可能死锁或阻塞普通查询",
    );
    process.exit(1);
  }

  try {
    const databaseMaxConnections = await readDatabaseMaxConnections();
    if (databaseMaxConnections !== null && databaseMaxConnections < connectionLimit) {
      logger.fatal(
        { connectionLimit, databaseMaxConnections, maxConcurrentModelJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS },
        "数据库 max_connections 小于后端连接池容量：请提高 MariaDB/MySQL max_connections，或降低 MAX_CONCURRENT_MODEL_JOBS/DB_CONNECTION_LIMIT",
      );
      process.exit(1);
    }
  } catch (error) {
    logger.fatal(
      {
        err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
      },
      "读取数据库 max_connections 失败，服务退出",
    );
    process.exit(1);
  }

  try {
    await runStartupEnvironmentChecks();
  } catch (error) {
    logger.fatal(
      {
        err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
      },
      "启动环境检查未通过，服务退出",
    );
    process.exit(1);
  }
}

if (import.meta.main) {
  await runServerStartupPreflight();

  process.on("unhandledRejection", (reason) => {
    try {
      logger.error(
        {
          err:
            reason instanceof Error
              ? { message: reason.message, stack: reason.stack }
              : { message: String(reason) },
        },
        "Unhandled promise rejection",
      );
    } catch {
      // Avoid cascading failures if serialization throws
    }
  });

  process.on("uncaughtException", (err) => {
    logger.fatal(
      { err: { message: err.message, stack: err.stack } },
      "Uncaught exception",
    );
    setTimeout(() => process.exit(1), 1000);
  });

  const server = Bun.serve(serverConfig);
  logger.info(`Server listening on ${server.url.href}`);
}

export { app };
