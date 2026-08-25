import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { cors } from "hono/cors";
import path from "path";
import { stat } from "node:fs/promises";
import type { RowDataPacket } from "mysql2/promise";
import { env, pool, connectionLimit } from "./config";
import { jwtAuth, requireStudent } from "./middleware";
import logger from "./utils/logger";
import { CONCURRENCY_LIMITS, DB_POOL_CONFIG } from "./utils/constants";
import { beginRequestObservation } from "./utils/request-observability";
import { runStartupEnvironmentChecks } from "./utils/startup-checks";
import type { JwtPayload, UserRole } from "./types";
// Import new modular controllers
import { authController } from "./modules/auth";
import { toolsController } from "./modules/tools";
import { manualsController } from "./modules/manuals";
import { reconcileManualDeletionJournals } from "./modules/manuals/manuals.service";
import { questionBankController } from "./modules/question-bank";
import { quizzesController } from "./modules/quizzes";
import { reportsController } from "./modules/reports";
import { holdPdfQueueSlotForTest } from "./modules/reports/reports.service";
import { runtimeController } from "./modules/runtime";
import { modelsController } from "./modules/models";
import {
  cleanupExpiredGuidedTrainingSessions,
  cleanupExpiredProductionPreparationSessions,
  cleanupStaleModelArtifacts,
} from "./modules/models/models.service";
import {
  withModelOperationContext,
  withModelOperationLease,
} from "./modules/models/model-operation-lease";
import { ModelCapacityBusyError } from "./modules/models/model-errors";
import { classesController } from "./modules/classes";
import { classMembersController } from "./modules/class-members";
import { datasetsController } from "./modules/datasets";
import { reconcileDatasetDeletionJournals } from "./modules/datasets/datasets.service";
import { experimentsController } from "./modules/experiments";
import { classGradesController } from "./modules/class-grades";
import {
  cleanupExpiredExportFiles,
  parseClassIdFromExportFilename,
} from "./modules/class-grades/class-grades.service";
import { usersController } from "./modules/users";
import { reconcileReportDeletionJournals } from "./modules/users/users.service";
import { academicTermsController } from "./modules/academic-terms";
import { cleanupStaleUploadTempFiles } from "./utils/runtime-cleanup";
import {
  hasLegacyGuidedTrainingCleanupTimerDisable,
  isRuntimeDiskCleanupTimerDisabled,
  shouldSkipGuidedTrainingCleanupForTrigger,
} from "./utils/runtime-cleanup-config";
import {
  getDatabaseManagedStorageRoot,
} from "./utils/managed-storage";
import { resolveExistingManagedFile } from "./utils/managed-file";

// --- Hono App Initialization ---
type AppEnv = {
  Variables: {
    requestId: string;
  };
};
const app = new Hono<AppEnv>();
const uploadsRoot = path.resolve(import.meta.dir, "../uploads");
const reportsRoot = path.resolve(uploadsRoot, "reports");
const exportsRoot = path.resolve(uploadsRoot, "exports");

interface StudentOwnerRow extends RowDataPacket {
  student_id: number;
  pdf_file_path: string;
}

interface DatasetFileRow extends RowDataPacket {
  file_path: string;
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
  file_path: string;
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
    return new Response(file.slice(range.start, range.end + 1, contentType), {
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

function escapeStoredPathBasename(filename: string): string {
  return filename
    .replaceAll("=", "==")
    .replaceAll("%", "=%")
    .replaceAll("_", "=_");
}

async function resolveAccessibleReportFile(
  filename: string,
  userId: number,
  role: UserRole,
): Promise<{ authorized: boolean; absolutePath: string | null }> {
  const [rows] = await pool.query<StudentOwnerRow[]>(
    `SELECT student_id, pdf_file_path
       FROM experiment_reports
      WHERE pdf_file_path = ?
         OR pdf_file_path LIKE ? ESCAPE '='
      ORDER BY report_id DESC`,
    [filename, `%/${escapeStoredPathBasename(filename)}`],
  );
  if (rows.length === 0) {
    return { authorized: false, absolutePath: null };
  }

  let hasAuthorizedOwner = false;
  for (const report of rows) {
    if (!report.student_id || !report.pdf_file_path) continue;
    const authorized = (
      role === "Admin"
      || (role === "Student" && report.student_id === userId)
      || (role === "Teacher" && await isTeacherOfStudent(report.student_id, userId))
      || (role === "Assistant" && await isAssistantOfStudent(report.student_id, userId))
    );
    if (!authorized) continue;
    hasAuthorizedOwner = true;
    const resolved = await resolveExistingManagedFile({
      storedPath: report.pdf_file_path,
      managedRoot: reportsRoot,
      environment: env,
      legacyFallbackDirectory: getDatabaseManagedStorageRoot(reportsRoot, env),
      enforceDisposableNamespace: true,
    }).catch(() => null);
    if (resolved) {
      return { authorized: true, absolutePath: resolved.absolutePath };
    }
  }
  return { authorized: hasAuthorizedOwner, absolutePath: null };
}

async function resolveDatasetDownloadPath(filename: string): Promise<string | null> {
  const [rows] = await pool.query<DatasetFileRow[]>(
    `SELECT file_path
       FROM datasets
      WHERE file_path = ?
         OR file_path LIKE ? ESCAPE '='
      ORDER BY dataset_id DESC
      LIMIT 1`,
    [filename, `%/${escapeStoredPathBasename(filename)}`],
  );
  if (!rows[0]?.file_path) return null;
  const resolved = await resolveExistingManagedFile({
    storedPath: rows[0].file_path,
    managedRoot: path.resolve(uploadsRoot, "datasets"),
    environment: env,
    legacyFallbackDirectory: getDatabaseManagedStorageRoot(
      path.resolve(uploadsRoot, "datasets"),
      env,
    ),
    enforceDisposableNamespace: true,
  }).catch(() => null);
  return resolved?.absolutePath ?? null;
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

async function resolveAccessibleManualFile(
  filename: string,
  role: UserRole,
): Promise<{ access: "allowed" | "forbidden" | "not_found"; absolutePath: string | null }> {
  const [rows] = await pool.query<ManualAccessRow[]>(
    `SELECT manual_id, is_active, file_path
       FROM experiment_manuals
      WHERE file_path = ?
         OR file_path LIKE ? ESCAPE '='
      ORDER BY manual_id DESC`,
    [filename, `%/${escapeStoredPathBasename(filename)}`],
  );

  if (rows.length === 0) return { access: "not_found", absolutePath: null };
  let hasAuthorizedManual = false;
  const manualsRoot = path.resolve(uploadsRoot, "manuals");
  for (const manual of rows) {
    if (role !== "Admin" && !manual.is_active) continue;
    hasAuthorizedManual = true;
    const resolved = await resolveExistingManagedFile({
      storedPath: manual.file_path,
      managedRoot: manualsRoot,
      environment: env,
      legacyFallbackDirectory: getDatabaseManagedStorageRoot(manualsRoot, env),
      enforceDisposableNamespace: true,
    }).catch(() => null);
    if (resolved) {
      return { access: "allowed", absolutePath: resolved.absolutePath };
    }
  }
  return {
    access: hasAuthorizedManual ? "allowed" : "forbidden",
    absolutePath: null,
  };
}

const directFileResponsePathPattern = /^\/(?:manuals|datasets|reports|exports)\//;
const CORS_ALLOW_HEADERS = ["Authorization", "Content-Type", "Range"];
const CORS_EXPOSE_HEADERS = ["Content-Length", "Content-Range", "Content-Disposition", "Accept-Ranges", "ETag", "Last-Modified", "X-Request-ID"];
function normalizeConfiguredCorsOrigin(rawOrigin: string): string {
  try {
    const parsedUrl = new URL(rawOrigin);
    if (!["http:", "https:"].includes(parsedUrl.protocol)) {
      throw new Error("unsupported protocol");
    }
    if (parsedUrl.pathname !== "/" || parsedUrl.search || parsedUrl.hash) {
      throw new Error("origin must not include path, query, or fragment");
    }
    return parsedUrl.origin;
  } catch (error) {
    throw new Error(`Invalid CORS_ALLOWED_ORIGINS origin "${rawOrigin}": ${error instanceof Error ? error.message : String(error)}`);
  }
}

function normalizeRequestCorsOrigin(origin: string | undefined | null): string | null | undefined {
  const trimmedOrigin = origin?.trim();
  if (!trimmedOrigin) return undefined;
  try {
    const parsedUrl = new URL(trimmedOrigin);
    if (!["http:", "https:"].includes(parsedUrl.protocol)) return null;
    return parsedUrl.origin;
  } catch {
    return null;
  }
}

const configuredCorsOrigins = (env.CORS_ALLOWED_ORIGINS ?? "")
  .split(",")
  .map((origin) => origin.trim())
  .filter((origin) => origin.length > 0)
  .map(normalizeConfiguredCorsOrigin);
const corsAllowedOriginSet = new Set(configuredCorsOrigins);
const isProductionRuntime = [process.env.NODE_ENV, process.env.BUN_ENV]
  .some((value) => String(value ?? "").toLowerCase() === "production");

if (isProductionRuntime && corsAllowedOriginSet.size === 0) {
  throw new Error("CORS_ALLOWED_ORIGINS must be set in production");
}

function resolveCorsOrigin(origin: string | undefined | null): string | undefined {
  const normalizedOrigin = normalizeRequestCorsOrigin(origin);
  if (normalizedOrigin === null) return undefined;
  if (!normalizedOrigin) return isProductionRuntime ? undefined : "*";
  if (!isProductionRuntime && corsAllowedOriginSet.size === 0) return normalizedOrigin;
  return corsAllowedOriginSet.has(normalizedOrigin) ? normalizedOrigin : undefined;
}

const defaultCors = cors({
  origin: (origin) => resolveCorsOrigin(origin),
  allowHeaders: CORS_ALLOW_HEADERS,
  exposeHeaders: CORS_EXPOSE_HEADERS,
});

function isDirectFileResponsePath(pathname: string): boolean {
  return directFileResponsePathPattern.test(pathname);
}

function appendVaryOrigin(headers: Headers): void {
  const currentVary = headers.get("Vary");
  if (!currentVary) {
    headers.set("Vary", "Origin");
    return;
  }
  if (!currentVary.toLowerCase().split(",").map((value) => value.trim()).includes("origin")) {
    headers.set("Vary", `${currentVary}, Origin`);
  }
}

function isCorsOriginRejected(origin: string | undefined | null): boolean {
  return Boolean(origin?.trim()) && resolveCorsOrigin(origin) === undefined;
}

function applyDirectFileCorsHeaders(headers: Headers, requestOrigin?: string | null): void {
  const allowedOrigin = resolveCorsOrigin(requestOrigin);
  if (allowedOrigin) {
    headers.set("Access-Control-Allow-Origin", allowedOrigin);
    appendVaryOrigin(headers);
  }
  headers.set("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS");
  headers.set("Access-Control-Allow-Headers", CORS_ALLOW_HEADERS.join(", "));
  headers.set("Access-Control-Expose-Headers", CORS_EXPOSE_HEADERS.join(", "));
}

// --- Global Middleware ---
app.use("*", async (c, next) => {
  const observation = beginRequestObservation(
    c.req.method,
    c.req.header("x-request-id"),
  );
  c.set("requestId", observation.requestId);
  c.header("X-Request-ID", observation.requestId);

  let failureStatus: number | null = null;
  try {
    await next();
  } catch (error) {
    failureStatus = error instanceof HTTPException ? error.status : 500;
    throw error;
  } finally {
    const status = failureStatus ?? c.res.status;
    const result = observation.finish(status);
    if (result) {
      const bindings = {
        requestId: observation.requestId,
        method: c.req.method,
        path: c.req.path,
        status,
        statusClass: result.statusClass,
        durationMs: Number(result.durationMs.toFixed(3)),
      };
      if (status >= 500) logger.warn(bindings, "HTTP request completed with server error");
      else logger.info(bindings, "HTTP request completed");
    }
  }
});
app.use("*", async (c, next) => {
  const requestOrigin = c.req.header("origin");
  if (!isDirectFileResponsePath(new URL(c.req.url).pathname)) {
    return defaultCors(c, next);
  }

  if (c.req.method === "OPTIONS") {
    if (isCorsOriginRejected(requestOrigin)) {
      return c.json({ error: "CORS origin not allowed" }, 403);
    }
    const headers = new Headers();
    applyDirectFileCorsHeaders(headers, requestOrigin);
    headers.set("Content-Length", "0");
    return new Response(null, { status: 204, headers });
  }

  await next();
  applyDirectFileCorsHeaders(c.res.headers, requestOrigin);
});
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

if (process.env.E2E_SERVER === "1") {
  app.post(
    "/api/v1/e2e/model-lease/hold",
    jwtAuth,
    requireStudent,
    async (c) => {
      const body = await c.req.json().catch(() => ({})) as {
        hold_ms?: unknown;
        experiment_id?: unknown;
      };
      const holdMs = Number(body.hold_ms ?? 3_000);
      if (!Number.isInteger(holdMs) || holdMs <= 0 || holdMs > 30_000) {
        return c.json({ error: "hold_ms must be an integer between 1 and 30000" }, 400);
      }
      const payload = c.get("jwtPayload") as JwtPayload;
      const executionExperimentId = body.experiment_id === undefined
        ? payload.sub
        : Number(body.experiment_id);
      if (!Number.isSafeInteger(executionExperimentId) || executionExperimentId <= 0) {
        return c.json({ error: "experiment_id must be a positive integer" }, 400);
      }

      try {
        await withModelOperationContext("prediction", async (context) =>
          await withModelOperationLease(
            context,
            {
              execution: {
                experimentId: executionExperimentId,
                modelType: "ma",
              },
            },
            async (lease) => {
              await Bun.sleep(holdMs);
              await lease.verifyOwned();
            },
          ));
        return c.json({ data: { held_ms: holdMs } });
      } catch (error) {
        if (error instanceof ModelCapacityBusyError) {
          c.header("Retry-After", String(error.retryAfterSeconds));
          return c.json({
            error: error.message,
            code: error.code,
            retryable: error.retryable,
          }, 429);
        }
        throw error;
      }
    },
  );

  app.post("/api/v1/e2e/pdf-slot/hold", async (c) => {
    const body = await c.req.json().catch(() => ({})) as { hold_ms?: unknown };
    const holdMs = Number(body.hold_ms ?? 5_000);
    if (!Number.isInteger(holdMs) || holdMs <= 0 || holdMs > 30_000) {
      return c.json({ error: "hold_ms must be an integer between 1 and 30000" }, 400);
    }

    await holdPdfQueueSlotForTest(holdMs);
    return c.json({ data: { held_ms: holdMs } });
  });
}

app.get("/manuals/:filename", jwtAuth, async (c) => {
  const payload = c.get("jwtPayload") as JwtPayload;
  const filename = c.req.param("filename");

  if (!isSafeFilename(filename)) return c.json({ error: "无效的文件名" }, 400);

  const manual = await resolveAccessibleManualFile(filename, payload.role);
  if (manual.access === "not_found") return c.json({ error: "文件不存在" }, 404);
  if (manual.access === "forbidden") return c.json({ error: "权限不足" }, 403);
  if (!manual.absolutePath) return c.json({ error: "文件不存在" }, 404);

  const response = await buildLocalFileResponse({
    absolutePath: manual.absolutePath,
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

  const absolutePath = await resolveDatasetDownloadPath(filename);
  if (!absolutePath) return c.json({ error: "文件不存在" }, 404);

  const response = await buildLocalFileResponse({
    absolutePath,
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

  const access = await resolveAccessibleReportFile(filename, payload.sub, payload.role);
  if (!access.authorized) return c.json({ error: "权限不足" }, 403);
  if (!access.absolutePath) return c.json({ error: "文件不存在" }, 404);

  const response = await buildLocalFileResponse({
    absolutePath: access.absolutePath,
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

  const resolvedExport = await resolveExistingManagedFile({
    storedPath: filename,
    managedRoot: exportsRoot,
    environment: env,
    legacyFallbackDirectory: getDatabaseManagedStorageRoot(exportsRoot, env),
    enforceDisposableNamespace: true,
  }).catch(() => null);
  if (!resolvedExport) return c.json({ error: "文件不存在" }, 404);

  const response = await buildLocalFileResponse({
    absolutePath: resolvedExport.absolutePath,
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
app.route("/", runtimeController);
app.route("/", modelsController);
app.route("/", classesController);
app.route("/", classMembersController);
app.route("/", datasetsController);
app.route("/", experimentsController);
app.route("/", classGradesController);
app.route("/", usersController);
app.route("/", academicTermsController);

// --- 404 & Error Handling ---
app.notFound((c) => c.json({ error: "接口不存在" }, 404));

app.onError((err, c) => {
  if (err instanceof HTTPException) {
    if (err.status >= 500) {
      logger.error(
        {
          err: { message: err.message, stack: err.stack },
          requestId: c.get("requestId"),
          path: c.req.path,
          method: c.req.method,
          status: err.status,
        },
        "HTTP exception",
      );
    } else if (err.status !== 401 && err.status !== 403) {
      logger.warn(
        {
          requestId: c.get("requestId"),
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
      requestId: c.get("requestId"),
      path: c.req.path,
      method: c.req.method,
    },
    "Unhandled error",
  );
  return c.json({ error: "服务器内部错误" }, 500);
});

const RUNTIME_DISK_CLEANUP_INTERVAL_MS = 24 * 60 * 60 * 1000;
let runtimeDiskCleanupTimer: ReturnType<typeof setInterval> | null = null;

const longRunningRequestPathPattern = /^(?:\/api\/v1\/tools\/adf|\/api\/v1\/models\/[^/]+\/(?:predict|(?:guided-training|production-preparation)\/sessions(?:\/[^/]+\/steps\/[^/]+\/run)?))$/;

export async function runGuidedTrainingSessionCleanup(trigger = "manual") {
  try {
    const result = await cleanupExpiredGuidedTrainingSessions();
    if (result.deletedSessions > 0 || result.deletedArtifactDirs > 0) {
      logger.info(
        { trigger, ...result },
        "已清理过期 guided training 会话",
      );
    }
    return result;
  } catch (error) {
    logger.warn(
      {
        trigger,
        err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
      },
      "清理过期 guided training 会话失败",
    );
    return null;
  }
}

const runCleanupStep = async <T>(
  trigger: string,
  name: string,
  cleanup: () => Promise<T>,
): Promise<T | null> => {
  try {
    const result = await cleanup();
    logger.info({ trigger, cleanup: name, result }, "运行时磁盘清理完成");
    return result;
  } catch (error) {
    logger.warn(
      {
        trigger,
        cleanup: name,
        err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
      },
      "运行时磁盘清理失败",
    );
    return null;
  }
};

interface RuntimeDiskCleanupOptions {
  skipDeletionJournals?: boolean;
}

const assertDeletionJournalResult = (
  name: string,
  result: Awaited<ReturnType<typeof reconcileDatasetDeletionJournals>>,
) => {
  if (result.failedJournals > 0 || result.skippedJournals > 0) {
    throw new Error(
      `${name} reconciliation left unresolved journals (failed=${result.failedJournals}, skipped=${result.skippedJournals}): ${result.failedJournalNames.join(", ")}`,
    );
  }
  return result;
};

async function reconcileDeletionJournalsStrictly() {
  const datasetDeletionJournals = assertDeletionJournalResult(
    "dataset deletion journal",
    await reconcileDatasetDeletionJournals(),
  );
  const reportDeletionJournals = assertDeletionJournalResult(
    "report deletion journal",
    await reconcileReportDeletionJournals(),
  );
  const manualDeletionJournals = assertDeletionJournalResult(
    "manual deletion journal",
    await reconcileManualDeletionJournals(),
  );
  return { datasetDeletionJournals, reportDeletionJournals, manualDeletionJournals };
}

const STARTUP_DELETION_RECONCILIATION_ATTEMPTS = 3;

export async function runStartupDeletionJournalReconciliation() {
  let lastError: unknown;
  for (let attempt = 1; attempt <= STARTUP_DELETION_RECONCILIATION_ATTEMPTS; attempt += 1) {
    try {
      return await reconcileDeletionJournalsStrictly();
    } catch (error) {
      lastError = error;
      logger.warn(
        {
          attempt,
          maxAttempts: STARTUP_DELETION_RECONCILIATION_ATTEMPTS,
          err: error instanceof Error ? { message: error.message, stack: error.stack } : { message: String(error) },
        },
        "启动时恢复删除日志失败",
      );
      if (attempt < STARTUP_DELETION_RECONCILIATION_ATTEMPTS) {
        await new Promise((resolve) => setTimeout(resolve, attempt * 100));
      }
    }
  }
  throw lastError instanceof Error
    ? lastError
    : new Error(`启动时恢复删除日志失败: ${String(lastError)}`);
}

export async function runRuntimeDiskCleanup(
  trigger = "manual",
  options: RuntimeDiskCleanupOptions = {},
) {
  // Durable transactions are recovery work, not retention work. Always settle
  // them before another cleanup can remove or rewrite related artifacts.
  const datasetDeletionJournals = options.skipDeletionJournals
    ? null
    : await runCleanupStep(
      trigger,
      "dataset-deletion-journals",
      reconcileDatasetDeletionJournals,
    );
  const reportDeletionJournals = options.skipDeletionJournals
    ? null
    : await runCleanupStep(
      trigger,
      "report-deletion-journals",
      reconcileReportDeletionJournals,
    );
  const manualDeletionJournals = options.skipDeletionJournals
    ? null
    : await runCleanupStep(
      trigger,
      "manual-deletion-journals",
      reconcileManualDeletionJournals,
    );
  const deletionRecoveryComplete = options.skipDeletionJournals || (
    datasetDeletionJournals !== null
    && reportDeletionJournals !== null
    && manualDeletionJournals !== null
    && datasetDeletionJournals.failedJournals === 0
    && datasetDeletionJournals.skippedJournals === 0
    && reportDeletionJournals.failedJournals === 0
    && reportDeletionJournals.skippedJournals === 0
    && manualDeletionJournals.failedJournals === 0
    && manualDeletionJournals.skippedJournals === 0
  );
  if (!deletionRecoveryComplete) {
    logger.warn(
      { trigger, datasetDeletionJournals, reportDeletionJournals, manualDeletionJournals },
      "删除日志尚未完全恢复，本轮跳过后续保留期清理",
    );
    return {
      guidedTraining: null,
      productionPreparation: null,
      exportFiles: null,
      uploadTempFiles: null,
      modelArtifacts: null,
      datasetDeletionJournals,
      reportDeletionJournals,
      manualDeletionJournals,
    };
  }
  const guidedTraining = shouldSkipGuidedTrainingCleanupForTrigger(trigger)
    ? null
    : await runGuidedTrainingSessionCleanup(trigger);
  const productionPreparation = await runCleanupStep(
    trigger,
    "production-preparation",
    cleanupExpiredProductionPreparationSessions,
  );
  const exportFiles = await runCleanupStep(trigger, "export-files", cleanupExpiredExportFiles);
  const uploadTempFiles = await runCleanupStep(trigger, "upload-temp-files", cleanupStaleUploadTempFiles);
  const modelArtifacts = await runCleanupStep(trigger, "model-artifacts", cleanupStaleModelArtifacts);

  return {
    guidedTraining,
    productionPreparation,
    exportFiles,
    uploadTempFiles,
    modelArtifacts,
    datasetDeletionJournals,
    reportDeletionJournals,
    manualDeletionJournals,
  };
}

export function startRuntimeDiskCleanupSchedule() {
  if (runtimeDiskCleanupTimer) {
    return;
  }

  if (isRuntimeDiskCleanupTimerDisabled()) {
    logger.info("运行时磁盘定时清理已通过环境变量禁用");
    return;
  }

  if (hasLegacyGuidedTrainingCleanupTimerDisable()) {
    logger.info("DISABLE_GUIDED_TRAINING_CLEANUP_TIMER 已废弃：仅跳过 interval 触发的 guided training 清理，其他运行时磁盘清理仍会执行");
  }

  runtimeDiskCleanupTimer = setInterval(() => {
    void runRuntimeDiskCleanup("interval");
  }, RUNTIME_DISK_CLEANUP_INTERVAL_MS);
  runtimeDiskCleanupTimer.unref?.();
}

export function startGuidedTrainingSessionCleanupSchedule() {
  startRuntimeDiskCleanupSchedule();
}

const resolveBunIdleTimeoutSeconds = () => {
  const rawValue = process.env.BUN_IDLE_TIMEOUT_SECONDS?.trim() || "255";
  const parsed = Number(rawValue);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 255) {
    logger.warn(
      { rawValue, fallback: 255 },
      "Invalid BUN_IDLE_TIMEOUT_SECONDS; using 255 seconds",
    );
    return 255;
  }
  return parsed;
};

export const serverConfig = {
  hostname: process.env.HOST?.trim() || "127.0.0.1",
  port: env.PORT,
  idleTimeout: resolveBunIdleTimeoutSeconds(),
  fetch(request: Request, server: Bun.Server<undefined>) {
    const { pathname } = new URL(request.url);
    if (longRunningRequestPathPattern.test(pathname)) {
      server.timeout(request, 0);
    }
    // Model services intentionally do not consume request.signal. Accepted work
    // continues after a browser disconnect and is bounded by its own deadline.
    return app.fetch(request);
  },
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
  // ---- 启动校验：单实例查询/事务连接池容量 ----
  const minRequired = DB_POOL_CONFIG.MIN_CONNECTION_LIMIT;
  if (connectionLimit < minRequired) {
    logger.fatal(
      {
        deploymentMode: "single_instance",
        connectionLimit,
        maxConcurrentModelJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS,
        minRequired,
      },
      "数据库连接池容量不足：单实例课堂并发下 DB_CONNECTION_LIMIT 不能低于计算出的最小查询/事务容量",
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

  // Startup is fail-closed: unlike the interval runner, a durable journal that
  // remains unreadable after retries prevents the service from accepting new
  // requests against an unresolved DB/filesystem state.
  await runStartupDeletionJournalReconciliation();
  await runRuntimeDiskCleanup("startup", { skipDeletionJournals: true });
  startRuntimeDiskCleanupSchedule();
}

if (import.meta.main) {
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

  // Bind before any database reconciliation or filesystem cleanup. A second
  // process configured for the required listener fails here, before two online
  // runtimes can touch process-coordinated state concurrently.
  let applicationReady = false;
  const server = Bun.serve({
    ...serverConfig,
    fetch(request, bunServer) {
      if (!applicationReady) {
        return new Response("Service startup in progress", {
          status: 503,
          headers: { "Retry-After": "1" },
        });
      }
      return serverConfig.fetch(request, bunServer);
    },
  });

  try {
    await runServerStartupPreflight();
    applicationReady = true;
    logger.info(
      { deploymentMode: "single_instance", coordinationScope: "process" },
      `Server listening on ${server.url.href}`,
    );
  } catch (error) {
    server.stop(true);
    throw error;
  }
}

export { app };
