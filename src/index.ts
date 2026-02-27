import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { cors } from "hono/cors";
import { serveStatic } from "hono/bun";
import path from "path";
import type { RowDataPacket } from "mysql2/promise";
import { env, pool } from "./config";
import { jwtAuth } from "./middleware";
import logger from "./utils/logger";
import type { JwtPayload, UserRole } from "./types";
// Import new modular controllers
import { authController } from "./modules/auth";
import { toolsController } from "./modules/tools";
import { manualsController } from "./modules/manuals";
import { questionBankController } from "./modules/question-bank";
import { quizzesController } from "./modules/quizzes";
import { reportsController } from "./modules/reports";
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

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSafeFilename(filename: string): boolean {
  return filename === path.basename(filename) && !filename.includes("\0");
}

function toHeaderFilename(filename: string): string {
  return filename.replace(/["\\]/g, "_");
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
app.get("/manuals/:filename", jwtAuth, async (c) => {
  const payload = c.get("jwtPayload") as JwtPayload;
  const filename = c.req.param("filename");

  if (!isSafeFilename(filename)) return c.json({ error: "无效的文件名" }, 400);

  const access = await checkManualFileAccess(filename, payload.role);
  if (access === "not_found") return c.json({ error: "文件不存在" }, 404);
  if (access === "forbidden") return c.json({ error: "权限不足" }, 403);

  const file = Bun.file(path.resolve(uploadsRoot, "manuals", filename));
  if (!(await file.exists())) return c.json({ error: "文件不存在" }, 404);

  return new Response(file, {
    headers: {
      "Content-Type": file.type || "application/pdf",
      "Content-Disposition": `inline; filename="${toHeaderFilename(filename)}"`,
      "Cache-Control": "private, max-age=60",
    },
  });
});

app.get("/datasets/:filename", jwtAuth, async (c) => {
  const payload = c.get("jwtPayload") as JwtPayload;
  if (payload.role !== "Admin") return c.json({ error: "权限不足" }, 403);

  const filename = c.req.param("filename");
  if (!isSafeFilename(filename)) return c.json({ error: "无效的文件名" }, 400);

  const file = Bun.file(path.resolve(uploadsRoot, "datasets", filename));
  if (!(await file.exists())) return c.json({ error: "文件不存在" }, 404);

  return new Response(file, {
    headers: {
      "Content-Type": file.type || "text/csv; charset=UTF-8",
      "Content-Disposition": `attachment; filename="${toHeaderFilename(filename)}"`,
      "Cache-Control": "private, max-age=60",
    },
  });
});
app.get("/reports/:filename", jwtAuth, async (c) => {
  const payload = c.get("jwtPayload") as JwtPayload;
  const filename = c.req.param("filename");

  if (!isSafeFilename(filename)) return c.json({ error: "无效的文件名" }, 400);

  const allowed = await canAccessReportFile(filename, payload.sub, payload.role);
  if (!allowed) return c.json({ error: "权限不足" }, 403);

  const file = Bun.file(path.resolve(reportsRoot, filename));
  if (!(await file.exists())) return c.json({ error: "文件不存在" }, 404);

  return new Response(file, {
    headers: {
      "Content-Type": file.type || "application/pdf",
      "Content-Disposition": `inline; filename="${toHeaderFilename(filename)}"`,
      "Cache-Control": "private, max-age=60",
    },
  });
});

app.get("/exports/:filename", jwtAuth, async (c) => {
  const payload = c.get("jwtPayload") as JwtPayload;
  const filename = c.req.param("filename");

  if (!isSafeFilename(filename)) return c.json({ error: "无效的文件名" }, 400);

  const allowed = await canAccessExportFile(filename, payload.sub, payload.role);
  if (!allowed) return c.json({ error: "权限不足" }, 403);

  const file = Bun.file(path.resolve(exportsRoot, filename));
  if (!(await file.exists())) return c.json({ error: "文件不存在" }, 404);

  return new Response(file, {
    headers: {
      "Content-Type": file.type || "application/octet-stream",
      "Content-Disposition": `attachment; filename="${toHeaderFilename(filename)}"`,
      "Cache-Control": "private, max-age=60",
    },
  });
});
app.use(
  "/js/*",
  serveStatic({
    root: path.resolve(import.meta.dir, "../cdn"),
  }),
);
app.use(
  "/images/*",
  serveStatic({
    root: path.resolve(import.meta.dir, "../cdn"),
  }),
);

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
  logger.error(
    {
      err: err instanceof Error ? { message: err.message, stack: err.stack } : err,
      path: c.req.path,
      method: c.req.method,
    },
    "Unhandled error",
  );
  if (err instanceof HTTPException) {
    return c.json({ error: err.message }, err.status);
  }
  return c.json({ error: "服务器内部错误" }, 500);
});

// --- Server Startup ---
logger.info(`Server listening on http://localhost:${env.PORT}`);

export default {
  port: env.PORT,
  fetch: app.fetch,
};
