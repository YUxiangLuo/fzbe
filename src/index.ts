import { Hono } from "hono";
import type { Context, Next } from "hono";
import { HTTPException } from "hono/http-exception";
import { cors } from "hono/cors";
import mysql from "mysql2/promise";
import { serveStatic } from "hono/bun";
import path from "path";
// Import route modules
import authRoutes from "./routes/auth";
import userRoutes from "./routes/users";
import classRoutes from "./routes/classes";
import classMembersRoutes from "./routes/class-members";
import classGradesRoutes from "./routes/class-grades";
import manualRoutes from "./routes/manuals";
import datasetRoutes from "./routes/datasets";
import questionBankRoutes from "./routes/question-bank";
import quizzesRoutes from "./routes/quizzes";
import reportsRoutes from "./routes/reports";
import modelsApi from "./routes/models";
import experimentsRoutes from "./routes/experiments";
import toolsRoutes from "./routes/tools";

// --- Custom Logger Middleware ---
const timestampedLogger = async (c: Context, next: Next) => {
  const start = Date.now();
  await next();
  const duration = Date.now() - start;
  const timestamp = new Date().toISOString();
  console.log(
    `[${timestamp}] ${c.req.method} ${c.req.path} | ${c.res.status} | ${duration}ms`,
  );
};

// --- Environment Configuration & DB Connection ---
const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE, JWT_SECRET, PORT } =
  process.env;

if (!JWT_SECRET) {
  throw new Error("JWT_SECRET environment variable not set!");
}

const pool = mysql.createPool({
  host: DB_HOST,
  user: DB_USER,
  password: DB_PASSWORD,
  database: DB_DATABASE,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  charset: 'utf8mb4'
});

console.log("Database connection pool created.");

// --- Hono App Initialization ---
const app = new Hono();

// --- Global Middleware ---
app.use(timestampedLogger);
app.use(cors());
app.use(
  "/manuals/*",
  serveStatic({
    root: path.resolve(import.meta.dir, "../uploads"),
  }),
);
app.use(
  "/datasets/*",
  serveStatic({
    root: path.resolve(import.meta.dir, "../uploads"),
  }),
);
app.use(
  "/reports/*",
  serveStatic({
    root: path.resolve(import.meta.dir, "../uploads"),
  }),
);
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
// All routes now use /api/ prefix directly, so we don't need a nested structure
app.route("/", authRoutes(pool, JWT_SECRET));
app.route("/", userRoutes(pool, JWT_SECRET));
app.route("/", classRoutes(pool, JWT_SECRET));
app.route("/", classMembersRoutes(pool, JWT_SECRET));
app.route("/", classGradesRoutes(pool, JWT_SECRET));
app.route("/", manualRoutes(pool, JWT_SECRET));
app.route("/", datasetRoutes(pool, JWT_SECRET));
app.route("/", questionBankRoutes(pool, JWT_SECRET));
app.route("/", quizzesRoutes(pool, JWT_SECRET));
app.route("/", reportsRoutes(pool, JWT_SECRET));
app.route("/", modelsApi(pool, JWT_SECRET));
app.route("/", experimentsRoutes(pool, JWT_SECRET));
app.route("/", toolsRoutes(pool, JWT_SECRET));

// --- 404 & Error Handling ---
app.notFound((c) => c.json({ message: "Not Found" }, 404));

app.onError((err, c) => {
  console.error("Global error handler:", err);
  if (err instanceof HTTPException) {
    return c.json({ error: err.message }, err.status);
  }
  return c.json({ error: "Internal Server Error" }, 500);
});

// --- Server Startup ---
const port = PORT ? parseInt(PORT, 10) : 3000;
console.log(`Server listening on http://localhost:${port}`);

export default {
  port,
  fetch: app.fetch,
};
