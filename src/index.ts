import { Hono } from "hono";
import type { Context, Next } from "hono";
import { HTTPException } from "hono/http-exception";
import { cors } from "hono/cors";
import mysql from "mysql2/promise";
import { serveStatic } from "hono/bun";
import path from "path";
// Import route modules
import userRoutes from "./routes/users";
import classRoutes from "./routes/classes";
import manualRoutes from "./routes/manuals";
import datasetRoutes from "./routes/datasets";
import forecastApi from "./routes/forecasting";

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
});

console.log("Database connection pool created.");

// --- Hono App Initialization ---
const app = new Hono();

// --- Global Middleware ---
app.use(timestampedLogger);
app.use(cors());
app.use(
  "/static/*",
  serveStatic({
    root: path.resolve(import.meta.dir, "../uploads"),
    onNotFound: (path, c) => {
      console.log(`${path} is not found, you access ${c.req.path}`);
    },
  }),
);

// --- Mount Routes ---
const api = new Hono();

api.route("/", userRoutes(pool, JWT_SECRET));
api.route("/", classRoutes(pool, JWT_SECRET));
api.route("/", manualRoutes(pool, JWT_SECRET));
api.route("/", datasetRoutes(pool, JWT_SECRET));
api.route("/", forecastApi(JWT_SECRET)); // Forecasting routes are already prefixed with /forecast and /tools

app.route("/api", api);

// --- 404 & Error Handling ---
app.notFound((c) => c.json({ message: "Not Found" }, 404));

app.onError((err, c) => {
  console.error("Global error handler:", err);
  if (err instanceof HTTPException) {
    return err.getResponse();
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
