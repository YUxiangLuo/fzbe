import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { cors } from "hono/cors";
import { serveStatic } from "hono/bun";
import path from "path";
import { env, pool } from "./config";
import logger from "./logger";
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

// --- Global Middleware ---
// app.use(timestampedLogger);
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
  "/exports/*",
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
app.notFound((c) => c.json({ message: "Not Found" }, 404));

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
  return c.json({ error: "Internal Server Error" }, 500);
});

// --- Server Startup ---
logger.info(`Server listening on http://localhost:${env.PORT}`);

export default {
  port: env.PORT,
  fetch: app.fetch,
};
