import { Hono } from "hono";
import { connectionLimit, env } from "../../config";
import { jwtAuth, requireAdmin } from "../../middleware";
import {
  ADF_HTTP_TIMEOUT_BUDGETS,
  CONCURRENCY_LIMITS,
  DB_POOL_CONFIG,
  MODEL_HTTP_TIMEOUT_BUDGETS,
  PYTHON_RUNTIME_CONFIG,
  TIMEOUTS,
} from "../../utils/constants";
import { getModelExecutionLockStats } from "../models/model-execution-lock";
import { getModelOperationStats } from "../models/model-operation-lease";
import { getPdfQueueStats } from "../reports/reports.service";

const runtimeController = new Hono();

const buildRuntimeInfo = () => ({
  app: "fangzhen-be",
  deploymentMode: "single_instance" as const,
  coordinationScope: "process" as const,
  e2eServer: process.env.E2E_SERVER === "1",
  database: env.DB_DATABASE,
  port: env.PORT,
  concurrency: {
    systemParallelism: CONCURRENCY_LIMITS.SYSTEM_PARALLELISM,
    maxConcurrentModelJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS,
    maxConcurrentPdfJobs: CONCURRENCY_LIMITS.MAX_CONCURRENT_PDF_JOBS,
    dbConnectionLimit: connectionLimit,
    dbMinimumConnectionLimit: DB_POOL_CONFIG.MIN_CONNECTION_LIMIT,
    pythonModelProcessThreads: PYTHON_RUNTIME_CONFIG.MODEL_PROCESS_THREADS,
    pythonTensorflowInterOpThreads: PYTHON_RUNTIME_CONFIG.TF_INTER_OP_THREADS,
  },
  timeouts: {
    modelExecutionHardCapMs: TIMEOUTS.MODEL_EXECUTION_HARD_CAP,
    modelSingleOperationMs: TIMEOUTS.MODEL_SINGLE_OPERATION,
    modelSingleOperationMaxMs: TIMEOUTS.MODEL_SINGLE_OPERATION_MAX,
    modelOperationCleanupBudgetMs: TIMEOUTS.MODEL_OPERATION_CLEANUP_BUDGET,
    modelFailureRecordMs: TIMEOUTS.MODEL_FAILURE_RECORD,
    modelNginxReadMs: MODEL_HTTP_TIMEOUT_BUDGETS.NGINX_READ,
    modelBrowserRequestMs: MODEL_HTTP_TIMEOUT_BUDGETS.BROWSER_REQUEST,
    modelProcessTerminationGraceMs: TIMEOUTS.MODEL_PROCESS_TERMINATION_GRACE,
    adfExecutionMs: TIMEOUTS.ADF,
    adfOperationMs: TIMEOUTS.ADF_OPERATION,
    adfOperationCleanupBudgetMs: TIMEOUTS.ADF_OPERATION_CLEANUP_BUDGET,
    adfNginxReadMs: ADF_HTTP_TIMEOUT_BUDGETS.NGINX_READ,
    adfBrowserRequestMs: ADF_HTTP_TIMEOUT_BUDGETS.BROWSER_REQUEST,
    modelTrainingMs: TIMEOUTS.MODEL_TRAINING,
    modelPredictionMs: TIMEOUTS.MODEL_PREDICTION,
    pdfQueueMs: TIMEOUTS.PDF_QUEUE,
    pdfGenerationMs: TIMEOUTS.PDF_GENERATION,
  },
  modelOperations: getModelOperationStats(),
  modelExecutionLocks: getModelExecutionLockStats(),
  pdfQueue: getPdfQueueStats(),
});

runtimeController.get(
  "/api/v1/health",
  (c) => c.json({ app: "fangzhen-be", status: "ok" }),
);

runtimeController.get(
  "/api/v1/client-config",
  jwtAuth,
  (c) => c.json({
    reportSubmission: {
      pdfQueueMs: TIMEOUTS.PDF_QUEUE,
      pdfGenerationMs: TIMEOUTS.PDF_GENERATION,
    },
  }),
);

if (process.env.E2E_SERVER === "1") {
  // E2E startup and concurrency guards must inspect the exact disposable
  // database/capacity before test credentials exist. Production rejects
  // E2E_SERVER at environment loading and never exposes this variant.
  runtimeController.get("/api/v1/runtime-info", (c) => c.json(buildRuntimeInfo()));
} else {
  runtimeController.get(
    "/api/v1/runtime-info",
    jwtAuth,
    requireAdmin,
    (c) => c.json(buildRuntimeInfo()),
  );
}

export default runtimeController;
