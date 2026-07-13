import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireStudent } from "../../middleware";
import { adfRequestSchema } from "./tools.schema";
import { performAdfTest } from "./tools.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import { AdfValidationError, type AdfInputPoint } from "../../utils/adf";
import {
  ModelCapacityBusyError,
  ModelOperationDeadlineError,
  ModelServiceUnavailableError,
  ModelTimeoutError,
} from "../models/model-errors";

const toolsController = new Hono();

toolsController.post(
  "/api/v1/tools/adf",
  jwtAuth,
  requireStudent,
  zValidator("json", adfRequestSchema),
  async (c) => {
    const { series } = c.req.valid("json") as { series: AdfInputPoint[] };

    try {
      const result = await performAdfTest(series);
      return c.json({ result });
    } catch (error) {
      if (error instanceof AdfValidationError) {
        return c.json({ error: error.message }, 400);
      }
      if (error instanceof ModelCapacityBusyError) {
        c.header("Retry-After", String(error.retryAfterSeconds));
        return c.json({
          error: error.message,
          code: error.code,
          retryable: error.retryable,
        }, 429);
      }
      if (error instanceof ModelOperationDeadlineError) {
        return c.json({
          error: "ADF 检测超过服务端时限，请重试",
          code: error.code,
          operation: error.operation,
          timeout_ms: error.timeoutMs,
          retryable: error.retryable,
        }, 504);
      }
      if (error instanceof ModelTimeoutError) {
        return c.json({
          error: "ADF 检测执行超时，请重试",
          code: error.code,
          operation: "adf",
          timeout_ms: error.timeoutMs,
          retryable: true,
        }, 504);
      }
      if (error instanceof ModelServiceUnavailableError) {
        return c.json({ error: error.message, code: error.code }, 503);
      }
      return handleUnexpectedError(c, error, "ADF 检测失败");
    }
  },
);

export default toolsController;
