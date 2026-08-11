import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireStudent } from "../../middleware";
import { experimentIdParamSchema, submitReportSchema } from "./reports.schema";
import {
  submitReport,
  getLatestReportStatus,
  HTMLSanitizationError,
  PdfGenerationTimeoutError,
} from "./reports.service";
import { SemaphoreTimeoutError } from "../../utils/semaphore";
import { handleUnexpectedError } from "../../utils/error-handler";
import type { JwtPayload } from "../../types";

const reportsController = new Hono();

reportsController.post(
  "/api/v1/experiment-runs/:experimentId/report",
  jwtAuth,
  requireStudent,
  zValidator("param", experimentIdParamSchema),
  zValidator("json", submitReportSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { experimentId } = c.req.valid("param");
      const { report_content } = c.req.valid("json");

      const result = await submitReport(
        parseInt(experimentId, 10),
        payload.sub,
        report_content,
      );

      if ("error" in result) {
        return c.json({ error: result.error }, result.status);
      }

      return c.json(result.data, result.isUpdate ? 200 : 201);
    } catch (error) {
      if (error instanceof SemaphoreTimeoutError) {
        return c.json({ error: `PDF 生成服务繁忙，排队等待超过 ${Math.ceil(error.timeoutMs / 1000)} 秒，请稍后再试` }, 503);
      }
      if (error instanceof PdfGenerationTimeoutError) {
        return c.json({ error: "PDF 生成超时，请稍后重试" }, 504);
      }
      if (error instanceof HTMLSanitizationError) {
        return c.json({ error: "无效的报告内容： " + error.message }, 400);
      }
      return handleUnexpectedError(c, error, "处理报告时发生内部服务器错误。");
    }
  },
);

reportsController.get(
  "/api/v1/my-latest-report-status",
  jwtAuth,
  requireStudent,
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const result = await getLatestReportStatus(payload.sub);
      return c.json(result);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取报告状态失败");
    }
  },
);

export default reportsController;
