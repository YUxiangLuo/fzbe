import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireStudent } from "../../middleware";
import { adfRequestSchema } from "./tools.schema";
import { performAdfTest } from "./tools.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import { AdfValidationError, type AdfInputPoint } from "../../utils/adf";
import { acquireModelSlot, releaseModelSlot } from "../models/model-locks";

const toolsController = new Hono();

toolsController.post(
  "/api/v1/tools/adf",
  jwtAuth,
  requireStudent,
  zValidator("json", adfRequestSchema),
  async (c) => {
    const { series } = c.req.valid("json") as { series: AdfInputPoint[] };

    let slotLock = null;
    try {
      slotLock = await acquireModelSlot();
      if (!slotLock) {
        return c.json({ error: "模型服务繁忙，请稍后再试" }, 429);
      }
      const result = await performAdfTest(series);
      return c.json({ result });
    } catch (error) {
      if (error instanceof AdfValidationError) {
        return c.json({ error: error.message }, 400);
      }
      return handleUnexpectedError(c, error, "ADF 检测失败");
    } finally {
      await releaseModelSlot(slotLock);
    }
  },
);

export default toolsController;
