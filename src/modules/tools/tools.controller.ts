import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth } from "../../middleware";
import { adfRequestSchema } from "./tools.schema";
import { performAdfTest } from "./tools.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import { AdfValidationError, type AdfInputPoint } from "../../utils/adf";

const toolsController = new Hono();

toolsController.post(
  "/api/v1/tools/adf",
  jwtAuth,
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
      return handleUnexpectedError(c, error, "ADF 检测失败");
    }
  },
);

export default toolsController;
