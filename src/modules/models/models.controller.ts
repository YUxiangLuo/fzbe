import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireStudent } from "../../middleware";
import { trainSchema, predictSchema, ALL_MODELS, type ModelType } from "./models.schema";
import { trainModel, predictModel, tryAcquireModelSlot, releaseModelSlot } from "./models.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import type { JwtPayload } from "../../types";

const modelsController = new Hono();

modelsController.post(
  "/api/v1/models/:model_type/training",
  jwtAuth,
  requireStudent,
  zValidator("json", trainSchema),
  async (c) => {
    const modelType = c.req.param("model_type") as ModelType;
    if (!ALL_MODELS.includes(modelType)) {
      return c.json({ error: "指定的模型类型无效" }, 400);
    }

    const payload = c.get("jwtPayload") as JwtPayload;
    const body = c.req.valid("json");

    if (!tryAcquireModelSlot()) {
      return c.json({ error: "模型服务繁忙，请稍后再试" }, 429);
    }

    try {
      const result = await trainModel(modelType, payload.sub, body);

      if ("error" in result && "dataset_size" in result) {
        return c.json({ error: result.error, dataset_size: result.dataset_size }, 400);
      }

      return c.json(result);
    } catch (error: unknown) {
      const err = error as Error;
      if (err.message?.includes("timed out")) {
        const isEnsembleModel = ["stacking", "weighted_avg", "boosting"].includes(modelType);
        const timeoutSeconds = isEnsembleModel ? 120 : 30;
        return c.json({ error: `训练超时（${timeoutSeconds}秒）` }, 504);
      }
      if (err.message?.startsWith("[PYTHON_VALIDATION]")) {
        return c.json({ error: err.message.substring(20) }, 400);
      }
      if (err.message?.startsWith("[PYTHON_DATA]")) {
        return c.json({ error: err.message.substring(14) }, 400);
      }
      const message = err.message || `${modelType.toUpperCase()} 训练失败`;
      return handleUnexpectedError(c, error, message);
    } finally {
      releaseModelSlot();
    }
  },
);

modelsController.post(
  "/api/v1/models/:model_type/predict",
  jwtAuth,
  zValidator("json", predictSchema),
  async (c) => {
    const modelType = c.req.param("model_type") as ModelType;
    if (!ALL_MODELS.includes(modelType)) {
      return c.json({ error: "指定的模型类型无效" }, 400);
    }

    const payload = c.get("jwtPayload") as JwtPayload;
    const body = c.req.valid("json");

    if (!tryAcquireModelSlot()) {
      return c.json({ error: "模型服务繁忙，请稍后再试" }, 429);
    }

    try {
      const result = await predictModel(modelType, payload.sub, body.forecast_steps);

      if ("error" in result) {
        return c.json({ error: result.error }, 404);
      }

      return c.json(result);
    } catch (error: unknown) {
      const err = error as Error;
      const message = err.message || `${modelType.toUpperCase()} 预测失败`;
      return handleUnexpectedError(c, error, message);
    } finally {
      releaseModelSlot();
    }
  },
);

export default modelsController;
