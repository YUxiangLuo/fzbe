import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireStudent } from "../../middleware";
import { trainSchema, predictSchema, isModelType } from "./models.schema";
import {
  trainModel,
  predictExperimentModel,
  acquireModelSlot,
  releaseModelSlot,
  acquireModelExecutionLock,
  releaseModelExecutionLock,
  ModelValidationError,
  ModelNotFoundError,
  ModelAccessError,
} from "./models.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import type { JwtPayload } from "../../types";
import { TIMEOUTS } from "../../utils/constants";

const modelsController = new Hono();

modelsController.post(
  "/api/v1/models/:model_type/training",
  jwtAuth,
  requireStudent,
  zValidator("json", trainSchema),
  async (c) => {
    const modelTypeParam = c.req.param("model_type");
    if (!isModelType(modelTypeParam)) {
      return c.json({ error: "指定的模型类型无效" }, 400);
    }
    const modelType = modelTypeParam;

    const payload = c.get("jwtPayload") as JwtPayload;
    const body = c.req.valid("json");

    let executionLock = null;
    let slotLock = null;

    try {
      executionLock = await acquireModelExecutionLock(body.experiment_id, modelType);
      if (!executionLock) {
        return c.json({ error: "同一模型正在训练或预测，请稍后再试" }, 409);
      }

      slotLock = await acquireModelSlot();
      if (!slotLock) {
        return c.json({ error: "模型服务繁忙，请稍后再试" }, 429);
      }

      const result = await trainModel(modelType, payload.sub, body);
      return c.json(result);
    } catch (error: unknown) {
      if (error instanceof ModelAccessError) {
        return c.json({ error: error.message }, error.status);
      }
      if (error instanceof ModelValidationError) {
        return c.json({ error: error.message, ...error.details }, 400);
      }
      const err = error as Error;
      if (err.message?.includes("timed out") || err.message?.includes("超时")) {
        const timeoutSeconds = Math.ceil(TIMEOUTS.MODEL_TRAINING / 1000);
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
      await releaseModelSlot(slotLock);
      await releaseModelExecutionLock(executionLock);
    }
  },
);

modelsController.post(
  "/api/v1/models/:model_type/predict",
  jwtAuth,
  requireStudent,
  zValidator("json", predictSchema),
  async (c) => {
    const modelTypeParam = c.req.param("model_type");
    if (!isModelType(modelTypeParam)) {
      return c.json({ error: "指定的模型类型无效" }, 400);
    }
    const modelType = modelTypeParam;

    const payload = c.get("jwtPayload") as JwtPayload;
    const body = c.req.valid("json");

    let executionLock = null;
    let slotLock = null;

    try {
      executionLock = await acquireModelExecutionLock(body.experiment_id, modelType);
      if (!executionLock) {
        return c.json({ error: "同一模型正在训练或预测，请稍后再试" }, 409);
      }

      slotLock = await acquireModelSlot();
      if (!slotLock) {
        return c.json({ error: "模型服务繁忙，请稍后再试" }, 429);
      }

      const result = await predictExperimentModel(
        modelType,
        payload.sub,
        body.experiment_id,
        body.forecast_steps,
      );
      return c.json(result);
    } catch (error: unknown) {
      if (error instanceof ModelAccessError) {
        return c.json({ error: error.message }, error.status);
      }
      if (error instanceof ModelNotFoundError) {
        return c.json({ error: error.message }, 404);
      }
      const err = error as Error;
      if (err.message?.includes("timed out") || err.message?.includes("超时")) {
        const timeoutSeconds = Math.ceil(TIMEOUTS.MODEL_PREDICTION / 1000);
        return c.json({ error: `预测超时（${timeoutSeconds}秒）` }, 504);
      }
      const message = err.message || `${modelType.toUpperCase()} 预测失败`;
      return handleUnexpectedError(c, error, message);
    } finally {
      await releaseModelSlot(slotLock);
      await releaseModelExecutionLock(executionLock);
    }
  },
);

export default modelsController;
