import { Hono } from "hono";
import { jwtAuth, requireStudent } from "../../middleware";
import { predictSchema, prepareProductionSchema, isModelType, getTrainSchemaForModel, type ModelType } from "./models.schema";
import {
  buildLocalizedPredictValidationDetails,
  buildLocalizedTrainValidationDetails,
  localizePredictValidationIssue,
  localizeTrainValidationIssue,
} from "./models.validation";
import {
  trainModel,
  createOrResumeGuidedTrainingSession,
  getGuidedTrainingSession,
  runGuidedTrainingStep,
  getReusableProductionModelArtifactResult,
  prepareExperimentModelForProduction,
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
const SLOT_FREE_REUSABLE_PRODUCTION_MODEL_TYPES = new Set<ModelType>(["ma", "es", "arima", "lstm"]);

type StreamSend = (event: object) => void;

const TRAINING_STREAM_HEARTBEAT_INTERVAL_MS = 5000;

const createNdjsonStreamResponse = (run: (send: StreamSend) => Promise<void>) => {
  const encoder = new TextEncoder();
  let closed = false;
  let heartbeatTimer: ReturnType<typeof setInterval> | undefined;

  const stopHeartbeat = () => {
    if (!heartbeatTimer) return;
    clearInterval(heartbeatTimer);
    heartbeatTimer = undefined;
  };

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const send: StreamSend = (event) => {
        if (closed) return;
        try {
          controller.enqueue(
            encoder.encode(`${JSON.stringify({ timestamp: new Date().toISOString(), ...event })}\n`),
          );
        } catch {
          closed = true;
          stopHeartbeat();
        }
      };

      heartbeatTimer = setInterval(() => {
        if (closed) {
          stopHeartbeat();
          return;
        }
        send({
          type: "heartbeat",
          source: "backend",
          message: "模型训练仍在进行中",
        });
      }, TRAINING_STREAM_HEARTBEAT_INTERVAL_MS);

      void (async () => {
        try {
          await run(send);
        } finally {
          closed = true;
          stopHeartbeat();
          try {
            controller.close();
          } catch {
            // The client may have already closed the stream.
          }
        }
      })();
    },
    cancel() {
      closed = true;
      stopHeartbeat();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
};

const normalizeTrainingStreamError = (error: unknown, modelType: ModelType) => {
  if (error instanceof ModelAccessError) {
    return { status: error.status, message: error.message };
  }
  if (error instanceof ModelValidationError) {
    return { status: 400, message: error.message, details: error.details };
  }

  const err = error as Error;
  if (err.message?.includes("timed out") || err.message?.includes("超时")) {
    const timeoutSeconds = Math.ceil(TIMEOUTS.MODEL_TRAINING / 1000);
    return { status: 504, message: `训练超时（${timeoutSeconds}秒）` };
  }
  if (err.message?.startsWith("[PYTHON_VALIDATION]")) {
    return { status: 400, message: err.message.substring(20) };
  }
  if (err.message?.startsWith("[PYTHON_DATA]")) {
    return { status: 400, message: err.message.substring(14) };
  }
  if (err.message?.startsWith("Python 执行失败")) {
    return { status: 500, message: `${modelType.toUpperCase()} 训练失败，请稍后重试` };
  }

  return {
    status: 500,
    message: err.message || `${modelType.toUpperCase()} 训练失败`,
  };
};

modelsController.post(
  "/api/v1/models/:model_type/training",
  jwtAuth,
  requireStudent,
  async (c) => {
    const modelTypeParam = c.req.param("model_type");
    if (!isModelType(modelTypeParam)) {
      return c.json({ error: "指定的模型类型无效" }, 400);
    }
    const modelType = modelTypeParam;

    const payload = c.get("jwtPayload") as JwtPayload;
    let rawBody: unknown;
    try {
      rawBody = await c.req.json();
    } catch {
      return c.json({ error: "请求体必须是有效的 JSON" }, 400);
    }

    const validation = getTrainSchemaForModel(modelType).safeParse(rawBody);
    if (!validation.success) {
      const localizedDetails = buildLocalizedTrainValidationDetails(validation.error.issues);
      return c.json(
        {
          error: validation.error.issues[0]
            ? localizeTrainValidationIssue(validation.error.issues[0])
            : "训练请求参数无效",
          details: localizedDetails,
        },
        400,
      );
    }

    const body = validation.data;

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
  "/api/v1/models/:model_type/training/stream",
  jwtAuth,
  requireStudent,
  async (c) => {
    const modelTypeParam = c.req.param("model_type");
    if (!isModelType(modelTypeParam)) {
      return c.json({ error: "指定的模型类型无效" }, 400);
    }
    const modelType = modelTypeParam;

    const payload = c.get("jwtPayload") as JwtPayload;
    let rawBody: unknown;
    try {
      rawBody = await c.req.json();
    } catch {
      return c.json({ error: "请求体必须是有效的 JSON" }, 400);
    }

    const validation = getTrainSchemaForModel(modelType).safeParse(rawBody);
    if (!validation.success) {
      const localizedDetails = buildLocalizedTrainValidationDetails(validation.error.issues);
      return c.json(
        {
          error: validation.error.issues[0]
            ? localizeTrainValidationIssue(validation.error.issues[0])
            : "训练请求参数无效",
          details: localizedDetails,
        },
        400,
      );
    }

    const body = validation.data;
    let executionLock = null;
    let slotLock = null;

    try {
      executionLock = await acquireModelExecutionLock(body.experiment_id, modelType);
      if (!executionLock) {
        return c.json({ error: "同一模型正在训练或预测，请稍后再试" }, 409);
      }

      slotLock = await acquireModelSlot();
      if (!slotLock) {
        await releaseModelExecutionLock(executionLock);
        executionLock = null;
        return c.json({ error: "模型服务繁忙，请稍后再试" }, 429);
      }
    } catch (error) {
      await releaseModelSlot(slotLock);
      await releaseModelExecutionLock(executionLock);
      return handleUnexpectedError(c, error, "启动训练流失败");
    }

    return createNdjsonStreamResponse(async (send) => {
      try {
        send({
          type: "progress",
          phase: "queued",
          percent: 2,
          source: "backend",
          message: "已获取模型训练执行资源，正在启动训练任务",
        });

        const result = await trainModel(modelType, payload.sub, body, (event) => {
          send(event);
        });

        send({
          type: "progress",
          phase: "done",
          percent: 100,
          source: "backend",
          message: "模型训练完成，正在返回结果",
        });
        send({ type: "result", data: result });
      } catch (error) {
        const normalizedError = normalizeTrainingStreamError(error, modelType);
        send({ type: "error", ...normalizedError });
      } finally {
        await releaseModelSlot(slotLock);
        await releaseModelExecutionLock(executionLock);
      }
    });
  },
);

modelsController.post(
  "/api/v1/models/:model_type/guided-training/sessions",
  jwtAuth,
  requireStudent,
  async (c) => {
    const modelTypeParam = c.req.param("model_type");
    if (!isModelType(modelTypeParam)) {
      return c.json({ error: "指定的模型类型无效" }, 400);
    }
    const modelType = modelTypeParam;

    const payload = c.get("jwtPayload") as JwtPayload;
    let rawBody: unknown;
    try {
      rawBody = await c.req.json();
    } catch {
      return c.json({ error: "请求体必须是有效的 JSON" }, 400);
    }

    const validation = getTrainSchemaForModel(modelType).safeParse(rawBody);
    if (!validation.success) {
      const localizedDetails = buildLocalizedTrainValidationDetails(validation.error.issues);
      return c.json(
        {
          error: validation.error.issues[0]
            ? localizeTrainValidationIssue(validation.error.issues[0])
            : "训练请求参数无效",
          details: localizedDetails,
        },
        400,
      );
    }

    try {
      const session = await createOrResumeGuidedTrainingSession(modelType, payload.sub, validation.data);
      return c.json(session);
    } catch (error: unknown) {
      if (error instanceof ModelAccessError) {
        return c.json({ error: error.message }, error.status);
      }
      if (error instanceof ModelValidationError) {
        return c.json({ error: error.message, ...error.details }, 400);
      }
      if (error instanceof ModelNotFoundError) {
        return c.json({ error: error.message }, 404);
      }
      return handleUnexpectedError(c, error, "创建分阶段训练会话失败");
    }
  },
);

modelsController.get(
  "/api/v1/models/:model_type/guided-training/sessions/:session_id",
  jwtAuth,
  requireStudent,
  async (c) => {
    const modelTypeParam = c.req.param("model_type");
    if (!isModelType(modelTypeParam)) {
      return c.json({ error: "指定的模型类型无效" }, 400);
    }
    const sessionId = c.req.param("session_id");
    if (!sessionId) {
      return c.json({ error: "训练会话 ID 无效" }, 400);
    }

    const payload = c.get("jwtPayload") as JwtPayload;
    try {
      const session = await getGuidedTrainingSession(
        modelTypeParam,
        payload.sub,
        sessionId,
      );
      return c.json(session);
    } catch (error: unknown) {
      if (error instanceof ModelAccessError) {
        return c.json({ error: error.message }, error.status);
      }
      if (error instanceof ModelValidationError) {
        return c.json({ error: error.message, ...error.details }, 400);
      }
      if (error instanceof ModelNotFoundError) {
        return c.json({ error: error.message }, 404);
      }
      return handleUnexpectedError(c, error, "获取分阶段训练会话失败");
    }
  },
);

modelsController.post(
  "/api/v1/models/:model_type/guided-training/sessions/:session_id/steps/:step_id/run",
  jwtAuth,
  requireStudent,
  async (c) => {
    const modelTypeParam = c.req.param("model_type");
    if (!isModelType(modelTypeParam)) {
      return c.json({ error: "指定的模型类型无效" }, 400);
    }
    const sessionId = c.req.param("session_id");
    const stepId = c.req.param("step_id");
    if (!sessionId || !stepId) {
      return c.json({ error: "训练会话或步骤 ID 无效" }, 400);
    }

    const payload = c.get("jwtPayload") as JwtPayload;
    try {
      const session = await runGuidedTrainingStep(
        modelTypeParam,
        payload.sub,
        sessionId,
        stepId,
      );
      return c.json(session);
    } catch (error: unknown) {
      if (error instanceof ModelAccessError) {
        return c.json({ error: error.message }, error.status);
      }
      if (error instanceof ModelValidationError) {
        if (error.message.includes("模型服务繁忙")) {
          return c.json({ error: error.message }, 429);
        }
        if (error.message.includes("同一模型正在训练或预测")) {
          return c.json({ error: error.message }, 409);
        }
        return c.json({ error: error.message, ...error.details }, 400);
      }
      if (error instanceof ModelNotFoundError) {
        return c.json({ error: error.message }, 404);
      }

      const normalizedError = normalizeTrainingStreamError(error, modelTypeParam);
      return c.json({ error: normalizedError.message, details: normalizedError }, normalizedError.status as any);
    }
  },
);

modelsController.post(
  "/api/v1/models/:model_type/prepare-production",
  jwtAuth,
  requireStudent,
  async (c) => {
    const modelTypeParam = c.req.param("model_type");
    if (!isModelType(modelTypeParam)) {
      return c.json({ error: "指定的模型类型无效" }, 400);
    }
    const modelType = modelTypeParam;

    const payload = c.get("jwtPayload") as JwtPayload;
    let rawBody: unknown;
    try {
      rawBody = await c.req.json();
    } catch {
      return c.json({ error: "请求体必须是有效的 JSON" }, 400);
    }

    const validation = prepareProductionSchema.safeParse(rawBody);
    if (!validation.success) {
      return c.json({ error: "生产预测准备请求参数无效" }, 400);
    }

    const body = validation.data;

    let executionLock = null;
    let slotLock = null;

    try {
      executionLock = await acquireModelExecutionLock(body.experiment_id, modelType);
      if (!executionLock) {
        return c.json({ error: "同一模型正在训练或预测，请稍后再试" }, 409);
      }

      if (SLOT_FREE_REUSABLE_PRODUCTION_MODEL_TYPES.has(modelType)) {
        const reusableArtifact = await getReusableProductionModelArtifactResult(modelType, payload.sub, body.experiment_id);
        if (reusableArtifact) {
          return c.json(reusableArtifact);
        }
      }

      slotLock = await acquireModelSlot();
      if (!slotLock) {
        return c.json({ error: "模型服务繁忙，请稍后再试" }, 429);
      }

      const result = await prepareExperimentModelForProduction(modelType, payload.sub, body.experiment_id);
      return c.json(result);
    } catch (error: unknown) {
      if (error instanceof ModelAccessError) {
        return c.json({ error: error.message }, error.status);
      }
      if (error instanceof ModelNotFoundError) {
        return c.json({ error: error.message }, 404);
      }
      if (error instanceof ModelValidationError) {
        return c.json({ error: error.message, ...error.details }, 400);
      }
      const err = error as Error;
      if (err.message?.includes("timed out") || err.message?.includes("超时")) {
        const timeoutSeconds = Math.ceil(TIMEOUTS.MODEL_TRAINING / 1000);
        return c.json({ error: `生产模型准备超时（${timeoutSeconds}秒）` }, 504);
      }
      if (err.message?.startsWith("[PYTHON_VALIDATION]")) {
        return c.json({ error: err.message.substring(20) }, 400);
      }
      if (err.message?.startsWith("[PYTHON_DATA]")) {
        return c.json({ error: err.message.substring(14) }, 400);
      }
      const message = err.message || `${modelType.toUpperCase()} 生产模型准备失败`;
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
  async (c) => {
    const modelTypeParam = c.req.param("model_type");
    if (!isModelType(modelTypeParam)) {
      return c.json({ error: "指定的模型类型无效" }, 400);
    }
    const modelType = modelTypeParam;

    const payload = c.get("jwtPayload") as JwtPayload;
    let rawBody: unknown;
    try {
      rawBody = await c.req.json();
    } catch {
      return c.json({ error: "请求体必须是有效的 JSON" }, 400);
    }

    const validation = predictSchema.safeParse(rawBody);
    if (!validation.success) {
      const localizedDetails = buildLocalizedPredictValidationDetails(validation.error.issues);
      return c.json(
        {
          error: validation.error.issues[0]
            ? localizePredictValidationIssue(validation.error.issues[0])
            : "预测请求参数无效",
          details: localizedDetails,
        },
        400,
      );
    }

    const body = validation.data;

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
      if (error instanceof ModelValidationError) {
        return c.json({ error: error.message, ...error.details }, 400);
      }
      const err = error as Error;
      if (err.message?.includes("timed out") || err.message?.includes("超时")) {
        const timeoutSeconds = Math.ceil(TIMEOUTS.MODEL_PREDICTION / 1000);
        return c.json({ error: `预测超时（${timeoutSeconds}秒）` }, 504);
      }
      if (err.message?.startsWith("[PYTHON_VALIDATION]")) {
        return c.json({ error: err.message.substring(20) }, 400);
      }
      if (err.message?.startsWith("[PYTHON_DATA]")) {
        return c.json({ error: err.message.substring(14) }, 400);
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
