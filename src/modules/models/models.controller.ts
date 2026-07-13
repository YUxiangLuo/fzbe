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
  createOrResumeGuidedTrainingSession,
  discardGuidedTrainingSession,
  getGuidedTrainingSession,
  runGuidedTrainingStep,
  getReusableProductionModelArtifactResult,
  prepareExperimentModelForProduction,
  predictExperimentModel,
  acquireModelSlot,
  releaseModelSlot,
  acquireModelExecutionLock,
  combineModelLockSignals,
  releaseModelExecutionLock,
  ModelValidationError,
  ModelNotFoundError,
  ModelAccessError,
  ModelConflictError,
  ModelServiceUnavailableError,
} from "./models.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import type { JwtPayload } from "../../types";
import { TIMEOUTS } from "../../utils/constants";
import { parsePythonServiceError } from "./model-error-protocol";

const modelsController = new Hono();
const SLOT_FREE_REUSABLE_PRODUCTION_MODEL_TYPES = new Set<ModelType>(["ma", "es", "arima", "lstm"]);

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

modelsController.delete(
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
      await discardGuidedTrainingSession(modelTypeParam, payload.sub, sessionId);
      return c.body(null, 204);
    } catch (error) {
      if (error instanceof ModelAccessError) {
        return c.json({ error: error.message }, error.status);
      }
      if (error instanceof ModelConflictError) {
        return c.json({ error: error.message, code: error.code }, 409);
      }
      if (error instanceof ModelServiceUnavailableError) {
        return c.json({ error: error.message, code: error.code }, 503);
      }
      if (error instanceof ModelNotFoundError) {
        return c.json({ error: error.message }, 404);
      }
      if (error instanceof ModelValidationError) {
        return c.json({ error: error.message, ...error.details }, 400);
      }
      return handleUnexpectedError(c, error, "删除分阶段训练会话失败");
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
      if (error instanceof ModelConflictError) {
        return c.json({ error: error.message, code: error.code, ...error.details }, 409);
      }
      if (error instanceof ModelServiceUnavailableError) {
        return c.json({ error: error.message, code: error.code }, 503);
      }
      if (error instanceof ModelNotFoundError) {
        return c.json({ error: error.message }, 404);
      }

      const err = error as Error;
      if (err.message?.includes("timed out") || err.message?.includes("超时")) {
        return c.json({ error: "训练步骤超时，请重试当前步骤" }, 504);
      }
      const pythonError = parsePythonServiceError(error);
      if (pythonError !== null) {
        return c.json({ error: pythonError || "Python 模型输入无效" }, 400);
      }
      return handleUnexpectedError(c, error, `${modelTypeParam.toUpperCase()} 训练步骤失败`);
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
        const reusableArtifact = await getReusableProductionModelArtifactResult(
          modelType,
          payload.sub,
          body.experiment_id,
          body.forecast_steps,
        );
        if (reusableArtifact) {
          return c.json(reusableArtifact);
        }
      }

      slotLock = await acquireModelSlot();
      if (!slotLock) {
        return c.json({ error: "模型服务繁忙，请稍后再试" }, 429);
      }

      const result = await prepareExperimentModelForProduction(
        modelType,
        payload.sub,
        body.experiment_id,
        body.forecast_steps,
        false,
        combineModelLockSignals(executionLock, slotLock),
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
      if (error instanceof ModelConflictError) {
        return c.json({ error: error.message, code: error.code, ...error.details }, 409);
      }
      if (error instanceof ModelServiceUnavailableError) {
        return c.json({ error: error.message, code: error.code }, 503);
      }
      const err = error as Error;
      if (err.message?.includes("timed out") || err.message?.includes("超时")) {
        return c.json({ error: "生产模型准备超时，请稍后重试" }, 504);
      }
      const pythonError = parsePythonServiceError(error);
      if (pythonError !== null) {
        return c.json({ error: pythonError || "Python 模型输入无效" }, 400);
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
        body.preparation_token,
        combineModelLockSignals(executionLock, slotLock),
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
      if (error instanceof ModelConflictError) {
        return c.json({ error: error.message, code: error.code, ...error.details }, 409);
      }
      if (error instanceof ModelServiceUnavailableError) {
        return c.json({ error: error.message, code: error.code }, 503);
      }
      const err = error as Error;
      if (err.message?.includes("timed out") || err.message?.includes("超时")) {
        const timeoutSeconds = Math.ceil(TIMEOUTS.MODEL_PREDICTION / 1000);
        return c.json({ error: `预测超时（${timeoutSeconds}秒）` }, 504);
      }
      const pythonError = parsePythonServiceError(error);
      if (pythonError !== null) {
        return c.json({ error: pythonError || "Python 模型输入无效" }, 400);
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
