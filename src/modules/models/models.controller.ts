import { Hono } from "hono";
import { jwtAuth, requireStudent } from "../../middleware";
import { predictSchema, prepareProductionSchema, isModelType, getTrainSchemaForModel } from "./models.schema";
import {
  buildLocalizedPredictValidationDetails,
  buildLocalizedTrainValidationDetails,
  localizePredictValidationIssue,
  localizeTrainValidationIssue,
} from "./models.validation";
import {
  createOrResumeGuidedTrainingSession,
  createOrResumeProductionPreparationSession,
  discardGuidedTrainingSession,
  getGuidedTrainingSession,
  getProductionPreparationSession,
  runGuidedTrainingStep,
  runProductionPreparationStep,
  predictExperimentModel,
  getExperimentModelObservability,
  ModelValidationError,
  ModelNotFoundError,
  ModelAccessError,
  ModelConflictError,
  ModelServiceUnavailableError,
  ModelTimeoutError,
  ModelCapacityBusyError,
  ModelOperationDeadlineError,
} from "./models.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import type { JwtPayload } from "../../types";
import { parsePythonServiceError } from "./model-error-protocol";

const modelsController = new Hono();

type ModelTimeoutOperation =
  | "guided_training_session"
  | "guided_training_step"
  | "production_preparation"
  | "prediction";

const buildModelTimeoutPayload = (
  error: ModelTimeoutError,
  operation: ModelTimeoutOperation,
  message: string,
) => ({
  error: message,
  code: error.code,
  operation,
  timeout_ms: error.timeoutMs,
  retryable: true,
});

const buildOperationDeadlinePayload = (
  error: ModelOperationDeadlineError,
  message: string,
) => ({
  error: message,
  code: error.code,
  operation: error.operation,
  timeout_ms: error.timeoutMs,
  retryable: error.retryable,
});

modelsController.get(
  "/api/v1/models/experiments/:experimentId/observability",
  jwtAuth,
  requireStudent,
  async (c) => {
    const experimentId = Number(c.req.param("experimentId"));
    if (!Number.isSafeInteger(experimentId) || experimentId <= 0) {
      return c.json({ error: "实验 ID 无效" }, 400);
    }
    const payload = c.get("jwtPayload") as JwtPayload;
    try {
      return c.json(await getExperimentModelObservability(experimentId, payload.sub));
    } catch (error: unknown) {
      if (error instanceof ModelAccessError) {
        return c.json({ error: error.message }, error.status);
      }
      return handleUnexpectedError(c, error, "读取模型可观测元数据失败");
    }
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
      if (error instanceof ModelCapacityBusyError) {
        c.header("Retry-After", String(error.retryAfterSeconds));
        return c.json({ error: error.message, code: error.code, retryable: error.retryable }, 429);
      }
      if (error instanceof ModelOperationDeadlineError) {
        return c.json(buildOperationDeadlinePayload(error, "恢复训练会话超过服务端时限，请重试"), 504);
      }
      if (error instanceof ModelAccessError) {
        return c.json({ error: error.message }, error.status);
      }
      if (error instanceof ModelValidationError) {
        return c.json({ error: error.message, ...error.details }, 400);
      }
      if (error instanceof ModelNotFoundError) {
        return c.json({ error: error.message }, 404);
      }
      if (error instanceof ModelConflictError) {
        return c.json({ error: error.message, code: error.code, ...error.details }, 409);
      }
      if (error instanceof ModelServiceUnavailableError) {
        return c.json({ error: error.message, code: error.code }, 503);
      }
      if (error instanceof ModelTimeoutError) {
        return c.json(
          buildModelTimeoutPayload(
            error,
            "guided_training_session",
            "恢复训练会话模型产物超时，请重试",
          ),
          504,
        );
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
      if (error instanceof ModelCapacityBusyError) {
        c.header("Retry-After", String(error.retryAfterSeconds));
        return c.json({ error: error.message, code: error.code, retryable: error.retryable }, 429);
      }
      if (error instanceof ModelOperationDeadlineError) {
        return c.json(buildOperationDeadlinePayload(error, "删除训练会话超过服务端时限，请重试"), 504);
      }
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
      if (error instanceof ModelCapacityBusyError) {
        c.header("Retry-After", String(error.retryAfterSeconds));
        return c.json({ error: error.message, code: error.code, retryable: error.retryable }, 429);
      }
      if (error instanceof ModelOperationDeadlineError) {
        return c.json(buildOperationDeadlinePayload(error, "训练步骤超过服务端时限，请重试当前步骤"), 504);
      }
      if (error instanceof ModelAccessError) {
        return c.json({ error: error.message }, error.status);
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
      if (error instanceof ModelNotFoundError) {
        return c.json({ error: error.message }, 404);
      }
      if (error instanceof ModelTimeoutError) {
        return c.json(
          buildModelTimeoutPayload(error, "guided_training_step", "训练步骤超时，请重试当前步骤"),
          504,
        );
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
  "/api/v1/models/:model_type/production-preparation/sessions",
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

    try {
      const result = await createOrResumeProductionPreparationSession(
        modelType,
        payload.sub,
        body.experiment_id,
        body.forecast_steps,
      );
      return c.json(result);
    } catch (error: unknown) {
      if (error instanceof ModelCapacityBusyError) {
        c.header("Retry-After", String(error.retryAfterSeconds));
        return c.json({ error: error.message, code: error.code, retryable: error.retryable }, 429);
      }
      if (error instanceof ModelOperationDeadlineError) {
        return c.json(buildOperationDeadlinePayload(error, "生产模型准备超过服务端时限，请稍后重试"), 504);
      }
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
      if (error instanceof ModelTimeoutError) {
        return c.json(
          buildModelTimeoutPayload(error, "production_preparation", "生产模型准备超时，请稍后重试"),
          504,
        );
      }
      const pythonError = parsePythonServiceError(error);
      if (pythonError !== null) {
        return c.json({ error: pythonError || "Python 模型输入无效" }, 400);
      }
      const err = error as Error;
      const message = err.message || `${modelType.toUpperCase()} 生产模型准备失败`;
      return handleUnexpectedError(c, error, message);
    }
  },
);

modelsController.get(
  "/api/v1/models/:model_type/production-preparation/sessions/:session_id",
  jwtAuth,
  requireStudent,
  async (c) => {
    const modelTypeParam = c.req.param("model_type");
    if (!isModelType(modelTypeParam)) {
      return c.json({ error: "指定的模型类型无效" }, 400);
    }
    const sessionId = c.req.param("session_id");
    if (!sessionId) return c.json({ error: "生产准备会话 ID 无效" }, 400);
    const payload = c.get("jwtPayload") as JwtPayload;
    try {
      return c.json(await getProductionPreparationSession(modelTypeParam, payload.sub, sessionId));
    } catch (error: unknown) {
      if (error instanceof ModelAccessError) return c.json({ error: error.message }, error.status);
      if (error instanceof ModelNotFoundError) return c.json({ error: error.message }, 404);
      if (error instanceof ModelValidationError) return c.json({ error: error.message, ...error.details }, 400);
      return handleUnexpectedError(c, error, "获取生产准备会话失败");
    }
  },
);

modelsController.post(
  "/api/v1/models/:model_type/production-preparation/sessions/:session_id/steps/:step_id/run",
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
      return c.json({ error: "生产准备会话或步骤 ID 无效" }, 400);
    }
    const payload = c.get("jwtPayload") as JwtPayload;
    try {
      return c.json(await runProductionPreparationStep(
        modelTypeParam,
        payload.sub,
        sessionId,
        stepId,
      ));
    } catch (error: unknown) {
      if (error instanceof ModelCapacityBusyError) {
        c.header("Retry-After", String(error.retryAfterSeconds));
        return c.json({ error: error.message, code: error.code, retryable: error.retryable }, 429);
      }
      if (error instanceof ModelOperationDeadlineError) {
        return c.json(buildOperationDeadlinePayload(error, "生产准备步骤超过服务端时限，请重试当前步骤"), 504);
      }
      if (error instanceof ModelAccessError) return c.json({ error: error.message }, error.status);
      if (error instanceof ModelNotFoundError) return c.json({ error: error.message }, 404);
      if (error instanceof ModelValidationError) return c.json({ error: error.message, ...error.details }, 400);
      if (error instanceof ModelConflictError) {
        return c.json({ error: error.message, code: error.code, ...error.details }, 409);
      }
      if (error instanceof ModelServiceUnavailableError) {
        return c.json({ error: error.message, code: error.code }, 503);
      }
      if (error instanceof ModelTimeoutError) {
        return c.json(
          buildModelTimeoutPayload(error, "production_preparation", "生产准备步骤超时，请重试当前步骤"),
          504,
        );
      }
      const pythonError = parsePythonServiceError(error);
      if (pythonError !== null) {
        return c.json({ error: pythonError || "Python 模型输入无效" }, 400);
      }
      return handleUnexpectedError(c, error, `${modelTypeParam.toUpperCase()} 生产准备步骤失败`);
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

    try {
      const result = await predictExperimentModel(
        modelType,
        payload.sub,
        body.experiment_id,
        body.forecast_steps,
        body.preparation_token,
      );
      return c.json(result);
    } catch (error: unknown) {
      if (error instanceof ModelCapacityBusyError) {
        c.header("Retry-After", String(error.retryAfterSeconds));
        return c.json({ error: error.message, code: error.code, retryable: error.retryable }, 429);
      }
      if (error instanceof ModelOperationDeadlineError) {
        return c.json(buildOperationDeadlinePayload(error, "预测超过服务端时限，请重试"), 504);
      }
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
      if (error instanceof ModelTimeoutError) {
        const timeoutSeconds = Math.ceil(error.timeoutMs / 1000);
        return c.json(
          buildModelTimeoutPayload(error, "prediction", `预测超时（${timeoutSeconds}秒）`),
          504,
        );
      }
      const pythonError = parsePythonServiceError(error);
      if (pythonError !== null) {
        return c.json({ error: pythonError || "Python 模型输入无效" }, 400);
      }
      const err = error as Error;
      const message = err.message || `${modelType.toUpperCase()} 预测失败`;
      return handleUnexpectedError(c, error, message);
    }
  },
);

export default modelsController;
