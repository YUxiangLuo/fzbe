process.env.DB_HOST = "127.0.0.1";
process.env.DB_USER = "test";
process.env.DB_PASSWORD = "test";
process.env.DB_DATABASE = "test";
process.env.JWT_SECRET = "test-jwt-secret";
process.env.PORT = "3001";

import { beforeEach, describe, expect, mock, test } from "bun:test";
import { Hono } from "hono";
import { sign } from "hono/jwt";

mock.restore();

const preparationToken = "123e4567-e89b-42d3-a456-426614174000";

const createOrResumeGuidedTrainingSession = mock(async () => ({
  session_id: "guided-1",
  experiment_id: 990002,
  model_type: "ma",
  status: "ready",
  current_step_id: "prepare_data",
  next_step_id: "prepare_data",
  steps: [],
  step_outputs: {},
  result: null,
  error_message: null,
}));
const getGuidedTrainingSession = mock(async () => ({
  session_id: "guided-1",
  experiment_id: 990002,
  model_type: "ma",
  status: "ready",
  current_step_id: "prepare_data",
  next_step_id: "prepare_data",
  steps: [],
  step_outputs: {},
  result: null,
  error_message: null,
}));
const runGuidedTrainingStep = mock(async () => ({
  session_id: "guided-1",
  experiment_id: 990002,
  model_type: "ma",
  status: "completed",
  current_step_id: null,
  next_step_id: null,
  steps: [],
  step_outputs: {},
  result: { status: "success", results: { metrics: { rmse: 1 } } },
  error_message: null,
}));
const discardGuidedTrainingSession = mock(async () => ({ status: "success" }));
const createOrResumeProductionPreparationSession = mock(async () => ({
  status: "success",
  results: {
    session_id: "production-1",
    session_status: "ready",
    completed_step_ids: [],
    total_steps: 2,
    next_step: { id: "initialize", label: "确认生产准备材料" },
    error_message: null,
  },
}));
const getProductionPreparationSession = mock(async () => ({
  status: "success",
  results: {
    session_id: "production-1",
    session_status: "ready",
    completed_step_ids: [],
    total_steps: 2,
    next_step: { id: "initialize", label: "确认生产准备材料" },
    error_message: null,
  },
}));
const runProductionPreparationStep = mock(async () => ({
  status: "success",
  results: {
    session_id: "production-1",
    session_status: "completed",
    completed_step_ids: ["initialize", "finalize"],
    total_steps: 2,
    next_step: null,
    error_message: null,
    preparation_token: preparationToken,
    prepared_forecast_steps: 6,
    reused_existing: false,
  },
}));
const predictExperimentModel = mock(async () => ({
  status: "success",
  results: { predictions: [] },
}));
const getExperimentModelObservability = mock(async () => ({
  models: {},
}));

class ModelValidationError extends Error {
  constructor(message: string, public details?: Record<string, unknown>) {
    super(message);
  }
}
class ModelNotFoundError extends Error {}
class ModelAccessError extends Error {
  constructor(message: string, public status: 400 | 403 | 404) {
    super(message);
  }
}
class ModelConflictError extends Error {
  constructor(message: string, public code: string, public details?: Record<string, unknown>) {
    super(message);
  }
}
class ModelServiceUnavailableError extends Error {
  constructor(message: string, public code: string) {
    super(message);
  }
}
class ModelCapacityBusyError extends Error {
  readonly code = "MODEL_CAPACITY_BUSY";
  readonly retryable = true;
  constructor(message = "模型服务繁忙，请稍后再试", public retryAfterSeconds = 2) {
    super(message);
  }
}
class ModelOperationDeadlineError extends Error {
  readonly code = "MODEL_OPERATION_DEADLINE_EXCEEDED";
  readonly retryable = true;
  constructor(public timeoutMs: number, public operation: string) {
    super("deadline");
  }
}
class ModelTimeoutError extends Error {
  readonly code = "MODEL_TIMEOUT";

  constructor(
    message: string,
    public timeoutMs: number,
    public source: "python_process" | "python_subprocess",
  ) {
    super(message);
  }
}

mock.module("../../../src/modules/models/models.service", () => ({
  createOrResumeGuidedTrainingSession,
  getGuidedTrainingSession,
  runGuidedTrainingStep,
  discardGuidedTrainingSession,
  createOrResumeProductionPreparationSession,
  getProductionPreparationSession,
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
}));

const { default: modelsController } = await import("../../../src/modules/models/models.controller");
const app = new Hono();
app.route("/", modelsController);

const authHeaders = async (role = "Student") => ({
  Authorization: `Bearer ${await sign({
    sub: 20240002,
    username: "20240002",
    role,
    exp: Math.floor(Date.now() / 1000) + 600,
  }, process.env.JWT_SECRET!)}`,
  "Content-Type": "application/json",
});

const request = async (
  pathname: string,
  method: "GET" | "POST" | "DELETE",
  body?: Record<string, unknown>,
) => app.request(pathname, {
  method,
  headers: await authHeaders(),
  body: body === undefined ? undefined : JSON.stringify(body),
});

const trainBody = () => ({
  experiment_id: 990002,
  selected_industry: "制造业",
  selected_company: "样本企业",
  selected_product: "样本产品",
  data_window_train_start_index: 0,
  data_window_train_end_index: 11,
  data_window_evaluate_start_index: 12,
  data_window_evaluate_end_index: 17,
  moving_average_window: 4,
});

describe("models controller", () => {
  beforeEach(() => {
    for (const fn of [
      createOrResumeGuidedTrainingSession,
      getGuidedTrainingSession,
      runGuidedTrainingStep,
      discardGuidedTrainingSession,
      createOrResumeProductionPreparationSession,
      getProductionPreparationSession,
      runProductionPreparationStep,
      predictExperimentModel,
      getExperimentModelObservability,
    ]) fn.mockClear();
  });

  test("returns the active model observability for the authenticated experiment owner", async () => {
    const response = await request(
      "/api/v1/models/experiments/990002/observability",
      "GET",
    );

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      models: {},
    });
    expect(getExperimentModelObservability).toHaveBeenCalledWith(990002, 20240002);
  });

  test("rejects an invalid observability experiment id before service access", async () => {
    const response = await request(
      "/api/v1/models/experiments/not-a-number/observability",
      "GET",
    );

    expect(response.status).toBe(400);
    expect(getExperimentModelObservability).not.toHaveBeenCalled();
  });

  test("requires authentication and the student role for observability", async () => {
    const unauthenticated = await app.request(
      "/api/v1/models/experiments/990002/observability",
    );
    const teacher = await app.request(
      "/api/v1/models/experiments/990002/observability",
      { headers: await authHeaders("Teacher") },
    );

    expect(unauthenticated.status).toBe(401);
    expect(teacher.status).toBe(403);
    expect(getExperimentModelObservability).not.toHaveBeenCalled();
  });

  test("maps observability ownership and missing-experiment errors", async () => {
    for (const expected of [
      { message: "权限不足：该实验不属于当前用户", status: 403 as const },
      { message: "实验不存在", status: 404 as const },
    ]) {
      getExperimentModelObservability.mockRejectedValueOnce(
        new ModelAccessError(expected.message, expected.status),
      );
      const response = await request(
        "/api/v1/models/experiments/990002/observability",
        "GET",
      );

      expect(response.status).toBe(expected.status);
      expect(await response.json()).toEqual({ error: expected.message });
    }
  });

  test("removes both legacy training endpoints", async () => {
    const plain = await request("/api/v1/models/ma/training", "POST", trainBody());
    const stream = await request("/api/v1/models/ma/training/stream", "POST", trainBody());
    expect(plain.status).toBe(404);
    expect(stream.status).toBe(404);
    expect(createOrResumeGuidedTrainingSession).not.toHaveBeenCalled();
  });

  test("creates a guided session without controller-side model locks", async () => {
    const response = await request(
      "/api/v1/models/ma/guided-training/sessions",
      "POST",
      trainBody(),
    );
    expect(response.status).toBe(200);
    expect((await response.json()).session_id).toBe("guided-1");
    expect(createOrResumeGuidedTrainingSession).toHaveBeenCalledWith(
      "ma",
      20240002,
      expect.objectContaining({ moving_average_window: 4 }),
    );
  });

  test("validates guided model-specific parameters before service execution", async () => {
    const response = await request(
      "/api/v1/models/weighted_avg/guided-training/sessions",
      "POST",
      trainBody(),
    );
    expect(response.status).toBe(400);
    expect(createOrResumeGuidedTrainingSession).not.toHaveBeenCalled();
  });

  test("returns a 409 when completed-session recovery finds the same model busy", async () => {
    createOrResumeGuidedTrainingSession.mockRejectedValueOnce(
      new ModelConflictError("同一模型正在训练或预测，请稍后再试", "MODEL_BUSY"),
    );
    const response = await request(
      "/api/v1/models/ma/guided-training/sessions",
      "POST",
      trainBody(),
    );

    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({
      error: "同一模型正在训练或预测，请稍后再试",
      code: "MODEL_BUSY",
    });
  });

  test("returns structured guided conflicts", async () => {
    runGuidedTrainingStep.mockRejectedValueOnce(
      new ModelConflictError("训练会话已失效", "GUIDED_SESSION_SUPERSEDED"),
    );
    const response = await request(
      "/api/v1/models/ma/guided-training/sessions/guided-1/steps/forecast/run",
      "POST",
    );
    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({
      error: "训练会话已失效",
      code: "GUIDED_SESSION_SUPERSEDED",
    });
  });

  test("parses Python error tags without depending on a following space", async () => {
    runGuidedTrainingStep.mockRejectedValueOnce(new Error("[PYTHON_VALIDATION]参数错误"));
    const response = await request(
      "/api/v1/models/ma/guided-training/sessions/guided-1/steps/forecast/run",
      "POST",
    );
    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "参数错误" });
  });

  test("discards a failed guided session", async () => {
    const response = await request(
      "/api/v1/models/ma/guided-training/sessions/guided-1",
      "DELETE",
    );
    expect(response.status).toBe(204);
    expect(discardGuidedTrainingSession).toHaveBeenCalledWith("ma", 20240002, "guided-1");
  });

  test("removes the aggregate production preparation endpoint", async () => {
    const response = await request("/api/v1/models/ma/prepare-production", "POST", {
      experiment_id: 990002,
      forecast_steps: 6,
    });
    expect(response.status).toBe(404);
    expect(createOrResumeProductionPreparationSession).not.toHaveBeenCalled();
  });

  test("creates a resumable production preparation session", async () => {
    const response = await request(
      "/api/v1/models/ma/production-preparation/sessions",
      "POST",
      { experiment_id: 990002, forecast_steps: 6 },
    );
    expect(response.status).toBe(200);
    expect((await response.json()).results).toMatchObject({
      session_id: "production-1",
      session_status: "ready",
      next_step: { id: "initialize" },
    });
    expect(createOrResumeProductionPreparationSession).toHaveBeenCalledWith(
      "ma",
      20240002,
      990002,
      6,
    );
  });

  test("preserves the retraining contract for a missing active model run", async () => {
    createOrResumeProductionPreparationSession.mockRejectedValueOnce(
      new ModelValidationError(
        "MA 模型缺少可验证的训练版本，请重新训练该模型",
        {
          code: "MODEL_TRAINING_RUN_REQUIRED",
          required_action: "retrain",
          model_type: "ma",
        },
      ),
    );

    const response = await request(
      "/api/v1/models/ma/production-preparation/sessions",
      "POST",
      { experiment_id: 990002, forecast_steps: 6 },
    );

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({
      error: "MA 模型缺少可验证的训练版本，请重新训练该模型",
      code: "MODEL_TRAINING_RUN_REQUIRED",
      required_action: "retrain",
      model_type: "ma",
    });
  });

  test("reads an existing production preparation session", async () => {
    const response = await request(
      "/api/v1/models/ma/production-preparation/sessions/production-1",
      "GET",
    );
    expect(response.status).toBe(200);
    expect(getProductionPreparationSession).toHaveBeenCalledWith(
      "ma",
      20240002,
      "production-1",
    );
  });

  test("executes exactly one production preparation step", async () => {
    const response = await request(
      "/api/v1/models/ma/production-preparation/sessions/production-1/steps/initialize/run",
      "POST",
    );
    expect(response.status).toBe(200);
    expect((await response.json()).results).toMatchObject({
      session_status: "completed",
      preparation_token: preparationToken,
    });
    expect(runProductionPreparationStep).toHaveBeenCalledWith(
      "ma",
      20240002,
      "production-1",
      "initialize",
    );
  });

  test("preserves the retraining contract when a production source artifact is invalid", async () => {
    runProductionPreparationStep.mockRejectedValueOnce(
      new ModelValidationError(
        "MA 已训练模型产物已损坏，请重新训练该模型",
        {
          code: "MODEL_ARTIFACT_INVALID",
          required_action: "retrain",
          model_type: "ma",
        },
      ),
    );

    const response = await request(
      "/api/v1/models/ma/production-preparation/sessions/production-1/steps/initialize/run",
      "POST",
    );

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({
      error: "MA 已训练模型产物已损坏，请重新训练该模型",
      code: "MODEL_ARTIFACT_INVALID",
      required_action: "retrain",
      model_type: "ma",
    });
  });

  test("predict requires and forwards the preparation token", async () => {
    const response = await request("/api/v1/models/ma/predict", "POST", {
      experiment_id: 990002,
      forecast_steps: 6,
      preparation_token: preparationToken,
    });
    expect(response.status).toBe(200);
    expect(predictExperimentModel).toHaveBeenCalledWith(
      "ma",
      20240002,
      990002,
      6,
      preparationToken,
    );
  });

  test("rejects prediction without a preparation token before taking locks", async () => {
    const response = await request("/api/v1/models/ma/predict", "POST", {
      experiment_id: 990002,
      forecast_steps: 6,
    });
    expect(response.status).toBe(400);
    expect(predictExperimentModel).not.toHaveBeenCalled();
  });

  test("returns 503 when process execution ownership is lost", async () => {
    runProductionPreparationStep.mockRejectedValueOnce(
      new ModelServiceUnavailableError("进程内模型执行锁已失效", "MODEL_EXECUTION_LOCK_LOST"),
    );
    const response = await request(
      "/api/v1/models/ma/production-preparation/sessions/production-1/steps/initialize/run",
      "POST",
    );
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "进程内模型执行锁已失效",
      code: "MODEL_EXECUTION_LOCK_LOST",
    });
  });

  test("returns structured 429 with Retry-After when admission is full", async () => {
    runProductionPreparationStep.mockRejectedValueOnce(new ModelCapacityBusyError());
    const response = await request(
      "/api/v1/models/ma/production-preparation/sessions/production-1/steps/initialize/run",
      "POST",
    );

    expect(response.status).toBe(429);
    expect(response.headers.get("Retry-After")).toBe("2");
    expect(await response.json()).toEqual({
      error: "模型服务繁忙，请稍后再试",
      code: "MODEL_CAPACITY_BUSY",
      retryable: true,
    });
  });

  test("returns structured 504 for the per-operation backend deadline", async () => {
    predictExperimentModel.mockRejectedValueOnce(
      new ModelOperationDeadlineError(660_000, "prediction"),
    );
    const response = await request("/api/v1/models/ma/predict", "POST", {
      experiment_id: 990002,
      forecast_steps: 6,
      preparation_token: preparationToken,
    });

    expect(response.status).toBe(504);
    expect(await response.json()).toEqual({
      error: "预测超过服务端时限，请重试",
      code: "MODEL_OPERATION_DEADLINE_EXCEEDED",
      operation: "prediction",
      timeout_ms: 660_000,
      retryable: true,
    });
  });

  test("returns a structured 504 when a guided step times out", async () => {
    runGuidedTrainingStep.mockRejectedValueOnce(
      new ModelTimeoutError("Python 执行超时", 300_000, "python_process"),
    );
    const response = await request(
      "/api/v1/models/ma/guided-training/sessions/guided-1/steps/forecast/run",
      "POST",
    );

    expect(response.status).toBe(504);
    expect(await response.json()).toEqual({
      error: "训练步骤超时，请重试当前步骤",
      code: "MODEL_TIMEOUT",
      operation: "guided_training_step",
      timeout_ms: 300_000,
      retryable: true,
    });
  });

  test("returns a structured 504 when guided artifact recovery times out", async () => {
    createOrResumeGuidedTrainingSession.mockRejectedValueOnce(
      new ModelTimeoutError("复制融合模型产物超时", 30_000, "python_process"),
    );
    const response = await request(
      "/api/v1/models/weighted_avg/guided-training/sessions",
      "POST",
      {
        ...trainBody(),
        models: "ma,es",
      },
    );

    expect(response.status).toBe(504);
    expect(await response.json()).toEqual({
      error: "恢复训练会话模型产物超时，请重试",
      code: "MODEL_TIMEOUT",
      operation: "guided_training_session",
      timeout_ms: 30_000,
      retryable: true,
    });
  });

  test("returns a structured 504 when production preparation times out", async () => {
    runProductionPreparationStep.mockRejectedValueOnce(
      new ModelTimeoutError("Python 执行超时", 600_000, "python_process"),
    );
    const response = await request(
      "/api/v1/models/weighted_avg/production-preparation/sessions/production-1/steps/fit_member_lstm/run",
      "POST",
    );

    expect(response.status).toBe(504);
    expect(await response.json()).toEqual({
      error: "生产准备步骤超时，请重试当前步骤",
      code: "MODEL_TIMEOUT",
      operation: "production_preparation",
      timeout_ms: 600_000,
      retryable: true,
    });
  });

  test("returns a structured 504 when prediction times out", async () => {
    predictExperimentModel.mockRejectedValueOnce(
      new ModelTimeoutError("Python 执行超时", 60_000, "python_subprocess"),
    );
    const response = await request("/api/v1/models/ma/predict", "POST", {
      experiment_id: 990002,
      forecast_steps: 6,
      preparation_token: preparationToken,
    });

    expect(response.status).toBe(504);
    expect(await response.json()).toEqual({
      error: "预测超时（60秒）",
      code: "MODEL_TIMEOUT",
      operation: "prediction",
      timeout_ms: 60_000,
      retryable: true,
    });
  });

  test("returns 409 when the service reports the experiment-scoped model is busy", async () => {
    runProductionPreparationStep.mockRejectedValueOnce(
      new ModelConflictError("同一模型正在训练或预测，请稍后再试", "MODEL_BUSY"),
    );
    const response = await request(
      "/api/v1/models/ma/production-preparation/sessions/production-1/steps/initialize/run",
      "POST",
    );
    expect(response.status).toBe(409);
  });

  test("rejects unsupported guided model types", async () => {
    const response = await request(
      "/api/v1/models/not-real/guided-training/sessions",
      "POST",
      trainBody(),
    );
    expect(response.status).toBe(400);
    expect(createOrResumeGuidedTrainingSession).not.toHaveBeenCalled();
  });
});
