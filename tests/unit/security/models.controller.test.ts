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
const combinedSignal = new AbortController().signal;
const executionLock = { connection: {} as never, name: "exec:1" };
const slotLock = { connection: {} as never, name: "slot:1" };

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
const getReusableProductionModelArtifactResult = mock(async () => null as unknown);
const prepareExperimentModelForProduction = mock(async () => ({
  status: "success",
  results: {
    preparation_token: preparationToken,
    prepared_forecast_steps: 6,
    reused_existing: false,
  },
}));
const predictExperimentModel = mock(async () => ({
  status: "success",
  results: { predictions: [] },
}));
const acquireModelExecutionLock = mock(async () => executionLock);
const releaseModelExecutionLock = mock(async () => {});
const acquireModelSlot = mock(async () => slotLock);
const releaseModelSlot = mock(async () => {});
const combineModelLockSignals = mock(() => combinedSignal);

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

mock.module("../../../src/modules/models/models.service", () => ({
  createOrResumeGuidedTrainingSession,
  getGuidedTrainingSession,
  runGuidedTrainingStep,
  discardGuidedTrainingSession,
  getReusableProductionModelArtifactResult,
  prepareExperimentModelForProduction,
  predictExperimentModel,
  acquireModelExecutionLock,
  releaseModelExecutionLock,
  acquireModelSlot,
  releaseModelSlot,
  combineModelLockSignals,
  ModelValidationError,
  ModelNotFoundError,
  ModelAccessError,
  ModelConflictError,
  ModelServiceUnavailableError,
}));

const { default: modelsController } = await import("../../../src/modules/models/models.controller");
const app = new Hono();
app.route("/", modelsController);

const authHeaders = async () => ({
  Authorization: `Bearer ${await sign({
    sub: 20240002,
    username: "20240002",
    role: "Student",
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
      getReusableProductionModelArtifactResult,
      prepareExperimentModelForProduction,
      predictExperimentModel,
      acquireModelExecutionLock,
      releaseModelExecutionLock,
      acquireModelSlot,
      releaseModelSlot,
      combineModelLockSignals,
    ]) fn.mockClear();
    getReusableProductionModelArtifactResult.mockResolvedValue(null);
    acquireModelExecutionLock.mockResolvedValue(executionLock);
    acquireModelSlot.mockResolvedValue(slotLock);
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
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
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

  test("prepares production explicitly and passes the lock-loss signal", async () => {
    const response = await request("/api/v1/models/ma/prepare-production", "POST", {
      experiment_id: 990002,
      forecast_steps: 6,
    });
    expect(response.status).toBe(200);
    expect(prepareExperimentModelForProduction).toHaveBeenCalledWith(
      "ma",
      20240002,
      990002,
      6,
      false,
      combinedSignal,
    );
    expect(releaseModelSlot).toHaveBeenCalledWith(slotLock);
    expect(releaseModelExecutionLock).toHaveBeenCalledWith(executionLock);
  });

  test("reuses a prepared base artifact without taking a global slot", async () => {
    getReusableProductionModelArtifactResult.mockResolvedValueOnce({
      status: "success",
      results: {
        preparation_token: preparationToken,
        prepared_forecast_steps: 6,
        reused_existing: true,
      },
    });
    const response = await request("/api/v1/models/ma/prepare-production", "POST", {
      experiment_id: 990002,
      forecast_steps: 6,
    });
    expect(response.status).toBe(200);
    expect(acquireModelSlot).not.toHaveBeenCalled();
    expect(prepareExperimentModelForProduction).not.toHaveBeenCalled();
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
      combinedSignal,
    );
  });

  test("rejects prediction without a preparation token before taking locks", async () => {
    const response = await request("/api/v1/models/ma/predict", "POST", {
      experiment_id: 990002,
      forecast_steps: 6,
    });
    expect(response.status).toBe(400);
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
    expect(predictExperimentModel).not.toHaveBeenCalled();
  });

  test("returns 503 when lock ownership is lost", async () => {
    prepareExperimentModelForProduction.mockRejectedValueOnce(
      new ModelServiceUnavailableError("数据库模型锁已失效", "MODEL_LOCK_LOST"),
    );
    const response = await request("/api/v1/models/ma/prepare-production", "POST", {
      experiment_id: 990002,
      forecast_steps: 6,
    });
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "数据库模型锁已失效",
      code: "MODEL_LOCK_LOST",
    });
  });

  test("returns 409 when the experiment-scoped lock is busy", async () => {
    acquireModelExecutionLock.mockResolvedValueOnce(null);
    const response = await request("/api/v1/models/ma/prepare-production", "POST", {
      experiment_id: 990002,
      forecast_steps: 6,
    });
    expect(response.status).toBe(409);
    expect(acquireModelSlot).not.toHaveBeenCalled();
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
