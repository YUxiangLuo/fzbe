process.env.DB_HOST ??= "127.0.0.1";
process.env.DB_USER ??= "test";
process.env.DB_PASSWORD ??= "test";
process.env.DB_DATABASE ??= "test";
process.env.JWT_SECRET ??= "test-jwt-secret";
process.env.PORT ??= "3001";
process.env.DEFAULT_PASSWORD_HASH ??= "test-default-hash";

import { beforeEach, describe, expect, mock, test } from "bun:test";
import { Hono } from "hono";
import { sign } from "hono/jwt";
import type { ModelType, TrainInput } from "../../../src/modules/models/models.schema";

const trainModel = mock(async () => ({
  status: "success",
  results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
}));
const predictExperimentModel = mock(async () => ({
  status: "success",
  results: { predictions: [] },
}));
const acquireModelSlot = mock(async () => ({ connection: {} as never, name: "slot:1" }));
const releaseModelSlot = mock(async () => {});
const acquireModelExecutionLock = mock(async () => ({ connection: {} as never, name: "exec:1" }));
const releaseModelExecutionLock = mock(async () => {});

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

mock.module("../../../src/modules/models/models.service", () => ({
  trainModel,
  predictExperimentModel,
  acquireModelSlot,
  releaseModelSlot,
  acquireModelExecutionLock,
  releaseModelExecutionLock,
  ModelValidationError,
  ModelNotFoundError,
  ModelAccessError,
}));

const { default: modelsController } = await import("../../../src/modules/models/models.controller");

const app = new Hono();
app.route("/", modelsController);

const createAuthHeaders = async () => {
  const token = await sign(
    {
      sub: 20240002,
      username: "20240002",
      role: "Student",
      exp: Math.floor(Date.now() / 1000) + 60 * 10,
    },
    process.env.JWT_SECRET!,
  );

  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
};

const buildTrainBody = (): TrainInput => ({
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

const requestJson = async (path: string, body: Record<string, unknown>) => {
  const headers = await createAuthHeaders();
  return app.request(path, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
};

describe("models controller", () => {
  beforeEach(() => {
    trainModel.mockReset();
    trainModel.mockResolvedValue({
      status: "success",
      results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
    });

    predictExperimentModel.mockReset();
    predictExperimentModel.mockResolvedValue({
      status: "success",
      results: { predictions: [] },
    });

    acquireModelSlot.mockReset();
    acquireModelSlot.mockResolvedValue({ connection: {} as never, name: "slot:1" });

    releaseModelSlot.mockReset();
    releaseModelSlot.mockResolvedValue(undefined);

    acquireModelExecutionLock.mockReset();
    acquireModelExecutionLock.mockResolvedValue({ connection: {} as never, name: "exec:1" });

    releaseModelExecutionLock.mockReset();
    releaseModelExecutionLock.mockResolvedValue(undefined);
  });

  test("returns 409 when the experiment-scoped execution lock is unavailable", async () => {
    acquireModelExecutionLock.mockResolvedValueOnce(null);

    const response = await requestJson("/api/v1/models/ma/training", buildTrainBody());
    const data = await response.json();

    expect(response.status).toBe(409);
    expect(data).toEqual({ error: "同一模型正在训练或预测，请稍后再试" });
    expect(acquireModelSlot).not.toHaveBeenCalled();
  });

  test("returns 429 and releases the execution lock when no model slot is available", async () => {
    const executionLock = { connection: {} as never, name: "exec:busy" };
    acquireModelExecutionLock.mockResolvedValueOnce(executionLock);
    acquireModelSlot.mockResolvedValueOnce(null);

    const response = await requestJson("/api/v1/models/ma/training", buildTrainBody());
    const data = await response.json();

    expect(response.status).toBe(429);
    expect(data).toEqual({ error: "模型服务繁忙，请稍后再试" });
    expect(releaseModelExecutionLock).toHaveBeenCalledWith(executionLock);
  });

  test("maps ModelAccessError from training to the declared HTTP status", async () => {
    trainModel.mockRejectedValueOnce(new ModelAccessError("权限不足：该实验不属于当前用户", 403));

    const response = await requestJson("/api/v1/models/ma/training", buildTrainBody());
    const data = await response.json();

    expect(response.status).toBe(403);
    expect(data).toEqual({ error: "权限不足：该实验不属于当前用户" });
    expect(releaseModelSlot).toHaveBeenCalledTimes(1);
    expect(releaseModelExecutionLock).toHaveBeenCalledTimes(1);
  });

  test("releases the execution lock when slot acquisition throws during training", async () => {
    const executionLock = { connection: {} as never, name: "exec:throw" };
    acquireModelExecutionLock.mockResolvedValueOnce(executionLock);
    acquireModelSlot.mockRejectedValueOnce(new Error("slot acquisition failed"));

    const response = await requestJson("/api/v1/models/ma/training", buildTrainBody());

    expect(response.status).toBe(500);
    expect(releaseModelExecutionLock).toHaveBeenCalledWith(executionLock);
    expect(releaseModelSlot).toHaveBeenCalledWith(null);
  });

  test("maps ModelNotFoundError from prediction to 404", async () => {
    predictExperimentModel.mockRejectedValueOnce(new ModelNotFoundError("未找到该实验的已训练 MA 模型"));

    const response = await requestJson("/api/v1/models/ma/predict", {
      experiment_id: 990002,
      forecast_steps: 6,
    });
    const data = await response.json();

    expect(response.status).toBe(404);
    expect(data).toEqual({ error: "未找到该实验的已训练 MA 模型" });
  });

  test("translates prediction timeouts to 504", async () => {
    predictExperimentModel.mockRejectedValueOnce(new Error("Python 执行超时 (30000ms)"));

    const response = await requestJson("/api/v1/models/ma/predict", {
      experiment_id: 990002,
      forecast_steps: 6,
    });
    const data = await response.json();

    expect(response.status).toBe(504);
    expect(data.error).toContain("预测超时");
  });

  test("releases the execution lock when slot acquisition throws during prediction", async () => {
    const executionLock = { connection: {} as never, name: "exec:predict-throw" };
    acquireModelExecutionLock.mockResolvedValueOnce(executionLock);
    acquireModelSlot.mockRejectedValueOnce(new Error("slot acquisition failed"));

    const response = await requestJson("/api/v1/models/ma/predict", {
      experiment_id: 990002,
      forecast_steps: 6,
    });

    expect(response.status).toBe(500);
    expect(releaseModelExecutionLock).toHaveBeenCalledWith(executionLock);
    expect(releaseModelSlot).toHaveBeenCalledWith(null);
  });

  test("rejects unsupported model types before invoking the service layer", async () => {
    const response = await requestJson("/api/v1/models/not-real/training", buildTrainBody());
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({ error: "指定的模型类型无效" });
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
    expect(trainModel).not.toHaveBeenCalled();
  });
});