process.env.DB_HOST = "127.0.0.1";
process.env.DB_USER = "test";
process.env.DB_PASSWORD = "test";
process.env.DB_DATABASE = "test";
process.env.JWT_SECRET = "test-jwt-secret";
process.env.PORT = "3001";

import { beforeEach, describe, expect, mock, test } from "bun:test";
import { Hono } from "hono";
import { sign } from "hono/jwt";
import type { ModelType, TrainInput } from "../../../src/modules/models/models.schema";

mock.restore();

const trainModel = mock(async () => ({
  status: "success",
  results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
}));
const prepareExperimentModelForProduction = mock(async () => ({
  status: "success",
  results: { saved_model: "/tmp/fangzhen-py/trained_models/example-production.pkl" },
}));
const getReusableProductionModelArtifactResult = mock(async () => null as null | {
  status: "success";
  results: { saved_model: string; artifact_phase: "production"; reused_existing: boolean };
});
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

const readNdjson = async (response: Response) => {
  const text = await response.text();
  return text
    .trim()
    .split(/\n+/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
};

describe("models controller", () => {
  beforeEach(() => {
    trainModel.mockReset();
    trainModel.mockResolvedValue({
      status: "success",
      results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
    });

    prepareExperimentModelForProduction.mockReset();
    prepareExperimentModelForProduction.mockResolvedValue({
      status: "success",
      results: { saved_model: "/tmp/fangzhen-py/trained_models/example-production.pkl" },
    });

    getReusableProductionModelArtifactResult.mockReset();
    getReusableProductionModelArtifactResult.mockResolvedValue(null);

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

  test("maps python validation failures from training to 400", async () => {
    trainModel.mockRejectedValueOnce(new Error("[PYTHON_VALIDATION] Weighted Average requires at least 2 base models."));

    const response = await requestJson("/api/v1/models/weighted_avg/training", {
      ...buildTrainBody(),
      models: "ma",
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({ error: "Weighted Average requires at least 2 base models." });
  });

  test("maps python data failures from training to 400", async () => {
    trainModel.mockRejectedValueOnce(new Error("[PYTHON_DATA] Failed to parse or process --data: bad column"));

    const response = await requestJson("/api/v1/models/lstm/training", {
      ...buildTrainBody(),
      lstmNormalization: "minmax",
      lstmTargetFeature: "销售数量",
      lstmFeatures: "bad column",
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({ error: "Failed to parse or process --data: bad column" });
  });

  test("translates training timeouts to 504", async () => {
    trainModel.mockRejectedValueOnce(new Error("Python 执行超时 (120000ms)"));

    const response = await requestJson("/api/v1/models/ma/training", buildTrainBody());
    const data = await response.json();

    expect(response.status).toBe(504);
    expect(data.error).toContain("训练超时");
  });

  test("rejects underspecified weighted ensemble training requests before invoking the service layer", async () => {
    const response = await requestJson("/api/v1/models/weighted_avg/training", {
      ...buildTrainBody(),
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({
      error: "基础模型列表不能为空",
      details: {
        formErrors: [],
        fieldErrors: {
          models: ["基础模型列表不能为空"],
        },
      },
    });
    expect(trainModel).not.toHaveBeenCalled();
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
  });

  test("rejects underspecified LSTM training requests before invoking the service layer", async () => {
    const response = await requestJson("/api/v1/models/lstm/training", {
      ...buildTrainBody(),
      lstmTargetFeature: "销售数量",
      lstmFeatures: "价格指数,产能利用率",
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({
      error: "LSTM 归一化方式必须是 minmax 或 zscore",
      details: {
        formErrors: [],
        fieldErrors: {
          lstmNormalization: ["LSTM 归一化方式必须是 minmax 或 zscore"],
        },
      },
    });
    expect(trainModel).not.toHaveBeenCalled();
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
  });

  test("accepts LSTM feature arrays and forwards them as JSON strings", async () => {
    const response = await requestJson("/api/v1/models/lstm/training", {
      ...buildTrainBody(),
      lstmNormalization: "minmax",
      lstmTargetFeature: "销售数量",
      lstmFeatures: ["价格指数", "外部,指标"],
    });
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toEqual({
      status: "success",
      results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
    });
    expect(trainModel).toHaveBeenCalledWith("lstm", 20240002, expect.objectContaining({
      lstmFeatures: JSON.stringify(["价格指数", "外部,指标"]),
    }));
  });

  test("streams training progress and the final result", async () => {
    trainModel.mockImplementationOnce(async (_modelType, _studentId, _body, reportProgress) => {
      reportProgress?.({
        type: "phase",
        phase: "python_training",
        percent: 50,
        source: "python",
        message: "正在执行 Python 训练",
      });
      return {
        status: "success",
        results: { metrics: { rmse: 2, mae: 1.5, r2: 0.8 } },
      };
    });

    const response = await requestJson("/api/v1/models/ma/training/stream", buildTrainBody());
    const events = await readNdjson(response);

    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("application/x-ndjson");
    expect(events.some((event) => event.type === "progress" && event.phase === "queued")).toBe(true);
    expect(events.some((event) => event.type === "phase" && event.message === "正在执行 Python 训练")).toBe(true);
    expect(events.at(-1)).toEqual({
      type: "result",
      timestamp: expect.any(String),
      data: {
        status: "success",
        results: { metrics: { rmse: 2, mae: 1.5, r2: 0.8 } },
      },
    });
    expect(trainModel).toHaveBeenCalledWith("ma", 20240002, expect.objectContaining({
      moving_average_window: 4,
    }), expect.any(Function));
    expect(releaseModelSlot).toHaveBeenCalledTimes(1);
    expect(releaseModelExecutionLock).toHaveBeenCalledTimes(1);
  });

  test("does not expose raw Python stderr in training stream errors", async () => {
    trainModel.mockRejectedValueOnce(new Error(
      "Python 执行失败: Traceback (most recent call last): File \"/opt/fangzhen2/fzbe/py/src/models/ma.py\"",
    ));

    const response = await requestJson("/api/v1/models/ma/training/stream", buildTrainBody());
    const events = await readNdjson(response);
    const errorEvent = events.find((event) => event.type === "error");

    expect(response.status).toBe(200);
    expect(errorEvent).toEqual({
      type: "error",
      timestamp: expect.any(String),
      status: 500,
      message: "MA 训练失败，请稍后重试",
    });
    expect(JSON.stringify(events)).not.toContain("Traceback");
    expect(JSON.stringify(events)).not.toContain("/opt/fangzhen2");
    expect(releaseModelSlot).toHaveBeenCalledTimes(1);
    expect(releaseModelExecutionLock).toHaveBeenCalledTimes(1);
  });

  test("returns localized field errors for missing moving-average params", async () => {
    const body = buildTrainBody();
    delete (body as Partial<TrainInput>).moving_average_window;

    const response = await requestJson("/api/v1/models/ma/training", body as Record<string, unknown>);
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({
      error: "移动平均时间窗口不能为空",
      details: {
        formErrors: [],
        fieldErrors: {
          moving_average_window: ["移动平均时间窗口不能为空"],
        },
      },
    });
    expect(trainModel).not.toHaveBeenCalled();
  });

  test("returns localized lower-bound errors for invalid moving-average params", async () => {
    const response = await requestJson("/api/v1/models/ma/training", {
      ...buildTrainBody(),
      moving_average_window: 1,
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({
      error: "移动平均时间窗口必须大于等于 2",
      details: {
        formErrors: [],
        fieldErrors: {
          moving_average_window: ["移动平均时间窗口必须大于等于 2"],
        },
      },
    });
    expect(trainModel).not.toHaveBeenCalled();
  });

  test("returns localized integer errors for decimal moving-average params", async () => {
    const response = await requestJson("/api/v1/models/ma/training", {
      ...buildTrainBody(),
      moving_average_window: 2.5,
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({
      error: "移动平均时间窗口必须为整数",
      details: {
        formErrors: [],
        fieldErrors: {
          moving_average_window: ["移动平均时间窗口必须为整数"],
        },
      },
    });
    expect(trainModel).not.toHaveBeenCalled();
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

  test("returns a reusable base production artifact without acquiring a global model slot", async () => {
    getReusableProductionModelArtifactResult.mockResolvedValueOnce({
      status: "success",
      results: {
        saved_model: "/tmp/fangzhen-py/trained_models/example-production.pkl",
        artifact_phase: "production",
        reused_existing: true,
      },
    });

    const response = await requestJson("/api/v1/models/ma/prepare-production", {
      experiment_id: 990002,
    });
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toEqual({
      status: "success",
      results: {
        saved_model: "/tmp/fangzhen-py/trained_models/example-production.pkl",
        artifact_phase: "production",
        reused_existing: true,
      },
    });
    expect(getReusableProductionModelArtifactResult).toHaveBeenCalledWith("ma", 20240002, 990002);
    expect(acquireModelSlot).not.toHaveBeenCalled();
    expect(prepareExperimentModelForProduction).not.toHaveBeenCalled();
  });

  test("checks reusable ensemble production artifacts behind the global model slot", async () => {
    const callOrder: string[] = [];
    const slotLock = { connection: {} as never, name: "slot:ensemble" };
    acquireModelSlot.mockImplementationOnce(async () => {
      callOrder.push("slot");
      return slotLock;
    });
    prepareExperimentModelForProduction.mockImplementationOnce(async () => {
      callOrder.push("prepare");
      return {
        status: "success",
        results: {
          saved_model: "/tmp/fangzhen-py/trained_models/example-production.pkl",
          artifact_phase: "production",
          reused_existing: true,
        },
      };
    });

    const response = await requestJson("/api/v1/models/weighted_avg/prepare-production", {
      experiment_id: 990002,
    });
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toEqual({
      status: "success",
      results: {
        saved_model: "/tmp/fangzhen-py/trained_models/example-production.pkl",
        artifact_phase: "production",
        reused_existing: true,
      },
    });
    expect(getReusableProductionModelArtifactResult).not.toHaveBeenCalled();
    expect(callOrder).toEqual(["slot", "prepare"]);
    expect(releaseModelSlot).toHaveBeenCalledWith(slotLock);
  });

  test("prepares a base model for production prediction", async () => {
    const response = await requestJson("/api/v1/models/ma/prepare-production", {
      experiment_id: 990002,
    });
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toEqual({
      status: "success",
      results: { saved_model: "/tmp/fangzhen-py/trained_models/example-production.pkl" },
    });
    expect(getReusableProductionModelArtifactResult).toHaveBeenCalledWith("ma", 20240002, 990002);
    expect(prepareExperimentModelForProduction).toHaveBeenCalledWith("ma", 20240002, 990002);
  });

  test("prepares an ensemble model for production prediction", async () => {
    const response = await requestJson("/api/v1/models/weighted_avg/prepare-production", {
      experiment_id: 990002,
    });
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toEqual({
      status: "success",
      results: { saved_model: "/tmp/fangzhen-py/trained_models/example-production.pkl" },
    });
    expect(getReusableProductionModelArtifactResult).not.toHaveBeenCalled();
    expect(acquireModelSlot).toHaveBeenCalledTimes(1);
    expect(prepareExperimentModelForProduction).toHaveBeenCalledWith("weighted_avg", 20240002, 990002);
  });

  test("rejects invalid production preparation payloads before invoking the service layer", async () => {
    const response = await requestJson("/api/v1/models/ma/prepare-production", {});
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({ error: "生产预测准备请求参数无效" });
    expect(prepareExperimentModelForProduction).not.toHaveBeenCalled();
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
  });

  test("maps ModelNotFoundError from production preparation to 404", async () => {
    prepareExperimentModelForProduction.mockRejectedValueOnce(
      new ModelNotFoundError("未找到该实验的已训练 MA 回测模型"),
    );

    const response = await requestJson("/api/v1/models/ma/prepare-production", {
      experiment_id: 990002,
    });
    const data = await response.json();

    expect(response.status).toBe(404);
    expect(data).toEqual({ error: "未找到该实验的已训练 MA 回测模型" });
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

  test("maps ModelValidationError from prediction to 400", async () => {
    predictExperimentModel.mockRejectedValueOnce(
      new ModelValidationError("移动平均基础模型尚未完成训练，无法准备生产预测模型"),
    );

    const response = await requestJson("/api/v1/models/ma/predict", {
      experiment_id: 990002,
      forecast_steps: 6,
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({ error: "移动平均基础模型尚未完成训练，无法准备生产预测模型" });
  });

  test("returns localized field errors for missing prediction params", async () => {
    const response = await requestJson("/api/v1/models/ma/predict", {
      experiment_id: 990002,
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({
      error: "预测步数不能为空",
      details: {
        formErrors: [],
        fieldErrors: {
          forecast_steps: ["预测步数不能为空"],
        },
      },
    });
    expect(predictExperimentModel).not.toHaveBeenCalled();
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
  });

  test("returns localized field errors for invalid prediction params", async () => {
    const response = await requestJson("/api/v1/models/ma/predict", {
      experiment_id: 990002,
      forecast_steps: 0,
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({
      error: "预测步数必须大于等于 1",
      details: {
        formErrors: [],
        fieldErrors: {
          forecast_steps: ["预测步数必须大于等于 1"],
        },
      },
    });
    expect(predictExperimentModel).not.toHaveBeenCalled();
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
  });

  test("maps python validation failures from prediction to 400", async () => {
    predictExperimentModel.mockRejectedValueOnce(
      new Error("[PYTHON_VALIDATION] Requested forecast_steps (20) exceeds the saved LSTM horizon (12)."),
    );

    const response = await requestJson("/api/v1/models/lstm/predict", {
      experiment_id: 990002,
      forecast_steps: 20,
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({ error: "Requested forecast_steps (20) exceeds the saved LSTM horizon (12)." });
  });

  test("maps python data failures from prediction to 400", async () => {
    predictExperimentModel.mockRejectedValueOnce(new Error("[PYTHON_DATA] Saved LSTM history is invalid"));

    const response = await requestJson("/api/v1/models/lstm/predict", {
      experiment_id: 990002,
      forecast_steps: 6,
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({ error: "Saved LSTM history is invalid" });
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
