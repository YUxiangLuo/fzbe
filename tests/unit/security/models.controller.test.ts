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
const activateGuidedTrainingSession = mock(async () => ({
  session_id: "guided-1",
  experiment_id: 990002,
  model_type: "ma",
  status: "completed",
  current_step_id: null,
  next_step_id: null,
  steps: [],
  step_outputs: {},
  result: { status: "success", results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } } },
  error_message: null,
  backtest_artifact_changed: true,
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
  result: { status: "success", results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } } },
  error_message: null,
}));
const completeExperimentModel = mock(async () => ({
  status: "success",
  artifact_revision: "11111111-1111-4111-8111-111111111111",
  experiment_state_version: 5,
}));
const acquireModelSlot = mock(async () => ({ connection: {} as never, name: "slot:1" }));
const releaseModelSlot = mock(async () => {});
const acquireModelExecutionLock = mock(async () => ({ connection: {} as never, name: "exec:1" }));
const releaseModelExecutionLock = mock(async () => {});
const acquireModelExecutionLockWithStatusImplementation = async () => {
  const lock = await acquireModelExecutionLock();
  return lock
    ? { status: "acquired" as const, lock }
    : { status: "conflict" as const };
};
const acquireModelExecutionLockWithStatus = mock(acquireModelExecutionLockWithStatusImplementation);
const acquireModelArtifactMutationLocks = mock(async () => ([
  { connection: {} as never, name: "exec:1" },
]));
const releaseModelArtifactMutationLocks = mock(async () => {});
const acquireModelJobLocksImplementation = async (experimentId: number, modelType: ModelType) => {
  const slotLock = await acquireModelSlot();
  if (!slotLock) return { status: "capacity" as const };

  const artifactLocks = await acquireModelArtifactMutationLocks(experimentId, modelType);
  if (!artifactLocks) {
    await releaseModelSlot(slotLock);
    return { status: "conflict" as const };
  }

  return {
    status: "acquired" as const,
    locks: { slotLock, artifactLocks },
  };
};
const acquireModelJobLocks = mock(acquireModelJobLocksImplementation);
const releaseModelJobLocksImplementation = async (locks: {
  slotLock: { connection: never; name: string };
  artifactLocks: Array<{ connection: never; name: string }>;
} | null) => {
  if (!locks) return;
  await releaseModelArtifactMutationLocks(locks.artifactLocks);
  await releaseModelSlot(locks.slotLock);
};
const releaseModelJobLocks = mock(releaseModelJobLocksImplementation);

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

class ModelCapacityError extends Error {}
class ModelConflictError extends Error {}
class ModelStaleAttemptError extends Error {}

mock.module("../../../src/modules/models/models.service", () => ({
  trainModel,
  getReusableProductionModelArtifactResult,
  prepareExperimentModelForProduction,
  predictExperimentModel,
  createOrResumeGuidedTrainingSession,
  getGuidedTrainingSession,
  activateGuidedTrainingSession,
  runGuidedTrainingStep,
  completeExperimentModel,
  acquireModelSlot,
  releaseModelSlot,
  acquireModelExecutionLock,
  acquireModelExecutionLockWithStatus,
  releaseModelExecutionLock,
  acquireModelArtifactMutationLocks,
  releaseModelArtifactMutationLocks,
  acquireModelJobLocks,
  releaseModelJobLocks,
  ModelValidationError,
  ModelNotFoundError,
  ModelAccessError,
  ModelCapacityError,
  ModelConflictError,
  ModelStaleAttemptError,
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

const requestGet = async (path: string) => {
  const headers = await createAuthHeaders();
  return app.request(path, { headers });
};

const requestPost = async (path: string) => {
  const headers = await createAuthHeaders();
  return app.request(path, { method: "POST", headers });
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

    createOrResumeGuidedTrainingSession.mockReset();
    createOrResumeGuidedTrainingSession.mockResolvedValue({
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
    });

    getGuidedTrainingSession.mockReset();
    getGuidedTrainingSession.mockResolvedValue({
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
    });

    activateGuidedTrainingSession.mockReset();
    activateGuidedTrainingSession.mockResolvedValue({
      session_id: "guided-1",
      experiment_id: 990002,
      model_type: "ma",
      status: "completed",
      current_step_id: null,
      next_step_id: null,
      steps: [],
      step_outputs: {},
      result: { status: "success", results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } } },
      error_message: null,
      backtest_artifact_changed: true,
    });

    runGuidedTrainingStep.mockReset();
    runGuidedTrainingStep.mockResolvedValue({
      session_id: "guided-1",
      experiment_id: 990002,
      model_type: "ma",
      status: "completed",
      current_step_id: null,
      next_step_id: null,
      steps: [],
      step_outputs: {},
      result: { status: "success", results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } } },
      error_message: null,
    });

    completeExperimentModel.mockReset();
    completeExperimentModel.mockResolvedValue({
      status: "success",
      artifact_revision: "11111111-1111-4111-8111-111111111111",
      experiment_state_version: 5,
    });

    acquireModelSlot.mockReset();
    acquireModelSlot.mockResolvedValue({ connection: {} as never, name: "slot:1" });

    releaseModelSlot.mockReset();
    releaseModelSlot.mockResolvedValue(undefined);

    acquireModelExecutionLock.mockReset();
    acquireModelExecutionLock.mockResolvedValue({ connection: {} as never, name: "exec:1" });

    acquireModelExecutionLockWithStatus.mockReset();
    acquireModelExecutionLockWithStatus.mockImplementation(acquireModelExecutionLockWithStatusImplementation);

    releaseModelExecutionLock.mockReset();
    releaseModelExecutionLock.mockResolvedValue(undefined);

    acquireModelArtifactMutationLocks.mockReset();
    acquireModelArtifactMutationLocks.mockResolvedValue([
      { connection: {} as never, name: "exec:1" },
    ]);

    releaseModelArtifactMutationLocks.mockReset();
    releaseModelArtifactMutationLocks.mockResolvedValue(undefined);

    acquireModelJobLocks.mockReset();
    acquireModelJobLocks.mockImplementation(acquireModelJobLocksImplementation);

    releaseModelJobLocks.mockReset();
    releaseModelJobLocks.mockImplementation(releaseModelJobLocksImplementation);
  });

  test("returns 409 and releases the slot when the experiment lock is unavailable", async () => {
    acquireModelArtifactMutationLocks.mockResolvedValueOnce(null);

    const response = await requestJson("/api/v1/models/ma/training", buildTrainBody());
    const data = await response.json();

    expect(response.status).toBe(409);
    expect(data).toEqual({ error: "同一模型正在训练或预测，请稍后再试" });
    expect(acquireModelSlot).toHaveBeenCalledTimes(1);
    expect(releaseModelSlot).toHaveBeenCalledTimes(1);
  });

  test("returns 429 before acquiring an experiment lock when no model slot is available", async () => {
    acquireModelSlot.mockResolvedValueOnce(null);

    const response = await requestJson("/api/v1/models/ma/training", buildTrainBody());
    const data = await response.json();

    expect(response.status).toBe(429);
    expect(data).toEqual({ error: "模型服务繁忙，请稍后再试" });
    expect(response.headers.get("retry-after")).toBe("3");
    expect(acquireModelArtifactMutationLocks).not.toHaveBeenCalled();
  });

  test("maps ModelAccessError from training to the declared HTTP status", async () => {
    trainModel.mockRejectedValueOnce(new ModelAccessError("权限不足：该实验不属于当前用户", 403));

    const response = await requestJson("/api/v1/models/ma/training", buildTrainBody());
    const data = await response.json();

    expect(response.status).toBe(403);
    expect(data).toEqual({ error: "权限不足：该实验不属于当前用户" });
    expect(releaseModelSlot).toHaveBeenCalledTimes(1);
    expect(releaseModelArtifactMutationLocks).toHaveBeenCalledTimes(1);
  });

  test("maps a stale direct-training activation to 409", async () => {
    trainModel.mockRejectedValueOnce(
      new ModelStaleAttemptError("训练期间模型版本已更新，本次结果已丢弃"),
    );

    const response = await requestJson("/api/v1/models/ma/training", buildTrainBody());

    expect(response.status).toBe(409);
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

  test("creates guided training sessions without acquiring the controller training slot", async () => {
    const response = await requestJson("/api/v1/models/ma/guided-training/sessions", buildTrainBody());
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toEqual({
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
    });
    expect(createOrResumeGuidedTrainingSession).toHaveBeenCalledWith(
      "ma",
      20240002,
      expect.objectContaining({ experiment_id: 990002, moving_average_window: 4 }),
    );
    expect(acquireModelArtifactMutationLocks).not.toHaveBeenCalled();
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
    expect(acquireModelSlot).not.toHaveBeenCalled();
  });

  test("maps guided-session creation cleanup conflicts to 409", async () => {
    createOrResumeGuidedTrainingSession.mockRejectedValueOnce(
      new ModelConflictError("该实验已有模型任务在执行，请稍后重试"),
    );

    const response = await requestJson(
      "/api/v1/models/ma/guided-training/sessions",
      buildTrainBody(),
    );

    expect(response.status).toBe(409);
  });

  test("gets a guided training session for the current student", async () => {
    const response = await requestGet("/api/v1/models/ma/guided-training/sessions/guided-1");
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.session_id).toBe("guided-1");
    expect(getGuidedTrainingSession).toHaveBeenCalledWith("ma", 20240002, "guided-1");
  });

  test("maps an invalidated guided-session lookup to 409", async () => {
    getGuidedTrainingSession.mockRejectedValueOnce(
      new ModelConflictError("实验训练输入或基础模型版本已更新，请重新开始分阶段训练"),
    );

    const response = await requestGet("/api/v1/models/ma/guided-training/sessions/guided-1");

    expect(response.status).toBe(409);
  });

  test("activates a completed guided training session explicitly", async () => {
    const response = await requestPost("/api/v1/models/ma/guided-training/sessions/guided-1/activate");
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.backtest_artifact_changed).toBe(true);
    expect(activateGuidedTrainingSession).toHaveBeenCalledWith("ma", 20240002, "guided-1");
  });

  test("maps a busy guided activation to 409", async () => {
    activateGuidedTrainingSession.mockRejectedValueOnce(
      new ModelConflictError("该实验已有模型任务在执行，请稍后重试"),
    );

    const response = await requestPost("/api/v1/models/ma/guided-training/sessions/guided-1/activate");

    expect(response.status).toBe(409);
  });

  test("runs the requested guided training step", async () => {
    const response = await requestPost("/api/v1/models/ma/guided-training/sessions/guided-1/steps/forecast/run");
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.status).toBe("completed");
    expect(runGuidedTrainingStep).toHaveBeenCalledWith("ma", 20240002, "guided-1", "forecast");
  });

  test("maps guided training busy errors to 429", async () => {
    runGuidedTrainingStep.mockRejectedValueOnce(new ModelCapacityError("模型服务繁忙，请稍后重试"));

    const response = await requestPost("/api/v1/models/ma/guided-training/sessions/guided-1/steps/forecast/run");
    const data = await response.json();

    expect(response.status).toBe(429);
    expect(data).toEqual({ error: "模型服务繁忙，请稍后重试" });
    expect(response.headers.get("retry-after")).toBe("3");
  });

  test("marks only the currently active artifact revision as completed", async () => {
    const revision = "11111111-1111-4111-8111-111111111111";
    const response = await requestJson("/api/v1/models/ma/complete", {
      experiment_id: 990002,
      artifact_revision: revision,
    });
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toEqual({
      status: "success",
      artifact_revision: revision,
      experiment_state_version: 5,
    });
    expect(acquireModelExecutionLock).toHaveBeenCalledWith(990002, "ma");
    expect(completeExperimentModel).toHaveBeenCalledWith(
      "ma",
      20240002,
      990002,
      revision,
    );
    expect(releaseModelExecutionLock).toHaveBeenCalledTimes(1);
  });

  test("returns 409 and releases the lock for a stale completion revision", async () => {
    completeExperimentModel.mockRejectedValueOnce(
      new ModelConflictError("模型结果版本已更新，请刷新页面后重试"),
    );
    const response = await requestJson("/api/v1/models/ma/complete", {
      experiment_id: 990002,
      artifact_revision: "11111111-1111-4111-8111-111111111111",
    });

    expect(response.status).toBe(409);
    expect(releaseModelExecutionLock).toHaveBeenCalledTimes(1);
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
    expect(releaseModelArtifactMutationLocks).toHaveBeenCalledTimes(1);
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
    expect(releaseModelArtifactMutationLocks).toHaveBeenCalledTimes(1);
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

  test("does not acquire an experiment lock when slot acquisition throws during training", async () => {
    acquireModelSlot.mockRejectedValueOnce(new Error("slot acquisition failed"));

    const response = await requestJson("/api/v1/models/ma/training", buildTrainBody());

    expect(response.status).toBe(500);
    expect(acquireModelArtifactMutationLocks).not.toHaveBeenCalled();
    expect(releaseModelArtifactMutationLocks).not.toHaveBeenCalled();
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

  test("maps a fenced production preparation attempt to 409", async () => {
    prepareExperimentModelForProduction.mockRejectedValueOnce(
      new ModelStaleAttemptError("回测模型版本已更新，本次生产模型结果已丢弃"),
    );

    const response = await requestJson("/api/v1/models/ma/prepare-production", {
      experiment_id: 990002,
    });

    expect(response.status).toBe(409);
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

  test("maps a fenced prediction result to 409", async () => {
    predictExperimentModel.mockRejectedValueOnce(
      new ModelStaleAttemptError("预测期间模型版本已更新，本次预测结果已丢弃"),
    );

    const response = await requestJson("/api/v1/models/ma/predict", {
      experiment_id: 990002,
      forecast_steps: 6,
    });

    expect(response.status).toBe(409);
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

  test("rejects prediction horizons above the fixed six-step production contract", async () => {
    const response = await requestJson("/api/v1/models/ma/predict", {
      experiment_id: 990002,
      forecast_steps: 7,
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({
      error: "预测步数必须小于等于 6",
      details: {
        formErrors: [],
        fieldErrors: {
          forecast_steps: ["预测步数必须小于等于 6"],
        },
      },
    });
    expect(predictExperimentModel).not.toHaveBeenCalled();
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
  });

  test("maps python validation failures from prediction to 400", async () => {
    predictExperimentModel.mockRejectedValueOnce(
      new Error("[PYTHON_VALIDATION] Saved LSTM model metadata is invalid."),
    );

    const response = await requestJson("/api/v1/models/lstm/predict", {
      experiment_id: 990002,
      forecast_steps: 6,
    });
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({ error: "Saved LSTM model metadata is invalid." });
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

  test("does not acquire an execution lock when slot acquisition throws during prediction", async () => {
    acquireModelSlot.mockRejectedValueOnce(new Error("slot acquisition failed"));

    const response = await requestJson("/api/v1/models/ma/predict", {
      experiment_id: 990002,
      forecast_steps: 6,
    });

    expect(response.status).toBe(500);
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
    expect(releaseModelExecutionLock).not.toHaveBeenCalled();
  });

  test("rejects unsupported model types before invoking the service layer", async () => {
    const response = await requestJson("/api/v1/models/not-real/training", buildTrainBody());
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({ error: "指定的模型类型无效" });
    expect(acquireModelArtifactMutationLocks).not.toHaveBeenCalled();
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
    expect(trainModel).not.toHaveBeenCalled();
  });
});
