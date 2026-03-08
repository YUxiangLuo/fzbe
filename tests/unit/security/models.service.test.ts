import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const pythonModulePath = path.join(srcRoot, "utils", "python.ts");
const modelsServiceModulePath = path.join(srcRoot, "modules", "models", "models.service.ts");
const freshModelsServiceModulePath = `${modelsServiceModulePath}?models-service-test`;

const poolQuery = mock(async () => [[]]);
const existsSync = mock((_: string) => false);
const createReadStream = mock(() => {
  throw new Error("createReadStream should not be called in this test");
});

mock.module(configModulePath, () => ({
  pool: {
    query: poolQuery,
  },
}));

mock.module("fs", () => ({
  default: {
    existsSync,
    createReadStream,
  },
  existsSync,
  createReadStream,
}));

mock.module(pythonModulePath, () => ({
  PY_DIR: "/tmp/fangzhen-py",
  resolvePythonExecutable: () => "/tmp/fangzhen-py/venv/bin/python",
}));

const {
  trainModel,
  predictExperimentModel,
  ModelAccessError,
  ModelNotFoundError,
} = await import(freshModelsServiceModulePath);

describe("models service", () => {
  beforeEach(() => {
    poolQuery.mockReset();
    poolQuery.mockResolvedValue([
      [{
        student_id: 20240002,
        status: "In Progress",
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
      }],
    ]);

    existsSync.mockReset();
    existsSync.mockReturnValue(false);

    createReadStream.mockClear();
  });

  test("blocks training for completed experiments before loading dataset data", async () => {
    poolQuery.mockResolvedValueOnce([
      [{
        student_id: 20240002,
        status: "Completed",
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
      }],
    ]);

    await expect(
      trainModel("ma", 20240002, {
        experiment_id: 990002,
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
        moving_average_window: 4,
      }),
    ).rejects.toEqual(
      expect.objectContaining({
        message: "已完成的实验不允许继续训练或预测模型",
        status: 400,
      }),
    );

    expect(createReadStream).not.toHaveBeenCalled();
  });

  test("rejects training when request payload does not match the persisted experiment selection", async () => {
    await expect(
      trainModel("ma", 20240002, {
        experiment_id: 990002,
        selected_industry: "制造业",
        selected_company: "其他企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
        moving_average_window: 4,
      }),
    ).rejects.toMatchObject({
      message: "请求中的企业与实验记录不一致，请刷新页面后重试",
    });

    expect(createReadStream).not.toHaveBeenCalled();
  });

  test("rejects training when the persisted experiment setup is incomplete", async () => {
    poolQuery.mockResolvedValueOnce([
      [{
        student_id: 20240002,
        status: "In Progress",
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: null,
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
      }],
    ]);

    await expect(
      trainModel("ma", 20240002, {
        experiment_id: 990002,
        moving_average_window: 4,
      }),
    ).rejects.toMatchObject({
      message: "实验尚未完成数据集选择，无法训练模型",
    });

    expect(createReadStream).not.toHaveBeenCalled();
  });

  test("rejects prediction when the experiment belongs to another student", async () => {
    poolQuery.mockResolvedValueOnce([
      [{
        student_id: 99999999,
        status: "In Progress",
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
      }],
    ]);

    const predictionPromise = predictExperimentModel("ma", 20240002, 990002, 6);

    await expect(predictionPromise).rejects.toBeInstanceOf(ModelAccessError);
    await expect(predictionPromise).rejects.toMatchObject({
      message: "权限不足：该实验不属于当前用户",
      status: 403,
    });
  });

  test("rejects prediction when the experiment does not exist", async () => {
    poolQuery.mockResolvedValueOnce([[]]);

    await expect(
      predictExperimentModel("ma", 20240002, 990002, 6),
    ).rejects.toMatchObject({
      message: "实验不存在",
      status: 404,
    });
  });

  test("raises ModelNotFoundError when the experiment-scoped model artifact is missing", async () => {
    await expect(
      predictExperimentModel("ma", 20240002, 990002, 6),
    ).rejects.toBeInstanceOf(ModelNotFoundError);

    expect(existsSync).toHaveBeenCalledWith(
      expect.stringContaining("exp990002-student20240002-ma.pkl"),
    );
  });
});