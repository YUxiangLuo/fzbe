import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import * as actualFs from "node:fs";
import { unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const pythonModulePath = path.join(srcRoot, "utils", "python.ts");
const modelsServiceModulePath = path.join(srcRoot, "modules", "models", "models.service.ts");
const freshModelsServiceModulePath = `${modelsServiceModulePath}?models-service-test`;
const datasetCsvPath = "/tmp/dataset.csv";
const modelDatasetCsv =
  "月份,销售数量,价格指数,产能利用率\n" +
  "2024-01,100,1.2,0.8\n" +
  "2024-02,110,1.3,0.9\n" +
  "2024-03,120,1.4,1.0\n" +
  "2024-04,130,1.5,1.1\n" +
  "2024-05,140,1.6,1.2\n" +
  "2024-06,150,1.7,1.3\n" +
  "2024-07,160,1.8,1.4\n" +
  "2024-08,170,1.9,1.5\n" +
  "2024-09,180,2.0,1.6\n" +
  "2024-10,190,2.1,1.7\n" +
  "2024-11,200,2.2,1.8\n" +
  "2024-12,210,2.3,1.9\n" +
  "2025-01,220,2.4,2.0\n" +
  "2025-02,230,2.5,2.1\n" +
  "2025-03,240,2.6,2.2\n" +
  "2025-04,250,2.7,2.3\n" +
  "2025-05,260,2.8,2.4\n" +
  "2025-06,270,2.9,2.5\n";

const poolQuery = mock(async () => [[]]);
const existsSync = mock((_: string) => false);
const createReadStream = mock(() => {
  throw new Error("createReadStream should not be called in this test");
});
const readdirSync = mock(() => []);
const unlinkSync = mock(() => {});
const copyFileSync = mock((_: string, __: string) => {});
const spawnMock = mock((_: string[], __: Record<string, unknown>) => ({
  stdout: new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('{"predictions":[]}'));
      controller.close();
    },
  }),
  stderr: new ReadableStream({
    start(controller) {
      controller.close();
    },
  }),
  exited: Promise.resolve(0),
  exitCode: 0,
  killed: false,
  pid: 4321,
  kill: mock(() => {}),
}));
const originalSpawn = Bun.spawn;

mock.module(configModulePath, () => ({
  env: {
    DB_HOST: process.env.DB_HOST ?? "127.0.0.1",
    DB_USER: process.env.DB_USER ?? "test",
    DB_PASSWORD: process.env.DB_PASSWORD ?? "test",
    DB_DATABASE: process.env.DB_DATABASE ?? "test",
    JWT_SECRET: "test-jwt-secret",
    PORT: Number(process.env.PORT ?? "3001"),
  },
  connectionLimit: 64,
  pool: {
    query: poolQuery,
  },
}));

mock.module("fs", () => ({
  ...actualFs,
  default: {
    ...actualFs.default,
    existsSync,
    createReadStream,
    readdirSync,
    unlinkSync,
    copyFileSync,
  },
  existsSync,
  createReadStream,
  readdirSync,
  unlinkSync,
  copyFileSync,
}));

mock.module(pythonModulePath, () => ({
  PY_DIR: "/tmp/fangzhen-py",
  resolvePythonExecutable: () => "/tmp/fangzhen-py/venv/bin/python",
}));

const {
  trainModel,
  runPythonModel,
  createOrResumeGuidedTrainingSession,
  prepareExperimentModelForProduction,
  predictExperimentModel,
  ModelAccessError,
  ModelNotFoundError,
} = await import(freshModelsServiceModulePath);

const buildExperimentContextRow = (overrides: Record<string, unknown> = {}) => ({
  student_id: 20240002,
  status: "In Progress",
  selected_industry: "制造业",
  selected_company: "样本企业",
  selected_product: "样本产品",
  data_window_train_start_index: 0,
  data_window_train_end_index: 11,
  data_window_evaluate_start_index: 12,
  data_window_evaluate_end_index: 17,
  moving_average_completed: 1,
  moving_average_window: 4,
  exponential_smoothing_completed: 1,
  exponential_smoothing_alpha: 0.35,
  arima_completed: 1,
  arima_d: 1,
  lstm_completed: 1,
  lstm_normalization: "minmax",
  lstm_features: JSON.stringify(["价格指数", "产能利用率"]),
  lstm_target_field: "销售数量",
  ensemble_weighted_completed: 1,
  ensemble_weighted_base_models: JSON.stringify(["ma", "lstm"]),
  ensemble_boosting_completed: 1,
  ensemble_boosting_base_models: JSON.stringify(["ma", "arima"]),
  ensemble_stacking_completed: 1,
  ensemble_stacking_base_models: JSON.stringify(["ma", "lstm"]),
  ...overrides,
});

describe("models service", () => {
  beforeEach(async () => {
    await writeFile(datasetCsvPath, modelDatasetCsv);

    poolQuery.mockReset();
    poolQuery.mockResolvedValue([
      [buildExperimentContextRow()],
    ]);

    existsSync.mockReset();
    existsSync.mockReturnValue(false);

    createReadStream.mockClear();
    readdirSync.mockClear();
    unlinkSync.mockClear();
    copyFileSync.mockClear();
    spawnMock.mockClear();
    (Bun as { spawn: typeof Bun.spawn }).spawn = spawnMock as typeof Bun.spawn;
  });

  afterEach(async () => {
    (Bun as { spawn: typeof Bun.spawn }).spawn = originalSpawn;
    await unlink(datasetCsvPath).catch(() => {});
  });

  test("reports only safe Python progress lines to callers", async () => {
    existsSync.mockImplementation((filePath: string) => filePath.endsWith("/run_model.py"));
    const progressMessages: string[] = [];

    spawnMock.mockImplementationOnce(() => ({
      stdout: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('{"predictions":[]}'));
          controller.close();
        },
      }),
      stderr: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode([
            "2026-06-28 06:00:00,000 - INFO - Starting model training and evaluation...",
            "2026-06-28 06:00:00,001 - INFO - MA run started with args: {'data': '[{\\\"销售数量\\\": 123}]'}",
            "Traceback (most recent call last):",
            "File \"/opt/fangzhen2/fzbe/py/src/models/ma.py\", line 1, in <module>",
            "ValueError: boom",
            "2026-06-28 06:00:00,002 - DEBUG - Training data: min=1, max=2, mean=1.5",
            "2026-06-28 06:00:00,003 - INFO - Evaluation metrics computed: RMSE=1.0000, MAE=1.0000, MAPE=1.0000, R²=0.9000",
          ].join("\n")));
          controller.close();
        },
      }),
      exited: Promise.resolve(0),
      exitCode: 0,
      killed: false,
      pid: 4321,
      kill: mock(() => {}),
    }));

    await runPythonModel("ma", { data: "[]" }, undefined, (event) => {
      if (event.source === "python") {
        progressMessages.push(event.message);
      }
    });

    expect(progressMessages).toEqual([
      "Starting model training and evaluation...",
      "Evaluation metrics computed: RMSE=1.0000, MAE=1.0000, MAPE=1.0000, R²=0.9000",
    ]);
  });

  test("does not include raw stderr when Python exits with an untagged failure", async () => {
    existsSync.mockImplementation((filePath: string) => filePath.endsWith("/run_model.py"));
    spawnMock.mockImplementationOnce(() => ({
      stdout: new ReadableStream({
        start(controller) {
          controller.close();
        },
      }),
      stderr: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode([
            "Traceback (most recent call last):",
            "File \"/opt/fangzhen2/fzbe/py/src/models/ma.py\", line 1, in <module>",
            "RuntimeError: internal failure with /opt/fangzhen2/secret/path",
          ].join("\n")));
          controller.close();
        },
      }),
      exited: Promise.resolve(1),
      exitCode: 1,
      killed: false,
      pid: 4321,
      kill: mock(() => {}),
    }));

    const error = await runPythonModel("ma", { data: "[]" }).catch((caughtError) => caughtError as Error);

    expect(error).toBeInstanceOf(Error);
    expect(error.message).toBe("Python 执行失败");
    expect(error.message).not.toContain("Traceback");
    expect(error.message).not.toContain("/opt/fangzhen2");
  });

  test("blocks training for completed experiments before loading dataset data", async () => {
    poolQuery.mockResolvedValueOnce([
      [buildExperimentContextRow({ status: "Completed" })],
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
      [buildExperimentContextRow({ selected_product: null })],
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

  test("restores the session-scoped guided artifact before reusing a completed session", async () => {
    const requestBody = {
      experiment_id: 990002,
      selected_industry: "制造业",
      selected_company: "样本企业",
      selected_product: "样本产品",
      data_window_train_start_index: 0,
      data_window_train_end_index: 11,
      data_window_evaluate_start_index: 12,
      data_window_evaluate_end_index: 17,
      moving_average_window: 4,
    };
    const artifactDir = "/tmp/fangzhen-py/guided_training/exp990002-student20240002-ma-guided-a";
    const sessionModelPath = `${artifactDir}/ma.pkl`;
    const productionModelPath = "/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma-production.pkl";

    poolQuery.mockReset();
    poolQuery
      .mockResolvedValueOnce([[buildExperimentContextRow()]])
      .mockResolvedValueOnce([[
        {
          session_id: "guided-a",
          experiment_id: 990002,
          student_id: 20240002,
          model_type: "ma",
          request_hash: "request-hash",
          request_payload: JSON.stringify(requestBody),
          status: "completed",
          current_step_id: null,
          completed_step_ids: JSON.stringify([
            "prepare_data",
            "build_windows",
            "forecast",
            "estimate_residuals",
            "evaluate_metrics",
            "save_artifact",
          ]),
          step_outputs: JSON.stringify({ save_artifact: { metrics: { rmse: 1 } } }),
          result_payload: JSON.stringify({
            status: "success",
            results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
          }),
          artifact_dir: artifactDir,
          error_message: null,
        },
      ]]);
    existsSync.mockImplementation((filePath: string) =>
      filePath === sessionModelPath || filePath === productionModelPath
    );

    const session = await createOrResumeGuidedTrainingSession("ma", 20240002, requestBody);

    expect(session.status).toBe("completed");
    expect(copyFileSync).toHaveBeenCalledWith(
      sessionModelPath,
      "/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma.pkl",
    );
    expect(unlinkSync).not.toHaveBeenCalledWith(productionModelPath);
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("rejects production preparation when an ensemble is incomplete", async () => {
    poolQuery.mockResolvedValueOnce([
      [buildExperimentContextRow({ ensemble_weighted_completed: 0 })],
    ]);

    await expect(
      prepareExperimentModelForProduction("weighted_avg", 20240002, 990002),
    ).rejects.toMatchObject({
      message: "加权平均融合模型尚未完成训练，无法准备生产预测模型",
    });

    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("fits an ensemble production artifact from the backtest model with fit-only parameters", async () => {
    poolQuery.mockReset();
    poolQuery
      .mockResolvedValueOnce([[
        buildExperimentContextRow({
          data_window_train_end_index: 0,
          data_window_evaluate_start_index: 1,
          data_window_evaluate_end_index: 1,
        }),
      ]])
      .mockResolvedValueOnce([[
        buildExperimentContextRow({
          data_window_train_end_index: 0,
          data_window_evaluate_start_index: 1,
          data_window_evaluate_end_index: 1,
        }),
      ]])
      .mockResolvedValueOnce([[{ file_path: "/tmp/dataset.csv" }]]);

    existsSync.mockImplementation((filePath: string) => !filePath.includes("-weighted_avg-production.pkl"));
    createReadStream.mockImplementation(() =>
      Readable.from([
        "月份,销售数量,价格指数,产能利用率\n2024-01,100,1.2,0.8\n2024-02,110,1.3,0.9\n",
      ]) as unknown as ReturnType<typeof createReadStream>
    );

    const result = await prepareExperimentModelForProduction("weighted_avg", 20240002, 990002);

    expect(result).toEqual({
      status: "success",
      results: {
        saved_model: "/tmp/fangzhen-py/trained_models/exp990002-student20240002-weighted_avg-production.pkl",
        artifact_phase: "production",
        reused_existing: false,
      },
    });

    const [command] = spawnMock.mock.calls[0] ?? [];
    expect(command).toContain("weighted_avg");
    expect(command).toContain("--fit-only");
    expect(command).toContain("--load-model");
    expect(command).toContain("/tmp/fangzhen-py/trained_models/exp990002-student20240002-weighted_avg.pkl");
    expect(command).toContain("--save-model");
    expect(command).toContain("/tmp/fangzhen-py/trained_models/exp990002-student20240002-weighted_avg-production.pkl");
    expect(command).toContain("--train-end");
    expect(command).toContain("1");
  });

  test("reuses an existing production artifact without invoking python", async () => {
    existsSync.mockImplementation((filePath: string) => filePath.includes("-ma-production.pkl"));

    const result = await prepareExperimentModelForProduction("ma", 20240002, 990002);

    expect(result).toEqual({
      status: "success",
      results: {
        saved_model: "/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma-production.pkl",
        artifact_phase: "production",
        reused_existing: true,
      },
    });
    expect(spawnMock).not.toHaveBeenCalled();
    expect(createReadStream).not.toHaveBeenCalled();
  });

  test("fits a production artifact from the backtest model with fit-only parameters", async () => {
    poolQuery.mockReset();
    poolQuery
      .mockResolvedValueOnce([[
        buildExperimentContextRow({
          data_window_train_end_index: 0,
          data_window_evaluate_start_index: 1,
          data_window_evaluate_end_index: 1,
        }),
      ]])
      .mockResolvedValueOnce([[
        buildExperimentContextRow({
          data_window_train_end_index: 0,
          data_window_evaluate_start_index: 1,
          data_window_evaluate_end_index: 1,
        }),
      ]])
      .mockResolvedValueOnce([[{ file_path: "/tmp/dataset.csv" }]]);

    existsSync.mockImplementation((filePath: string) => !filePath.includes("-ma-production.pkl"));
    createReadStream.mockImplementation(() =>
      Readable.from([
        "月份,销售数量,价格指数,产能利用率\n2024-01,100,1.2,0.8\n2024-02,110,1.3,0.9\n",
      ]) as unknown as ReturnType<typeof createReadStream>
    );

    const result = await prepareExperimentModelForProduction("ma", 20240002, 990002);

    expect(result).toEqual({
      status: "success",
      results: {
        saved_model: "/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma-production.pkl",
        artifact_phase: "production",
        reused_existing: false,
      },
    });

    const [command, options] = spawnMock.mock.calls[0] ?? [];
    expect(command).toContain("ma");
    expect(command).toContain("--fit-only");
    expect(command).toContain("--load-model");
    expect(command).toContain("/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma.pkl");
    expect(command).toContain("--save-model");
    expect(command).toContain("/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma-production.pkl");
    expect(command).toContain("--train-end");
    expect(command).toContain("1");
    expect(options).toMatchObject({
      cwd: "/tmp/fangzhen-py",
      stdout: "pipe",
      stderr: "pipe",
      env: {
        OMP_NUM_THREADS: "1",
        OPENBLAS_NUM_THREADS: "1",
        MKL_NUM_THREADS: "1",
        NUMEXPR_NUM_THREADS: "1",
        TF_NUM_INTRAOP_THREADS: "1",
        TF_NUM_INTEROP_THREADS: "1",
      },
    });
  });

  test("rejects prediction when the experiment belongs to another student", async () => {
    poolQuery.mockResolvedValueOnce([
      [buildExperimentContextRow({ student_id: 99999999 })],
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

  test("rejects prediction when the model is not ready for production", async () => {
    poolQuery.mockResolvedValueOnce([
      [buildExperimentContextRow({ moving_average_completed: 0 })],
    ]);

    await expect(
      predictExperimentModel("ma", 20240002, 990002, 6),
    ).rejects.toMatchObject({
      message: "移动平均基础模型尚未完成训练，无法准备生产预测模型",
    });

    expect(spawnMock).not.toHaveBeenCalled();
    expect(createReadStream).not.toHaveBeenCalled();
  });

  test("prepares a missing production artifact before prediction", async () => {
    poolQuery.mockReset();
    poolQuery
      .mockResolvedValueOnce([[
        buildExperimentContextRow({
          data_window_train_end_index: 0,
          data_window_evaluate_start_index: 1,
          data_window_evaluate_end_index: 1,
        }),
      ]])
      .mockResolvedValueOnce([[
        buildExperimentContextRow({
          data_window_train_end_index: 0,
          data_window_evaluate_start_index: 1,
          data_window_evaluate_end_index: 1,
        }),
      ]])
      .mockResolvedValueOnce([[
        buildExperimentContextRow({
          data_window_train_end_index: 0,
          data_window_evaluate_start_index: 1,
          data_window_evaluate_end_index: 1,
        }),
      ]])
      .mockResolvedValueOnce([[{ file_path: "/tmp/dataset.csv" }]]);

    let productionArtifactPrepared = false;
    existsSync.mockImplementation((filePath: string) => (
      filePath.endsWith("/run_model.py")
      || (filePath.includes("-ma.pkl") && !filePath.includes("-production"))
      || (productionArtifactPrepared && filePath.includes("-ma-production.pkl"))
    ));
    createReadStream.mockImplementation(() =>
      Readable.from([
        "月份,销售数量,价格指数,产能利用率\n2024-01,100,1.2,0.8\n2024-02,110,1.3,0.9\n",
      ]) as unknown as ReturnType<typeof createReadStream>
    );
    spawnMock
      .mockImplementationOnce(() => ({
        stdout: new ReadableStream({
          start(controller) {
            productionArtifactPrepared = true;
            controller.enqueue(new TextEncoder().encode('{"saved_model":"/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma-production.pkl"}'));
            controller.close();
          },
        }),
        stderr: new ReadableStream({
          start(controller) {
            controller.close();
          },
        }),
        exited: Promise.resolve(0),
        exitCode: 0,
        killed: false,
        pid: 4321,
        kill: mock(() => {}),
      }))
      .mockImplementationOnce(() => ({
        stdout: new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode('{"predictions":[{"prediction":101.2},{"prediction":98.4}]}'));
            controller.close();
          },
        }),
        stderr: new ReadableStream({
          start(controller) {
            controller.close();
          },
        }),
        exited: Promise.resolve(0),
        exitCode: 0,
        killed: false,
        pid: 4321,
        kill: mock(() => {}),
      }));

    const result = await predictExperimentModel("ma", 20240002, 990002, 6);

    expect(result).toEqual({
      status: "success",
      results: {
        predictions: [{ prediction: 101.2 }, { prediction: 98.4 }],
      },
    });
    expect(spawnMock).toHaveBeenCalledTimes(2);

    const [prepareCommand] = spawnMock.mock.calls[0] ?? [];
    expect(prepareCommand).toContain("--fit-only");
    expect(prepareCommand.some((arg) => arg.includes("exp990002-student20240002-ma-production.pkl"))).toBe(true);

    const [predictCommand] = spawnMock.mock.calls[1] ?? [];
    expect(predictCommand).toContain("--forecast-steps");
    expect(predictCommand.some((arg) => arg.includes("exp990002-student20240002-ma-production.pkl"))).toBe(true);
  });

  test("runs prediction with the experiment-scoped model artifact and returns python predictions", async () => {
    existsSync.mockImplementation(() => true);
    spawnMock.mockImplementationOnce((command: string[]) => ({
      stdout: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('{"predictions":[{"prediction":101.2},{"prediction":98.4}]}'));
          controller.close();
        },
      }),
      stderr: new ReadableStream({
        start(controller) {
          controller.close();
        },
      }),
      exited: Promise.resolve(0),
      exitCode: 0,
      killed: false,
      pid: 4321,
      kill: mock(() => {}),
    }));

    const result = await predictExperimentModel("ma", 20240002, 990002, 6);

    expect(result).toEqual({
      status: "success",
      results: {
        predictions: [{ prediction: 101.2 }, { prediction: 98.4 }],
      },
    });

    const [command, options] = spawnMock.mock.calls[0] ?? [];
    expect(command).toContain("ma");
    expect(command).toContain("--load-model");
    expect(command).toContain("--forecast-steps");
    expect(command).toContain("6");
    expect(command).toContain("--pretty");
    expect(command).toContain("/tmp/fangzhen-py/run_model.py");
    expect(command.some((arg) => arg.includes("exp990002-student20240002-ma-production.pkl"))).toBe(true);
    expect(options).toMatchObject({
      cwd: "/tmp/fangzhen-py",
      stdout: "pipe",
      stderr: "pipe",
    });
  });

  test("passes available base model artifacts through to ensemble training", async () => {
    poolQuery.mockReset();
    poolQuery
      .mockResolvedValueOnce([[buildExperimentContextRow()]])
      .mockResolvedValueOnce([[{ file_path: "/tmp/dataset.csv" }]]);

    existsSync.mockImplementation((filePath: string) => (
      filePath.endsWith("/run_model.py")
      || filePath.includes("-ma.pkl")
      || filePath.includes("-lstm.pkl")
      || filePath.includes("-lstm_keras_model.h5")
    ));
    createReadStream.mockImplementation(() =>
      Readable.from([
        "月份,销售数量,价格指数,产能利用率\n2024-01,100,1.2,0.8\n2024-02,110,1.3,0.9\n2024-03,120,1.4,1.0\n2024-04,130,1.5,1.1\n2024-05,140,1.6,1.2\n2024-06,150,1.7,1.3\n2024-07,160,1.8,1.4\n2024-08,170,1.9,1.5\n2024-09,180,2.0,1.6\n2024-10,190,2.1,1.7\n2024-11,200,2.2,1.8\n2024-12,210,2.3,1.9\n2025-01,220,2.4,2.0\n2025-02,230,2.5,2.1\n2025-03,240,2.6,2.2\n2025-04,250,2.7,2.3\n2025-05,260,2.8,2.4\n2025-06,270,2.9,2.5\n",
      ]) as unknown as ReturnType<typeof createReadStream>
    );
    spawnMock.mockImplementationOnce(() => ({
      stdout: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('{"metrics":{"rmse":1,"mae":1,"r2":0.9},"models":["ma","lstm"],"weights":{"ma":0.4,"lstm":0.6},"eval_y_true":[],"predictions":[]}'));
          controller.close();
        },
      }),
      stderr: new ReadableStream({
        start(controller) {
          controller.close();
        },
      }),
      exited: Promise.resolve(0),
      exitCode: 0,
      killed: false,
      pid: 4321,
      kill: mock(() => {}),
    }));

    await trainModel("weighted_avg", 20240002, {
      experiment_id: 990002,
      selected_industry: "制造业",
      selected_company: "样本企业",
      selected_product: "样本产品",
      data_window_train_start_index: 0,
      data_window_train_end_index: 11,
      data_window_evaluate_start_index: 12,
      data_window_evaluate_end_index: 17,
      models: "ma,lstm",
      moving_average_window: 4,
      lstmNormalization: "minmax",
      lstmTargetFeature: "销售数量",
      lstmFeatures: "价格指数,产能利用率",
    });

    const [command] = spawnMock.mock.calls[0] ?? [];
    const baseModelFilesArgIndex = command.indexOf("--base-model-files");
    expect(baseModelFilesArgIndex).toBeGreaterThan(-1);

    const encodedBaseModelFiles = command[baseModelFilesArgIndex + 1];
    expect(JSON.parse(encodedBaseModelFiles)).toEqual({
      ma: "/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma.pkl",
      lstm: "/tmp/fangzhen-py/trained_models/exp990002-student20240002-lstm.pkl",
    });
  });

  test("passes persisted LSTM feature lists to ensemble training as JSON to preserve comma-containing field names", async () => {
    poolQuery.mockReset();
    poolQuery
      .mockResolvedValueOnce([[buildExperimentContextRow({
        lstm_features: JSON.stringify(["价格指数", "外部,指标"]),
      })]])
      .mockResolvedValueOnce([[{ file_path: "/tmp/dataset.csv" }]]);

    existsSync.mockImplementation((filePath: string) => (
      filePath.endsWith("/run_model.py")
      || filePath.includes("-ma.pkl")
      || filePath.includes("-lstm.pkl")
      || filePath.includes("-lstm_keras_model.h5")
    ));
    spawnMock.mockImplementationOnce(() => ({
      stdout: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('{"metrics":{"rmse":1,"mae":1,"r2":0.9},"models":["ma","lstm"],"weights":{"ma":0.4,"lstm":0.6},"eval_y_true":[],"predictions":[]}'));
          controller.close();
        },
      }),
      stderr: new ReadableStream({
        start(controller) {
          controller.close();
        },
      }),
      exited: Promise.resolve(0),
      exitCode: 0,
      killed: false,
      pid: 4321,
      kill: mock(() => {}),
    }));

    await trainModel("weighted_avg", 20240002, {
      experiment_id: 990002,
      selected_industry: "制造业",
      selected_company: "样本企业",
      selected_product: "样本产品",
      data_window_train_start_index: 0,
      data_window_train_end_index: 11,
      data_window_evaluate_start_index: 12,
      data_window_evaluate_end_index: 17,
      models: "ma,lstm",
      moving_average_window: 4,
      lstmNormalization: "minmax",
      lstmTargetFeature: "销售数量",
      lstmFeatures: JSON.stringify(["价格指数", "外部,指标"]),
    });

    const [command] = spawnMock.mock.calls[0] ?? [];
    const lstmFeaturesArgIndex = command.indexOf("--lstm-features");
    expect(lstmFeaturesArgIndex).toBeGreaterThan(-1);
    expect(JSON.parse(command[lstmFeaturesArgIndex + 1])).toEqual(["价格指数", "外部,指标"]);
  });

  test("rejects comma-collision LSTM feature lists that differ from persisted config", async () => {
    poolQuery.mockReset();
    poolQuery.mockResolvedValueOnce([[buildExperimentContextRow({
      lstm_features: JSON.stringify(["价格指数", "外部,指标"]),
    })]]);

    await expect(trainModel("weighted_avg", 20240002, {
      experiment_id: 990002,
      selected_industry: "制造业",
      selected_company: "样本企业",
      selected_product: "样本产品",
      data_window_train_start_index: 0,
      data_window_train_end_index: 11,
      data_window_evaluate_start_index: 12,
      data_window_evaluate_end_index: 17,
      models: "ma,lstm",
      moving_average_window: 4,
      lstmNormalization: "minmax",
      lstmTargetFeature: "销售数量",
      lstmFeatures: JSON.stringify(["价格指数,外部", "指标"]),
    })).rejects.toThrow("请求中的LSTM 输入特征与已完成基础模型记录不一致");

    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("returns an empty prediction list when python omits predictions", async () => {
    existsSync.mockImplementation(() => true);
    spawnMock.mockImplementationOnce(() => ({
      stdout: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('{"status":"ok"}'));
          controller.close();
        },
      }),
      stderr: new ReadableStream({
        start(controller) {
          controller.close();
        },
      }),
      exited: Promise.resolve(0),
      exitCode: 0,
      killed: false,
      pid: 4321,
      kill: mock(() => {}),
    }));

    const result = await predictExperimentModel("lstm", 20240002, 990002, 3);

    expect(result).toEqual({
      status: "success",
      results: {
        predictions: [],
      },
    });
  });

  test("rejects ensemble training when a requested base model is not completed", async () => {
    poolQuery.mockResolvedValueOnce([
      [buildExperimentContextRow({ arima_completed: 0 })],
    ]);

    await expect(
      trainModel("stacking", 20240002, {
        experiment_id: 990002,
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
        models: "ma,arima",
        moving_average_window: 4,
        arimaD: 1,
      }),
    ).rejects.toMatchObject({
      message: "融合模型依赖的 ARIMA 基础模型尚未完成训练",
    });

    expect(createReadStream).not.toHaveBeenCalled();
  });

  test("rejects ensemble training when persisted LSTM config is incomplete", async () => {
    poolQuery.mockResolvedValueOnce([
      [buildExperimentContextRow({ lstm_features: JSON.stringify([]) })],
    ]);

    await expect(
      trainModel("weighted_avg", 20240002, {
        experiment_id: 990002,
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
        models: "ma,lstm",
        moving_average_window: 4,
        lstmNormalization: "minmax",
        lstmTargetFeature: "销售数量",
        lstmFeatures: "价格指数,产能利用率",
      }),
    ).rejects.toMatchObject({
      message: "融合模型依赖的 LSTM 基础模型缺少特征或归一化配置",
    });

    expect(createReadStream).not.toHaveBeenCalled();
  });

  test("rejects ensemble training when client ARIMA params differ from persisted config", async () => {
    await expect(
      trainModel("boosting", 20240002, {
        experiment_id: 990002,
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
        models: "es,arima",
        exponential_smoothing_alpha: 0.35,
        arimaD: 2,
      }),
    ).rejects.toMatchObject({
      message: "请求中的ARIMA 差分阶数与已完成基础模型记录不一致，请刷新页面后重试",
    });

    expect(createReadStream).not.toHaveBeenCalled();
  });
});
