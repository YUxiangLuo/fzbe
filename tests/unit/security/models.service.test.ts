import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import * as actualFs from "node:fs";
import { unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { Readable } from "node:stream";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const pythonModulePath = path.join(srcRoot, "utils", "python.ts");
const modelRuntimeModulePath = path.join(srcRoot, "modules", "models", "model-runtime.ts");
const modelLocksModulePath = path.join(srcRoot, "modules", "models", "model-locks.ts");
const modelsServiceModulePath = path.join(srcRoot, "modules", "models", "models.service.ts");
const freshModelsServiceModulePath = `${modelsServiceModulePath}?models-service-test`;
const datasetCsvPath = "/tmp/dataset.csv";
const mockedPyDir = "/tmp/fangzhen-py";
const mockedGuidedTrainingRoot = path.join(mockedPyDir, "guided_training");
const mockedTrainedModelsRoot = path.join(mockedPyDir, "trained_models");
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
const beginTransaction = mock(async () => {});
const commit = mock(async () => {});
const rollback = mock(async () => {});
const release = mock(() => {});
const getConnection = mock(async () => ({
  query: poolQuery,
  beginTransaction,
  commit,
  rollback,
  release,
}));
const existsSync = mock((_: string) => false);
const createReadStream = mock(() => {
  throw new Error("createReadStream should not be called in this test");
});
const readdirSync = mock(() => []);
const statSync = mock((_: string) => ({
  isDirectory: () => false,
  isFile: () => false,
  mtimeMs: 0,
}));
const unlinkSync = mock(() => {});
const copyFileSync = mock((_: string, __: string) => {});
const readFileSync = mock((_: string, __?: unknown) => "");
const writeFileSync = mock((_: string, __: unknown, ___?: unknown) => {});
const mkdirSync = mock((_: string, __?: unknown) => undefined);
const rmSync = mock((_: string, __?: unknown) => undefined);
const defaultSpawnImplementation = (_: string[], __: Record<string, unknown>) => ({
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
});
const spawnResult = (payload: unknown) => ({
  stdout: new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(JSON.stringify(payload)));
      controller.close();
    },
  }),
  stderr: new ReadableStream({ start: (controller) => controller.close() }),
  exited: Promise.resolve(0),
  exitCode: 0,
  killed: false,
  pid: 4321,
  kill: mock(() => {}),
});
const spawnMock = mock(defaultSpawnImplementation);
const originalSpawn = Bun.spawn;
const acquireModelExecutionLock = mock(async () => ({ connection: {} as never, name: "exec:1" }));
const releaseModelExecutionLock = mock(async () => {});
const acquireModelArtifactMutationLocks = mock(async () => ([
  { connection: {} as never, name: "exec:1" },
]));
const releaseModelArtifactMutationLocks = mock(async () => {});
const acquireModelSlot = mock(async () => ({ connection: {} as never, name: "slot:1" }));
const releaseModelSlot = mock(async () => {});

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
    getConnection,
  },
}));

mock.module("fs", () => ({
  ...actualFs,
  default: {
    ...actualFs.default,
    existsSync,
    createReadStream,
    readdirSync,
    statSync,
    unlinkSync,
    copyFileSync,
    readFileSync,
    writeFileSync,
    mkdirSync,
    rmSync,
  },
  existsSync,
  createReadStream,
  readdirSync,
  statSync,
  unlinkSync,
  copyFileSync,
  readFileSync,
  writeFileSync,
  mkdirSync,
  rmSync,
}));

mock.module(pythonModulePath, () => ({
  PY_DIR: mockedPyDir,
  resolvePythonExecutable: () => path.join(mockedPyDir, "venv/bin/python"),
}));

mock.module(modelRuntimeModulePath, () => ({
  GUIDED_TRAINING_ROOT: mockedGuidedTrainingRoot,
  buildExperimentScopedModelPath: (
    experimentId: number,
    studentId: number,
    modelType: string,
    phase: "backtest" | "production" = "backtest",
  ) => path.join(
    mockedTrainedModelsRoot,
    `exp${experimentId}-student${studentId}-${modelType}${phase === "production" ? "-production" : ""}.pkl`,
  ),
  buildVersionedModelArtifactPath: (
    experimentId: number,
    studentId: number,
    modelType: string,
    revisionToken: string,
    phase: "backtest" | "production" = "backtest",
  ) => path.join(
    mockedTrainedModelsRoot,
    `exp${experimentId}-student${studentId}-${modelType}-${phase}-${revisionToken}.pkl`,
  ),
  buildGuidedTrainingArtifactDir: (
    experimentId: number,
    studentId: number,
    modelType: string,
    sessionId: string,
  ) => path.join(
    mockedGuidedTrainingRoot,
    `exp${experimentId}-student${studentId}-${modelType}-${sessionId}`,
  ),
  buildGuidedTrainingStatePath: (
    experimentId: number,
    studentId: number,
    modelType: string,
    sessionId: string,
  ) => path.join(
    mockedGuidedTrainingRoot,
    `exp${experimentId}-student${studentId}-${modelType}-${sessionId}`,
    "state.pkl",
  ),
  buildModelExecutionLockName: (experimentId: number) =>
    `fangzhen-model:exp:${experimentId}`,
  buildModelSlotLockName: (slot: number) => `fangzhen-model:slot:${slot}`,
}));

mock.module(modelLocksModulePath, () => ({
  acquireModelExecutionLock,
  releaseModelExecutionLock,
  acquireModelArtifactMutationLocks,
  releaseModelArtifactMutationLocks,
  acquireModelSlot,
  releaseModelSlot,
}));

const {
  trainModel,
  runPythonModel,
  createOrResumeGuidedTrainingSession,
  getGuidedTrainingSession,
  activateGuidedTrainingSession,
  cleanupExpiredGuidedTrainingSessions,
  cleanupStaleModelArtifacts,
  completeExperimentModel,
  prepareExperimentModelForProduction,
  predictExperimentModel,
  ModelAccessError,
  ModelNotFoundError,
} = await import(freshModelsServiceModulePath);

const buildExperimentContextRow = (overrides: Record<string, unknown> = {}) => ({
  student_id: 20240002,
  status: "In Progress",
  state_version: 4,
  training_input_version: 2,
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

const buildActiveArtifactRow = (
  modelType: string = "ma",
  overrides: Record<string, unknown> = {},
) => ({
  experiment_id: 990002,
  student_id: 20240002,
  model_type: modelType,
  revision_token: `revision-${modelType}`,
  training_input_version: 2,
  source_session_id: null,
  backtest_artifact_path: path.join(
    mockedTrainedModelsRoot,
    `exp990002-student20240002-${modelType}.pkl`,
  ),
  request_payload: JSON.stringify({}),
  result_payload: JSON.stringify({ status: "success", results: {} }),
  production_artifact_path: path.join(
    mockedTrainedModelsRoot,
    `exp990002-student20240002-${modelType}-production.pkl`,
  ),
  production_revision_token: `revision-${modelType}`,
  ...overrides,
});

const stableJsonStringify = (value: unknown): string => {
  if (Array.isArray(value)) {
    return `[${value.map(stableJsonStringify).join(",")}]`;
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => (
      `${JSON.stringify(key)}:${stableJsonStringify(record[key])}`
    )).join(",")}}`;
  }
  return JSON.stringify(value);
};

const buildGuidedRequestHash = (
  modelType: string,
  body: Record<string, unknown>,
  targetRevision: string | null,
  baseRevisions: Array<{ model_type: string; revision_token: string | null }> = [],
) => crypto.createHash("sha256").update(stableJsonStringify({
  modelType,
  body,
  trainingInputVersion: 2,
  dependencyRevisions: {
    target_revision: targetRevision,
    base_revisions: baseRevisions,
  },
})).digest("hex");

const defaultPoolQuery = async (sqlValue: unknown, params: unknown[] = []) => {
  const sql = String(sqlValue);
  if (/^\s*(?:INSERT|UPDATE|DELETE)/i.test(sql)) {
    return [{ affectedRows: 1, insertId: 1 }];
  }
  if (sql.includes("FROM experiment_status")) {
    return [[buildExperimentContextRow()]];
  }
  if (sql.includes("FROM experiment_model_artifacts")) {
    const modelType = typeof params[2] === "string" ? params[2] : "ma";
    return [[buildActiveArtifactRow(modelType)]];
  }
  if (sql.includes("file_path") && sql.includes("dataset_segments")) {
    return [[{ file_path: datasetCsvPath }]];
  }
  return [[]];
};

const installArtifactPoolMock = ({
  modelType,
  experimentOverrides = {},
  productionReady,
}: {
  modelType: string;
  experimentOverrides?: Record<string, unknown>;
  productionReady: boolean;
}) => {
  poolQuery.mockImplementation(async (sqlValue: unknown, params: unknown[] = []) => {
    const sql = String(sqlValue);
    if (/^\s*(?:INSERT|UPDATE|DELETE)/i.test(sql)) {
      return [{ affectedRows: 1, insertId: 1 }];
    }
    if (sql.includes("FROM experiment_status")) {
      return [[buildExperimentContextRow(experimentOverrides)]];
    }
    if (sql.includes("FROM experiment_model_artifacts")) {
      const requestedModelType = typeof params[2] === "string" ? params[2] : modelType;
      return [[buildActiveArtifactRow(requestedModelType, productionReady ? {} : {
        production_artifact_path: null,
        production_revision_token: null,
      })]];
    }
    if (sql.includes("file_path") && sql.includes("dataset_segments")) {
      return [[{ file_path: datasetCsvPath }]];
    }
    return defaultPoolQuery(sqlValue, params);
  });
};

const installTrainingPoolMock = (experimentOverrides: Record<string, unknown> = {}) => {
  poolQuery.mockImplementation(async (sqlValue: unknown, params: unknown[] = []) => {
    const sql = String(sqlValue);
    if (/^\s*(?:INSERT|UPDATE|DELETE)/i.test(sql)) {
      return [{ affectedRows: 1, insertId: 1 }];
    }
    if (sql.includes("FROM experiment_status")) {
      return [[buildExperimentContextRow(experimentOverrides)]];
    }
    if (sql.includes("FROM experiment_model_artifacts")) {
      const modelType = typeof params[2] === "string" ? params[2] : "ma";
      return [[buildActiveArtifactRow(modelType)]];
    }
    if (sql.includes("file_path") && sql.includes("dataset_segments")) {
      return [[{ file_path: datasetCsvPath }]];
    }
    return defaultPoolQuery(sqlValue, params);
  });
};

describe("models service", () => {
  beforeEach(async () => {
    await writeFile(datasetCsvPath, modelDatasetCsv);

    poolQuery.mockReset();
    poolQuery.mockImplementation(defaultPoolQuery);

    existsSync.mockReset();
    existsSync.mockReturnValue(false);

    createReadStream.mockClear();
    readdirSync.mockClear();
    statSync.mockClear();
    statSync.mockReturnValue({
      isDirectory: () => false,
      isFile: () => false,
      mtimeMs: 0,
    });
    unlinkSync.mockClear();
    copyFileSync.mockClear();
    readFileSync.mockClear();
    writeFileSync.mockClear();
    mkdirSync.mockClear();
    rmSync.mockClear();
    spawnMock.mockReset();
    spawnMock.mockImplementation(defaultSpawnImplementation);
    acquireModelExecutionLock.mockClear();
    releaseModelExecutionLock.mockClear();
    acquireModelArtifactMutationLocks.mockReset();
    acquireModelArtifactMutationLocks.mockResolvedValue([
      { connection: {} as never, name: "exec:1" },
    ]);
    releaseModelArtifactMutationLocks.mockClear();
    acquireModelSlot.mockClear();
    releaseModelSlot.mockClear();
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    release.mockClear();
    getConnection.mockClear();
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
    const [command, options] = spawnMock.mock.calls[0] ?? [];
    expect(command).toContain("--data-stdin");
    expect(command).not.toContain("--data");
    expect(options).toMatchObject({ maxBuffer: 16 * 1024 * 1024 });
    expect(options.stdin).toBeInstanceOf(Blob);
    expect(await (options.stdin as Blob).text()).toBe("[]");
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

  test("rejects training when the persisted evaluation range contains only one point", async () => {
    poolQuery.mockResolvedValueOnce([[
      buildExperimentContextRow({
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 12,
      }),
    ]]);

    await expect(
      trainModel("ma", 20240002, {
        experiment_id: 990002,
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 12,
        moving_average_window: 4,
      }),
    ).rejects.toMatchObject({
      message: "实验记录中的评估区间至少需要 2 个数据点，请重新选择数据窗口后重试",
    });

    expect(createReadStream).not.toHaveBeenCalled();
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("rejects LSTM training that cannot preserve the fixed six-step production strategy", async () => {
    poolQuery.mockResolvedValueOnce([[
      buildExperimentContextRow({
        data_window_train_start_index: 0,
        data_window_train_end_index: 4,
        data_window_evaluate_start_index: 5,
        data_window_evaluate_end_index: 6,
      }),
    ]]);

    await expect(
      createOrResumeGuidedTrainingSession("lstm", 20240002, {
        experiment_id: 990002,
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 4,
        data_window_evaluate_start_index: 5,
        data_window_evaluate_end_index: 6,
        lstmNormalization: "minmax",
        lstmTargetFeature: "销售数量",
        lstmFeatures: JSON.stringify(["价格指数"]),
      }),
    ).rejects.toMatchObject({
      message: expect.stringContaining("LSTM 生产预测固定为 6 期"),
    });

    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("keeps completed-session lookup pure and only restores its artifact on explicit activation", async () => {
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

    const completedSessionRow = {
      session_id: "guided-a",
      experiment_id: 990002,
      student_id: 20240002,
      model_type: "ma",
      request_hash: buildGuidedRequestHash("ma", requestBody, null),
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
      step_plan: JSON.stringify([]),
      result_payload: JSON.stringify({
        status: "success",
        results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
      }),
      artifact_dir: artifactDir,
      error_message: null,
      training_input_version: 2,
    };

    poolQuery.mockImplementation(async (sqlValue: unknown, params: unknown[] = []) => {
      const sql = String(sqlValue);
      if (sql.includes("FROM model_guided_training_sessions")) {
        return [[completedSessionRow]];
      }
      if (sql.includes("FROM experiment_model_artifacts")) {
        return [[]];
      }
      return defaultPoolQuery(sqlValue, params);
    });
    existsSync.mockImplementation((filePath: string) =>
      filePath === sessionModelPath || filePath === productionModelPath
    );

    const lookedUpSession = await createOrResumeGuidedTrainingSession("ma", 20240002, requestBody);

    expect(lookedUpSession.status).toBe("completed");
    expect(lookedUpSession.backtest_artifact_changed).toBeFalse();
    expect(copyFileSync).not.toHaveBeenCalled();
    expect(unlinkSync).not.toHaveBeenCalledWith(productionModelPath);
    expect(writeFileSync).not.toHaveBeenCalled();
    expect(acquireModelArtifactMutationLocks).not.toHaveBeenCalled();

    const activatedSession = await activateGuidedTrainingSession("ma", 20240002, "guided-a");

    expect(activatedSession.status).toBe("completed");
    expect(activatedSession.backtest_artifact_changed).toBeTrue();
    expect(activatedSession.artifact_revision).toBe("guided-a");
    expect(activatedSession.experiment_state_version).toBe(5);
    expect(acquireModelArtifactMutationLocks).toHaveBeenCalledWith(990002, "ma");
    expect(releaseModelArtifactMutationLocks).toHaveBeenCalledTimes(1);
    expect(copyFileSync).not.toHaveBeenCalled();
    expect(unlinkSync).not.toHaveBeenCalledWith(productionModelPath);
    expect(writeFileSync).not.toHaveBeenCalled();
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("INSERT INTO experiment_model_artifacts")
    )).toBeTrue();
    const stateResetCall = poolQuery.mock.calls.find(([sql]) =>
      String(sql).includes("UPDATE experiment_status")
    );
    expect(stateResetCall).toBeDefined();
    expect(String(stateResetCall?.[0])).toContain("moving_average_completed");
    expect(String(stateResetCall?.[0])).toContain("ensemble_weighted_completed");
    expect(String(stateResetCall?.[0])).toContain("selected_best_model");
    expect(String(stateResetCall?.[0])).toContain("quiz_about_plan_completed");
    expect(stateResetCall?.[1]?.slice(-2)).toEqual([990002, 20240002]);
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("makes repeated activation of the same completed guided session idempotent", async () => {
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
    const backtestModelPath = "/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma.pkl";
    const sourceMarkerPath = `${backtestModelPath}.guided-session`;
    const productionModelPath = "/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma-production.pkl";

    const completedSessionRow = {
      session_id: "guided-a",
      experiment_id: 990002,
      student_id: 20240002,
      model_type: "ma",
      request_hash: buildGuidedRequestHash("ma", requestBody, "guided-a"),
      request_payload: JSON.stringify(requestBody),
      status: "completed",
      current_step_id: null,
      completed_step_ids: JSON.stringify(["save_artifact"]),
      step_outputs: JSON.stringify({}),
      step_plan: JSON.stringify([]),
      result_payload: JSON.stringify({ status: "success", results: {} }),
      artifact_dir: artifactDir,
      error_message: null,
      training_input_version: 2,
    };
    poolQuery.mockImplementation(async (sqlValue: unknown, params: unknown[] = []) => {
      const sql = String(sqlValue);
      if (sql.includes("FROM model_guided_training_sessions")) {
        return [[completedSessionRow]];
      }
      if (sql.includes("FROM experiment_model_artifacts")) {
        return [[buildActiveArtifactRow("ma", {
          revision_token: "guided-a",
          source_session_id: "guided-a",
          backtest_artifact_path: sessionModelPath,
          production_artifact_path: productionModelPath,
          production_revision_token: "guided-a",
        })]];
      }
      return defaultPoolQuery(sqlValue, params);
    });
    existsSync.mockImplementation((filePath: string) => [
      sessionModelPath,
      backtestModelPath,
      sourceMarkerPath,
      productionModelPath,
    ].includes(filePath));
    readFileSync.mockImplementation((filePath: string) =>
      filePath === sourceMarkerPath ? "guided-a" : ""
    );

    const session = await activateGuidedTrainingSession("ma", 20240002, "guided-a");

    expect(session.status).toBe("completed");
    expect(session.backtest_artifact_changed).toBeFalse();
    expect(copyFileSync).not.toHaveBeenCalled();
    expect(unlinkSync).not.toHaveBeenCalledWith(productionModelPath);
    expect(writeFileSync).not.toHaveBeenCalled();
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("UPDATE experiment_status")
    )).toBeFalse();
    expect(releaseModelArtifactMutationLocks).toHaveBeenCalledTimes(1);
  });

  test("fences explicit activation when training inputs change after dependency validation", async () => {
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
    const artifactDir = "/tmp/fangzhen-py/guided_training/exp990002-student20240002-ma-guided-race";
    const completedSessionRow = {
      session_id: "guided-race",
      experiment_id: 990002,
      student_id: 20240002,
      model_type: "ma",
      request_hash: buildGuidedRequestHash("ma", requestBody, null),
      request_payload: JSON.stringify(requestBody),
      status: "completed",
      current_step_id: null,
      completed_step_ids: JSON.stringify(["save_artifact"]),
      step_outputs: JSON.stringify({}),
      step_plan: JSON.stringify([]),
      result_payload: JSON.stringify({ status: "success", results: {} }),
      artifact_dir: artifactDir,
      error_message: null,
      training_input_version: 2,
    };
    let experimentSelectCount = 0;
    poolQuery.mockImplementation(async (sqlValue: unknown, params: unknown[] = []) => {
      const sql = String(sqlValue);
      if (sql.includes("FROM model_guided_training_sessions")) {
        return [[completedSessionRow]];
      }
      if (sql.includes("FROM experiment_status")) {
        experimentSelectCount += 1;
        return [[buildExperimentContextRow({
          training_input_version: experimentSelectCount >= 3 ? 3 : 2,
        })]];
      }
      if (sql.includes("FROM experiment_model_artifacts")) {
        return [[]];
      }
      return defaultPoolQuery(sqlValue, params);
    });
    existsSync.mockImplementation((filePath: string) => filePath === `${artifactDir}/ma.pkl`);

    await expect(
      activateGuidedTrainingSession("ma", 20240002, "guided-race"),
    ).rejects.toMatchObject({
      message: "训练期间实验输入已更新，本次结果未激活；请重新开始训练",
    });

    expect(rollback).toHaveBeenCalledTimes(1);
    expect(releaseModelArtifactMutationLocks).toHaveBeenCalledTimes(1);
  });

  test("makes repeated completion of the same active artifact idempotent", async () => {
    const result = await completeExperimentModel(
      "ma",
      20240002,
      990002,
      "revision-ma",
    );

    expect(result).toEqual({
      status: "success",
      artifact_revision: "revision-ma",
      experiment_state_version: 4,
    });
    expect(commit).toHaveBeenCalledTimes(1);
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("UPDATE experiment_status")
    )).toBeFalse();
  });

  test("rejects an old ensemble session after a base model artifact changes", async () => {
    const requestBody = {
      experiment_id: 990002,
      selected_industry: "制造业",
      selected_company: "样本企业",
      selected_product: "样本产品",
      data_window_train_start_index: 0,
      data_window_train_end_index: 11,
      data_window_evaluate_start_index: 12,
      data_window_evaluate_end_index: 17,
      models: "ma,es",
      moving_average_window: 4,
      exponential_smoothing_alpha: 0.35,
    };
    const artifactDir = "/tmp/fangzhen-py/guided_training/exp990002-student20240002-weighted_avg-guided-old";

    const staleSessionRow = {
          session_id: "guided-old",
          experiment_id: 990002,
          student_id: 20240002,
          model_type: "weighted_avg",
          request_hash: "stale-base-model-revision",
          request_payload: JSON.stringify(requestBody),
          status: "completed",
          current_step_id: null,
          completed_step_ids: JSON.stringify(["save_artifact"]),
          step_outputs: JSON.stringify({}),
          step_plan: JSON.stringify([]),
          result_payload: JSON.stringify({ status: "success", results: {} }),
          artifact_dir: artifactDir,
          error_message: null,
          training_input_version: 2,
        };
    poolQuery.mockImplementation(async (sqlValue: unknown, params: unknown[] = []) => {
      const sql = String(sqlValue);
      if (sql.includes("FROM model_guided_training_sessions")) {
        return [[staleSessionRow]];
      }
      if (sql.includes("FROM experiment_model_artifacts")) {
        const modelType = typeof params[2] === "string" ? params[2] : "weighted_avg";
        return [[buildActiveArtifactRow(modelType)]];
      }
      return defaultPoolQuery(sqlValue, params);
    });
    existsSync.mockImplementation((filePath: string) => (
      filePath.endsWith("-ma.pkl") || filePath.endsWith("-es.pkl")
    ));
    readFileSync.mockReturnValue("current-base-model-revision");

    await expect(
      getGuidedTrainingSession("weighted_avg", 20240002, "guided-old"),
    ).rejects.toMatchObject({
      message: "实验训练输入或基础模型版本已更新，请重新开始分阶段训练",
    });

    expect(copyFileSync).not.toHaveBeenCalled();
    expect(writeFileSync).not.toHaveBeenCalled();
  });

  test("recreates a completed guided session when its reusable artifact is missing", async () => {
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
    const expiredArtifactDir = "/tmp/fangzhen-py/guided_training/exp990002-student20240002-ma-guided-expired";
    const freshArtifactDir = "/tmp/fangzhen-py/guided_training/exp990002-student20240002-ma-guided-fresh";
    const stepPlan = JSON.stringify([
      {
        id: "prepare_data",
        label: "确认训练材料",
        description: "检查训练输入。",
        actionLabel: "校验数据",
      },
    ]);

    const completedSessionRow = {
      session_id: "guided-expired",
      experiment_id: 990002,
      student_id: 20240002,
      model_type: "ma",
      request_hash: buildGuidedRequestHash("ma", requestBody, null),
      request_payload: JSON.stringify(requestBody),
      status: "completed",
      current_step_id: null,
      completed_step_ids: JSON.stringify(["prepare_data"]),
      step_outputs: JSON.stringify({ prepare_data: { ok: true } }),
      step_plan: stepPlan,
      result_payload: JSON.stringify({
        status: "success",
        results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
      }),
      artifact_dir: expiredArtifactDir,
      error_message: null,
      training_input_version: 2,
    };
    const freshSessionRow = {
      ...completedSessionRow,
      session_id: "guided-fresh",
      status: "ready",
      current_step_id: "prepare_data",
      completed_step_ids: JSON.stringify([]),
      step_outputs: JSON.stringify({}),
      result_payload: null,
      artifact_dir: freshArtifactDir,
    };

    poolQuery.mockImplementation(async (sqlValue: unknown, params: unknown[] = []) => {
      const sql = String(sqlValue);
      if (/^\s*(?:INSERT|UPDATE|DELETE)/i.test(sql)) {
        return [{ affectedRows: 1, insertId: 1 }];
      }
      if (sql.includes("FROM experiment_model_artifacts")) {
        return [[]];
      }
      if (sql.includes("FROM model_guided_training_sessions")) {
        return [[
          params.length === 4 || params[0] === "guided-expired"
            ? completedSessionRow
            : freshSessionRow,
        ]];
      }
      return defaultPoolQuery(sqlValue, params);
    });
    existsSync.mockReturnValue(false);

    const session = await createOrResumeGuidedTrainingSession("ma", 20240002, requestBody);

    expect(session.session_id).toBe("guided-fresh");
    expect(session.status).toBe("ready");
    expect(copyFileSync).not.toHaveBeenCalled();
    expect(rmSync).toHaveBeenCalledWith(
      expiredArtifactDir,
      { recursive: true, force: true },
    );
    expect(mkdirSync).not.toHaveBeenCalled();
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("DELETE FROM model_guided_training_sessions"),
    )).toBe(true);
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("SET source_session_id = NULL"),
    )).toBe(true);
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("INSERT INTO model_guided_training_sessions"),
    )).toBe(true);
  });

  test("cleans up expired guided sessions and only removes guided artifact directories", async () => {
    const safeArtifactDir = "/tmp/fangzhen-py/guided_training/exp990002-student20240002-ma-expired";
    const unsafeArtifactDir = "/tmp/fangzhen-py/trained_models/not-a-guided-session";
    const now = new Date("2026-07-05T00:00:00.000Z");

    poolQuery.mockClear();
    poolQuery
      .mockResolvedValueOnce([[
        { session_id: "expired-safe", experiment_id: 990002, model_type: "ma", artifact_dir: safeArtifactDir },
        { session_id: "expired-unsafe", experiment_id: 990002, model_type: "ma", artifact_dir: unsafeArtifactDir },
      ]])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([{ affectedRows: 1 }]);

    const result = await cleanupExpiredGuidedTrainingSessions({
      now,
      limit: 50,
    });

    expect(result).toEqual({
      deletedSessions: 2,
      deletedArtifactDirs: 1,
    });
    expect(rmSync).toHaveBeenCalledTimes(1);
    expect(rmSync).toHaveBeenCalledWith(
      safeArtifactDir,
      { recursive: true, force: true },
    );
    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("expires_at <= ?");
    expect(poolQuery.mock.calls[0]?.[1]).toEqual([now, now]);
    expect(poolQuery.mock.calls[1]?.[1]).toEqual(["expired-safe", now, now]);
    expect(poolQuery.mock.calls[2]?.[1]).toEqual(["expired-unsafe", now, now]);
  });

  test("uses database time for guided session expiration in production cleanup", async () => {
    poolQuery.mockClear();
    poolQuery.mockResolvedValueOnce([[]]);
    readdirSync.mockReturnValueOnce([]);

    await cleanupExpiredGuidedTrainingSessions({ limit: 50 });

    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("expires_at <= NOW()");
    expect(poolQuery.mock.calls[0]?.[1]).toEqual([]);
  });

  test("keeps cleanup progress recoverable when expired artifact deletion fails", async () => {
    const artifactDir = "/tmp/fangzhen-py/guided_training/exp990002-student20240002-ma-stuck";

    poolQuery.mockClear();
    poolQuery
      .mockResolvedValueOnce([[
        {
          session_id: "expired-stuck",
          experiment_id: 990002,
          model_type: "ma",
          artifact_dir: artifactDir,
        },
      ]])
      .mockResolvedValueOnce([{ affectedRows: 1 }]);
    rmSync.mockImplementationOnce(() => {
      throw new Error("permission denied");
    });

    const result = await cleanupExpiredGuidedTrainingSessions({
      now: new Date("2026-07-05T00:00:00.000Z"),
      limit: 50,
    });

    expect(result).toEqual({
      deletedSessions: 1,
      deletedArtifactDirs: 0,
    });
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("DELETE FROM model_guided_training_sessions"),
    )).toBe(true);
  });

  test("removes old orphan guided artifact directories", async () => {
    const orphanName = "exp990002-student20240002-ma-orphan";
    const orphanDir = `/tmp/fangzhen-py/guided_training/${orphanName}`;

    poolQuery.mockClear();
    poolQuery
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([[]]);
    readdirSync.mockReturnValueOnce([orphanName]);
    statSync.mockReturnValueOnce({
      isDirectory: () => true,
      isFile: () => false,
      mtimeMs: new Date("2026-07-03T00:00:00.000Z").getTime(),
    });

    const result = await cleanupExpiredGuidedTrainingSessions({
      now: new Date("2026-07-05T00:00:00.000Z"),
      limit: 50,
      orphanMaxAgeMs: 60_000,
    });

    expect(result).toEqual({
      deletedSessions: 0,
      deletedArtifactDirs: 1,
    });
    expect(rmSync).toHaveBeenCalledWith(
      orphanDir,
      { recursive: true, force: true },
    );
  });

  test("cleans stale model artifact groups for completed experiments", async () => {
    const trainedModelsDir = "/tmp/fangzhen-py/trained_models";
    const modelFile = "exp990002-student20240002-ma.pkl";
    const productionFile = "exp990002-student20240002-ma-production.pkl";

    poolQuery.mockClear();
    poolQuery.mockResolvedValueOnce([[
      {
        can_delete: 1,
      },
    ]]).mockResolvedValueOnce([[]]);
    readdirSync.mockReturnValueOnce([modelFile, productionFile, "unrelated.pkl"]);
    statSync.mockImplementation((filePath: string) => ({
      isDirectory: () => false,
      isFile: () => true,
      mtimeMs: String(filePath).includes("unrelated")
        ? new Date("2026-07-05T00:00:00.000Z").getTime()
        : new Date("2026-06-01T00:00:00.000Z").getTime(),
    }));

    const result = await cleanupStaleModelArtifacts({
      now: new Date("2026-07-05T00:00:00.000Z"),
      maxAgeMs: 7 * 24 * 60 * 60 * 1000,
    });

    expect(result).toEqual({
      scannedGroups: 1,
      deletedGroups: 1,
      deletedFiles: 2,
      failedFiles: 0,
    });
    expect(unlinkSync).toHaveBeenCalledWith(path.join(trainedModelsDir, modelFile));
    expect(unlinkSync).toHaveBeenCalledWith(path.join(trainedModelsDir, productionFile));
    expect(unlinkSync).not.toHaveBeenCalledWith(path.join(trainedModelsDir, "unrelated.pkl"));
    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("TIMESTAMPDIFF");
    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("NOW()");
    expect(poolQuery.mock.calls[0]?.[1]).toEqual([
      7 * 24 * 60 * 60,
      990002,
      20240002,
    ]);
  });

  test("keeps stale model artifact groups when database time says the experiment is still active", async () => {
    const modelFile = "exp990003-student20240003-ma.pkl";
    const productionFile = "exp990003-student20240003-ma-production.pkl";

    poolQuery.mockClear();
    poolQuery.mockResolvedValueOnce([[{ can_delete: 0 }]]);
    readdirSync.mockReturnValueOnce([modelFile, productionFile]);
    statSync.mockReturnValue({
      isDirectory: () => false,
      isFile: () => true,
      mtimeMs: new Date("2026-06-01T00:00:00.000Z").getTime(),
    });

    const result = await cleanupStaleModelArtifacts({
      now: new Date("2026-07-05T00:00:00.000Z"),
      maxAgeMs: 7 * 24 * 60 * 60 * 1000,
    });

    expect(result).toEqual({
      scannedGroups: 1,
      deletedGroups: 0,
      deletedFiles: 0,
      failedFiles: 0,
    });
    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("TIMESTAMPDIFF");
    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("NOW()");
    expect(poolQuery.mock.calls[0]?.[1]).toEqual([
      7 * 24 * 60 * 60,
      990003,
      20240003,
    ]);
    expect(unlinkSync).not.toHaveBeenCalled();
  });

  test("preserves sidecars and ensemble members belonging to an active artifact", async () => {
    const mainFile = "exp990004-student20240004-weighted_avg-backtest-revision.pkl";
    const memberFile = "exp990004-student20240004-weighted_avg-backtest-revision_base_ma.pkl";
    const lstmMemberFile = "exp990004-student20240004-weighted_avg-backtest-revision_base_lstm.pkl";
    const lstmSidecarFile = "exp990004-student20240004-weighted_avg-backtest-revision_base_lstm_keras_model.h5";
    const unrelatedFile = "exp990004-student20240004-ma-backtest-orphan.pkl";
    const activeMainPath = path.join(mockedTrainedModelsRoot, mainFile);

    poolQuery.mockClear();
    poolQuery
      .mockResolvedValueOnce([[{ can_delete: 1 }]])
      .mockResolvedValueOnce([[
        {
          backtest_artifact_path: activeMainPath,
          production_artifact_path: null,
        },
      ]]);
    readdirSync.mockReturnValueOnce([
      mainFile,
      memberFile,
      lstmMemberFile,
      lstmSidecarFile,
      unrelatedFile,
    ]);
    statSync.mockReturnValue({
      isDirectory: () => false,
      isFile: () => true,
      mtimeMs: new Date("2026-06-01T00:00:00.000Z").getTime(),
    });

    const result = await cleanupStaleModelArtifacts({
      now: new Date("2026-07-05T00:00:00.000Z"),
      maxAgeMs: 7 * 24 * 60 * 60 * 1000,
    });

    expect(result.deletedFiles).toBe(1);
    expect(unlinkSync).toHaveBeenCalledTimes(1);
    expect(unlinkSync).toHaveBeenCalledWith(
      path.join(mockedTrainedModelsRoot, unrelatedFile),
    );
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
    installArtifactPoolMock({
      modelType: "weighted_avg",
      experimentOverrides: {
        data_window_train_end_index: 0,
        data_window_evaluate_start_index: 1,
        data_window_evaluate_end_index: 2,
      },
      productionReady: false,
    });

    existsSync.mockImplementation((filePath: string) => !filePath.includes("-weighted_avg-production.pkl"));
    createReadStream.mockImplementation(() =>
      Readable.from([
        "月份,销售数量,价格指数,产能利用率\n2024-01,100,1.2,0.8\n2024-02,110,1.3,0.9\n2024-03,120,1.4,1.0\n",
      ]) as unknown as ReturnType<typeof createReadStream>
    );
    spawnMock
      .mockImplementationOnce(() => spawnResult({ valid: true }))
      .mockImplementationOnce(defaultSpawnImplementation)
      .mockImplementationOnce(() => spawnResult({ valid: true }));

    const result = await prepareExperimentModelForProduction("weighted_avg", 20240002, 990002);

    expect(result).toEqual({
      status: "success",
      results: {
        saved_model: expect.stringMatching(
          /^\/tmp\/fangzhen-py\/trained_models\/exp990002-student20240002-weighted_avg-production-revision-weighted_avg-[0-9a-f-]{36}\.pkl$/,
        ),
        artifact_phase: "production",
        reused_existing: false,
        artifact_revision: "revision-weighted_avg",
      },
    });

    const fitCall = spawnMock.mock.calls.find(([command]) => command.includes("--fit-only"));
    const [command = []] = fitCall ?? [];
    expect(command).toContain("weighted_avg");
    expect(command).toContain("--fit-only");
    expect(command).toContain("--load-model");
    expect(command).toContain("/tmp/fangzhen-py/trained_models/exp990002-student20240002-weighted_avg.pkl");
    expect(command).toContain("--save-model");
    expect(command).toContain(result.results.saved_model);
    expect(command).toContain("--train-end");
    expect(command).toContain("2");
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
        artifact_revision: "revision-ma",
      },
    });
    expect(spawnMock).not.toHaveBeenCalled();
    expect(createReadStream).not.toHaveBeenCalled();
  });

  test("rejects an active artifact after training inputs change", async () => {
    installArtifactPoolMock({
      modelType: "ma",
      experimentOverrides: { training_input_version: 3 },
      productionReady: true,
    });
    existsSync.mockReturnValue(true);

    await expect(
      prepareExperimentModelForProduction("ma", 20240002, 990002),
    ).rejects.toMatchObject({
      message: "实验训练输入已更新，当前模型产物不再匹配；请重新训练模型",
    });
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("fits a production artifact from the backtest model with fit-only parameters", async () => {
    installArtifactPoolMock({
      modelType: "ma",
      experimentOverrides: {
        data_window_train_end_index: 0,
        data_window_evaluate_start_index: 1,
        data_window_evaluate_end_index: 2,
      },
      productionReady: false,
    });

    existsSync.mockImplementation((filePath: string) => !filePath.includes("-ma-production.pkl"));
    createReadStream.mockImplementation(() =>
      Readable.from([
        "月份,销售数量,价格指数,产能利用率\n2024-01,100,1.2,0.8\n2024-02,110,1.3,0.9\n2024-03,120,1.4,1.0\n",
      ]) as unknown as ReturnType<typeof createReadStream>
    );

    const result = await prepareExperimentModelForProduction("ma", 20240002, 990002);

    expect(result).toEqual({
      status: "success",
      results: {
        saved_model: expect.stringMatching(
          /^\/tmp\/fangzhen-py\/trained_models\/exp990002-student20240002-ma-production-revision-ma-[0-9a-f-]{36}\.pkl$/,
        ),
        artifact_phase: "production",
        reused_existing: false,
        artifact_revision: "revision-ma",
      },
    });

    const [command, options] = spawnMock.mock.calls[0] ?? [];
    expect(command).toContain("ma");
    expect(command).toContain("--fit-only");
    expect(command).toContain("--load-model");
    expect(command).toContain("/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma.pkl");
    expect(command).toContain("--save-model");
    expect(command).toContain(result.results.saved_model);
    expect(command).toContain("--train-end");
    expect(command).toContain("2");
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

  test("fences an obsolete production worker before it can publish its artifact", async () => {
    let finalArtifactPath: string | null = null;
    poolQuery.mockImplementation(async (sqlValue: unknown, params: unknown[] = []) => {
      const sql = String(sqlValue);
      if (sql.includes("SET production_artifact_path = ?")) {
        finalArtifactPath = String(params[0]);
        return [{ affectedRows: 0, insertId: 0 }];
      }
      if (/^\s*(?:INSERT|UPDATE|DELETE)/i.test(sql)) {
        return [{ affectedRows: 1, insertId: 1 }];
      }
      if (sql.includes("FROM experiment_status")) {
        return [[buildExperimentContextRow({
          data_window_train_end_index: 0,
          data_window_evaluate_start_index: 1,
          data_window_evaluate_end_index: 2,
        })]];
      }
      if (sql.includes("FROM experiment_model_artifacts")) {
        return [[buildActiveArtifactRow("ma", {
          production_artifact_path: null,
          production_revision_token: null,
        })]];
      }
      if (sql.includes("file_path") && sql.includes("dataset_segments")) {
        return [[{ file_path: datasetCsvPath }]];
      }
      return defaultPoolQuery(sqlValue, params);
    });
    existsSync.mockImplementation((filePath: string) => (
      filePath.endsWith("/run_model.py")
      || filePath.includes("-ma.pkl")
      || filePath.includes("-ma-production-revision-ma-")
    ));
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );

    await expect(
      prepareExperimentModelForProduction("ma", 20240002, 990002),
    ).rejects.toMatchObject({
      message: "回测模型版本已更新，本次生产模型结果已丢弃",
    });

    expect(finalArtifactPath).toContain("-ma-production-revision-ma-");
    expect(unlinkSync).toHaveBeenCalledWith(finalArtifactPath);
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
    let registeredProductionPath: string | null = null;
    poolQuery.mockImplementation(async (sqlValue: unknown, params: unknown[] = []) => {
      const sql = String(sqlValue);
      if (/^\s*(?:INSERT|UPDATE|DELETE)/i.test(sql)) {
        if (sql.includes("SET production_artifact_path = ?")) {
          registeredProductionPath = String(params[0]);
        }
        return [{ affectedRows: 1, insertId: 1 }];
      }
      if (sql.includes("FROM experiment_status")) {
        return [[buildExperimentContextRow({
          data_window_train_end_index: 0,
          data_window_evaluate_start_index: 1,
          data_window_evaluate_end_index: 2,
        })]];
      }
      if (sql.includes("FROM experiment_model_artifacts")) {
        return [[buildActiveArtifactRow("ma", registeredProductionPath ? {
          production_artifact_path: registeredProductionPath,
          production_revision_token: "revision-ma",
        } : {
          production_artifact_path: null,
          production_revision_token: null,
        })]];
      }
      if (sql.includes("file_path") && sql.includes("dataset_segments")) {
        return [[{ file_path: datasetCsvPath }]];
      }
      return defaultPoolQuery(sqlValue, params);
    });
    existsSync.mockImplementation((filePath: string) => (
      filePath.endsWith("/run_model.py")
      || (filePath.includes("-ma.pkl") && !filePath.includes("-production"))
      || filePath === registeredProductionPath
      || filePath.includes("-ma-production-revision-ma-")
    ));
    createReadStream.mockImplementation(() =>
      Readable.from([
        "月份,销售数量,价格指数,产能利用率\n2024-01,100,1.2,0.8\n2024-02,110,1.3,0.9\n2024-03,120,1.4,1.0\n",
      ]) as unknown as ReturnType<typeof createReadStream>
    );
    spawnMock
      .mockImplementationOnce(() => ({
        stdout: new ReadableStream({
          start(controller) {
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
      results: expect.objectContaining({
        predictions: [{ prediction: 101.2 }, { prediction: 98.4 }],
      }),
    });
    expect(spawnMock).toHaveBeenCalledTimes(2);

    const [prepareCommand] = spawnMock.mock.calls[0] ?? [];
    expect(prepareCommand).toContain("--fit-only");
    expect(prepareCommand.some((arg) => arg.includes("exp990002-student20240002-ma-production-revision-ma-"))).toBe(true);

    const [predictCommand] = spawnMock.mock.calls[1] ?? [];
    expect(predictCommand).toContain("--forecast-steps");
    expect(registeredProductionPath).not.toBeNull();
    expect(predictCommand).toContain(registeredProductionPath!);
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
      results: expect.objectContaining({
        predictions: [{ prediction: 101.2 }, { prediction: 98.4 }],
      }),
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
    installTrainingPoolMock();

    existsSync.mockImplementation((filePath: string) => (
      filePath.endsWith("/run_model.py")
      || filePath.includes("-weighted_avg-backtest-")
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
    })).mockImplementationOnce(() => spawnResult({ valid: true }));

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

  test("does not reuse base-model artifacts from an older training-input version", async () => {
    poolQuery.mockImplementation(async (sqlValue: unknown, params: unknown[] = []) => {
      const sql = String(sqlValue);
      if (/^\s*(?:INSERT|UPDATE|DELETE)/i.test(sql)) {
        return [{ affectedRows: 1, insertId: 1 }];
      }
      if (sql.includes("FROM experiment_status")) {
        return [[buildExperimentContextRow()]];
      }
      if (sql.includes("FROM experiment_model_artifacts")) {
        const requestedModelType = typeof params[2] === "string" ? params[2] : "ma";
        return [[buildActiveArtifactRow(requestedModelType, { training_input_version: 1 })]];
      }
      if (sql.includes("file_path") && sql.includes("dataset_segments")) {
        return [[{ file_path: datasetCsvPath }]];
      }
      return defaultPoolQuery(sqlValue, params);
    });
    existsSync.mockImplementation((filePath: string) => (
      filePath.endsWith("/run_model.py")
      || filePath.includes("-weighted_avg-backtest-")
    ));
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    spawnMock
      .mockImplementationOnce(() => spawnResult({
        metrics: { rmse: 1, mae: 1, r2: 0.9 },
        models: ["ma", "lstm"],
        weights: { ma: 0.4, lstm: 0.6 },
        eval_y_true: [],
        predictions: [],
      }))
      .mockImplementationOnce(() => spawnResult({ valid: true }));

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
    expect(command).not.toContain("--base-model-files");
  });

  test("invalidates dependent ensemble artifacts after a base model is retrained", async () => {
    installTrainingPoolMock();
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );

    const artifactPaths = new Set([
      "/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma-production.pkl",
      "/tmp/fangzhen-py/trained_models/exp990002-student20240002-weighted_avg.pkl",
      "/tmp/fangzhen-py/trained_models/exp990002-student20240002-weighted_avg-production.pkl",
      "/tmp/fangzhen-py/trained_models/exp990002-student20240002-boosting.pkl",
      "/tmp/fangzhen-py/trained_models/exp990002-student20240002-boosting-production.pkl",
      "/tmp/fangzhen-py/trained_models/exp990002-student20240002-stacking.pkl",
      "/tmp/fangzhen-py/trained_models/exp990002-student20240002-stacking-production.pkl",
    ]);
    existsSync.mockImplementation((filePath: string) =>
      filePath.endsWith("/run_model.py")
      || filePath.includes("-ma-backtest-")
      || artifactPaths.has(filePath)
    );
    spawnMock.mockImplementationOnce(() => ({
      stdout: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(JSON.stringify({
            metrics: { rmse: 1, mae: 1, mape: 1, r2: 0.9 },
            eval_y_true: [100, 110],
            predictions: [
              { prediction: 101, std_dev: 1 },
              { prediction: 109, std_dev: 1 },
            ],
          })));
          controller.close();
        },
      }),
      stderr: new ReadableStream({ start: (controller) => controller.close() }),
      exited: Promise.resolve(0),
      exitCode: 0,
      killed: false,
      pid: 4321,
      kill: mock(() => {}),
    })).mockImplementationOnce(() => spawnResult({ valid: true }));

    await trainModel("ma", 20240002, {
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

    expect(unlinkSync).not.toHaveBeenCalled();
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("DELETE FROM experiment_model_artifacts")
      && String(sql).includes("weighted_avg")
    )).toBeTrue();

    const stateResetCall = poolQuery.mock.calls.find(([sql]) =>
      String(sql).includes("UPDATE experiment_status")
    );
    expect(stateResetCall).toBeDefined();
    expect(String(stateResetCall?.[0])).toContain("moving_average_completed");
    expect(String(stateResetCall?.[0])).toContain("ensemble_weighted_completed");
    expect(String(stateResetCall?.[0])).toContain("ensemble_boosting_completed");
    expect(String(stateResetCall?.[0])).toContain("ensemble_stacking_completed");
    expect(String(stateResetCall?.[0])).toContain("quiz_about_plan_completed");
    expect(stateResetCall?.[1]?.slice(-2)).toEqual([990002, 20240002]);
  });

  test("passes persisted LSTM feature lists to ensemble training as JSON to preserve comma-containing field names", async () => {
    installTrainingPoolMock({
      lstm_features: JSON.stringify(["价格指数", "外部,指标"]),
    });

    existsSync.mockImplementation((filePath: string) => (
      filePath.endsWith("/run_model.py")
      || filePath.includes("-weighted_avg-backtest-")
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
    })).mockImplementationOnce(() => spawnResult({ valid: true }));

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
    poolQuery.mockClear();
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
      results: expect.objectContaining({
        predictions: [],
      }),
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
