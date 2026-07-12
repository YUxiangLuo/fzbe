import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import * as actualFs from "node:fs";
import { unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const pythonModulePath = path.join(srcRoot, "utils", "python.ts");
const modelRuntimeModulePath = path.join(srcRoot, "modules", "models", "model-runtime.ts");
const modelRunStoreModulePath = path.join(srcRoot, "modules", "models", "model-run-store.ts");
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
const renameSync = mock((_: string, __: string) => {});
const mkdirSync = mock((_: string, __?: unknown) => undefined);
const rmSync = mock((_: string, __?: unknown) => undefined);
const createSpawnResult = (stdout = "", stderr = "", exitCode = 0) => ({
  stdout: new ReadableStream({
    start(controller) {
      if (stdout) controller.enqueue(new TextEncoder().encode(stdout));
      controller.close();
    },
  }),
  stderr: new ReadableStream({
    start(controller) {
      if (stderr) controller.enqueue(new TextEncoder().encode(stderr));
      controller.close();
    },
  }),
  exited: Promise.resolve(exitCode),
  exitCode,
  killed: false,
  pid: 4321,
  kill: mock(() => {}),
});
const spawnMock = mock((_: string[], __: Record<string, unknown>) => (
  createSpawnResult('{"predictions":[]}')
));
const getActiveModelTrainingRun = mock(async (
  _experimentId: number,
  _studentId: number,
  _modelType: string,
) => null as null | {
  run_id: string;
  artifact_path: string;
  production_artifact_path: string | null;
  model_type?: string;
  dataset_revision?: string;
  runtime_revision?: string;
});
const setProductionArtifactForRun = mock(async (_runId: string, _artifactPath: string) => true);
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
    statSync,
    unlinkSync,
    copyFileSync,
    renameSync,
    mkdirSync,
    rmSync,
  },
  existsSync,
  createReadStream,
  readdirSync,
  statSync,
  unlinkSync,
  copyFileSync,
  renameSync,
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
  buildVersionedModelPath: (
    experimentId: number,
    studentId: number,
    modelType: string,
    runId: string,
    phase: "backtest" | "production" = "backtest",
  ) => path.join(
    mockedTrainedModelsRoot,
    `exp${experimentId}-student${studentId}-${modelType}-${runId}-${phase}.pkl`,
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
  buildModelExecutionLockName: (experimentId: number, modelType: string) =>
    `fangzhen-model:exp:${experimentId}:${modelType}`,
  buildModelSlotLockName: (slot: number) => `fangzhen-model:slot:${slot}`,
}));

mock.module(modelRunStoreModulePath, () => ({
  buildExperimentTrainingRevision: mock(() => "experiment-revision"),
  buildModelTrainingRequestHash: mock(() => "request-hash"),
  commitModelTrainingRun: mock(async ({ runId, sessionId }: { runId?: string; sessionId?: string }) => ({
    runId: runId ?? sessionId ?? "run-test",
    stateVersion: 1,
    reusedExistingRun: false,
  })),
  getActiveModelTrainingRun,
  setProductionArtifactForRun,
}));

const {
  trainModel,
  runPythonModel,
  createOrResumeGuidedTrainingSession,
  cleanupExpiredGuidedTrainingSessions,
  cleanupStaleModelArtifacts,
  getReusableProductionModelArtifactResult,
  getModelRuntimeRevision,
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

const buildActiveRun = (
  modelType: string,
  overrides: Record<string, unknown> = {},
) => {
  const runId = `${modelType}-run-current`;
  return {
    run_id: runId,
    artifact_path: path.join(
      mockedTrainedModelsRoot,
      `exp990002-student20240002-${modelType}-${runId}-backtest.pkl`,
    ),
    production_artifact_path: null,
    model_type: modelType,
    ...overrides,
  };
};

const useActiveRuns = (...runs: ReturnType<typeof buildActiveRun>[]) => {
  const runsByModel = new Map(runs.map((run) => [run.model_type, run]));
  getActiveModelTrainingRun.mockImplementation(async (_experimentId, _studentId, modelType) => (
    runsByModel.get(modelType) ?? null
  ));
};

const mockExperimentAndDatasetQueries = (
  experiment = buildExperimentContextRow(),
) => {
  poolQuery.mockReset();
  poolQuery.mockImplementation(async (sql: string) => (
    sql.includes("FROM dataset_segments")
      ? [[{ file_path: datasetCsvPath }]]
      : [[experiment]]
  ));
};

describe("models service", () => {
  beforeEach(async () => {
    await writeFile(datasetCsvPath, modelDatasetCsv);

    poolQuery.mockReset();
    poolQuery.mockResolvedValue([
      [buildExperimentContextRow()],
    ]);

    getActiveModelTrainingRun.mockReset();
    getActiveModelTrainingRun.mockResolvedValue(null);
    setProductionArtifactForRun.mockReset();
    setProductionArtifactForRun.mockResolvedValue(true);

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
    renameSync.mockClear();
    mkdirSync.mockClear();
    rmSync.mockClear();
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
      .mockResolvedValueOnce([[{ file_path: datasetCsvPath }]])
      .mockResolvedValueOnce([[
        {
          session_id: "guided-a",
          experiment_id: 990002,
          student_id: 20240002,
          model_type: "ma",
          request_hash: "request-hash",
          dataset_revision: "0".repeat(64),
          runtime_revision: "0".repeat(64),
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
    const versionedPath = "/tmp/fangzhen-py/trained_models/exp990002-student20240002-ma-guided-a-backtest.pkl";
    expect(copyFileSync).toHaveBeenCalledWith(
      sessionModelPath,
      expect.stringContaining(`${versionedPath}.tmp-`),
    );
    expect(renameSync).toHaveBeenCalledWith(
      expect.stringContaining(`${versionedPath}.tmp-`),
      versionedPath,
    );
    expect(unlinkSync).not.toHaveBeenCalledWith(productionModelPath);
    expect(spawnMock).not.toHaveBeenCalled();
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
      request_hash: "request-hash",
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

    poolQuery.mockReset();
    poolQuery
      .mockResolvedValueOnce([[buildExperimentContextRow()]])
      .mockResolvedValueOnce([[{ file_path: datasetCsvPath }]])
      .mockResolvedValueOnce([[completedSessionRow]])
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([[freshSessionRow]]);
    existsSync.mockReturnValue(false);

    const session = await createOrResumeGuidedTrainingSession("ma", 20240002, requestBody);

    expect(session.session_id).toBe("guided-fresh");
    expect(session.status).toBe("ready");
    expect(copyFileSync).not.toHaveBeenCalled();
    expect(rmSync).toHaveBeenCalledWith(
      expiredArtifactDir,
      { recursive: true, force: true },
    );
    expect(mkdirSync).toHaveBeenCalledWith(
      expect.stringContaining("/tmp/fangzhen-py/guided_training/exp990002-student20240002-ma-"),
      { recursive: true },
    );
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("DELETE FROM model_guided_training_sessions"),
    )).toBe(true);
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("INSERT INTO model_guided_training_sessions"),
    )).toBe(true);
  });

  test("cleans up expired guided sessions and only removes guided artifact directories", async () => {
    const safeArtifactDir = "/tmp/fangzhen-py/guided_training/exp990002-student20240002-ma-expired";
    const unsafeArtifactDir = "/tmp/fangzhen-py/trained_models/not-a-guided-session";
    const now = new Date("2026-07-05T00:00:00.000Z");

    poolQuery.mockReset();
    poolQuery
      .mockResolvedValueOnce([[
        { session_id: "expired-safe", artifact_dir: safeArtifactDir },
        { session_id: "expired-unsafe", artifact_dir: unsafeArtifactDir },
      ]])
      .mockResolvedValueOnce([[]]);

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
    expect(poolQuery.mock.calls[0]?.[1]).toEqual([now]);
    expect(poolQuery.mock.calls[1]?.[1]).toEqual([
      "expired-safe",
      "expired-unsafe",
    ]);
  });

  test("uses database time for guided session expiration in production cleanup", async () => {
    poolQuery.mockReset();
    poolQuery.mockResolvedValueOnce([[]]);
    readdirSync.mockReturnValueOnce([]);

    await cleanupExpiredGuidedTrainingSessions({ limit: 50 });

    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("expires_at <= NOW()");
    expect(poolQuery.mock.calls[0]?.[1]).toEqual([]);
  });

  test("keeps expired guided session rows when artifact deletion fails", async () => {
    const artifactDir = "/tmp/fangzhen-py/guided_training/exp990002-student20240002-ma-stuck";

    poolQuery.mockReset();
    poolQuery.mockResolvedValueOnce([[
      { session_id: "expired-stuck", artifact_dir: artifactDir },
    ]]);
    rmSync.mockImplementationOnce(() => {
      throw new Error("permission denied");
    });

    const result = await cleanupExpiredGuidedTrainingSessions({
      now: new Date("2026-07-05T00:00:00.000Z"),
      limit: 50,
    });

    expect(result).toEqual({
      deletedSessions: 0,
      deletedArtifactDirs: 0,
    });
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("DELETE FROM model_guided_training_sessions"),
    )).toBe(false);
  });

  test("removes old orphan guided artifact directories", async () => {
    const orphanName = "exp990002-student20240002-ma-orphan";
    const orphanDir = `/tmp/fangzhen-py/guided_training/${orphanName}`;

    poolQuery.mockReset();
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

    poolQuery.mockReset();
    poolQuery.mockResolvedValueOnce([[
      {
        can_delete: 1,
      },
    ]]);
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

    poolQuery.mockReset();
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

  test("rejects a reusable production artifact after the model runtime changes", async () => {
    getActiveModelTrainingRun.mockResolvedValue({
      run_id: "ma-run-stale-runtime",
      artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-stale-runtime.pkl",
      production_artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-stale-runtime-production.pkl",
      model_type: "ma",
      dataset_revision: "dataset-revision",
      runtime_revision: "stale-runtime-revision",
    });
    poolQuery
      .mockResolvedValueOnce([[buildExperimentContextRow()]])
      .mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );

    await expect(
      getReusableProductionModelArtifactResult("ma", 20240002, 990002),
    ).rejects.toThrow("MA 模型运行环境已变化，请重新训练该模型");
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("rejects a reusable production artifact after the dataset changes", async () => {
    getActiveModelTrainingRun.mockResolvedValue({
      run_id: "ma-run-stale-data",
      artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-stale-data.pkl",
      production_artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-stale-data-production.pkl",
      model_type: "ma",
      dataset_revision: "stale-dataset-revision",
      runtime_revision: getModelRuntimeRevision(),
    });
    poolQuery
      .mockResolvedValueOnce([[buildExperimentContextRow()]])
      .mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );

    await expect(
      getReusableProductionModelArtifactResult("ma", 20240002, 990002),
    ).rejects.toThrow("MA 模型对应的数据集版本已变化，请重新训练该模型");
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("rejects ensemble training when a selected base run belongs to an old runtime", async () => {
    getActiveModelTrainingRun.mockResolvedValue({
      run_id: "base-run-stale-runtime",
      artifact_path: "/tmp/fangzhen-py/trained_models/base-run-stale-runtime.pkl",
      production_artifact_path: null,
      model_type: "ma",
      dataset_revision: "dataset-revision",
      runtime_revision: "stale-runtime-revision",
    });
    poolQuery
      .mockResolvedValueOnce([[buildExperimentContextRow()]])
      .mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );

    await expect(trainModel("weighted_avg", 20240002, {
      experiment_id: 990002,
      models: "ma,lstm",
    })).rejects.toThrow("MA 模型运行环境已变化，请重新训练该模型");
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("rejects undersized training and evaluation windows before loading data", async () => {
    poolQuery.mockResolvedValueOnce([[
      buildExperimentContextRow({
        data_window_train_end_index: 6,
        data_window_evaluate_start_index: 7,
        data_window_evaluate_end_index: 8,
      }),
    ]]);

    await expect(trainModel("es", 20240002, {
      experiment_id: 990002,
      data_window_train_start_index: 0,
      data_window_train_end_index: 6,
      data_window_evaluate_start_index: 7,
      data_window_evaluate_end_index: 8,
      exponential_smoothing_alpha: 0.35,
    })).rejects.toMatchObject({
      message: "训练数据窗口至少需要 8 个数据点，请重新选择数据窗口后重试",
      details: { training_points: 7, minimum_training_points: 8 },
    });

    poolQuery.mockResolvedValueOnce([[
      buildExperimentContextRow({
        data_window_train_end_index: 7,
        data_window_evaluate_start_index: 8,
        data_window_evaluate_end_index: 8,
      }),
    ]]);
    await expect(trainModel("es", 20240002, {
      experiment_id: 990002,
      data_window_train_start_index: 0,
      data_window_train_end_index: 7,
      data_window_evaluate_start_index: 8,
      data_window_evaluate_end_index: 8,
      exponential_smoothing_alpha: 0.35,
    })).rejects.toMatchObject({
      message: "评估数据窗口至少需要 2 个数据点，请重新选择数据窗口后重试",
      details: { evaluation_points: 1, minimum_evaluation_points: 2 },
    });
    expect(createReadStream).not.toHaveBeenCalled();
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("fits an ensemble production artifact from the backtest model with fit-only parameters", async () => {
    const activeRun = buildActiveRun("weighted_avg");
    useActiveRuns(activeRun);
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    spawnMock
      .mockImplementationOnce(() => createSpawnResult('{"mode":"weighted_avg_fit_only"}'))
      .mockImplementationOnce(() => createSpawnResult('{"valid":true}'));

    const result = await prepareExperimentModelForProduction("weighted_avg", 20240002, 990002);

    const [command] = spawnMock.mock.calls[0] ?? [];
    const saveModelIndex = command.indexOf("--save-model");
    const candidatePath = command[saveModelIndex + 1];
    expect(candidatePath).toMatch(
      /^\/tmp\/fangzhen-py\/trained_models\/exp990002-student20240002-weighted_avg-weighted_avg-run-current-production-[\w-]+\.pkl$/,
    );
    expect(result).toEqual({
      status: "success",
      results: {
        saved_model: candidatePath,
        artifact_phase: "production",
        reused_existing: false,
      },
    });
    expect(command).toContain("weighted_avg");
    expect(command).toContain("--fit-only");
    expect(command).toContain("--load-model");
    expect(command).toContain(activeRun.artifact_path);
    expect(command).toContain("--save-model");
    expect(command).toContain(candidatePath);
    expect(command).toContain("--train-end");
    expect(command).toContain("17");
    expect(command).toContain("--fit-forecast-steps");
    expect(command).toContain("6");
    expect(setProductionArtifactForRun).toHaveBeenCalledWith(activeRun.run_id, candidatePath);
  });

  test("rejects a legacy production artifact without a verifiable active run", async () => {
    existsSync.mockImplementation((filePath: string) => filePath.includes("-ma-production.pkl"));

    await expect(
      prepareExperimentModelForProduction("ma", 20240002, 990002),
    ).rejects.toThrow("MA 模型缺少可验证的训练版本，请重新训练该模型");

    expect(spawnMock).not.toHaveBeenCalled();
    expect(createReadStream).not.toHaveBeenCalled();
  });

  test("reuses a revision-checked production artifact from the active run", async () => {
    const productionPath = path.join(mockedTrainedModelsRoot, "ma-current-production.pkl");
    const activeRun = buildActiveRun("ma", { production_artifact_path: productionPath });
    useActiveRuns(activeRun);
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );

    const result = await prepareExperimentModelForProduction("ma", 20240002, 990002);

    expect(result).toEqual({
      status: "success",
      results: {
        saved_model: productionPath,
        artifact_phase: "production",
        reused_existing: true,
      },
    });
    expect(spawnMock).not.toHaveBeenCalled();
    expect(poolQuery).toHaveBeenCalledWith(
      expect.stringContaining("FROM dataset_segments"),
      ["制造业", "样本企业", "样本产品"],
    );
  });

  test("fits a production artifact from the backtest model with fit-only parameters", async () => {
    const activeRun = buildActiveRun("ma");
    useActiveRuns(activeRun);
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );

    const result = await prepareExperimentModelForProduction("ma", 20240002, 990002);

    const [command, options] = spawnMock.mock.calls[0] ?? [];
    const saveModelIndex = command.indexOf("--save-model");
    const candidatePath = command[saveModelIndex + 1];
    expect(result.results.saved_model).toBe(candidatePath);
    expect(command).toContain("ma");
    expect(command).toContain("--fit-only");
    expect(command).toContain("--load-model");
    expect(command).toContain(activeRun.artifact_path);
    expect(command).toContain("--save-model");
    expect(command).toContain(candidatePath);
    expect(command).toContain("--train-end");
    expect(command).toContain("17");
    expect(command).toContain("--fit-forecast-steps");
    expect(command).toContain("6");
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
    expect(setProductionArtifactForRun).toHaveBeenCalledWith(activeRun.run_id, candidatePath);
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
    const activeRun = buildActiveRun("ma");
    useActiveRuns(activeRun);

    await expect(
      predictExperimentModel("ma", 20240002, 990002, 6),
    ).rejects.toBeInstanceOf(ModelNotFoundError);

    expect(existsSync).toHaveBeenCalledWith(
      activeRun.artifact_path,
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
    const activeRun = buildActiveRun("ma");
    useActiveRuns(activeRun);
    mockExperimentAndDatasetQueries();

    let productionArtifactPrepared = false;
    existsSync.mockImplementation((filePath: string) => (
      filePath.endsWith("/run_model.py")
      || filePath === activeRun.artifact_path
      || (productionArtifactPrepared && filePath.includes("-production-"))
    ));
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    spawnMock
      .mockImplementationOnce(() => ({
        stdout: new ReadableStream({
          start(controller) {
            productionArtifactPrepared = true;
            controller.enqueue(new TextEncoder().encode('{"mode":"ma_fit_only"}'));
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
    const saveModelIndex = prepareCommand.indexOf("--save-model");
    const candidatePath = prepareCommand[saveModelIndex + 1];
    expect(prepareCommand).toContain("--fit-only");
    expect(prepareCommand).toContain("--fit-forecast-steps");
    expect(prepareCommand).toContain("6");
    expect(candidatePath).toContain("exp990002-student20240002-ma-ma-run-current-production-");

    const [predictCommand] = spawnMock.mock.calls[1] ?? [];
    expect(predictCommand).toContain("--forecast-steps");
    expect(predictCommand).toContain(candidatePath);
    expect(setProductionArtifactForRun).toHaveBeenCalledWith(activeRun.run_id, candidatePath);
  });

  test("runs prediction with the experiment-scoped model artifact and returns python predictions", async () => {
    const productionPath = path.join(mockedTrainedModelsRoot, "ma-current-production.pkl");
    useActiveRuns(buildActiveRun("ma", { production_artifact_path: productionPath }));
    mockExperimentAndDatasetQueries();
    existsSync.mockImplementation(() => true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
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
    expect(command).toContain(productionPath);
    expect(options).toMatchObject({
      cwd: "/tmp/fangzhen-py",
      stdout: "pipe",
      stderr: "pipe",
    });
  });

  test("floors negative production demand predictions at the API boundary", async () => {
    const productionPath = path.join(mockedTrainedModelsRoot, "ma-current-production.pkl");
    useActiveRuns(buildActiveRun("ma", { production_artifact_path: productionPath }));
    mockExperimentAndDatasetQueries();
    existsSync.mockImplementation(() => true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    spawnMock.mockImplementationOnce(() => ({
      stdout: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(
            '{"predictions":[{"prediction":-4.2,"std_dev":1.5},{"prediction":3.1,"std_dev":0.8}]}',
          ));
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

    const result = await predictExperimentModel("ma", 20240002, 990002, 2);

    expect(result.results.predictions).toEqual([
      { prediction: 0, std_dev: 1.5 },
      { prediction: 3.1, std_dev: 0.8 },
    ]);
  });

  test("refits and retries when a cached LSTM production horizon is too short", async () => {
    const previousProductionPath = path.join(mockedTrainedModelsRoot, "lstm-current-production.pkl");
    const activeRun = buildActiveRun("lstm", {
      production_artifact_path: previousProductionPath,
    });
    useActiveRuns(activeRun);
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    spawnMock
      .mockImplementationOnce(() => ({
        stdout: new ReadableStream({
          start(controller) {
            controller.close();
          },
        }),
        stderr: new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(
              "[VALIDATION] Requested forecast_steps (8) exceeds the saved LSTM horizon (6).",
            ));
            controller.close();
          },
        }),
        exited: Promise.resolve(1),
        exitCode: 1,
        killed: false,
        pid: 4321,
        kill: mock(() => {}),
      }))
      .mockImplementationOnce(() => ({
        stdout: new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode('{"mode":"lstm_fit_only"}'));
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
            controller.enqueue(new TextEncoder().encode(
              '{"predictions":[{"prediction":12.5,"std_dev":1.2}]}',
            ));
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

    const result = await predictExperimentModel("lstm", 20240002, 990002, 8);

    expect(result.results.predictions).toEqual([{ prediction: 12.5, std_dev: 1.2 }]);
    expect(spawnMock).toHaveBeenCalledTimes(3);
    const [refitCommand] = spawnMock.mock.calls[1] ?? [];
    const saveModelIndex = refitCommand.indexOf("--save-model");
    const candidatePath = refitCommand[saveModelIndex + 1];
    expect(refitCommand).toContain("--fit-only");
    expect(refitCommand).toContain("--fit-forecast-steps");
    expect(refitCommand).toContain("8");
    expect(candidatePath).not.toBe(previousProductionPath);
    expect(setProductionArtifactForRun).toHaveBeenCalledWith(activeRun.run_id, candidatePath);
    expect(unlinkSync).toHaveBeenCalledWith(previousProductionPath);
  });

  test("keeps the current production artifact untouched when horizon refit fails", async () => {
    const previousProductionPath = path.join(mockedTrainedModelsRoot, "lstm-current-production.pkl");
    const activeRun = buildActiveRun("lstm", {
      production_artifact_path: previousProductionPath,
    });
    useActiveRuns(activeRun);
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    spawnMock
      .mockImplementationOnce(() => createSpawnResult(
        "",
        "[VALIDATION] Requested forecast_steps (8) exceeds the saved LSTM horizon (6).",
        1,
      ))
      .mockImplementationOnce(() => createSpawnResult(
        "",
        "[VALIDATION] synthetic production refit failure",
        1,
      ));

    await expect(
      predictExperimentModel("lstm", 20240002, 990002, 8),
    ).rejects.toThrow("synthetic production refit failure");

    const [refitCommand] = spawnMock.mock.calls[1] ?? [];
    const saveModelIndex = refitCommand.indexOf("--save-model");
    const candidatePath = refitCommand[saveModelIndex + 1];
    expect(candidatePath).not.toBe(previousProductionPath);
    expect(unlinkSync).toHaveBeenCalledWith(candidatePath);
    expect(unlinkSync).not.toHaveBeenCalledWith(previousProductionPath);
    expect(setProductionArtifactForRun).not.toHaveBeenCalled();
  });

  test("rejects a prediction result when its active model version changes in flight", async () => {
    const productionPath = "/tmp/fangzhen-py/trained_models/ma-run-a-production.pkl";
    const runA = {
      run_id: "ma-run-a",
      artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-a.pkl",
      production_artifact_path: productionPath,
      model_type: "ma",
    };
    const runB = {
      run_id: "ma-run-b",
      artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-b.pkl",
      production_artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-b-production.pkl",
      model_type: "ma",
    };
    getActiveModelTrainingRun
      .mockResolvedValueOnce(runA)
      .mockResolvedValueOnce(runA)
      .mockResolvedValueOnce(runB);
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    spawnMock.mockImplementationOnce(() => ({
      stdout: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('{"predictions":[{"prediction":101.2}]}'));
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

    await expect(
      predictExperimentModel("ma", 20240002, 990002, 6),
    ).rejects.toMatchObject({
      message: "预测期间模型版本已变化，请重新预测",
    });
  });

  test("passes available base model artifacts through to ensemble training", async () => {
    const maRun = buildActiveRun("ma");
    const lstmRun = buildActiveRun("lstm");
    useActiveRuns(maRun, lstmRun);
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
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
    expect(command).toContain("--data-file");
    expect(command).not.toContain("--data");
    const baseModelFilesArgIndex = command.indexOf("--base-model-files");
    expect(baseModelFilesArgIndex).toBeGreaterThan(-1);

    const encodedBaseModelFiles = command[baseModelFilesArgIndex + 1];
    expect(JSON.parse(encodedBaseModelFiles)).toEqual({
      ma: maRun.artifact_path,
      lstm: lstmRun.artifact_path,
    });
    expect(getActiveModelTrainingRun).toHaveBeenCalledTimes(2);
  });

  test("passes persisted LSTM feature lists to ensemble training as JSON to preserve comma-containing field names", async () => {
    useActiveRuns(buildActiveRun("ma"), buildActiveRun("lstm"));
    mockExperimentAndDatasetQueries(buildExperimentContextRow({
      lstm_features: JSON.stringify(["价格指数", "外部,指标"]),
    }));
    existsSync.mockReturnValue(true);
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

  test("rejects legacy base artifacts without active training runs", async () => {
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );

    await expect(trainModel("weighted_avg", 20240002, {
      experiment_id: 990002,
      models: "ma,lstm",
    })).rejects.toThrow("MA 模型缺少可验证的训练版本，请重新训练该模型");

    expect(spawnMock).not.toHaveBeenCalled();
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
    const productionPath = path.join(mockedTrainedModelsRoot, "lstm-current-production.pkl");
    useActiveRuns(buildActiveRun("lstm", { production_artifact_path: productionPath }));
    mockExperimentAndDatasetQueries();
    existsSync.mockImplementation(() => true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
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
