import { afterEach, beforeEach, describe, expect, jest, mock, test } from "bun:test";
import crypto from "node:crypto";
import * as actualFs from "node:fs";
import { unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { TIMEOUTS } from "../../../src/utils/constants";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const pythonModulePath = path.join(srcRoot, "utils", "python.ts");
const modelRuntimeModulePath = path.join(srcRoot, "modules", "models", "model-runtime.ts");
const modelRunStoreModulePath = path.join(srcRoot, "modules", "models", "model-run-store.ts");
const modelLocksModulePath = path.join(srcRoot, "modules", "models", "model-locks.ts");
const modelsServiceModulePath = path.join(srcRoot, "modules", "models", "models.service.ts");
const freshModelsServiceModulePath = `${modelsServiceModulePath}?models-service-test`;
const datasetCsvPath = "/tmp/dataset.csv";
const mockedPyDir = "/tmp/fangzhen-py";
const mockedGuidedTrainingRoot = path.join(mockedPyDir, "guided_training");
const mockedTrainedModelsRoot = path.join(mockedPyDir, "trained_models");
const preparationToken = "123e4567-e89b-42d3-a456-426614174000";
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

const ENSEMBLE_MODEL_TYPES = ["weighted_avg", "boosting", "stacking"];
// The ensemble manifest is produced inside the (mocked) Python digest script, so
// tests only need a stable value shared between buildActiveRun and the spawn stub.
const ENSEMBLE_ARTIFACT_DIGEST = "e".repeat(64);

const mockArtifactSha256 = (modelType: string) => {
  if (ENSEMBLE_MODEL_TYPES.includes(modelType)) {
    return ENSEMBLE_ARTIFACT_DIGEST;
  }
  const fileSha256 = crypto.createHash("sha256").update(modelDatasetCsv).digest("hex");
  const manifest = crypto.createHash("sha256");
  manifest.update("fz-base-model-artifact-v1");
  manifest.update("\0");
  manifest.update(modelType);
  manifest.update("\0state\0");
  manifest.update(fileSha256);
  if (modelType === "lstm") {
    manifest.update("\0keras\0");
    manifest.update(fileSha256);
  }
  return manifest.digest("hex");
};

const poolQuery = mock(async () => [[]]);
const poolGetConnection = mock(async () => {
  throw new Error("pool.getConnection should not be called in this test");
});
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
// Ensemble verification spawns auxiliary Python (digest + member validation +
// sidecar canonicalization). Route those by their inlined script markers so a
// test only has to describe the actual model run(s).
const withEnsembleAuxiliarySpawns = (
  modelRun: (command: string[], options: Record<string, unknown>) => ReturnType<typeof createSpawnResult>,
) => (command: string[], options: Record<string, unknown>) => {
  const joined = command.join(" ");
  if (joined.includes("fz-ensemble-model-artifact-v1")) {
    return createSpawnResult(JSON.stringify({ ok: true, digest: ENSEMBLE_ARTIFACT_DIGEST }));
  }
  if (joined.includes("keras path is not the canonical sidecar")) {
    return createSpawnResult('{"ok":true}');
  }
  if (joined.includes("base_model_states")) {
    return createSpawnResult('{"valid":true}');
  }
  return modelRun(command, options);
};
const spawnMock = mock((_: string[], __: Record<string, unknown>) => (
  createSpawnResult('{"predictions":[]}')
));
const getActiveModelTrainingRun = mock(async (
  _experimentId: number,
  _studentId: number,
  _modelType: string,
) => null as null | {
  run_id: string;
  session_id?: string | null;
  request_hash?: string;
  artifact_path: string;
  production_artifact_path: string | null;
  production_artifact_forecast_steps?: number | null;
  production_revision?: string | null;
  model_type?: string;
  dataset_revision?: string;
  runtime_revision?: string;
});
const setProductionArtifactForRun = mock(async (_runId: string, _artifactPath: string) => true);
const backfillArtifactSha256ForRun = mock(async (_runId: string, _sha256: string) => true);
const backfillProductionArtifactSha256ForRun = mock(async (
  _runId: string,
  _productionRevision: string,
  _sha256: string,
) => true);
const clearProductionArtifactForRun = mock(async (_runId: string, _productionRevision: string) => true);
const originalSpawn = Bun.spawn;
const acquireModelExecutionLock = mock(async () => ({
  name: "test-execution-lock",
  connection: {},
  signal: new AbortController().signal,
}));
const acquireModelSlot = mock(async () => ({
  name: "test-slot-lock",
  connection: {},
  signal: new AbortController().signal,
}));
const releaseModelExecutionLock = mock(async () => undefined);
const releaseModelSlot = mock(async () => undefined);
const verifyModelLockOwned = mock(async () => undefined);

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
    getConnection: poolGetConnection,
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
  buildGuidedTrainingAttemptPaths: (artifactDir: string, stepId: string, attemptId: string) => ({
    candidateStatePath: path.join(artifactDir, "checkpoints", `${stepId}-${attemptId}.pkl`),
    receiptPath: path.join(artifactDir, "checkpoints", `${stepId}-${attemptId}.receipt.json`),
  }),
  buildModelExecutionLockName: (experimentId: number, modelType: string) =>
    `fangzhen-model:exp:${experimentId}:${modelType}`,
  buildModelSlotLockName: (slot: number) => `fangzhen-model:slot:${slot}`,
}));

const commitModelTrainingRun = mock(async ({ runId, sessionId }: { runId?: string; sessionId?: string }) => ({
  runId: runId ?? sessionId ?? "run-test",
  stateVersion: 1,
  reusedExistingRun: false,
}));

const buildExperimentTrainingRevision = mock(() => "experiment-revision");
const buildModelTrainingRequestHash = mock(() => "request-hash");
const loadCurrentExperimentStatePatch = mock(async () => ({ state_version: 1 }));

mock.module(modelRunStoreModulePath, () => ({
  backfillArtifactSha256ForRun,
  backfillProductionArtifactSha256ForRun,
  buildExperimentTrainingRevision,
  buildModelTrainingRequestHash,
  clearProductionArtifactForRun,
  commitModelTrainingRun,
  getActiveModelTrainingRun,
  loadCurrentExperimentStatePatch,
  setProductionArtifactForRun,
}));

mock.module(modelLocksModulePath, () => ({
  acquireModelExecutionLock,
  acquireModelSlot,
  releaseModelExecutionLock,
  releaseModelSlot,
  verifyModelLockOwned,
  getModelLockStats: () => ({}),
  combineModelLockSignals: (...handles: Array<{ signal?: AbortSignal } | null | undefined>) =>
    handles.find((handle) => handle?.signal)?.signal,
}));

const {
  runPythonModel,
  createOrResumeGuidedTrainingSession,
  getGuidedTrainingSession,
  runGuidedTrainingStep,
  resolveGuidedStepTimeoutMs,
  createOrResumeProductionPreparationSession,
  getProductionPreparationSession,
  runProductionPreparationStep,
  waitForAuxiliaryPythonProcess,
  cleanupExpiredGuidedTrainingSessions,
  cleanupExpiredProductionPreparationSessions,
  cleanupStaleModelArtifacts,
  getModelRuntimeRevision,
  hashDatasetSnapshot,
  loadDatasetFromDb,
  predictExperimentModel,
  ModelAccessError,
  ModelArtifactError,
  ModelNotFoundError,
  ModelTimeoutError,
} = await import(freshModelsServiceModulePath);

const trainModel = createOrResumeGuidedTrainingSession;

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

const buildGuidedSessionRow = (overrides: Record<string, unknown> = {}) => ({
  session_id: "guided-revision-test",
  experiment_id: 990002,
  student_id: 20240002,
  model_type: "ma",
  request_hash: "request-hash",
  dataset_revision: "dataset-revision",
  runtime_revision: getModelRuntimeRevision(),
  protocol_version: 3,
  request_payload: JSON.stringify({
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
  status: "ready",
  current_step_id: "prepare_data",
  completed_step_ids: "[]",
  step_outputs: "{}",
  step_plan: "[]",
  result_payload: null,
  artifact_dir: "/tmp/fangzhen-py/guided_training/guided-revision-test",
  committed_state_path: null,
  committed_state_sha256: null,
  active_attempt_id: null,
  error_message: null,
  superseded_reason: null,
  experiment_state_version: null,
  ...overrides,
});

const buildActiveRun = (
  modelType: string,
  overrides: Record<string, unknown> = {},
) => {
  const runId = `${modelType}-run-current`;
  const productionArtifactPath = typeof overrides.production_artifact_path === "string"
    ? overrides.production_artifact_path
    : null;
  return {
    run_id: runId,
    artifact_path: path.join(
      mockedTrainedModelsRoot,
      `exp990002-student20240002-${modelType}-${runId}-backtest.pkl`,
    ),
    artifact_sha256: mockArtifactSha256(modelType),
    production_artifact_path: productionArtifactPath,
    production_artifact_sha256: productionArtifactPath ? mockArtifactSha256(modelType) : null,
    production_artifact_forecast_steps: productionArtifactPath ? 12 : null,
    production_revision: productionArtifactPath ? preparationToken : null,
    parameters: {},
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

const guidedSessionArtifactRecoveryCases = [
  {
    modelType: "ma",
    requestParams: { moving_average_window: 4 },
  },
  {
    modelType: "es",
    requestParams: { exponential_smoothing_alpha: 0.35 },
  },
  {
    modelType: "arima",
    requestParams: { arimaD: 1 },
  },
  {
    modelType: "lstm",
    requestParams: {
      lstmNormalization: "minmax",
      lstmTargetFeature: "销售数量",
      lstmFeatures: JSON.stringify(["价格指数", "产能利用率"]),
    },
  },
  {
    modelType: "weighted_avg",
    requestParams: {
      models: "ma,es",
      moving_average_window: 4,
      exponential_smoothing_alpha: 0.35,
    },
  },
  {
    modelType: "boosting",
    requestParams: {
      models: "ma,es",
      moving_average_window: 4,
      exponential_smoothing_alpha: 0.35,
    },
  },
  {
    modelType: "stacking",
    requestParams: {
      models: "ma,es",
      moving_average_window: 4,
      exponential_smoothing_alpha: 0.35,
    },
  },
] as const;

const invalidLegacyStdDevCases = [
  { label: "negative", value: -1 },
  { label: "nonnumeric", value: "invalid" },
] as const;

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
    poolGetConnection.mockReset();
    poolGetConnection.mockImplementation(async () => {
      throw new Error("pool.getConnection should not be called in this test");
    });

    getActiveModelTrainingRun.mockReset();
    getActiveModelTrainingRun.mockResolvedValue(null);
    buildExperimentTrainingRevision.mockReset();
    buildExperimentTrainingRevision.mockReturnValue("experiment-revision");
    buildModelTrainingRequestHash.mockReset();
    buildModelTrainingRequestHash.mockReturnValue("request-hash");
    loadCurrentExperimentStatePatch.mockReset();
    loadCurrentExperimentStatePatch.mockResolvedValue({ state_version: 1 });
    setProductionArtifactForRun.mockReset();
    setProductionArtifactForRun.mockResolvedValue(true);
    backfillArtifactSha256ForRun.mockReset();
    backfillArtifactSha256ForRun.mockResolvedValue(true);
    backfillProductionArtifactSha256ForRun.mockReset();
    backfillProductionArtifactSha256ForRun.mockResolvedValue(true);
    clearProductionArtifactForRun.mockReset();
    clearProductionArtifactForRun.mockResolvedValue(true);
    commitModelTrainingRun.mockReset();
    commitModelTrainingRun.mockImplementation(async ({ runId, sessionId }: {
      runId?: string;
      sessionId?: string;
    }) => ({
      runId: runId ?? sessionId ?? "run-test",
      stateVersion: 1,
      reusedExistingRun: false,
    }));
    acquireModelExecutionLock.mockClear();
    acquireModelSlot.mockClear();
    releaseModelExecutionLock.mockClear();
    releaseModelSlot.mockClear();
    verifyModelLockOwned.mockClear();

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
    spawnMock.mockReset();
    spawnMock.mockImplementation(() => createSpawnResult('{"predictions":[]}'));
    (Bun as { spawn: typeof Bun.spawn }).spawn = spawnMock as typeof Bun.spawn;
  });

  test("rejects malformed model dataset rows instead of silently skipping them", async () => {
    const cases = [
      {
        csv: "年份,月份,销售数量\n2024,13,100\n",
        field: "年份/月份",
      },
      {
        csv: "年份,月份,销售数量\n2024,1,Infinity\n",
        field: "销售数量",
      },
    ];

    for (const testCase of cases) {
      await writeFile(datasetCsvPath, testCase.csv);
      poolQuery.mockReset();
      poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);

      await expect(
        loadDatasetFromDb("制造业", "样本企业", "样本产品"),
      ).rejects.toMatchObject({
        details: {
          row: 1,
          field: testCase.field,
          code: "DATASET_ROW_INVALID",
        },
      });
    }
  });

  test("gives each guided ensemble subtask its own bounded timeout", () => {
    const ensembleBody = {
      models: "ma,es,arima,lstm",
    } as never;

    expect(resolveGuidedStepTimeoutMs("ma", "fit", {} as never)).toBe(300_000);
    expect(
      resolveGuidedStepTimeoutMs("weighted_avg", "validation_prediction_lstm", ensembleBody),
    ).toBe(630_000);
    expect(
      resolveGuidedStepTimeoutMs("stacking", "level1_prediction_lstm", ensembleBody),
    ).toBe(630_000);
    expect(
      resolveGuidedStepTimeoutMs("boosting", "retrain_chain", ensembleBody),
    ).toBe(630_000);

  });

  test("classifies auxiliary Python process deadlines as typed model timeouts", async () => {
    let stdoutController!: ReadableStreamDefaultController<Uint8Array>;
    let stderrController!: ReadableStreamDefaultController<Uint8Array>;
    let resolveExit!: (exitCode: number) => void;
    const exited = new Promise<number>((resolve) => {
      resolveExit = resolve;
    });
    let exitCode: number | null = null;
    const kill = mock(() => {
      exitCode = 143;
      stdoutController.close();
      stderrController.close();
      resolveExit(exitCode);
    });
    const proc = {
      stdout: new ReadableStream<Uint8Array>({
        start(controller) { stdoutController = controller; },
      }),
      stderr: new ReadableStream<Uint8Array>({
        start(controller) { stderrController = controller; },
      }),
      exited,
      get exitCode() { return exitCode; },
      pid: 12345,
      kill,
    };

    const error = await waitForAuxiliaryPythonProcess(proc, "复制融合模型产物", {
      timeoutMs: 1,
      terminationGraceMs: 1,
    }).catch((caughtError) => caughtError);

    expect(error).toBeInstanceOf(ModelTimeoutError);
    expect(error).toMatchObject({
      code: "MODEL_TIMEOUT",
      timeoutMs: 1,
      source: "python_process",
    });
    expect(kill).toHaveBeenCalledWith("SIGTERM");
  });

  test("waits for an aborted auxiliary Python process to exit before rejecting", async () => {
    let stdoutController!: ReadableStreamDefaultController<Uint8Array>;
    let stderrController!: ReadableStreamDefaultController<Uint8Array>;
    let resolveExit!: (exitCode: number) => void;
    let exitCode: number | null = null;
    const exited = new Promise<number>((resolve) => {
      resolveExit = resolve;
    });
    const kill = mock(() => {
      stdoutController.error(new Error("stdout pipe closed"));
      stderrController.error(new Error("stderr pipe closed"));
    });
    const proc = {
      stdout: new ReadableStream<Uint8Array>({
        start(controller) { stdoutController = controller; },
      }),
      stderr: new ReadableStream<Uint8Array>({
        start(controller) { stderrController = controller; },
      }),
      exited,
      get exitCode() { return exitCode; },
      pid: 12345,
      kill,
    };
    const controller = new AbortController();
    const lockLost = Object.assign(new Error("数据库模型锁已失效"), {
      code: "MODEL_LOCK_LOST",
    });

    const result = waitForAuxiliaryPythonProcess(proc, "校验融合模型产物", {
      terminationSignal: controller.signal,
      terminationGraceMs: 100,
    });
    controller.abort(lockLost);
    let settled = false;
    void result.finally(() => { settled = true; }).catch(() => {});
    await Bun.sleep(1);
    expect(settled).toBe(false);

    exitCode = 143;
    resolveExit(exitCode);
    await expect(result).rejects.toBe(lockLost);
    expect(kill).toHaveBeenCalledWith("SIGTERM");
  });

  afterEach(async () => {
    jest.useRealTimers();
    (Bun as { spawn: typeof Bun.spawn }).spawn = originalSpawn;
    await unlink(datasetCsvPath).catch(() => {});
  });

  test("terminates an in-flight Python process when the model lock signal is lost", async () => {
    existsSync.mockImplementation((filePath: string) => filePath.endsWith("/run_model.py"));
    let stdoutController!: ReadableStreamDefaultController<Uint8Array>;
    let stderrController!: ReadableStreamDefaultController<Uint8Array>;
    let resolveExit!: (exitCode: number) => void;
    const exited = new Promise<number>((resolve) => {
      resolveExit = resolve;
    });
    const proc = {
      stdout: new ReadableStream<Uint8Array>({
        start(controller) { stdoutController = controller; },
      }),
      stderr: new ReadableStream<Uint8Array>({
        start(controller) { stderrController = controller; },
      }),
      exited,
      exitCode: null as number | null,
      killed: false,
      pid: 999_999_999,
      kill: mock(() => {
        proc.killed = true;
        proc.exitCode = 137;
        stdoutController.close();
        stderrController.close();
        resolveExit(137);
      }),
    };
    spawnMock.mockImplementationOnce(() => proc);
    const lockSignal = new AbortController();
    const lockLostError = Object.assign(new Error("数据库模型锁已失效"), {
      code: "MODEL_LOCK_LOST",
    });

    const execution = runPythonModel("ma", { data: "[]" }, undefined, undefined, lockSignal.signal);
    await Promise.resolve();
    lockSignal.abort(lockLostError);

    await expect(execution).rejects.toBe(lockLostError);
    expect(proc.kill).toHaveBeenCalledWith("SIGTERM");
  });

  test("terminates the Python process before rejecting with a typed timeout", async () => {
    existsSync.mockImplementation((filePath: string) => filePath.endsWith("/run_model.py"));
    let stdoutController!: ReadableStreamDefaultController<Uint8Array>;
    let stderrController!: ReadableStreamDefaultController<Uint8Array>;
    let resolveExit!: (exitCode: number) => void;
    const exited = new Promise<number>((resolve) => {
      resolveExit = resolve;
    });
    const proc = {
      stdout: new ReadableStream<Uint8Array>({
        start(controller) { stdoutController = controller; },
      }),
      stderr: new ReadableStream<Uint8Array>({
        start(controller) { stderrController = controller; },
      }),
      exited,
      exitCode: null as number | null,
      killed: false,
      pid: 999_999_998,
      kill: mock(() => {
        proc.killed = true;
        proc.exitCode = 137;
        stdoutController.close();
        stderrController.close();
        resolveExit(137);
      }),
    };
    let inheritedDeadline: number | null = null;
    spawnMock.mockImplementationOnce((_command, options) => {
      inheritedDeadline = Number((options.env as Record<string, string>).FZ_MODEL_DEADLINE_EPOCH_MS);
      return proc;
    });

    const startedAt = Date.now();
    const error = await runPythonModel("ma", { data: "[]" }, 10).catch(
      (caughtError) => caughtError,
    );

    expect(error).toBeInstanceOf(ModelTimeoutError);
    expect(error).toMatchObject({
      code: "MODEL_TIMEOUT",
      timeoutMs: 10,
      source: "python_process",
    });
    expect(proc.kill).toHaveBeenCalledWith("SIGTERM");
    expect(inheritedDeadline).toBeGreaterThanOrEqual(startedAt + 10);
    expect(proc.exitCode).toBe(137);
  });

  test("maps a nested Python timeout tag to the typed timeout error", async () => {
    existsSync.mockImplementation((filePath: string) => filePath.endsWith("/run_model.py"));
    spawnMock.mockImplementationOnce(() => createSpawnResult("", "[TIMEOUT] Base model lstm timed out", 1));

    const error = await runPythonModel("weighted_avg", { data: "[]" }, 600_000).catch(
      (caughtError) => caughtError,
    );

    expect(error).toBeInstanceOf(ModelTimeoutError);
    expect(error).toMatchObject({
      code: "MODEL_TIMEOUT",
      timeoutMs: 600_000,
      source: "python_subprocess",
    });
  });

  test("maps a Python model artifact tag to the typed artifact error", async () => {
    existsSync.mockReturnValue(true);
    spawnMock.mockImplementationOnce(() =>
      createSpawnResult("", "[MODEL] Failed to load model: invalid pickle", 1)
    );

    const error = await runPythonModel("ma", { data: "[]" }, 5_000)
      .catch((caughtError) => caughtError);

    expect(error).toBeInstanceOf(ModelArtifactError);
    expect(error).toMatchObject({ code: "MODEL_ARTIFACT_INVALID" });
  });

  test("clamps a direct Python runner timeout to the immutable hard cap", async () => {
    existsSync.mockImplementation((filePath: string) => filePath.endsWith("/run_model.py"));
    let inheritedDeadline = 0;
    spawnMock.mockImplementationOnce((_command, options) => {
      inheritedDeadline = Number((options.env as Record<string, string>).FZ_MODEL_DEADLINE_EPOCH_MS);
      return createSpawnResult('{"predictions":[]}', "", 0);
    });

    const startedAt = Date.now();
    await runPythonModel("ma", { data: "[]" }, 900_000);

    expect(inheritedDeadline).toBeGreaterThanOrEqual(startedAt + 600_000);
    expect(inheritedDeadline).toBeLessThanOrEqual(Date.now() + 600_000);
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

  test("does not create a guided session after its operation deadline expires during dataset loading", async () => {
    const originalSetTimeout = globalThis.setTimeout;
    let triggerOperationDeadline: (() => void) | undefined;
    const setTimeoutSpy = jest.spyOn(globalThis, "setTimeout");
    setTimeoutSpy.mockImplementation(((...timerArgs: Parameters<typeof setTimeout>) => {
      const timer = originalSetTimeout(...timerArgs);
      const [callback, timeout, ...callbackArgs] = timerArgs;
      if (timeout === TIMEOUTS.MODEL_SINGLE_OPERATION) {
        triggerOperationDeadline = () => callback(...callbackArgs);
      }
      return timer;
    }) as typeof setTimeout);

    let resolveDatasetQuery!: (value: unknown) => void;
    const datasetQuery = new Promise((resolve) => {
      resolveDatasetQuery = resolve;
    });
    poolQuery.mockImplementation((sql: string) => {
      if (sql.includes("FROM dataset_segments")) return datasetQuery as never;
      return Promise.resolve([[buildExperimentContextRow()]]) as never;
    });
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );

    try {
      const result = trainModel("ma", 20240002, {
        experiment_id: 990002,
        moving_average_window: 4,
      });
      for (let attempt = 0; attempt < 10 && poolQuery.mock.calls.length < 2; attempt += 1) {
        await Promise.resolve();
      }
      expect(poolQuery.mock.calls.length).toBe(2);

      expect(triggerOperationDeadline).toBeDefined();
      triggerOperationDeadline?.();
      resolveDatasetQuery([[{ file_path: datasetCsvPath }]]);

      await expect(result).rejects.toMatchObject({
        code: "MODEL_OPERATION_DEADLINE_EXCEEDED",
        operation: "guided_training_session",
      });
      expect(mkdirSync).not.toHaveBeenCalled();
      expect(poolQuery.mock.calls.some(([sql]) => String(sql).includes("INSERT INTO model_guided_training_sessions")))
        .toBe(false);
    } finally {
      setTimeoutSpy.mockRestore();
    }
  });

  test("keeps a completed guided session current after its own authoritative config commit", async () => {
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

    poolQuery.mockReset();
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
    const completedSession = buildGuidedSessionRow({
      session_id: "guided-own-commit",
      request_hash: "request-hash-before-own-commit",
      dataset_revision: hashDatasetSnapshot(JSON.stringify(currentDataset)),
      runtime_revision: getModelRuntimeRevision(true),
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
    });
    const activeRun = buildActiveRun("ma", {
      session_id: completedSession.session_id,
      request_hash: completedSession.request_hash,
      dataset_revision: completedSession.dataset_revision,
      runtime_revision: completedSession.runtime_revision,
    });
    useActiveRuns(activeRun);
    buildModelTrainingRequestHash.mockReturnValue("request-hash-after-own-commit");
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM model_guided_training_sessions")) return [[completedSession]];
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      return [[buildExperimentContextRow()]];
    });
    existsSync.mockImplementation((filePath: string) => filePath === activeRun.artifact_path);

    const session = await getGuidedTrainingSession(
      "ma",
      20240002,
      completedSession.session_id,
    );

    expect(session.status).toBe("completed");
    expect(poolQuery.mock.calls.some(([sql]) => String(sql).includes("SET status = 'superseded'"))).toBe(false);
    expect(copyFileSync).not.toHaveBeenCalled();
    expect(commitModelTrainingRun).not.toHaveBeenCalled();
  });

  test("keeps legacy artifact validation out of guided session metadata reads", async () => {
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
    poolQuery.mockReset();
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
    const completedSession = buildGuidedSessionRow({
      session_id: "guided-legacy-metadata-read",
      dataset_revision: hashDatasetSnapshot(JSON.stringify(currentDataset)),
      runtime_revision: getModelRuntimeRevision(true),
      request_payload: JSON.stringify(requestBody),
      status: "completed",
      current_step_id: null,
      result_payload: JSON.stringify({
        status: "success",
        results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
      }),
    });
    const activeRun = buildActiveRun("ma", {
      session_id: completedSession.session_id,
      request_hash: completedSession.request_hash,
      dataset_revision: completedSession.dataset_revision,
      runtime_revision: completedSession.runtime_revision,
      artifact_sha256: null,
    });
    useActiveRuns(activeRun);
    let artifactReadCount = 0;
    createReadStream.mockImplementation((filePath: string) => {
      if (filePath === activeRun.artifact_path) artifactReadCount += 1;
      return Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>;
    });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM model_guided_training_sessions")) return [[completedSession]];
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      return [[buildExperimentContextRow()]];
    });
    existsSync.mockReturnValue(true);

    const session = await getGuidedTrainingSession(
      "ma",
      20240002,
      completedSession.session_id,
    );

    expect(session.status).toBe("completed");
    expect(artifactReadCount).toBe(0);
    expect(spawnMock).not.toHaveBeenCalled();
    expect(backfillArtifactSha256ForRun).not.toHaveBeenCalled();
    expect(acquireModelSlot).not.toHaveBeenCalled();
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
  });

  test("refreshes the experiment patch before reusing a completed guided session", async () => {
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

    poolQuery.mockReset();
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
    const historicalPatch = {
      state_version: 3,
      ensemble_weighted_completed: false,
      selected_best_model: null,
      production_plan_completed: false,
    };
    const currentPatch = {
      state_version: 11,
      ensemble_weighted_completed: true,
      selected_best_model: "ensemble_weighted",
      production_plan_completed: true,
    };
    const completedSession = buildGuidedSessionRow({
      session_id: "guided-current-patch",
      dataset_revision: hashDatasetSnapshot(JSON.stringify(currentDataset)),
      runtime_revision: getModelRuntimeRevision(true),
      status: "completed",
      current_step_id: null,
      result_payload: JSON.stringify({
        status: "success",
        results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
        experiment_state_patch: historicalPatch,
      }),
      experiment_state_version: 3,
    });
    const activeRun = buildActiveRun("ma", {
      session_id: completedSession.session_id,
      request_hash: completedSession.request_hash,
      dataset_revision: completedSession.dataset_revision,
      runtime_revision: completedSession.runtime_revision,
    });
    useActiveRuns(activeRun);
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      if (sql.includes("FROM model_guided_training_sessions")) return [[completedSession]];
      return [[buildExperimentContextRow()]];
    });
    existsSync.mockImplementation((filePath: string) => filePath === activeRun.artifact_path);
    loadCurrentExperimentStatePatch.mockResolvedValue(currentPatch);

    const session = await createOrResumeGuidedTrainingSession("ma", 20240002, requestBody);

    expect(session.result?.experiment_state_patch).toEqual(currentPatch);
    expect(session.experiment_state_version).toBe(11);
    expect(loadCurrentExperimentStatePatch).toHaveBeenCalledWith(990002, 20240002);
    expect(commitModelTrainingRun).not.toHaveBeenCalled();
  });

  test("resumes the active completed session when its post-commit request hash no longer matches", async () => {
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
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    poolQuery.mockReset();
    poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
    const row = buildGuidedSessionRow({
      session_id: "guided-post-commit-resume",
      request_hash: "request-hash-before-own-commit",
      status: "completed",
      current_step_id: null,
      completed_step_ids: JSON.stringify(["prepare_data"]),
      step_outputs: JSON.stringify({ prepare_data: { ok: true } }),
      result_payload: JSON.stringify({
        status: "success",
        results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
      }),
      request_payload: JSON.stringify(requestBody),
      dataset_revision: hashDatasetSnapshot(JSON.stringify(currentDataset)),
      runtime_revision: getModelRuntimeRevision(true),
    });
    const activeRun = buildActiveRun("ma", {
      session_id: row.session_id,
      request_hash: row.request_hash,
      dataset_revision: row.dataset_revision,
      runtime_revision: row.runtime_revision,
    });
    useActiveRuns(activeRun);
    buildModelTrainingRequestHash.mockReturnValue("request-hash-after-own-commit");
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      if (sql.includes("FROM model_guided_training_sessions") && sql.includes("request_hash = ?")) return [[]];
      if (sql.includes("FROM model_guided_training_sessions")) return [[row]];
      return [[buildExperimentContextRow()]];
    });
    existsSync.mockImplementation((filePath: string) => filePath === activeRun.artifact_path);

    const session = await createOrResumeGuidedTrainingSession("ma", 20240002, requestBody);

    expect(session.session_id).toBe(row.session_id);
    expect(session.status).toBe("completed");
    expect(poolQuery.mock.calls.some(([sql]) => String(sql).includes("INSERT INTO model_guided_training_sessions"))).toBe(false);
    expect(commitModelTrainingRun).not.toHaveBeenCalled();
  });

  for (const recoveryCase of guidedSessionArtifactRecoveryCases) {
    test(`rebuilds a completed ${recoveryCase.modelType} guided session when its own artifact is corrupt`, async () => {
      const requestBody = {
        experiment_id: 990002,
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
        ...recoveryCase.requestParams,
      };
      createReadStream.mockImplementation(() =>
        Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
      );
      poolQuery.mockReset();
      poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
      const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
      const datasetRevision = hashDatasetSnapshot(JSON.stringify(currentDataset));
      const runtimeRevision = getModelRuntimeRevision(true);
      const oldSessionId = `guided-corrupt-${recoveryCase.modelType}`;
      const oldArtifactDir = path.join(
        mockedGuidedTrainingRoot,
        `exp990002-student20240002-${recoveryCase.modelType}-${oldSessionId}`,
      );
      const completedSession = buildGuidedSessionRow({
        session_id: oldSessionId,
        model_type: recoveryCase.modelType,
        request_hash: "request-hash",
        dataset_revision: datasetRevision,
        runtime_revision: runtimeRevision,
        request_payload: JSON.stringify(requestBody),
        status: "completed",
        current_step_id: null,
        completed_step_ids: JSON.stringify(["save_artifact"]),
        result_payload: JSON.stringify({
          status: "success",
          results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
        }),
        artifact_dir: oldArtifactDir,
      });
      const activeRun = buildActiveRun(recoveryCase.modelType, {
        session_id: oldSessionId,
        request_hash: "request-hash",
        dataset_revision: datasetRevision,
        runtime_revision: runtimeRevision,
        ...(recoveryCase.modelType === "ma"
          || recoveryCase.modelType === "es"
          || recoveryCase.modelType === "arima"
          || recoveryCase.modelType === "lstm"
          ? { artifact_sha256: "0".repeat(64) }
          : {}),
      });
      const dependencyRuns = recoveryCase.modelType === "weighted_avg"
        || recoveryCase.modelType === "boosting"
        || recoveryCase.modelType === "stacking"
        ? [
          buildActiveRun("ma", { dataset_revision: datasetRevision, runtime_revision: runtimeRevision }),
          buildActiveRun("es", { dataset_revision: datasetRevision, runtime_revision: runtimeRevision }),
        ]
        : [];
      useActiveRuns(activeRun, ...dependencyRuns);
      existsSync.mockReturnValue(true);
      if (dependencyRuns.length > 0) {
        spawnMock.mockImplementation(() => createSpawnResult('{"valid":false}'));
      }

      let insertedSession: Record<string, unknown> | null = null;
      let newArtifactDir: string | null = null;
      poolQuery.mockImplementation(async (sql: string, params: unknown[] = []) => {
        if (sql.includes("FROM dataset_segments")) {
          return [[{ file_path: datasetCsvPath }]];
        }
        if (sql.includes("SET status = 'superseded'")) {
          return [{ affectedRows: 1 }];
        }
        if (sql.includes("DELETE FROM model_guided_training_sessions")) {
          return [{ affectedRows: 1 }];
        }
        if (sql.includes("INSERT INTO model_guided_training_sessions")) {
          newArtifactDir = String(params[13]);
          insertedSession = buildGuidedSessionRow({
            session_id: String(params[0]),
            experiment_id: params[1],
            student_id: params[2],
            model_type: params[3],
            request_hash: params[4],
            dataset_revision: params[5],
            runtime_revision: params[6],
            protocol_version: 3,
            request_payload: params[7],
            status: "ready",
            current_step_id: params[8],
            completed_step_ids: params[9],
            step_outputs: params[10],
            step_plan: params[11],
            result_payload: null,
            artifact_dir: newArtifactDir,
          });
          return [{ affectedRows: 1 }];
        }
        if (sql.includes("FROM model_guided_training_sessions")) {
          if (sql.includes("WHERE session_id = ?")) {
            return [insertedSession ? [insertedSession] : []];
          }
          return [[completedSession]];
        }
        return [[buildExperimentContextRow()]];
      });

      try {
        const session = await createOrResumeGuidedTrainingSession(
          recoveryCase.modelType,
          20240002,
          requestBody,
        );

        expect(session).toMatchObject({
          status: "ready",
          result: null,
        });
        expect(session.session_id).not.toBe(oldSessionId);
        expect(session.next_step_id).toBeTruthy();
        expect(rmSync).toHaveBeenCalledWith(oldArtifactDir, { recursive: true, force: true });
        expect(acquireModelSlot).toHaveBeenCalledTimes(1);
        expect(acquireModelExecutionLock).toHaveBeenCalledTimes(1);

        const queries = poolQuery.mock.calls.map(([sql]) => String(sql));
        const supersedeIndex = queries.findIndex((sql) => sql.includes("SET status = 'superseded'"));
        const deleteIndex = queries.findIndex((sql) => sql.includes("DELETE FROM model_guided_training_sessions"));
        const insertIndex = queries.findIndex((sql) => sql.includes("INSERT INTO model_guided_training_sessions"));
        expect(supersedeIndex).toBeGreaterThanOrEqual(0);
        expect(deleteIndex).toBeGreaterThan(supersedeIndex);
        expect(insertIndex).toBeGreaterThan(deleteIndex);
        expect(insertedSession).toMatchObject({
          session_id: session.session_id,
          model_type: recoveryCase.modelType,
          status: "ready",
        });
      } finally {
        if (newArtifactDir) {
          await actualFs.promises.rm(newArtifactDir, { recursive: true, force: true });
        }
      }
    });
  }

  test("supersedes a completed guided session that has no matching active run", async () => {
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    poolQuery.mockReset();
    poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
    let status: "completed" | "superseded" = "completed";
    const completedSessionRow = buildGuidedSessionRow({
      session_id: "guided-missing-run",
      dataset_revision: hashDatasetSnapshot(JSON.stringify(currentDataset)),
      runtime_revision: getModelRuntimeRevision(true),
      status: "completed",
      current_step_id: null,
      completed_step_ids: JSON.stringify(["prepare_data"]),
      step_outputs: JSON.stringify({ prepare_data: { ok: true } }),
      result_payload: JSON.stringify({
        status: "success",
        results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
      }),
    });

    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      if (sql.includes("SET status = 'superseded'")) {
        status = "superseded";
        return [{ affectedRows: 1 }];
      }
      if (sql.includes("FROM model_guided_training_sessions")) {
        return [[{ ...completedSessionRow, status }]];
      }
      return [[buildExperimentContextRow()]];
    });

    const session = await getGuidedTrainingSession(
      "ma",
      20240002,
      completedSessionRow.session_id,
    );

    expect(session.status).toBe("superseded");
    expect(session.error_code).toBe("GUIDED_SESSION_SUPERSEDED");
    expect(copyFileSync).not.toHaveBeenCalled();
    expect(commitModelTrainingRun).not.toHaveBeenCalled();
  });

  test("supersedes a completed ensemble session when a base run dependency changes", async () => {
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    poolQuery.mockReset();
    poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
    const datasetRevision = hashDatasetSnapshot(JSON.stringify(currentDataset));
    const runtimeRevision = getModelRuntimeRevision(true);
    let status: "completed" | "superseded" = "completed";
    const completedSessionRow = buildGuidedSessionRow({
      session_id: "guided-stale-ensemble-dependency",
      model_type: "weighted_avg",
      request_hash: "weighted-request-hash",
      dataset_revision: datasetRevision,
      runtime_revision: runtimeRevision,
      request_payload: JSON.stringify({
        experiment_id: 990002,
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
        models: "ma,es",
      }),
      status,
      current_step_id: null,
      completed_step_ids: JSON.stringify(["prepare_data", "save_artifact"]),
      result_payload: JSON.stringify({
        status: "success",
        results: { metrics: { rmse: 1, mae: 1, r2: 0.9 } },
      }),
    });
    const ensembleRun = buildActiveRun("weighted_avg", {
      session_id: completedSessionRow.session_id,
      request_hash: completedSessionRow.request_hash,
      dataset_revision: datasetRevision,
      runtime_revision: runtimeRevision,
    });
    const currentMaRun = buildActiveRun("ma", { dataset_revision: datasetRevision, runtime_revision: runtimeRevision });
    const currentEsRun = buildActiveRun("es", { dataset_revision: datasetRevision, runtime_revision: runtimeRevision });
    useActiveRuns(ensembleRun, currentMaRun, currentEsRun);
    existsSync.mockReturnValue(true);
    spawnMock.mockImplementationOnce(() => createSpawnResult('{"valid":true}'));

    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      if (sql.includes("FROM model_training_run_dependencies")) {
        return [[
          { base_model_type: "ma", base_run_id: "ma-run-replaced" },
          { base_model_type: "es", base_run_id: currentEsRun.run_id },
        ]];
      }
      if (sql.includes("SET status = 'superseded'")) {
        status = "superseded";
        return [{ affectedRows: 1 }];
      }
      if (sql.includes("FROM model_guided_training_sessions")) {
        return [[{ ...completedSessionRow, status }]];
      }
      return [[buildExperimentContextRow()]];
    });

    const session = await getGuidedTrainingSession(
      "weighted_avg",
      20240002,
      completedSessionRow.session_id,
    );

    expect(session.status).toBe("superseded");
    expect(session.error_code).toBe("GUIDED_SESSION_SUPERSEDED");
    expect(poolQuery.mock.calls.some(([sql]) => String(sql).includes("model_training_run_dependencies"))).toBe(true);
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

  test("cleans expired production preparation sessions without escaping their artifact root", async () => {
    const safeArtifactDir = "/tmp/fangzhen-py/production_preparation/exp990002-student20240002-ma-expired";
    const unsafeArtifactDir = "/tmp/fangzhen-py/trained_models/not-a-production-session";
    const safeCandidatePath = path.join(mockedTrainedModelsRoot, "ma-expired-candidate.pkl");
    const unsafeCandidatePath = path.join(mockedTrainedModelsRoot, "ma-unsafe-candidate.pkl");
    const now = new Date("2026-07-05T00:00:00.000Z");

    poolQuery.mockReset();
    poolQuery
      .mockResolvedValueOnce([[
        {
          session_id: "production-expired-safe",
          artifact_dir: safeArtifactDir,
          status: "failed",
          candidate_model_path: safeCandidatePath,
          candidate_is_active: 0,
        },
        {
          session_id: "production-expired-unsafe",
          artifact_dir: unsafeArtifactDir,
          status: "failed",
          candidate_model_path: unsafeCandidatePath,
          candidate_is_active: 0,
        },
      ]])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([[]]);
    readdirSync.mockReturnValueOnce([]);
    existsSync.mockImplementation((filePath: string) => filePath === safeCandidatePath);

    const result = await cleanupExpiredProductionPreparationSessions({
      now,
      limit: 50,
    });

    expect(result).toEqual({ deletedSessions: 1, deletedArtifactDirs: 1 });
    expect(rmSync).toHaveBeenCalledTimes(1);
    expect(rmSync).toHaveBeenCalledWith(safeArtifactDir, { recursive: true, force: true });
    expect(unlinkSync).toHaveBeenCalledWith(safeCandidatePath);
    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("expires_at <= ?");
    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("attempts.lease_expires_at > ?");
    expect(poolQuery.mock.calls[0]?.[1]).toEqual([now, now]);
    expect(poolQuery.mock.calls[1]?.[1]).toEqual(["production-expired-safe", now, now]);
    expect(poolQuery.mock.calls[3]?.[1]).toEqual(["production-expired-safe"]);
  });

  test("does not delete a production preparation session renewed after cleanup selection", async () => {
    const artifactDir = "/tmp/fangzhen-py/production_preparation/exp990002-student20240002-ma-renewed";
    const now = new Date("2026-07-05T00:00:00.000Z");
    poolQuery.mockReset();
    poolQuery
      .mockResolvedValueOnce([[
        {
          session_id: "production-renewed",
          artifact_dir: artifactDir,
          status: "ready",
          candidate_model_path: path.join(mockedTrainedModelsRoot, "ma-renewed.pkl"),
          candidate_is_active: 0,
        },
      ]])
      .mockResolvedValueOnce([{ affectedRows: 0 }]);
    readdirSync.mockReturnValueOnce([]);

    const result = await cleanupExpiredProductionPreparationSessions({ now, limit: 50 });

    expect(result).toEqual({ deletedSessions: 0, deletedArtifactDirs: 0 });
    expect(rmSync).not.toHaveBeenCalled();
    expect(unlinkSync).not.toHaveBeenCalled();
  });

  test("keeps an expired session candidate when an active run references it", async () => {
    const artifactDir = "/tmp/fangzhen-py/production_preparation/exp990002-student20240002-ma-active";
    const candidatePath = path.join(mockedTrainedModelsRoot, "ma-active-production.pkl");
    const now = new Date("2026-07-05T00:00:00.000Z");
    poolQuery.mockReset();
    poolQuery
      .mockResolvedValueOnce([[
        {
          session_id: "production-active-artifact",
          artifact_dir: artifactDir,
          status: "completed",
          candidate_model_path: candidatePath,
          candidate_is_active: 1,
        },
      ]])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([[]]);
    readdirSync.mockReturnValueOnce([]);
    existsSync.mockReturnValue(true);

    const result = await cleanupExpiredProductionPreparationSessions({ now, limit: 50 });

    expect(result).toEqual({ deletedSessions: 1, deletedArtifactDirs: 1 });
    expect(rmSync).toHaveBeenCalledWith(artifactDir, { recursive: true, force: true });
    expect(unlinkSync).not.toHaveBeenCalledWith(candidatePath);
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

  test("rejects production session creation when an ensemble is incomplete", async () => {
    poolQuery.mockResolvedValueOnce([
      [buildExperimentContextRow({ ensemble_weighted_completed: 0 })],
    ]);

    await expect(
      createOrResumeProductionPreparationSession("weighted_avg", 20240002, 990002),
    ).rejects.toMatchObject({
      message: "加权平均融合模型尚未完成训练，无法准备生产预测模型",
    });

    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("rejects production session creation after the model runtime changes", async () => {
    getActiveModelTrainingRun.mockResolvedValue({
      run_id: "ma-run-stale-runtime",
      artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-stale-runtime.pkl",
      production_artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-stale-runtime-production.pkl",
      production_artifact_forecast_steps: 6,
      production_revision: preparationToken,
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
      createOrResumeProductionPreparationSession("ma", 20240002, 990002, 6),
    ).rejects.toThrow("MA 模型运行环境已变化，请重新训练该模型");
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("rejects production session creation after the dataset changes", async () => {
    getActiveModelTrainingRun.mockResolvedValue({
      run_id: "ma-run-stale-data",
      artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-stale-data.pkl",
      production_artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-stale-data-production.pkl",
      production_artifact_forecast_steps: 6,
      production_revision: preparationToken,
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
      createOrResumeProductionPreparationSession("ma", 20240002, 990002, 6),
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

  test("defers ensemble split feasibility to the dynamic Python planner", async () => {
    useActiveRuns(buildActiveRun("ma"), buildActiveRun("es"));
    const experiment = buildExperimentContextRow({ moving_average_window: 11 });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      if (sql.includes("FROM model_guided_training_sessions")) {
        return [[buildGuidedSessionRow({
          session_id: "guided-dynamic-split",
          model_type: "weighted_avg",
        })]];
      }
      return [[experiment]];
    });
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );

    const session = await trainModel("weighted_avg", 20240002, {
      experiment_id: 990002,
      models: "ma,es",
      moving_average_window: 11,
      exponential_smoothing_alpha: 0.35,
    });

    expect(session.session_id).toBe("guided-dynamic-split");
    expect(createReadStream).toHaveBeenCalled();
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("supersedes a guided session before its next step when dataset contents change", async () => {
    existsSync.mockReturnValue(false);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    const row = buildGuidedSessionRow({
      dataset_revision: "dataset-before-mutation",
      runtime_revision: getModelRuntimeRevision(true),
    });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM model_guided_training_sessions")) return [[row]];
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      if (sql.includes("SET status = 'superseded'")) return [{ affectedRows: 1 }];
      return [[buildExperimentContextRow()]];
    });
    const error = await runGuidedTrainingStep(
      "ma",
      20240002,
      row.session_id,
      "prepare_data",
    ).catch((caughtError) => caughtError as Error & { code?: string });

    expect(error).toMatchObject({ code: "GUIDED_SESSION_SUPERSEDED" });
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("SET status = 'superseded'"),
    )).toBe(true);
    expect(acquireModelExecutionLock).toHaveBeenCalledTimes(1);
    expect(releaseModelExecutionLock).toHaveBeenCalledTimes(1);
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("does not supersede a guided session for a transient identity read failure", async () => {
    const row = buildGuidedSessionRow();
    poolQuery
      .mockResolvedValueOnce([[row]])
      .mockRejectedValueOnce(new Error("database connection reset"));

    await expect(runGuidedTrainingStep(
      "ma",
      20240002,
      row.session_id,
      "prepare_data",
    )).rejects.toThrow("database connection reset");

    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("SET status = 'superseded'"),
    )).toBe(false);
    expect(releaseModelExecutionLock).toHaveBeenCalledTimes(1);
  });

  test("re-reads a guided session under the execution lock before replaying a step", async () => {
    const initialRow = buildGuidedSessionRow();
    const advancedRow = buildGuidedSessionRow({
      current_step_id: "build_windows",
      completed_step_ids: JSON.stringify(["prepare_data"]),
      step_outputs: JSON.stringify({ prepare_data: { prepared_rows: 18 } }),
    });
    let sessionReads = 0;
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM model_guided_training_sessions")) {
        sessionReads += 1;
        return [[sessionReads === 1 ? initialRow : advancedRow]];
      }
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      return [[buildExperimentContextRow()]];
    });

    const result = await runGuidedTrainingStep(
      "ma",
      20240002,
      initialRow.session_id,
      "prepare_data",
    );

    expect(sessionReads).toBe(2);
    expect(result.next_step_id).toBe("build_windows");
    expect(result.steps.find((step) => step.id === "prepare_data")?.status).toBe("completed");
    expect(result.step_outputs.prepare_data).toEqual({ prepared_rows: 18 });
    expect(acquireModelExecutionLock).toHaveBeenCalledTimes(1);
    expect(poolGetConnection).not.toHaveBeenCalled();
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("supersedes a guided session before its next step when runtime revision changes", async () => {
    const previousRuntimeRevision = process.env.MODEL_RUNTIME_REVISION;
    try {
      process.env.MODEL_RUNTIME_REVISION = "runtime-before-mutation";
      const storedRuntimeRevision = getModelRuntimeRevision(true);
      createReadStream.mockImplementation(() =>
        Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
      );
      poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
      const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
      const row = buildGuidedSessionRow({
        dataset_revision: hashDatasetSnapshot(JSON.stringify(currentDataset)),
        runtime_revision: storedRuntimeRevision,
      });
      process.env.MODEL_RUNTIME_REVISION = "runtime-after-mutation";

      poolQuery.mockImplementation(async (sql: string) => {
        if (sql.includes("FROM model_guided_training_sessions")) return [[row]];
        if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
        if (sql.includes("SET status = 'superseded'")) return [{ affectedRows: 1 }];
        return [[buildExperimentContextRow()]];
      });

      const error = await runGuidedTrainingStep(
        "ma",
        20240002,
        row.session_id,
        "prepare_data",
      ).catch((caughtError) => caughtError as Error & { code?: string });

      expect(error).toMatchObject({ code: "GUIDED_SESSION_SUPERSEDED" });
      expect(poolQuery.mock.calls.some(([sql]) =>
        String(sql).includes("SET status = 'superseded'"),
      )).toBe(true);
      expect(spawnMock).not.toHaveBeenCalled();
    } finally {
      if (previousRuntimeRevision === undefined) delete process.env.MODEL_RUNTIME_REVISION;
      else process.env.MODEL_RUNTIME_REVISION = previousRuntimeRevision;
    }
  });

  test("supersedes a completed guided session before materializing a stale runtime artifact", async () => {
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
    const row = buildGuidedSessionRow({
      status: "completed",
      current_step_id: null,
      completed_step_ids: JSON.stringify(["save_artifact"]),
      result_payload: JSON.stringify({ status: "success", results: {} }),
      dataset_revision: hashDatasetSnapshot(JSON.stringify(currentDataset)),
      runtime_revision: "stale-runtime-revision",
    });
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM model_guided_training_sessions")) return [[row]];
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      if (sql.includes("SET status = 'superseded'")) return [{ affectedRows: 1 }];
      return [[buildExperimentContextRow()]];
    });
    existsSync.mockReturnValue(true);

    const error = await runGuidedTrainingStep(
      "ma",
      20240002,
      row.session_id,
      "save_artifact",
    ).catch((caughtError) => caughtError as Error & { code?: string });

    expect(error).toMatchObject({ code: "GUIDED_SESSION_SUPERSEDED" });
    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("AND status <> 'superseded'"),
    )).toBe(true);
    expect(copyFileSync).not.toHaveBeenCalled();
  });

  test("rechecks the dataset revision after Python receipt recovery and before DB commit", async () => {
    const candidatePath = `/tmp/guided-revision-candidate-${process.pid}.pkl`;
    const receiptPath = `/tmp/guided-revision-receipt-${process.pid}.json`;
    const attemptId = "attempt-before-final-commit";
    try {
      createReadStream.mockImplementation(() =>
        Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
      );
      poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
      const originalDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
      await writeFile(candidatePath, "candidate-checkpoint");
      const candidateSha256 = crypto
        .createHash("sha256")
        .update("candidate-checkpoint")
        .digest("hex");
      await writeFile(receiptPath, JSON.stringify({
        protocol_version: 3,
        attempt_id: attemptId,
        step_id: "prepare_data",
        candidate_state_path: candidatePath,
        candidate_state_sha256: candidateSha256,
        input_state_sha256: null,
        output: { prepared_rows: 18 },
      }));

      const row = buildGuidedSessionRow({
        dataset_revision: hashDatasetSnapshot(JSON.stringify(originalDataset)),
        runtime_revision: getModelRuntimeRevision(true),
        active_attempt_id: attemptId,
      });
      const attempt = {
        attempt_id: attemptId,
        session_id: row.session_id,
        step_id: "prepare_data",
        status: "running",
        input_state_path: null,
        input_state_sha256: null,
        candidate_state_path: candidatePath,
        receipt_path: receiptPath,
        lease_expires_at: new Date(Date.now() + 60_000),
        error_message: null,
      };
      const mutatedDatasetCsv = modelDatasetCsv.replace("2024-01,100,1.2,0.8", "2024-01,999,1.2,0.8");
      let datasetMutated = false;
      const connection = {
        beginTransaction: mock(async () => {
          await writeFile(datasetCsvPath, mutatedDatasetCsv);
          datasetMutated = true;
        }),
        commit: mock(async () => undefined),
        rollback: mock(async () => undefined),
        release: mock(() => undefined),
        query: mock(async (sql: string) => {
          if (sql.includes("FROM model_guided_training_sessions")) return [[row]];
          if (sql.includes("FROM model_guided_training_step_attempts")) return [[attempt]];
          if (sql.includes("UPDATE model_guided_training_sessions")) return [{ affectedRows: 1 }];
          return [[]];
        }),
      };
      poolGetConnection.mockResolvedValue(connection as never);

      existsSync.mockImplementation((filePath: string) =>
        filePath === candidatePath || filePath === receiptPath,
      );
      poolQuery.mockImplementation(async (sql: string) => {
        if (sql.includes("FROM model_guided_training_sessions")) return [[row]];
        if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
        if (sql.includes("SET status = 'superseded'")) return [{ affectedRows: 1 }];
        if (sql.includes("UPDATE model_guided_training_step_attempts")) return [{ affectedRows: 1 }];
        return [[buildExperimentContextRow()]];
      });

      const error = await runGuidedTrainingStep(
        "ma",
        20240002,
        row.session_id,
        "prepare_data",
      ).catch((caughtError) => caughtError as Error & { code?: string });

      expect(error).toMatchObject({ code: "GUIDED_SESSION_SUPERSEDED" });
      expect(datasetMutated).toBe(true);
      expect(commitModelTrainingRun).not.toHaveBeenCalled();
      expect(acquireModelSlot).toHaveBeenCalledTimes(1);
      expect(spawnMock).not.toHaveBeenCalled();
      const failureSql = connection.query.mock.calls.map(([sql]) => String(sql));
      const attemptFailureSql = failureSql.find((sql) =>
        sql.includes("SET status = IF(status = 'python_completed', status, 'failed')"));
      const sessionFailureSql = failureSql.find((sql) =>
        sql.includes("SET status = 'failed'") && sql.includes("active_attempt_id"));
      expect(attemptFailureSql).toContain("status IN ('running', 'python_completed')");
      expect(sessionFailureSql).toContain("active_attempt_id = ?");
      expect(sessionFailureSql).toContain("status = 'running'");
      expect(sessionFailureSql).not.toContain("active_attempt_id IS NULL");

      // Re-run the same protected failure with pool acquisition unavailable for
      // the best-effort status write. The original model error must still win.
      await writeFile(datasetCsvPath, modelDatasetCsv);
      datasetMutated = false;
      connection.query.mockClear();
      poolGetConnection.mockReset();
      const failureRecordPoolError = new Error("failure record pool unavailable");
      poolGetConnection
        .mockResolvedValueOnce(connection as never)
        .mockRejectedValueOnce(failureRecordPoolError);

      const originalError = await runGuidedTrainingStep(
        "ma",
        20240002,
        row.session_id,
        "prepare_data",
      ).catch((caughtError) => caughtError as Error & { code?: string });

      expect(originalError).toMatchObject({ code: "GUIDED_SESSION_SUPERSEDED" });
      expect(originalError).not.toBe(failureRecordPoolError);
      expect(datasetMutated).toBe(true);
      expect(poolGetConnection).toHaveBeenCalledTimes(2);
    } finally {
      await unlink(candidatePath).catch(() => {});
      await unlink(receiptPath).catch(() => {});
    }
  });

  test("commits an existing Python receipt without spawning the step again", async () => {
    const candidatePath = `/tmp/guided-replay-candidate-${process.pid}.pkl`;
    const receiptPath = `/tmp/guided-replay-receipt-${process.pid}.json`;
    const attemptId = "attempt-after-db-response-loss";
    try {
      poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
      const originalDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
      await writeFile(candidatePath, "replayable-checkpoint");
      const candidateSha256 = crypto
        .createHash("sha256")
        .update("replayable-checkpoint")
        .digest("hex");
      const receiptOutput = { prepared_rows: 18 };
      await writeFile(receiptPath, JSON.stringify({
        protocol_version: 3,
        attempt_id: attemptId,
        step_id: "prepare_data",
        candidate_state_path: candidatePath,
        candidate_state_sha256: candidateSha256,
        input_state_sha256: null,
        output: receiptOutput,
      }));
      const row = buildGuidedSessionRow({
        dataset_revision: hashDatasetSnapshot(JSON.stringify(originalDataset)),
        runtime_revision: getModelRuntimeRevision(true),
        active_attempt_id: attemptId,
      });
      const attempt = {
        attempt_id: attemptId,
        session_id: row.session_id,
        step_id: "prepare_data",
        status: "python_completed",
        input_state_path: null,
        input_state_sha256: null,
        candidate_state_path: candidatePath,
        receipt_path: receiptPath,
        lease_expires_at: new Date(Date.now() - 1_000),
        error_message: null,
      };
      const beginConnection = {
        beginTransaction: mock(async () => undefined),
        commit: mock(async () => undefined),
        rollback: mock(async () => undefined),
        release: mock(() => undefined),
        query: mock(async (sql: string) => {
          if (sql.includes("FROM model_guided_training_sessions")) return [[row]];
          if (sql.includes("FROM model_guided_training_step_attempts")) return [[attempt]];
          return [{ affectedRows: 1 }];
        }),
      };
      let committed = false;
      const commitConnection = {
        beginTransaction: mock(async () => undefined),
        commit: mock(async () => undefined),
        rollback: mock(async () => undefined),
        release: mock(() => undefined),
        query: mock(async (sql: string) => {
          if (sql.includes("SELECT active_attempt_id")) {
            return [[{ active_attempt_id: attemptId, status: "running" }]];
          }
          if (sql.includes("UPDATE model_guided_training_sessions")) committed = true;
          return [{ affectedRows: 1 }];
        }),
      };
      poolGetConnection
        .mockResolvedValueOnce(beginConnection as never)
        .mockResolvedValueOnce(commitConnection as never);
      existsSync.mockImplementation((filePath: string) =>
        filePath === candidatePath || filePath === receiptPath,
      );
      poolQuery.mockImplementation(async (sql: string) => {
        if (sql.includes("FROM model_guided_training_sessions")) {
          return [[committed
            ? {
              ...row,
              active_attempt_id: null,
              current_step_id: "build_windows",
              completed_step_ids: JSON.stringify(["prepare_data"]),
              step_outputs: JSON.stringify({ prepare_data: receiptOutput }),
              committed_state_path: candidatePath,
              committed_state_sha256: candidateSha256,
            }
            : row]];
        }
        if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
        if (sql.trimStart().startsWith("UPDATE")) return [{ affectedRows: 1 }];
        return [[buildExperimentContextRow()]];
      });

      const result = await runGuidedTrainingStep("ma", 20240002, row.session_id, "prepare_data");

      expect(result.status).toBe("ready");
      expect(result.next_step_id).toBe("build_windows");
      expect(result.step_outputs.prepare_data).toEqual(receiptOutput);
      expect(poolGetConnection).toHaveBeenCalledTimes(2);
      expect(commitConnection.commit).toHaveBeenCalledTimes(1);
      expect(acquireModelSlot).toHaveBeenCalledTimes(1);
      expect(spawnMock).not.toHaveBeenCalled();
    } finally {
      await unlink(candidatePath).catch(() => {});
      await unlink(receiptPath).catch(() => {});
    }
  });

  test("resumes production preparation metadata without spawning model work", async () => {
    const activeRun = buildActiveRun("weighted_avg");
    const sessionRow = {
      session_id: "production-existing",
      experiment_id: 990002,
      student_id: 20240002,
      run_id: activeRun.run_id,
      model_type: "weighted_avg",
      request_hash: "production-request-hash",
      dataset_revision: "dataset-revision",
      runtime_revision: getModelRuntimeRevision(),
      protocol_version: 1,
      forecast_steps: 6,
      status: "ready",
      current_step_id: "initialize",
      completed_step_ids: JSON.stringify([]),
      step_plan: JSON.stringify([
        { id: "initialize", label: "确认生产准备材料" },
        { id: "finalize", label: "组装并校验生产产物" },
      ]),
      artifact_dir: "/tmp/fangzhen-py/production_preparation/production-existing",
      candidate_model_path: path.join(mockedTrainedModelsRoot, "weighted-production.pkl"),
      committed_state_path: null,
      committed_state_sha256: null,
      active_attempt_id: null,
      production_revision: null,
      error_message: null,
      superseded_reason: null,
    };
    useActiveRuns(activeRun);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      if (sql.includes("FROM model_production_preparation_sessions")) return [[sessionRow]];
      return [[buildExperimentContextRow()]];
    });

    const result = await createOrResumeProductionPreparationSession(
      "weighted_avg",
      20240002,
      990002,
      6,
    );

    expect(result.results).toMatchObject({
      session_id: "production-existing",
      session_status: "ready",
      next_step: { id: "initialize" },
    });
    expect(acquireModelSlot).not.toHaveBeenCalled();
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("reads production preparation progress as metadata only", async () => {
    const sessionRow = {
      session_id: "production-read",
      experiment_id: 990002,
      student_id: 20240002,
      run_id: "ma-run-current",
      model_type: "ma",
      request_hash: "production-request-hash",
      dataset_revision: "dataset-revision",
      runtime_revision: getModelRuntimeRevision(),
      protocol_version: 1,
      forecast_steps: 6,
      status: "ready",
      current_step_id: "fit_model",
      completed_step_ids: JSON.stringify(["initialize"]),
      step_plan: JSON.stringify([
        { id: "initialize", label: "确认生产准备材料" },
        { id: "fit_model", label: "重拟合生产模型" },
        { id: "finalize", label: "组装并校验生产产物" },
      ]),
      artifact_dir: "/tmp/fangzhen-py/production_preparation/production-read",
      candidate_model_path: path.join(mockedTrainedModelsRoot, "ma-production.pkl"),
      committed_state_path: "/tmp/fangzhen-py/production_preparation/production-read/state.pkl",
      committed_state_sha256: "a".repeat(64),
      active_attempt_id: null,
      production_revision: null,
      error_message: null,
      superseded_reason: null,
    };
    poolQuery
      .mockResolvedValueOnce([[sessionRow]])
      .mockResolvedValueOnce([[buildExperimentContextRow()]]);

    const result = await getProductionPreparationSession("ma", 20240002, sessionRow.session_id);

    expect(result.results).toEqual({
      session_id: "production-read",
      session_status: "ready",
      completed_step_ids: ["initialize"],
      total_steps: 3,
      next_step: { id: "fit_model", label: "重拟合生产模型" },
      error_message: null,
    });
    expect(acquireModelSlot).not.toHaveBeenCalled();
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("rejects out-of-order production steps before acquiring execution capacity", async () => {
    poolQuery.mockResolvedValueOnce([[
      {
        session_id: "production-order",
        experiment_id: 990002,
        student_id: 20240002,
        run_id: "ma-run-current",
        model_type: "ma",
        request_hash: "production-request-hash",
        dataset_revision: "dataset-revision",
        runtime_revision: getModelRuntimeRevision(),
        protocol_version: 1,
        forecast_steps: 6,
        status: "ready",
        current_step_id: "initialize",
        completed_step_ids: JSON.stringify([]),
        step_plan: JSON.stringify([
          { id: "initialize", label: "确认生产准备材料" },
          { id: "finalize", label: "组装并校验生产产物" },
        ]),
        artifact_dir: "/tmp/fangzhen-py/production_preparation/production-order",
        candidate_model_path: path.join(mockedTrainedModelsRoot, "ma-production.pkl"),
        committed_state_path: null,
        committed_state_sha256: null,
        active_attempt_id: null,
        production_revision: null,
        error_message: null,
        superseded_reason: null,
      },
    ]]);

    await expect(
      runProductionPreparationStep(
        "ma",
        20240002,
        "production-order",
        "finalize",
      ),
    ).rejects.toMatchObject({
      code: "PRODUCTION_PREPARATION_STEP_CONFLICT",
    });
    expect(acquireModelSlot).not.toHaveBeenCalled();
    expect(acquireModelExecutionLock).not.toHaveBeenCalled();
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("supersedes production preparation before a step when the current dataset changed", async () => {
    const staleDatasetRevision = "d".repeat(64);
    const activeRun = buildActiveRun("ma", {
      dataset_revision: staleDatasetRevision,
      runtime_revision: getModelRuntimeRevision(true),
    });
    const sessionRow = {
      session_id: "production-stale-dataset",
      experiment_id: 990002,
      student_id: 20240002,
      run_id: activeRun.run_id,
      model_type: "ma",
      request_hash: "stale-production-request",
      dataset_revision: staleDatasetRevision,
      runtime_revision: activeRun.runtime_revision,
      protocol_version: 1,
      forecast_steps: 6,
      status: "ready",
      current_step_id: "initialize",
      completed_step_ids: JSON.stringify([]),
      step_plan: JSON.stringify([
        { id: "initialize", label: "确认生产准备材料" },
        { id: "finalize", label: "组装并校验生产产物" },
      ]),
      artifact_dir: "/tmp/fangzhen-py/production_preparation/production-stale-dataset",
      candidate_model_path: path.join(mockedTrainedModelsRoot, "ma-stale-candidate.pkl"),
      committed_state_path: null,
      committed_state_sha256: null,
      active_attempt_id: null,
      observed_production_revision: null,
      production_revision: null,
      error_message: null,
      superseded_reason: null,
    };
    useActiveRuns(activeRun);
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM model_production_preparation_sessions")) return [[sessionRow]];
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      if (sql.includes("SET expires_at = CASE")) return [{ affectedRows: 1 }];
      if (sql.includes("SET status = 'superseded'")) return [{ affectedRows: 1 }];
      return [[buildExperimentContextRow()]];
    });

    await expect(runProductionPreparationStep(
      "ma",
      20240002,
      sessionRow.session_id,
      "initialize",
    )).rejects.toMatchObject({ code: "PRODUCTION_PREPARATION_STALE" });

    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("SET status = 'superseded'"),
    )).toBe(true);
    expect(setProductionArtifactForRun).not.toHaveBeenCalled();
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("reuses a longer production horizon before an older shorter session can finalize", async () => {
    poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    createReadStream.mockImplementationOnce(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
    const datasetRevision = hashDatasetSnapshot(JSON.stringify(currentDataset));
    const runtimeRevision = getModelRuntimeRevision(true);
    const productionPath = path.join(mockedTrainedModelsRoot, "ma-twelve-period-production.pkl");
    const activeRun = buildActiveRun("ma", {
      dataset_revision: datasetRevision,
      runtime_revision: runtimeRevision,
      production_artifact_path: productionPath,
      production_artifact_forecast_steps: 12,
      production_revision: preparationToken,
    });
    const requestHash = crypto.createHash("sha256").update(JSON.stringify({
      runId: activeRun.run_id,
      datasetRevision,
      runtimeRevision,
      forecastSteps: 6,
    })).digest("hex");
    const sessionRow = {
      session_id: "production-shorter-finalize",
      experiment_id: 990002,
      student_id: 20240002,
      run_id: activeRun.run_id,
      model_type: "ma",
      request_hash: requestHash,
      dataset_revision: datasetRevision,
      runtime_revision: runtimeRevision,
      protocol_version: 1,
      forecast_steps: 6,
      status: "ready",
      current_step_id: "finalize",
      completed_step_ids: JSON.stringify(["initialize", "fit_model"]),
      step_plan: JSON.stringify([
        { id: "initialize", label: "确认生产准备材料" },
        { id: "fit_model", label: "重拟合生产模型" },
        { id: "finalize", label: "组装并校验生产产物" },
      ]),
      artifact_dir: "/tmp/fangzhen-py/production_preparation/production-shorter-finalize",
      candidate_model_path: path.join(mockedTrainedModelsRoot, "ma-six-period-candidate.pkl"),
      committed_state_path: null,
      committed_state_sha256: null,
      active_attempt_id: null,
      observed_production_revision: null,
      production_revision: null,
      error_message: null,
      superseded_reason: null,
    };
    const completedRow = {
      ...sessionRow,
      status: "completed",
      current_step_id: null,
      completed_step_ids: JSON.stringify(["initialize", "fit_model", "finalize"]),
      observed_production_revision: preparationToken,
      production_revision: preparationToken,
    };
    useActiveRuns(activeRun);
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    let sessionReads = 0;
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM model_production_preparation_sessions")) {
        sessionReads += 1;
        return [[sessionReads <= 2 ? sessionRow : completedRow]];
      }
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      if (sql.trimStart().startsWith("UPDATE")) return [{ affectedRows: 1 }];
      return [[buildExperimentContextRow()]];
    });

    const result = await runProductionPreparationStep(
      "ma",
      20240002,
      sessionRow.session_id,
      "finalize",
    );

    expect(result.results).toMatchObject({
      session_status: "completed",
      preparation_token: preparationToken,
      prepared_forecast_steps: 6,
      reused_existing: true,
    });
    expect(setProductionArtifactForRun).not.toHaveBeenCalled();
    expect(spawnMock).not.toHaveBeenCalled();
    expect(unlinkSync).toHaveBeenCalledWith(sessionRow.candidate_model_path);
  });

  test("rejects prediction when the experiment belongs to another student", async () => {
    poolQuery.mockResolvedValueOnce([
      [buildExperimentContextRow({ student_id: 99999999 })],
    ]);

    const predictionPromise = predictExperimentModel("ma", 20240002, 990002, 6, preparationToken);

    await expect(predictionPromise).rejects.toBeInstanceOf(ModelAccessError);
    await expect(predictionPromise).rejects.toMatchObject({
      message: "权限不足：该实验不属于当前用户",
      status: 403,
    });
  });

  test("rejects prediction when the experiment does not exist", async () => {
    poolQuery.mockResolvedValueOnce([[]]);

    await expect(
      predictExperimentModel("ma", 20240002, 990002, 6, preparationToken),
    ).rejects.toMatchObject({
      message: "实验不存在",
      status: 404,
    });
  });

  test("rejects prediction when the prepared artifact is missing", async () => {
    const activeRun = buildActiveRun("ma", {
      production_artifact_path: path.join(mockedTrainedModelsRoot, "ma-missing-production.pkl"),
    });
    useActiveRuns(activeRun);

    await expect(
      predictExperimentModel("ma", 20240002, 990002, 6, preparationToken),
    ).rejects.toMatchObject({ code: "PRODUCTION_PREPARATION_REQUIRED" });

    expect(existsSync).toHaveBeenCalledWith(
      activeRun.production_artifact_path,
    );
  });

  test("invalidates a corrupt production artifact before prediction", async () => {
    const productionPath = path.join(mockedTrainedModelsRoot, "ma-corrupt-production.pkl");
    const activeRun = buildActiveRun("ma", {
      production_artifact_path: productionPath,
      production_artifact_sha256: "f".repeat(64),
    });
    useActiveRuns(activeRun);
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );

    await expect(
      predictExperimentModel("ma", 20240002, 990002, 6, preparationToken),
    ).rejects.toMatchObject({
      code: "PRODUCTION_PREPARATION_REQUIRED",
      details: { required_action: "prepare" },
    });
    expect(clearProductionArtifactForRun).toHaveBeenCalledWith(activeRun.run_id, preparationToken);
    expect(unlinkSync).toHaveBeenCalledWith(productionPath);
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("treats a changed LSTM sidecar as production artifact corruption", async () => {
    const productionPath = path.join(mockedTrainedModelsRoot, "lstm-current-production.pkl");
    const activeRun = buildActiveRun("lstm", {
      production_artifact_path: productionPath,
    });
    useActiveRuns(activeRun);
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation((filePath: string) =>
      Readable.from([
        filePath.endsWith("_keras_model.h5") ? "tampered-keras-sidecar" : modelDatasetCsv,
      ]) as unknown as ReturnType<typeof createReadStream>
    );

    await expect(
      predictExperimentModel("lstm", 20240002, 990002, 3, preparationToken),
    ).rejects.toMatchObject({ code: "PRODUCTION_PREPARATION_REQUIRED" });

    expect(createReadStream).toHaveBeenCalledWith(
      productionPath.replace(/\.pkl$/, "_keras_model.h5"),
    );
    expect(clearProductionArtifactForRun).toHaveBeenCalledWith(activeRun.run_id, preparationToken);
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("rejects prediction when the model is not ready for production", async () => {
    poolQuery.mockResolvedValueOnce([
      [buildExperimentContextRow({ moving_average_completed: 0 })],
    ]);

    await expect(
      predictExperimentModel("ma", 20240002, 990002, 6, preparationToken),
    ).rejects.toMatchObject({
      message: "移动平均基础模型尚未完成训练，无法准备生产预测模型",
    });

    expect(spawnMock).not.toHaveBeenCalled();
    expect(createReadStream).not.toHaveBeenCalled();
  });

  test("requires explicit production preparation before prediction", async () => {
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

    await expect(
      predictExperimentModel("ma", 20240002, 990002, 6, preparationToken),
    ).rejects.toMatchObject({ code: "PRODUCTION_PREPARATION_STALE" });
    expect(spawnMock).not.toHaveBeenCalled();
    expect(setProductionArtifactForRun).not.toHaveBeenCalled();
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
          controller.enqueue(new TextEncoder().encode(
            '{"predictions":[{"prediction":101.2,"std_dev":1.1,"upper_error_p99":2.1,"uncertainty_source":"model","calibration_mean_error":1.0,"calibration_count":8},{"prediction":98.4,"std_dev":1.2,"upper_error_p99":2.2,"uncertainty_source":"model","calibration_mean_error":1.0,"calibration_count":8},{"prediction":99.1,"std_dev":1.3,"upper_error_p99":2.3,"uncertainty_source":"model","calibration_mean_error":1.0,"calibration_count":8},{"prediction":100.3,"std_dev":1.4,"upper_error_p99":2.4,"uncertainty_source":"model","calibration_mean_error":1.0,"calibration_count":8},{"prediction":102.0,"std_dev":1.5,"upper_error_p99":2.5,"uncertainty_source":"model","calibration_mean_error":1.0,"calibration_count":8},{"prediction":103.2,"std_dev":1.6,"upper_error_p99":2.6,"uncertainty_source":"model","calibration_mean_error":1.0,"calibration_count":8}]}',
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

    const result = await predictExperimentModel("ma", 20240002, 990002, 6, preparationToken);

    expect(result).toEqual({
      status: "success",
      results: {
        predictions: [
          { prediction: 101.2, std_dev: 1.1, upper_error_p99: 2.1, uncertainty_source: "model", calibration_mean_error: 1, calibration_count: 8 },
          { prediction: 98.4, std_dev: 1.2, upper_error_p99: 2.2, uncertainty_source: "model", calibration_mean_error: 1, calibration_count: 8 },
          { prediction: 99.1, std_dev: 1.3, upper_error_p99: 2.3, uncertainty_source: "model", calibration_mean_error: 1, calibration_count: 8 },
          { prediction: 100.3, std_dev: 1.4, upper_error_p99: 2.4, uncertainty_source: "model", calibration_mean_error: 1, calibration_count: 8 },
          { prediction: 102, std_dev: 1.5, upper_error_p99: 2.5, uncertainty_source: "model", calibration_mean_error: 1, calibration_count: 8 },
          { prediction: 103.2, std_dev: 1.6, upper_error_p99: 2.6, uncertainty_source: "model", calibration_mean_error: 1, calibration_count: 8 },
        ],
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

  for (const invalidStdDev of invalidLegacyStdDevCases) {
    test(`invalidates checksummed production output with ${invalidStdDev.label} std_dev`, async () => {
      const productionPath = path.join(
        mockedTrainedModelsRoot,
        `ma-invalid-${invalidStdDev.label}-std-dev-production.pkl`,
      );
      const activeRun = buildActiveRun("ma", { production_artifact_path: productionPath });
      useActiveRuns(activeRun);
      mockExperimentAndDatasetQueries();
      existsSync.mockReturnValue(true);
      createReadStream.mockImplementation(() =>
        Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
      );
      spawnMock.mockImplementationOnce(() => createSpawnResult(JSON.stringify({
        predictions: [{ prediction: 101.2, std_dev: invalidStdDev.value }],
      })));

      await expect(
        predictExperimentModel("ma", 20240002, 990002, 1, preparationToken),
      ).rejects.toMatchObject({
        code: "PRODUCTION_PREPARATION_REQUIRED",
        details: { required_action: "prepare" },
      });
      expect(clearProductionArtifactForRun).toHaveBeenCalledWith(
        activeRun.run_id,
        preparationToken,
      );
      expect(unlinkSync).toHaveBeenCalledWith(productionPath);
    });
  }

  test("invalidates production output with a partial bias diagnostic", async () => {
    const productionPath = path.join(mockedTrainedModelsRoot, "ma-partial-bias-production.pkl");
    const activeRun = buildActiveRun("ma", { production_artifact_path: productionPath });
    useActiveRuns(activeRun);
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    spawnMock.mockImplementationOnce(() => createSpawnResult(JSON.stringify({
      predictions: [{
        prediction: 101.2,
        std_dev: 1.1,
        upper_error_p99: 2.1,
        uncertainty_source: "empirical",
        calibration_mean_error: 1.5,
        calibration_count: null,
      }],
    })));

    await expect(
      predictExperimentModel("ma", 20240002, 990002, 1, preparationToken),
    ).rejects.toMatchObject({
      code: "PRODUCTION_PREPARATION_REQUIRED",
      details: { required_action: "prepare" },
    });
    expect(clearProductionArtifactForRun).toHaveBeenCalledWith(
      activeRun.run_id,
      preparationToken,
    );
    expect(unlinkSync).toHaveBeenCalledWith(productionPath);
  });

  test("invalidates a production artifact that changes during prediction", async () => {
    const productionPath = path.join(mockedTrainedModelsRoot, "ma-changing-production.pkl");
    const activeRun = buildActiveRun("ma", { production_artifact_path: productionPath });
    useActiveRuns(activeRun);
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
    let artifactRead = 0;
    createReadStream.mockImplementation(() => Readable.from([
      artifactRead++ === 0 ? modelDatasetCsv : "mutated-production-artifact",
    ]) as unknown as ReturnType<typeof createReadStream>);
    spawnMock.mockImplementationOnce(() => createSpawnResult(
      '{"predictions":[{"prediction":101.2},{"prediction":98.4}]}',
    ));

    await expect(
      predictExperimentModel("ma", 20240002, 990002, 2, preparationToken),
    ).rejects.toMatchObject({
      code: "PRODUCTION_PREPARATION_REQUIRED",
      details: { required_action: "prepare" },
    });
    expect(clearProductionArtifactForRun).toHaveBeenCalledWith(activeRun.run_id, preparationToken);
    expect(unlinkSync).toHaveBeenCalledWith(productionPath);
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
            '{"predictions":[{"prediction":-4.2,"std_dev":1.5,"upper_error_p99":3.5,"uncertainty_source":"model","calibration_mean_error":null,"calibration_count":null},{"prediction":3.1,"std_dev":0.8,"upper_error_p99":1.8,"uncertainty_source":"model","calibration_mean_error":null,"calibration_count":null}]}',
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

    const result = await predictExperimentModel("ma", 20240002, 990002, 2, preparationToken);

    expect(result.results.predictions).toEqual([
      { prediction: 0, std_dev: 1.5, upper_error_p99: 3.5, uncertainty_source: "model", calibration_mean_error: null, calibration_count: null },
      { prediction: 3.1, std_dev: 0.8, upper_error_p99: 1.8, uncertainty_source: "model", calibration_mean_error: null, calibration_count: null },
    ]);
  });

  test("requires a new preparation when the cached production horizon is too short", async () => {
    const previousProductionPath = path.join(mockedTrainedModelsRoot, "lstm-current-production.pkl");
    const activeRun = buildActiveRun("lstm", {
      production_artifact_path: previousProductionPath,
      production_artifact_forecast_steps: 6,
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
              '{"predictions":[{"prediction":12.5,"std_dev":1.2,"upper_error_p99":2.8,"uncertainty_source":"model","calibration_mean_error":null,"calibration_count":null}]}',
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

    await expect(
      predictExperimentModel("lstm", 20240002, 990002, 8, preparationToken),
    ).rejects.toMatchObject({ code: "PRODUCTION_PREPARATION_STALE" });
    expect(spawnMock).not.toHaveBeenCalled();
    expect(setProductionArtifactForRun).not.toHaveBeenCalled();
  });

  test("does not mutate the current artifact when preparation capacity is insufficient", async () => {
    const previousProductionPath = path.join(mockedTrainedModelsRoot, "lstm-current-production.pkl");
    const activeRun = buildActiveRun("lstm", {
      production_artifact_path: previousProductionPath,
      production_artifact_forecast_steps: 6,
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
      predictExperimentModel("lstm", 20240002, 990002, 8, preparationToken),
    ).rejects.toMatchObject({ code: "PRODUCTION_PREPARATION_STALE" });

    expect(spawnMock).not.toHaveBeenCalled();
    expect(unlinkSync).not.toHaveBeenCalledWith(previousProductionPath);
    expect(setProductionArtifactForRun).not.toHaveBeenCalled();
  });

  test("rejects a prediction result when its active model version changes in flight", async () => {
    const productionPath = "/tmp/fangzhen-py/trained_models/ma-run-a-production.pkl";
    const runA = {
      run_id: "ma-run-a",
      artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-a.pkl",
      production_artifact_path: productionPath,
      production_artifact_sha256: mockArtifactSha256("ma"),
      production_artifact_forecast_steps: 6,
      production_revision: preparationToken,
      model_type: "ma",
    };
    const runB = {
      run_id: "ma-run-b",
      artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-b.pkl",
      production_artifact_path: "/tmp/fangzhen-py/trained_models/ma-run-b-production.pkl",
      production_artifact_sha256: mockArtifactSha256("ma"),
      production_artifact_forecast_steps: 6,
      production_revision: "223e4567-e89b-42d3-a456-426614174000",
      model_type: "ma",
    };
    getActiveModelTrainingRun
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
      predictExperimentModel("ma", 20240002, 990002, 6, preparationToken),
    ).rejects.toMatchObject({
      message: "预测期间模型版本已变化，请重新准备生产模型",
    });
  });

  test("verifies available base model artifacts when creating an ensemble guided session", async () => {
    const maRun = buildActiveRun("ma");
    const lstmRun = buildActiveRun("lstm");
    useActiveRuns(maRun, lstmRun);
    const experiment = buildExperimentContextRow();
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      if (sql.includes("FROM model_guided_training_sessions")) {
        return [[{
          session_id: "guided-ensemble-existing",
          experiment_id: 990002,
          student_id: 20240002,
          model_type: "weighted_avg",
          request_hash: "request-hash",
          dataset_revision: "dataset-revision",
          runtime_revision: getModelRuntimeRevision(),
          protocol_version: 3,
          request_payload: "{}",
          status: "ready",
          current_step_id: "prepare_data",
          completed_step_ids: "[]",
          step_outputs: "{}",
          step_plan: "[]",
          result_payload: null,
          artifact_dir: "/tmp/fangzhen-py/guided_training/guided-ensemble-existing",
          error_message: null,
        }]];
      }
      return [[experiment]];
    });
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

    const session = await trainModel("weighted_avg", 20240002, {
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

    expect(session.session_id).toBe("guided-ensemble-existing");
    expect(spawnMock).not.toHaveBeenCalled();
    expect(getActiveModelTrainingRun).toHaveBeenCalledTimes(2);
  });

  test("rejects a tampered base artifact before fingerprinting an ensemble session", async () => {
    const maRun = buildActiveRun("ma", { artifact_sha256: "0".repeat(64) });
    const lstmRun = buildActiveRun("lstm");
    useActiveRuns(maRun, lstmRun);
    mockExperimentAndDatasetQueries();
    existsSync.mockReturnValue(true);
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );

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
      lstmFeatures: "价格指数,产能利用率",
    })).rejects.toMatchObject({
      message: "融合模型依赖的 MA 基础模型产物已损坏，请重新训练该基础模型",
      details: {
        code: "MODEL_ARTIFACT_INVALID",
        required_action: "retrain",
        base_model: "ma",
      },
    });

    expect(getActiveModelTrainingRun).toHaveBeenCalledTimes(1);
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("rechecks base artifact digests immediately before an ensemble Python step", async () => {
    const ensembleBody = {
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
    };
    const maRun = buildActiveRun("ma");
    const lstmRun = buildActiveRun("lstm");
    useActiveRuns(maRun, lstmRun);

    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
    const row = buildGuidedSessionRow({
      model_type: "weighted_avg",
      request_payload: JSON.stringify(ensembleBody),
      dataset_revision: hashDatasetSnapshot(JSON.stringify(currentDataset)),
      runtime_revision: getModelRuntimeRevision(true),
    });

    let attempt: Record<string, unknown> | null = null;
    const connection = {
      beginTransaction: mock(async () => undefined),
      commit: mock(async () => undefined),
      rollback: mock(async () => undefined),
      release: mock(() => undefined),
      query: mock(async (sql: string, params: unknown[] = []) => {
        if (sql.includes("FROM model_guided_training_sessions")) return [[row]];
        if (sql.includes("INSERT INTO model_guided_training_step_attempts")) {
          attempt = {
            attempt_id: params[0],
            session_id: params[1],
            step_id: params[2],
            status: "running",
            input_state_path: params[3],
            input_state_sha256: params[4],
            candidate_state_path: params[5],
            receipt_path: params[6],
            lease_expires_at: new Date(Date.now() + 60_000),
            error_message: null,
          };
          return [{ affectedRows: 1 }];
        }
        if (sql.includes("FROM model_guided_training_step_attempts")) return [[attempt]];
        return [{ affectedRows: 1 }];
      }),
    };
    poolGetConnection.mockResolvedValue(connection as never);
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM model_guided_training_sessions")) return [[row]];
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      return [[buildExperimentContextRow()]];
    });
    existsSync.mockImplementation((filePath: string) =>
      !filePath.includes(`${path.sep}checkpoints${path.sep}`)
    );
    let maArtifactReads = 0;
    createReadStream.mockImplementation((filePath: string) =>
      Readable.from([
        filePath === maRun.artifact_path && maArtifactReads++ > 0
          ? "tampered-ma-artifact"
          : modelDatasetCsv,
      ]) as unknown as ReturnType<typeof createReadStream>
    );

    await expect(runGuidedTrainingStep(
      "weighted_avg",
      20240002,
      row.session_id,
      "prepare_data",
    )).rejects.toMatchObject({
      message: "融合模型依赖的 MA 基础模型产物已损坏，请重新训练该基础模型",
      details: {
        code: "MODEL_ARTIFACT_INVALID",
        required_action: "retrain",
        base_model: "ma",
      },
    });

    expect(maArtifactReads).toBe(2);
    expect(getActiveModelTrainingRun).toHaveBeenCalledTimes(3);
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("accepts persisted comma-containing LSTM feature names for guided ensemble creation", async () => {
    useActiveRuns(buildActiveRun("ma"), buildActiveRun("lstm"));
    const experiment = buildExperimentContextRow({
      lstm_features: JSON.stringify(["价格指数", "外部,指标"]),
    });
    createReadStream.mockImplementation(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    poolQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
      if (sql.includes("FROM model_guided_training_sessions")) {
        return [[{
          session_id: "guided-features-existing",
          experiment_id: 990002,
          student_id: 20240002,
          model_type: "weighted_avg",
          request_hash: "request-hash",
          dataset_revision: "dataset-revision",
          runtime_revision: getModelRuntimeRevision(),
          protocol_version: 3,
          request_payload: "{}",
          status: "ready",
          current_step_id: "prepare_data",
          completed_step_ids: "[]",
          step_outputs: "{}",
          step_plan: "[]",
          result_payload: null,
          artifact_dir: "/tmp/fangzhen-py/guided_training/guided-features-existing",
          error_message: null,
        }]];
      }
      return [[experiment]];
    });
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

    const session = await trainModel("weighted_avg", 20240002, {
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

    expect(session.session_id).toBe("guided-features-existing");
    expect(spawnMock).not.toHaveBeenCalled();
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

  test("rejects production output when python omits predictions", async () => {
    const productionPath = path.join(mockedTrainedModelsRoot, "lstm-current-production.pkl");
    const activeRun = buildActiveRun("lstm", { production_artifact_path: productionPath });
    useActiveRuns(activeRun);
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

    await expect(
      predictExperimentModel("lstm", 20240002, 990002, 3, preparationToken),
    ).rejects.toMatchObject({
      code: "PRODUCTION_PREPARATION_REQUIRED",
      details: { required_action: "prepare" },
    });
    expect(clearProductionArtifactForRun).toHaveBeenCalledWith(
      activeRun.run_id,
      preparationToken,
    );
    expect(unlinkSync).toHaveBeenCalledWith(productionPath);
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
      [buildExperimentContextRow({ lstm_features: null })],
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
      message: "融合模型依赖的 LSTM 基础模型缺少特征配置、目标字段或归一化配置",
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
