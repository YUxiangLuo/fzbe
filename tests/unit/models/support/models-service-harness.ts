import { afterEach, beforeEach, describe, expect, jest, mock, test } from "bun:test";
import crypto from "node:crypto";
import * as actualFs from "node:fs";
import { unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { TIMEOUTS } from "../../../../src/utils/constants";

mock.restore();

export {
  afterEach,
  beforeEach,
  describe,
  expect,
  jest,
  mock,
  test,
  crypto,
  actualFs,
  unlink,
  writeFile,
  path,
  Readable,
  TIMEOUTS,
};


export const srcRoot = path.resolve(import.meta.dir, "../../../../src");
export const configModulePath = path.join(srcRoot, "config", "index.ts");
export const pythonModulePath = path.join(srcRoot, "utils", "python.ts");
export const modelRuntimeModulePath = path.join(srcRoot, "modules", "models", "model-runtime.ts");
export const modelRunStoreModulePath = path.join(srcRoot, "modules", "models", "model-run-store.ts");
export const modelLocksModulePath = path.join(srcRoot, "modules", "models", "model-locks.ts");
export const modelsServiceModulePath = path.join(srcRoot, "modules", "models", "models.service.ts");
// Model dataset reads intentionally reject absolute paths outside the managed
// upload root. Keep the shared service fixture under that same root so these
// tests exercise the production path contract instead of bypassing it with a
// now-invalid /tmp path.
export const datasetCsvPath = path.resolve(
  srcRoot,
  "../uploads/datasets/.unit-model-dataset.csv",
);
export const mockedPyDir = "/tmp/fangzhen-py";
export const mockedGuidedTrainingRoot = path.join(mockedPyDir, "guided_training");
export const mockedTrainedModelsRoot = path.join(mockedPyDir, "trained_models");
export const preparationToken = "123e4567-e89b-42d3-a456-426614174000";
export const modelDatasetCsv =
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

export const ENSEMBLE_MODEL_TYPES = ["weighted_avg", "boosting", "stacking"];
// The ensemble manifest is produced inside the (mocked) Python digest script, so
// tests only need a stable value shared between buildActiveRun and the spawn stub.
export const ENSEMBLE_ARTIFACT_DIGEST = "e".repeat(64);

export const mockArtifactSha256 = (modelType: string) => {
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

export const poolQuery = mock(async (_sql?: string, _values?: unknown[]) => [[]]);
export const poolConnectionBeginTransaction = mock(async () => {});
export const poolConnectionCommit = mock(async () => {});
export const poolConnectionRollback = mock(async () => {});
export const poolConnectionRelease = mock(() => {});
export const poolConnectionDestroy = mock(() => {});
const buildDefaultPoolConnection = () => ({
  beginTransaction: poolConnectionBeginTransaction,
  commit: poolConnectionCommit,
  rollback: poolConnectionRollback,
  release: poolConnectionRelease,
  destroy: poolConnectionDestroy,
  query: async (sql: string, values?: unknown[]) => {
    const result = await poolQuery(sql, values);
    if (sql.includes("FROM users") && sql.includes("FOR UPDATE")) {
      const rows = Array.isArray(result[0]) ? result[0] as Record<string, unknown>[] : [];
      if (!rows[0]?.role) {
        return [[{ user_id: Number(values?.[0]), role: "Student" }]];
      }
    }
    return result;
  },
});
export const poolGetConnection = mock(async () => buildDefaultPoolConnection());
export const existsSync = mock((_: string) => false);
export const createReadStream = mock(() => {
  throw new Error("createReadStream should not be called in this test");
});
export const readdirSync = mock(() => []);
export const statSync = mock((_: string) => ({
  isDirectory: () => false,
  isFile: () => false,
  mtimeMs: 0,
}));
export const unlinkSync = mock(() => {});
export const copyFileSync = mock((_: string, __: string) => {});
export const renameSync = mock((_: string, __: string) => {});
export const mkdirSync = mock((_: string, __?: unknown) => undefined);
export const rmSync = mock((_: string, __?: unknown) => undefined);
export const createSpawnResult = (stdout = "", stderr = "", exitCode = 0) => ({
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
export const withEnsembleAuxiliarySpawns = (
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
export const spawnMock = mock((_: string[], __: Record<string, unknown>) => (
  createSpawnResult('{"predictions":[]}')
));
export const getActiveModelTrainingRun = mock(async (
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
export const setProductionArtifactForRun = mock(async (_runId: string, _artifactPath: string) => true);
export const commitProductionPreparationResult = mock(async () => ({
  committed: true as const,
  replacedProductionArtifactPath: null,
}));
export const backfillArtifactSha256ForRun = mock(async (_runId: string, _sha256: string) => true);
export const backfillProductionArtifactSha256ForRun = mock(async (
  _runId: string,
  _productionRevision: string,
  _sha256: string,
) => true);
export const clearProductionArtifactForRun = mock(async (_runId: string, _productionRevision: string) => true);
export const originalSpawn = Bun.spawn;
export const acquireModelExecutionLock = mock(async () => ({
  name: "test-execution-lock",
  connection: {},
  signal: new AbortController().signal,
}));
export const acquireModelSlot = mock(async () => ({
  name: "test-slot-lock",
  connection: {},
  signal: new AbortController().signal,
}));
export const releaseModelExecutionLock = mock(async () => undefined);
export const releaseModelSlot = mock(async () => undefined);
export const verifyModelLockOwned = mock(async () => undefined);

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

export const commitModelTrainingRun = mock(async ({ runId, sessionId }: { runId?: string; sessionId?: string }) => ({
  runId: runId ?? sessionId ?? "run-test",
  stateVersion: 1,
  reusedExistingRun: false,
}));

export const buildExperimentTrainingRevision = mock(() => "experiment-revision");
export const buildModelTrainingRequestHash = mock(() => "request-hash");
export const loadCurrentExperimentStatePatch = mock(async () => ({ state_version: 1 }));

mock.module(modelRunStoreModulePath, () => ({
  backfillArtifactSha256ForRun,
  backfillProductionArtifactSha256ForRun,
  buildExperimentTrainingRevision,
  buildModelTrainingRequestHash,
  clearProductionArtifactForRun,
  commitProductionPreparationResult,
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

type ModelsServiceModule = typeof import(
  "../../../../src/modules/models/models.service"
);

let initializedModelsService: ModelsServiceModule | null = null;

export const initializeModelsServiceHarness = async (suiteKey: string) => {
  const loaded = await import(
    `${modelsServiceModulePath}?models-service-${suiteKey}`
  ) as ModelsServiceModule;
  initializedModelsService = loaded;
  return loaded;
};

const getHarnessModelRuntimeRevision = () => {
  if (!initializedModelsService) {
    throw new Error("models service test harness has not been initialized");
  }
  return initializedModelsService.getModelRuntimeRevision();
};

export const buildExperimentContextRow = (overrides: Record<string, unknown> = {}) => ({
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

export const buildGuidedSessionRow = (overrides: Record<string, unknown> = {}) => ({
  session_id: "guided-revision-test",
  experiment_id: 990002,
  student_id: 20240002,
  model_type: "ma",
  request_hash: "request-hash",
  dataset_revision: "dataset-revision",
  runtime_revision: getHarnessModelRuntimeRevision(),
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

export const buildActiveRun = (
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

export const useActiveRuns = (...runs: ReturnType<typeof buildActiveRun>[]) => {
  const runsByModel = new Map(runs.map((run) => [run.model_type, run]));
  getActiveModelTrainingRun.mockImplementation(async (_experimentId, _studentId, modelType) => (
    runsByModel.get(modelType) ?? null
  ));
};

export const guidedSessionArtifactRecoveryCases = [
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

export const invalidLegacyStdDevCases = [
  { label: "negative", value: -1 },
  { label: "nonnumeric", value: "invalid" },
] as const;

export const mockExperimentAndDatasetQueries = (
  experiment = buildExperimentContextRow(),
) => {
  poolQuery.mockReset();
  poolQuery.mockImplementation(async (sql: string) => (
    sql.includes("FROM dataset_segments")
      ? [[{ file_path: datasetCsvPath }]]
      : [[experiment]]
  ));
};


export const setupModelsServiceTestHooks = () => {
  beforeEach(async () => {
    // Git does not preserve the empty uploads/datasets directory in a clean
    // checkout. Keep this shared fixture self-bootstrapping so the unit suite
    // exercises the managed upload-root contract without depending on local
    // runtime leftovers.
    await actualFs.promises.mkdir(path.dirname(datasetCsvPath), { recursive: true });
    await writeFile(datasetCsvPath, modelDatasetCsv);

    poolQuery.mockReset();
    poolQuery.mockResolvedValue([
      [buildExperimentContextRow()],
    ]);
    poolConnectionBeginTransaction.mockClear();
    poolConnectionCommit.mockClear();
    poolConnectionRollback.mockClear();
    poolConnectionRelease.mockClear();
    poolConnectionDestroy.mockClear();
    poolGetConnection.mockReset();
    poolGetConnection.mockImplementation(async () => buildDefaultPoolConnection());

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
    commitProductionPreparationResult.mockReset();
    commitProductionPreparationResult.mockResolvedValue({
      committed: true,
      replacedProductionArtifactPath: null,
    });
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
  afterEach(async () => {
    jest.useRealTimers();
    (Bun as { spawn: typeof Bun.spawn }).spawn = originalSpawn;
    try {
      await unlink(datasetCsvPath);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        throw error;
      }
    }
  });

};
