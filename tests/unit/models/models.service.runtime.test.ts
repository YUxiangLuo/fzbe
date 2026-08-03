import * as harness from "./support/models-service-harness";

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
} = await harness.initializeModelsServiceHarness("runtime");
const trainModel = createOrResumeGuidedTrainingSession;

const {
  actualFs,
  Readable,
  TIMEOUTS,
  crypto,
  path,
  mock,
  jest,
  expect,
  test,
  describe,
  writeFile,
  unlink,
  datasetCsvPath,
  mockedGuidedTrainingRoot,
  mockedTrainedModelsRoot,
  preparationToken,
  modelDatasetCsv,
  ENSEMBLE_ARTIFACT_DIGEST,
  mockArtifactSha256,
  poolQuery,
  poolGetConnection,
  existsSync,
  createReadStream,
  readdirSync,
  statSync,
  unlinkSync,
  copyFileSync,
  renameSync,
  mkdirSync,
  rmSync,
  createSpawnResult,
  withEnsembleAuxiliarySpawns,
  spawnMock,
  getActiveModelTrainingRun,
  setProductionArtifactForRun,
  backfillArtifactSha256ForRun,
  backfillProductionArtifactSha256ForRun,
  clearProductionArtifactForRun,
  acquireModelExecutionLock,
  acquireModelSlot,
  releaseModelExecutionLock,
  releaseModelSlot,
  verifyModelLockOwned,
  commitModelTrainingRun,
  buildExperimentTrainingRevision,
  buildModelTrainingRequestHash,
  loadCurrentExperimentStatePatch,
  buildExperimentContextRow,
  buildGuidedSessionRow,
  buildActiveRun,
  useActiveRuns,
  guidedSessionArtifactRecoveryCases,
  invalidLegacyStdDevCases,
  mockExperimentAndDatasetQueries,
  setupModelsServiceTestHooks,
} = harness;

describe("models service runtime and artifacts", () => {
  setupModelsServiceTestHooks();
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
    expect(
      resolveGuidedStepTimeoutMs("boosting", "initialize_refit", ensembleBody),
    ).toBe(630_000);
    expect(
      resolveGuidedStepTimeoutMs("boosting", "combine_evaluate", ensembleBody),
    ).toBe(630_000);
    expect(resolveGuidedStepTimeoutMs(
      "boosting",
      "initialize_refit",
      { models: "ma,es" } as never,
    )).toBe(300_000);

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

});
