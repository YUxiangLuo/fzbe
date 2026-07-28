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
} = await harness.initializeModelsServiceHarness("prediction");
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

describe("models service prediction", () => {
  setupModelsServiceTestHooks();
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

});
