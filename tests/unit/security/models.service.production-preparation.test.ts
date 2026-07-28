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
} = await harness.initializeModelsServiceHarness("production-preparation");
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

describe("models service production preparation", () => {
  setupModelsServiceTestHooks();
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

});
