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
  ModelCommitOutcomeUnknownError,
  ModelNotFoundError,
  ModelTimeoutError,
} = await harness.initializeModelsServiceHarness("guided-training");
const { beginGuidedStepAttempt } = await import(
  "../../../src/modules/models/guided-training-state"
);
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

describe("models service guided training", () => {
  setupModelsServiceTestHooks();

  test("uses the database-computed lease flag when date strings have no timezone", async () => {
    const row = buildGuidedSessionRow({
      status: "running",
      current_step_id: "prepare_data",
      active_attempt_id: "guided-active-attempt",
    });
    const activeAttempt = {
      attempt_id: "guided-active-attempt",
      session_id: row.session_id,
      step_id: "prepare_data",
      status: "running",
      input_state_path: null,
      input_state_sha256: null,
      candidate_state_path: "/tmp/guided-active-attempt.pkl",
      candidate_state_sha256: null,
      receipt_path: "/tmp/guided-active-attempt.receipt.json",
      output_payload: null,
      final_result_payload: null,
      error_message: null,
      run_count: 1,
      // This intentionally looks expired if parsed as an application-local Date.
      lease_expires_at: "2000-01-01 00:00:00",
      lease_is_active: 1,
    };
    const connection = {
      beginTransaction: mock(async () => undefined),
      commit: mock(async () => undefined),
      rollback: mock(async () => undefined),
      release: mock(() => undefined),
      destroy: mock(() => undefined),
      query: mock(async (sql: string) => {
        if (sql.includes("FROM model_guided_training_sessions")) return [[row]];
        if (sql.includes("FROM model_guided_training_step_attempts")) {
          return [[activeAttempt]];
        }
        return [{ affectedRows: 1 }];
      }),
    };
    poolGetConnection.mockResolvedValueOnce(connection as never);

    await expect(
      beginGuidedStepAttempt(row as never, "prepare_data", 60_000),
    ).rejects.toMatchObject({ code: "GUIDED_STEP_IN_PROGRESS" });

    const attemptSelect = connection.query.mock.calls.find(([sql]) =>
      String(sql).includes("FROM model_guided_training_step_attempts")
    );
    expect(String(attemptSelect?.[0])).toContain("lease_expires_at > NOW() AS lease_is_active");
    expect(connection.query.mock.calls.some(([sql]) =>
      String(sql).includes("INSERT INTO model_guided_training_step_attempts")
    )).toBeFalse();
    expect(connection.rollback).toHaveBeenCalledTimes(1);
    expect(connection.commit).not.toHaveBeenCalled();
    expect(connection.release).toHaveBeenCalledTimes(1);
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
        const userLockIndex = queries.findIndex((sql) =>
          sql.includes("FROM users") && sql.includes("FOR UPDATE")
        );
        const experimentLockIndex = queries.findIndex((sql) =>
          sql.includes("FROM experiment_status") && sql.includes("FOR UPDATE")
        );
        const insertIndex = queries.findIndex((sql) => sql.includes("INSERT INTO model_guided_training_sessions"));
        expect(supersedeIndex).toBeGreaterThanOrEqual(0);
        expect(deleteIndex).toBeGreaterThan(supersedeIndex);
        expect(userLockIndex).toBeGreaterThan(deleteIndex);
        expect(experimentLockIndex).toBeGreaterThan(userLockIndex);
        expect(insertIndex).toBeGreaterThan(deleteIndex);
        expect(insertIndex).toBeGreaterThan(experimentLockIndex);
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

  test("preserves a final model and checkpoint when COMMIT acknowledgement is lost", async () => {
    const sessionId = `guided-ambiguous-${process.pid}`;
    const attemptId = `attempt-ambiguous-${process.pid}`;
    const artifactDir = path.join(mockedGuidedTrainingRoot, sessionId);
    const sourceModelPath = path.join(artifactDir, "ma.pkl");
    const candidatePath = path.join(artifactDir, "attempts", attemptId, "candidate.pkl");
    const receiptPath = path.join(artifactDir, "attempts", attemptId, "receipt.json");
    const versionedArtifactPath = path.join(
      mockedTrainedModelsRoot,
      `exp990002-student20240002-ma-${sessionId}-backtest.pkl`,
    );
    try {
      await actualFs.promises.mkdir(path.dirname(candidatePath), { recursive: true });
      await actualFs.promises.mkdir(mockedTrainedModelsRoot, { recursive: true });
      await writeFile(sourceModelPath, "durable-model-artifact");
      await writeFile(candidatePath, "durable-final-checkpoint");
      // Keep a real destination file for the synchronous manifest hash, while
      // making the first existence probe report false so this request owns the
      // cleanup responsibility for a newly copied artifact.
      await writeFile(versionedArtifactPath, "new-versioned-model-artifact");
      let versionedExistsChecks = 0;
      existsSync.mockImplementation((filePath: string) => {
        if (filePath === versionedArtifactPath) {
          versionedExistsChecks += 1;
          return versionedExistsChecks > 1;
        }
        return [sourceModelPath, candidatePath, receiptPath].includes(filePath);
      });
      createReadStream.mockImplementation((filePath: string) =>
        Readable.from([
          filePath === versionedArtifactPath
            ? "new-versioned-model-artifact"
            : modelDatasetCsv,
        ]) as unknown as ReturnType<typeof createReadStream>);
      const candidateSha256 = crypto
        .createHash("sha256")
        .update("durable-final-checkpoint")
        .digest("hex");
      const predictions = Array.from({ length: 6 }, (_, index) => ({
        prediction: 220 + index,
        std_dev: 2,
        uncertainty_source: "empirical",
        interval_lower: 216 + index,
        interval_upper: 224 + index,
        interval_level: 0.95,
        interval_kind: "empirical_residual_quantile",
        coverage_claim: "none",
      }));
      await writeFile(receiptPath, JSON.stringify({
        protocol_version: 3,
        attempt_id: attemptId,
        step_id: "save_artifact",
        candidate_state_path: candidatePath,
        candidate_state_sha256: candidateSha256,
        input_state_path: null,
        input_state_sha256: null,
        output: { saved: true },
        final_result: {
          metrics: {},
          predictions,
          eval_y_true: [220, 221, 222, 223, 224, 225],
          method_name: "递推移动平均",
          forecast_strategy: "recursive_roll_forward",
        },
      }));

      poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
      const originalDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
      const row = buildGuidedSessionRow({
        session_id: sessionId,
        artifact_dir: artifactDir,
        dataset_revision: hashDatasetSnapshot(JSON.stringify(originalDataset)),
        runtime_revision: getModelRuntimeRevision(true),
        status: "running",
        current_step_id: "save_artifact",
        completed_step_ids: JSON.stringify([
          "prepare_data",
          "build_windows",
          "forecast",
          "estimate_residuals",
          "evaluate_metrics",
        ]),
        active_attempt_id: attemptId,
      });
      const attempt = {
        attempt_id: attemptId,
        session_id: sessionId,
        step_id: "save_artifact",
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
        destroy: mock(() => undefined),
        query: mock(async (sql: string) => {
          if (sql.includes("FROM model_guided_training_sessions")) return [[row]];
          if (sql.includes("FROM model_guided_training_step_attempts")) return [[attempt]];
          return [{ affectedRows: 1 }];
        }),
      };
      poolGetConnection.mockResolvedValueOnce(beginConnection as never);
      poolQuery.mockImplementation(async (sql: string) => {
        if (sql.includes("FROM model_guided_training_sessions")) return [[row]];
        if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
        if (sql.trimStart().startsWith("UPDATE")) return [{ affectedRows: 1 }];
        return [[buildExperimentContextRow()]];
      });
      commitModelTrainingRun.mockRejectedValueOnce(
        new ModelCommitOutcomeUnknownError(),
      );

      await expect(runGuidedTrainingStep(
        "ma",
        20240002,
        sessionId,
        "save_artifact",
      )).rejects.toMatchObject({ code: "MODEL_COMMIT_OUTCOME_UNKNOWN" });

      expect(await Bun.file(versionedArtifactPath).exists()).toBeTrue();
      expect(await Bun.file(candidatePath).exists()).toBeTrue();
      expect(await Bun.file(receiptPath).exists()).toBeTrue();
      expect(unlinkSync.mock.calls.some(([filePath]) => filePath === versionedArtifactPath)).toBeFalse();
      expect(poolGetConnection).toHaveBeenCalledTimes(1);
    } finally {
      await actualFs.promises.rm(artifactDir, { recursive: true, force: true });
      await actualFs.promises.rm(versionedArtifactPath, { force: true });
      existsSync.mockImplementation(() => false);
    }
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
    })).rejects.toMatchObject({
      message: "MA 模型缺少可验证的训练版本，请重新训练该模型",
      details: {
        code: "MODEL_TRAINING_RUN_REQUIRED",
        required_action: "retrain",
        model_type: "ma",
      },
    });

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
