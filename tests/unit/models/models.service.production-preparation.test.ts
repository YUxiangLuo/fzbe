import * as harness from "./support/models-service-harness";

const {
  runPythonModel,
  createOrResumeGuidedTrainingSession,
  getGuidedTrainingSession,
  runGuidedTrainingStep,
  resolveGuidedStepTimeoutMs,
  createOrResumeProductionPreparationSession,
  commitProductionPreparationIntermediateAttempt,
  expandProductionPreparationPlan,
  getProductionPreparationSession,
  installProductionPreparationAttempt,
  recordProductionPreparationAttemptFailure,
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
  poolConnectionCommit,
  poolConnectionRollback,
  poolConnectionRelease,
  poolConnectionDestroy,
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
  commitProductionPreparationResult,
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

  test("lets baseline-only Boosting proceed directly from initialize to finalize", () => {
    const initialPlan = [
      { id: "initialize", label: "确认生产准备材料" },
      { id: "finalize", label: "组装并校验生产产物" },
    ];

    expect(expandProductionPreparationPlan(
      initialPlan,
      { work_items: [], direct_finalize: true },
      "boosting",
    )).toEqual(initialPlan);
  });

  test("rejects empty production work for every non-baseline-only contract", () => {
    const initialPlan = [
      { id: "initialize", label: "确认生产准备材料" },
      { id: "finalize", label: "组装并校验生产产物" },
    ];

    expect(() => expandProductionPreparationPlan(
      initialPlan,
      { work_items: [] },
      "boosting",
    )).toThrow("重复或空工作项");
    expect(() => expandProductionPreparationPlan(
      initialPlan,
      { work_items: [], direct_finalize: true },
      "weighted_avg",
    )).toThrow("重复或空工作项");
    expect(() => expandProductionPreparationPlan(
      initialPlan,
      {
        work_items: [{ id: "fit_stage_1_ma", model: "ma" }],
        direct_finalize: true,
      },
      "boosting",
    )).toThrow("重复或空工作项");
  });

  test("rejects production session creation when an ensemble is incomplete", async () => {
    poolQuery.mockResolvedValueOnce([
      [buildExperimentContextRow({ ensemble_weighted_completed: 0 })],
    ]);

    await expect(
      createOrResumeProductionPreparationSession("weighted_avg", 20240002, 990002),
    ).rejects.toMatchObject({
      message: "加权平均融合模型尚未完成训练，无法准备生产预测模型",
      details: {
        code: "MODEL_RETRAINING_REQUIRED",
        required_action: "retrain",
        model_type: "weighted_avg",
      },
    });

    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("requires retraining when a base model completion record is incomplete", async () => {
    poolQuery.mockResolvedValueOnce([[
      buildExperimentContextRow({ moving_average_completed: 0 }),
    ]]);

    await expect(
      createOrResumeProductionPreparationSession("ma", 20240002, 990002),
    ).rejects.toMatchObject({
      message: "移动平均基础模型尚未完成训练，无法准备生产预测模型",
      details: {
        code: "MODEL_RETRAINING_REQUIRED",
        required_action: "retrain",
        model_type: "ma",
      },
    });

    expect(getActiveModelTrainingRun).not.toHaveBeenCalled();
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("requires retraining instead of retrying when the selected model has no active run", async () => {
    poolQuery.mockResolvedValueOnce([
      [buildExperimentContextRow({ moving_average_completed: 1, moving_average_window: 4 })],
    ]);

    await expect(
      createOrResumeProductionPreparationSession("ma", 20240002, 990002, 6),
    ).rejects.toMatchObject({
      message: "MA 模型缺少可验证的训练版本，请重新训练该模型",
      details: {
        code: "MODEL_TRAINING_RUN_REQUIRED",
        required_action: "retrain",
        model_type: "ma",
      },
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
    ).rejects.toMatchObject({
      message: "MA 模型运行环境已变化，请重新训练该模型",
      details: {
        code: "MODEL_RETRAINING_REQUIRED",
        required_action: "retrain",
        model_type: "ma",
      },
    });
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
    ).rejects.toMatchObject({
      message: "MA 模型对应的数据集版本已变化，请重新训练该模型",
      details: {
        code: "MODEL_RETRAINING_REQUIRED",
        required_action: "retrain",
        model_type: "ma",
      },
    });
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("locks student, experiment, and active run before creating a production session", async () => {
    const sessionId = "123e4567-e89b-42d3-a456-426614174999";
    const artifactDir = path.join(
      "/tmp/fangzhen-py/production_preparation",
      `exp990002-student20240002-ma-${sessionId}`,
    );
    const randomUuidSpy = jest.spyOn(crypto, "randomUUID").mockReturnValue(sessionId);
    let insertedSession: Record<string, unknown> | null = null;
    try {
      await actualFs.promises.mkdir(artifactDir, { recursive: true });
      createReadStream.mockImplementation(() =>
        Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
      );
      poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
      const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
      const datasetRevision = hashDatasetSnapshot(JSON.stringify(currentDataset));
      const activeRun = buildActiveRun("ma", {
        request_hash: "training-request-current",
        dataset_revision: datasetRevision,
        runtime_revision: getModelRuntimeRevision(true),
        production_revision: null,
        is_active: 1,
      });
      useActiveRuns(activeRun);

      poolQuery.mockImplementation(async (sql: string, params: unknown[] = []) => {
        if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
        if (sql.includes("FROM model_training_runs") && sql.includes("FOR UPDATE")) {
          return [[activeRun]];
        }
        if (sql.includes("INSERT INTO model_production_preparation_sessions")) {
          insertedSession = {
            session_id: params[0],
            experiment_id: params[1],
            student_id: params[2],
            run_id: params[3],
            model_type: params[4],
            request_hash: params[5],
            dataset_revision: params[6],
            runtime_revision: params[7],
            protocol_version: params[8],
            forecast_steps: params[9],
            status: "ready",
            current_step_id: "initialize",
            completed_step_ids: params[10],
            step_plan: params[11],
            artifact_dir: params[12],
            candidate_model_path: params[13],
            committed_state_path: null,
            committed_state_sha256: null,
            active_attempt_id: null,
            observed_production_revision: params[14],
            production_revision: null,
            error_message: null,
            superseded_reason: null,
          };
          return [{ affectedRows: 1 }];
        }
        if (sql.includes("FROM model_production_preparation_sessions")) {
          return sql.includes("WHERE session_id = ?") && insertedSession
            ? [[insertedSession]]
            : [[]];
        }
        return [[buildExperimentContextRow()]];
      });

      const result = await createOrResumeProductionPreparationSession(
        "ma",
        20240002,
        990002,
        6,
      );

      expect(result.status).toBe("success");
      const statements = poolQuery.mock.calls.map(([sql]) => String(sql));
      const userLockIndex = statements.findIndex((sql) =>
        sql.includes("FROM users") && sql.includes("FOR UPDATE")
      );
      const experimentLockIndex = statements.findIndex((sql) =>
        sql.includes("FROM experiment_status") && sql.includes("FOR UPDATE")
      );
      const runLockIndex = statements.findIndex((sql) =>
        sql.includes("FROM model_training_runs") && sql.includes("FOR UPDATE")
      );
      const insertIndex = statements.findIndex((sql) =>
        sql.includes("INSERT INTO model_production_preparation_sessions")
      );
      expect(userLockIndex).toBeGreaterThanOrEqual(0);
      expect(experimentLockIndex).toBeGreaterThan(userLockIndex);
      expect(runLockIndex).toBeGreaterThan(experimentLockIndex);
      expect(insertIndex).toBeGreaterThan(runLockIndex);
      expect(insertedSession).toMatchObject({
        session_id: sessionId,
        run_id: activeRun.run_id,
        status: "ready",
      });
    } finally {
      randomUuidSpy.mockRestore();
      await actualFs.promises.rm(artifactDir, { recursive: true, force: true });
    }
  });

  test("removes an uncommitted production session directory after a known INSERT failure", async () => {
    const sessionId = "123e4567-e89b-42d3-a456-426614174998";
    const artifactDir = path.join(
      "/tmp/fangzhen-py/production_preparation",
      `exp990002-student20240002-ma-${sessionId}`,
    );
    const randomUuidSpy = jest.spyOn(crypto, "randomUUID").mockReturnValue(sessionId);
    try {
      await actualFs.promises.mkdir(artifactDir, { recursive: true });
      createReadStream.mockImplementation(() =>
        Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
      );
      poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
      const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
      const datasetRevision = hashDatasetSnapshot(JSON.stringify(currentDataset));
      const activeRun = buildActiveRun("ma", {
        request_hash: "training-request-current",
        dataset_revision: datasetRevision,
        runtime_revision: getModelRuntimeRevision(true),
        production_revision: null,
        is_active: 1,
      });
      useActiveRuns(activeRun);

      poolQuery.mockImplementation(async (sql: string) => {
        if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
        if (sql.includes("FROM model_production_preparation_sessions")) return [[]];
        if (sql.includes("FROM model_training_runs") && sql.includes("FOR UPDATE")) {
          return [[activeRun]];
        }
        if (sql.includes("INSERT INTO model_production_preparation_sessions")) {
          throw new Error("known production session INSERT failure");
        }
        return [[buildExperimentContextRow()]];
      });

      await expect(
        createOrResumeProductionPreparationSession("ma", 20240002, 990002, 6),
      ).rejects.toThrow("known production session INSERT failure");

      expect(rmSync).toHaveBeenCalledWith(artifactDir, { recursive: true, force: true });
      expect(poolConnectionRollback).toHaveBeenCalledTimes(1);
      expect(poolConnectionCommit).not.toHaveBeenCalled();
      expect(poolConnectionRelease).toHaveBeenCalledTimes(1);
      expect(poolConnectionDestroy).not.toHaveBeenCalled();
    } finally {
      randomUuidSpy.mockRestore();
      await actualFs.promises.rm(artifactDir, { recursive: true, force: true });
    }
  });

  test("preserves the production session directory when COMMIT outcome is unknown", async () => {
    const sessionId = "123e4567-e89b-42d3-a456-426614174997";
    const artifactDir = path.join(
      "/tmp/fangzhen-py/production_preparation",
      `exp990002-student20240002-ma-${sessionId}`,
    );
    const randomUuidSpy = jest.spyOn(crypto, "randomUUID").mockReturnValue(sessionId);
    try {
      await actualFs.promises.mkdir(artifactDir, { recursive: true });
      createReadStream.mockImplementation(() =>
        Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
      );
      poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
      const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
      const datasetRevision = hashDatasetSnapshot(JSON.stringify(currentDataset));
      const activeRun = buildActiveRun("ma", {
        request_hash: "training-request-current",
        dataset_revision: datasetRevision,
        runtime_revision: getModelRuntimeRevision(true),
        production_revision: null,
        is_active: 1,
      });
      useActiveRuns(activeRun);

      poolQuery.mockImplementation(async (sql: string) => {
        if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
        if (sql.includes("FROM model_production_preparation_sessions")) return [[]];
        if (sql.includes("FROM model_training_runs") && sql.includes("FOR UPDATE")) {
          return [[activeRun]];
        }
        if (sql.includes("INSERT INTO model_production_preparation_sessions")) {
          return [{ affectedRows: 1 }];
        }
        return [[buildExperimentContextRow()]];
      });
      poolConnectionCommit.mockImplementationOnce(async () => {
        throw new Error("COMMIT acknowledgement lost");
      });

      await expect(
        createOrResumeProductionPreparationSession("ma", 20240002, 990002, 6),
      ).rejects.toMatchObject({ code: "MODEL_COMMIT_OUTCOME_UNKNOWN" });

      expect(rmSync).not.toHaveBeenCalledWith(artifactDir, { recursive: true, force: true });
      expect(poolConnectionRollback).not.toHaveBeenCalled();
      expect(poolConnectionRelease).not.toHaveBeenCalled();
      expect(poolConnectionDestroy).toHaveBeenCalledTimes(1);
    } finally {
      randomUuidSpy.mockRestore();
      await actualFs.promises.rm(artifactDir, { recursive: true, force: true });
    }
  });

  test("installs a production attempt atomically under the session lock", async () => {
    const session = {
      session_id: "production-attempt-session",
      experiment_id: 990002,
      student_id: 20240002,
      run_id: "ma-run-current",
      model_type: "ma",
      request_hash: "production-request",
      dataset_revision: "dataset-revision",
      runtime_revision: getModelRuntimeRevision(true),
      protocol_version: 1,
      forecast_steps: 6,
      status: "ready",
      current_step_id: "initialize",
      completed_step_ids: JSON.stringify([]),
      step_plan: JSON.stringify([
        { id: "initialize", label: "确认材料" },
        { id: "finalize", label: "完成" },
      ]),
      artifact_dir: "/tmp/fangzhen-py/production_preparation/production-attempt-session",
      candidate_model_path: "/tmp/fangzhen-py/trained_models/ma-production.pkl",
      committed_state_path: null,
      committed_state_sha256: null,
      active_attempt_id: null,
      observed_production_revision: null,
      production_revision: null,
      error_message: null,
      superseded_reason: null,
      expires_at: null,
    };
    const attempt = {
      attempt_id: "production-attempt-new",
      session_id: session.session_id,
      step_id: "initialize",
      status: "running",
      input_state_path: null,
      input_state_sha256: null,
      candidate_state_path: "/tmp/production-attempt-new/candidate.pkl",
      candidate_state_sha256: null,
      receipt_path: "/tmp/production-attempt-new/receipt.json",
      output_payload: null,
      error_message: null,
      lease_expires_at: null,
    };
    const connection = {
      beginTransaction: mock(async () => undefined),
      commit: mock(async () => undefined),
      rollback: mock(async () => undefined),
      release: mock(() => undefined),
      destroy: mock(() => undefined),
      query: mock(async (sql: string) => {
        if (sql.includes("SELECT *") && sql.includes("FROM model_production_preparation_sessions")) {
          return [[session]];
        }
        return [{ affectedRows: 1 }];
      }),
    };
    poolGetConnection.mockResolvedValueOnce(connection as never);

    await installProductionPreparationAttempt({
      session: session as never,
      attempt: attempt as never,
      stepId: "initialize",
      observedProductionRevision: null,
      leaseSeconds: 120,
    });

    const statements = connection.query.mock.calls.map(([sql]) => String(sql));
    const lockIndex = statements.findIndex((sql) =>
      sql.includes("FROM model_production_preparation_sessions") && sql.includes("FOR UPDATE")
    );
    const insertIndex = statements.findIndex((sql) =>
      sql.includes("INSERT INTO model_production_preparation_step_attempts")
    );
    const sessionUpdateIndex = statements.findIndex((sql) =>
      sql.includes("UPDATE model_production_preparation_sessions")
    );
    expect(lockIndex).toBe(0);
    expect(insertIndex).toBeGreaterThan(lockIndex);
    expect(sessionUpdateIndex).toBeGreaterThan(insertIndex);
    expect(statements[sessionUpdateIndex]).toContain("active_attempt_id <=> ?");
    expect(connection.commit).toHaveBeenCalledTimes(1);
    expect(connection.rollback).not.toHaveBeenCalled();
  });

  test("uses the database-computed production lease flag when date strings have no timezone", async () => {
    const session = {
      session_id: "production-lease-session",
      experiment_id: 990002,
      student_id: 20240002,
      run_id: "ma-run-current",
      model_type: "ma",
      request_hash: "production-request",
      dataset_revision: "dataset-revision",
      runtime_revision: getModelRuntimeRevision(true),
      protocol_version: 1,
      forecast_steps: 6,
      status: "running",
      current_step_id: "initialize",
      completed_step_ids: JSON.stringify([]),
      step_plan: JSON.stringify([
        { id: "initialize", label: "确认材料" },
        { id: "finalize", label: "完成" },
      ]),
      artifact_dir: "/tmp/fangzhen-py/production_preparation/production-lease-session",
      candidate_model_path: "/tmp/fangzhen-py/trained_models/ma-production.pkl",
      committed_state_path: null,
      committed_state_sha256: null,
      active_attempt_id: "production-active-attempt",
      observed_production_revision: null,
      production_revision: null,
      error_message: null,
      superseded_reason: null,
      expires_at: null,
    };
    const previousAttempt = {
      attempt_id: "production-active-attempt",
      session_id: session.session_id,
      step_id: "initialize",
      status: "running",
      input_state_path: null,
      input_state_sha256: null,
      candidate_state_path: "/tmp/production-active-attempt.pkl",
      candidate_state_sha256: null,
      receipt_path: "/tmp/production-active-attempt.receipt.json",
      output_payload: null,
      error_message: null,
      // This intentionally looks expired if parsed as an application-local Date.
      lease_expires_at: "2000-01-01 00:00:00",
      lease_is_active: 1,
    };
    const nextAttempt = {
      ...previousAttempt,
      attempt_id: "production-replacement-attempt",
      candidate_state_path: "/tmp/production-replacement-attempt.pkl",
      receipt_path: "/tmp/production-replacement-attempt.receipt.json",
      lease_expires_at: null,
      lease_is_active: null,
    };
    const connection = {
      beginTransaction: mock(async () => undefined),
      commit: mock(async () => undefined),
      rollback: mock(async () => undefined),
      release: mock(() => undefined),
      destroy: mock(() => undefined),
      query: mock(async (sql: string) => {
        if (sql.includes("FROM model_production_preparation_sessions")) return [[session]];
        if (sql.includes("FROM model_production_preparation_step_attempts")) {
          return [[previousAttempt]];
        }
        return [{ affectedRows: 1 }];
      }),
    };
    poolGetConnection.mockResolvedValueOnce(connection as never);

    await expect(installProductionPreparationAttempt({
      session: session as never,
      attempt: nextAttempt as never,
      stepId: "initialize",
      observedProductionRevision: null,
      leaseSeconds: 120,
    })).rejects.toMatchObject({ code: "PRODUCTION_PREPARATION_STEP_IN_PROGRESS" });

    const attemptSelect = connection.query.mock.calls.find(([sql]) =>
      String(sql).includes("FROM model_production_preparation_step_attempts")
    );
    expect(String(attemptSelect?.[0])).toContain("lease_expires_at > NOW() AS lease_is_active");
    expect(connection.query.mock.calls.some(([sql]) =>
      String(sql).includes("INSERT INTO model_production_preparation_step_attempts")
    )).toBeFalse();
    expect(connection.rollback).toHaveBeenCalledTimes(1);
    expect(connection.commit).not.toHaveBeenCalled();
    expect(connection.release).toHaveBeenCalledTimes(1);
  });

  test("commits an intermediate production checkpoint with attempt ownership CAS", async () => {
    const attemptId = "production-intermediate-attempt";
    const connection = {
      beginTransaction: mock(async () => undefined),
      commit: mock(async () => undefined),
      rollback: mock(async () => undefined),
      release: mock(() => undefined),
      destroy: mock(() => undefined),
      query: mock(async (sql: string) => {
        if (sql.includes("SELECT active_attempt_id")) {
          return [[{ active_attempt_id: attemptId, status: "running" }]];
        }
        return [{ affectedRows: 1 }];
      }),
    };
    poolGetConnection.mockResolvedValueOnce(connection as never);

    await commitProductionPreparationIntermediateAttempt({
      sessionId: "production-session",
      attemptId,
      status: "ready",
      nextStepId: "finalize",
      completedStepIds: ["initialize"],
      stepPlan: [
        { id: "initialize", label: "确认材料" },
        { id: "finalize", label: "完成" },
      ],
      committedStatePath: "/tmp/checkpoint.pkl",
      committedStateSha256: "a".repeat(64),
      outputPayload: { prepared: true },
    });

    const statements = connection.query.mock.calls.map(([sql]) => String(sql));
    expect(statements[0]).toContain("FOR UPDATE");
    expect(statements[1]).toContain("UPDATE model_production_preparation_step_attempts");
    expect(statements[1]).toContain("AND session_id = ?");
    expect(statements[2]).toContain("UPDATE model_production_preparation_sessions");
    expect(statements[2]).toContain("AND active_attempt_id = ?");
    expect(statements[2]).toContain("AND status = 'running'");
    expect(connection.commit).toHaveBeenCalledTimes(1);
  });

  test("does not let an old production failure overwrite a replacement attempt", async () => {
    const connection = {
      beginTransaction: mock(async () => undefined),
      commit: mock(async () => undefined),
      rollback: mock(async () => undefined),
      release: mock(() => undefined),
      destroy: mock(() => undefined),
      query: mock(async (sql: string) => {
        if (sql.includes("SELECT active_attempt_id")) {
          return [[{ active_attempt_id: "replacement-attempt", status: "running" }]];
        }
        throw new Error(`unexpected stale failure update: ${sql}`);
      }),
    };
    poolGetConnection.mockResolvedValueOnce(connection as never);

    await recordProductionPreparationAttemptFailure({
      sessionId: "production-session",
      attemptId: "old-attempt",
      stepId: "initialize",
      message: "旧请求失败",
    });

    expect(connection.query).toHaveBeenCalledTimes(1);
    expect(connection.commit).toHaveBeenCalledTimes(1);
    expect(connection.rollback).not.toHaveBeenCalled();
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

  test("supersedes production preparation and requires retraining when the dataset changed", async () => {
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
    )).rejects.toMatchObject({
      message: "MA 模型对应的数据集版本已变化，请重新训练该模型",
      details: {
        code: "MODEL_RETRAINING_REQUIRED",
        required_action: "retrain",
        model_type: "ma",
      },
    });

    expect(poolQuery.mock.calls.some(([sql]) =>
      String(sql).includes("SET status = 'superseded'"),
    )).toBe(true);
    expect(setProductionArtifactForRun).not.toHaveBeenCalled();
    expect(commitProductionPreparationResult).not.toHaveBeenCalled();
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("requires retraining instead of returning 500 when the source artifact is corrupt", async () => {
    poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    createReadStream.mockImplementationOnce(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
    const datasetRevision = hashDatasetSnapshot(JSON.stringify(currentDataset));
    const runtimeRevision = getModelRuntimeRevision(true);
    const activeRun = buildActiveRun("ma", {
      artifact_sha256: "0".repeat(64),
      dataset_revision: datasetRevision,
      runtime_revision: runtimeRevision,
    });
    const requestHash = crypto.createHash("sha256").update(JSON.stringify({
      runId: activeRun.run_id,
      datasetRevision,
      runtimeRevision,
      forecastSteps: 6,
    })).digest("hex");
    const sessionRow = {
      session_id: "production-corrupt-source",
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
      current_step_id: "initialize",
      completed_step_ids: JSON.stringify([]),
      step_plan: JSON.stringify([
        { id: "initialize", label: "确认生产准备材料" },
        { id: "finalize", label: "组装并校验生产产物" },
      ]),
      artifact_dir: "/tmp/fangzhen-py/production_preparation/production-corrupt-source",
      candidate_model_path: path.join(mockedTrainedModelsRoot, "ma-corrupt-candidate.pkl"),
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
      if (sql.trimStart().startsWith("UPDATE")) return [{ affectedRows: 1 }];
      return [[buildExperimentContextRow()]];
    });

    await expect(runProductionPreparationStep(
      "ma",
      20240002,
      sessionRow.session_id,
      "initialize",
    )).rejects.toMatchObject({
      message: "MA 已训练模型产物已损坏，请重新训练该模型",
      details: {
        code: "MODEL_ARTIFACT_INVALID",
        required_action: "retrain",
        model_type: "ma",
      },
    });

    expect(spawnMock).not.toHaveBeenCalled();
  });

  test("preserves final production artifacts when COMMIT acknowledgement is lost", async () => {
    poolQuery.mockResolvedValueOnce([[{ file_path: datasetCsvPath }]]);
    createReadStream.mockImplementationOnce(() =>
      Readable.from([modelDatasetCsv]) as unknown as ReturnType<typeof createReadStream>
    );
    const currentDataset = await loadDatasetFromDb("制造业", "样本企业", "样本产品");
    const datasetJson = JSON.stringify(currentDataset);
    const datasetRevision = hashDatasetSnapshot(datasetJson);
    const runtimeRevision = getModelRuntimeRevision(true);
    const sessionId = `production-ambiguous-finalize-${process.pid}`;
    const artifactDir = path.join(
      "/tmp/fangzhen-py/production_preparation",
      sessionId,
    );
    const datasetSnapshotPath = path.join(artifactDir, "dataset.json");
    const sourceModelPath = path.join(
      mockedTrainedModelsRoot,
      `${sessionId}-source.pkl`,
    );
    const candidateModelPath = path.join(
      mockedTrainedModelsRoot,
      `${sessionId}-candidate.pkl`,
    );
    const activeRun = buildActiveRun("ma", {
      artifact_path: sourceModelPath,
      dataset_revision: datasetRevision,
      runtime_revision: runtimeRevision,
    });
    const requestHash = crypto.createHash("sha256").update(JSON.stringify({
      runId: activeRun.run_id,
      datasetRevision,
      runtimeRevision,
      forecastSteps: 6,
    })).digest("hex");
    const sessionRow = {
      session_id: sessionId,
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
      artifact_dir: artifactDir,
      candidate_model_path: candidateModelPath,
      committed_state_path: null,
      committed_state_sha256: null,
      active_attempt_id: null,
      observed_production_revision: null,
      production_revision: null,
      error_message: null,
      superseded_reason: null,
    };
    const installConnection = {
      beginTransaction: mock(async () => undefined),
      commit: mock(async () => undefined),
      rollback: mock(async () => undefined),
      release: mock(() => undefined),
      destroy: mock(() => undefined),
      query: mock(async (sql: string) => {
        if (sql.includes("FROM model_production_preparation_sessions")) {
          return [[sessionRow]];
        }
        return [{ affectedRows: 1 }];
      }),
    };
    const checkpointContents = "durable-production-final-checkpoint";
    const productionModelContents = "possibly-published-production-model";
    const checkpointSha256 = crypto
      .createHash("sha256")
      .update(checkpointContents)
      .digest("hex");
    let candidateStatePath: string | null = null;
    let receiptPath: string | null = null;
    const commitError = new ModelCommitOutcomeUnknownError();

    try {
      await actualFs.promises.mkdir(artifactDir, { recursive: true });
      await actualFs.promises.mkdir(mockedTrainedModelsRoot, { recursive: true });
      await writeFile(datasetSnapshotPath, datasetJson);
      await writeFile(sourceModelPath, modelDatasetCsv);
      await writeFile(candidateModelPath, productionModelContents);
      existsSync.mockImplementation((filePath: string) => (
        filePath.endsWith("production_prepare_model.py")
        || filePath === sourceModelPath
        || filePath === candidateModelPath
      ));
      createReadStream.mockImplementation((filePath: string) =>
        Readable.from([
          filePath === datasetSnapshotPath
            ? datasetJson
            : filePath === candidateStatePath
              ? checkpointContents
              : filePath === candidateModelPath
                ? productionModelContents
                : modelDatasetCsv,
        ]) as unknown as ReturnType<typeof createReadStream>
      );
      useActiveRuns(activeRun);
      poolGetConnection.mockResolvedValueOnce(installConnection as never);
      poolQuery.mockImplementation(async (sql: string) => {
        if (sql.includes("FROM model_production_preparation_sessions")) return [[sessionRow]];
        if (sql.includes("FROM dataset_segments")) return [[{ file_path: datasetCsvPath }]];
        if (sql.includes("SET expires_at = CASE")) return [{ affectedRows: 1 }];
        return [[buildExperimentContextRow()]];
      });
      spawnMock.mockImplementation((command: string[]) => {
        const argumentAfter = (flag: string) => {
          const index = command.indexOf(flag);
          if (index < 0 || !command[index + 1]) throw new Error(`missing ${flag}`);
          return command[index + 1]!;
        };
        const attemptId = argumentAfter("--attempt-id");
        candidateStatePath = argumentAfter("--output-state-file");
        receiptPath = argumentAfter("--receipt-file");
        const saveModelPath = argumentAfter("--save-model");
        expect(saveModelPath).toBe(candidateModelPath);
        const receipt = {
          protocol_version: 1,
          attempt_id: attemptId,
          step_id: "finalize",
          candidate_state_path: candidateStatePath,
          candidate_state_sha256: checkpointSha256,
          input_state_path: null,
          input_state_sha256: null,
          output: { finalized: true },
          final_result: { prepared: true },
        };
        return createSpawnResult(JSON.stringify(receipt));
      });
      commitProductionPreparationResult.mockRejectedValueOnce(commitError);

      let observedError: unknown;
      try {
        await runProductionPreparationStep("ma", 20240002, sessionId, "finalize");
      } catch (error) {
        observedError = error;
      }

      expect(observedError).toBe(commitError);
      expect(observedError).toMatchObject({
        code: "MODEL_COMMIT_OUTCOME_UNKNOWN",
        retryable: true,
      });
      expect(commitProductionPreparationResult).toHaveBeenCalledTimes(1);
      expect(commitProductionPreparationResult).toHaveBeenCalledWith(
        expect.objectContaining({
          experimentId: 990002,
          studentId: 20240002,
          runId: activeRun.run_id,
          sessionId,
          attempt: expect.objectContaining({
            artifactPath: candidateModelPath,
            committedStatePath: candidateStatePath,
          }),
        }),
      );
      // A second transaction here would be the best-effort failure recorder,
      // which must not overwrite a transaction that may already have committed.
      expect(poolGetConnection).toHaveBeenCalledTimes(1);
      expect(installConnection.query.mock.calls.some(([sql]) =>
        String(sql).includes("SET status = 'failed'")
        || String(sql).includes("error_message = ?"),
      )).toBeFalse();
      expect(await Bun.file(candidateModelPath).exists()).toBeTrue();
      expect(candidateStatePath).not.toBeNull();
      expect(receiptPath).not.toBeNull();
      expect(rmSync.mock.calls.some(([filePath]) => filePath === artifactDir)).toBeFalse();
      expect(unlinkSync.mock.calls.some(([filePath]) =>
        filePath === candidateModelPath
        || filePath === candidateStatePath
        || filePath === receiptPath,
      )).toBeFalse();
    } finally {
      await actualFs.promises.rm(artifactDir, { recursive: true, force: true });
      await actualFs.promises.rm(sourceModelPath, { force: true });
      await actualFs.promises.rm(candidateModelPath, { force: true });
      existsSync.mockImplementation(() => false);
    }
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
    expect(commitProductionPreparationResult).toHaveBeenCalledWith(
      expect.objectContaining({
        experimentId: 990002,
        studentId: 20240002,
        runId: activeRun.run_id,
        sessionId: sessionRow.session_id,
        productionRevision: preparationToken,
      }),
    );
    expect(spawnMock).not.toHaveBeenCalled();
    expect(unlinkSync).toHaveBeenCalledWith(sessionRow.candidate_model_path);
  });

});
