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
} = await harness.initializeModelsServiceHarness("cleanup");
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

describe("models service cleanup", () => {
  setupModelsServiceTestHooks();
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

});
