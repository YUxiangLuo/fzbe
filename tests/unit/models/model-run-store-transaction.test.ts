import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";
import type { TrainInput } from "../../../src/modules/models/models.schema";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const storeModulePath = path.join(srcRoot, "modules", "models", "model-run-store.ts");
const freshStoreModulePath = `${storeModulePath}?transaction-test`;

const query = mock(async () => [[]]);
const poolQuery = mock(async () => [[]]);
const beginTransaction = mock(async () => {});
const commit = mock(async () => {});
const rollback = mock(async () => {});
const release = mock(() => {});
const destroy = mock(() => {});
const connection = { query, beginTransaction, commit, rollback, release, destroy };
const getConnection = mock(async () => connection);

mock.module(configModulePath, () => ({
  pool: {
    getConnection,
    query: poolQuery,
  },
}));

const {
  backfillArtifactSha256ForRun,
  backfillProductionArtifactSha256ForRun,
  buildExperimentTrainingRevision,
  buildModelTrainingRequestHash,
  clearProductionArtifactForRun,
  commitProductionPreparationResult,
  commitModelTrainingRun,
  loadCurrentExperimentStatePatch,
  setProductionArtifactForRun,
} = await import(freshStoreModulePath);

const baseExperiment = {
  experiment_id: 42,
  student_id: 20240002,
  status: "In Progress",
  state_version: 4,
  selected_industry: "制造业",
  selected_company: "示例企业",
  selected_product: "示例产品",
  data_window_train_start_index: 0,
  data_window_train_end_index: 11,
  data_window_evaluate_start_index: 12,
  data_window_evaluate_end_index: 17,
  moving_average_window: 6,
  exponential_smoothing_alpha: 0.3,
  ensemble_weighted_base_models: JSON.stringify(["moving_average", "exponential_smoothing"]),
};

const baseBody = {
  experiment_id: 42,
  selected_industry: "制造业",
  selected_company: "示例企业",
  selected_product: "示例产品",
  data_window_train_start_index: 0,
  data_window_train_end_index: 11,
  data_window_evaluate_start_index: 12,
  data_window_evaluate_end_index: 17,
} satisfies TrainInput;

const successResult = () => ({
  status: "success",
  results: {
    metrics: { rmse: 1.2, mae: 0.8, r2: 0.91 },
  },
});

const productionCommitInput = () => ({
  experimentId: 42,
  studentId: 20240002,
  modelType: "ma" as const,
  runId: "run-ma-v1",
  sessionId: "production-session-v1",
  forecastSteps: 6,
  expectedProductionRevision: "revision-old",
  productionRevision: "revision-new",
  completedStepIds: ["initialize", "fit_model", "finalize"],
  stepPlan: [
    { id: "initialize", label: "确认材料" },
    { id: "fit_model", label: "拟合模型" },
    { id: "finalize", label: "提交产物" },
  ],
  attempt: {
    attemptId: "production-attempt-v1",
    artifactPath: "/models/run-ma-v1-production-new.pkl",
    artifactSha256: "b".repeat(64),
    committedStatePath: "/checkpoints/final.pkl",
    committedStateSha256: "c".repeat(64),
    outputPayload: { prepared: true },
  },
});

const createBarrier = () => {
  let release!: () => void;
  const promise = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { promise, release };
};

describe("model run transactional activation", () => {
  beforeEach(() => {
    query.mockReset();
    poolQuery.mockReset();
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    release.mockClear();
    destroy.mockClear();
    getConnection.mockReset();
    getConnection.mockResolvedValue(connection);
  });

  test("loads a complete current experiment patch without rewriting the model run", async () => {
    poolQuery.mockResolvedValueOnce([[
      {
        ...baseExperiment,
        state_version: 11,
        moving_average_completed: 1,
        ensemble_weighted_completed: 1,
        ensemble_weighted_base_models: JSON.stringify(["moving_average", "exponential_smoothing"]),
        selected_best_model: "ensemble_weighted",
        production_plan_completed: 1,
      },
    ]]);

    const patch = await loadCurrentExperimentStatePatch(42, 20240002);

    expect(patch).toMatchObject({
      state_version: 11,
      moving_average_completed: true,
      ensemble_weighted_completed: true,
      ensemble_weighted_base_models: ["moving_average", "exponential_smoothing"],
      selected_best_model: "ensemble_weighted",
      production_plan_completed: true,
    });
    expect(poolQuery).toHaveBeenCalledWith(
      expect.stringContaining("FROM experiment_status"),
      [42, 20240002],
    );
    expect(beginTransaction).not.toHaveBeenCalled();
  });

  test("activates the immutable run and authoritative experiment patch in one transaction", async () => {
    const body: TrainInput = { ...baseBody, moving_average_window: 6 };
    const requestHash = buildModelTrainingRequestHash({
      modelType: "ma",
      body,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      experimentRevision: buildExperimentTrainingRevision("ma", baseExperiment, body),
      baseRunFingerprint: {},
    });
    query.mockImplementation(async (sql: unknown) => {
      const statement = String(sql);
      if (statement.includes("FROM users")) {
        return [[{ user_id: 20240002, role: "Student" }]];
      }
      if (statement.includes("SELECT *") && statement.includes("FROM experiment_status")) {
        return [[baseExperiment]];
      }
      if (statement.includes("SELECT state_version")) {
        return [[{ state_version: 5 }]];
      }
      return [{ affectedRows: 1 }];
    });
    const resultPayload = successResult();

    const committed = await commitModelTrainingRun({
      runId: "run-ma-v1",
      sessionId: null,
      experimentId: 42,
      studentId: 20240002,
      modelType: "ma",
      requestHash,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      body,
      result: resultPayload,
      artifactPath: "/models/run-ma-v1.pkl",
      artifactSha256: "a".repeat(64),
    });

    expect(committed).toEqual({
      runId: "run-ma-v1",
      stateVersion: 5,
      reusedExistingRun: false,
    });
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledTimes(1);
    const insertRunCall = query.mock.calls.find(([sql]) =>
      String(sql).includes("INSERT INTO model_training_runs")
    );
    expect(insertRunCall).toBeDefined();
    expect(String(insertRunCall?.[0])).toContain("artifact_sha256");
    expect(insertRunCall?.[1]).toContain("a".repeat(64));
    const statements = query.mock.calls.map(([sql]) => String(sql));
    const userLockIndex = statements.findIndex((sql) =>
      sql.includes("FROM users") && sql.includes("FOR UPDATE")
    );
    const experimentLockIndex = statements.findIndex((sql) =>
      sql.includes("FROM experiment_status") && sql.includes("FOR UPDATE")
    );
    const insertRunIndex = statements.findIndex((sql) =>
      sql.includes("INSERT INTO model_training_runs")
    );
    expect(userLockIndex).toBeGreaterThanOrEqual(0);
    expect(userLockIndex).toBeLessThan(experimentLockIndex);
    expect(experimentLockIndex).toBeLessThan(insertRunIndex);
    expect(query.mock.calls[userLockIndex]?.[1]).toEqual([20240002]);
    expect(query.mock.calls.some(([sql]) => String(sql).includes("UPDATE experiment_status"))).toBe(true);
    expect(resultPayload.experiment_state_patch).toMatchObject({
      moving_average_completed: true,
      moving_average_window: 6,
      moving_average_metrics_rmse: 1.2,
      state_version: 5,
    });
  });

  test("rejects a missing or non-student identity before locking the experiment", async () => {
    query.mockImplementation(async (sql: unknown) => {
      const statement = String(sql);
      if (statement.includes("FROM users")) {
        return [[{ user_id: 20240002, role: "Assistant" }]];
      }
      throw new Error(`Unexpected query after identity rejection: ${statement}`);
    });

    await expect(commitModelTrainingRun({
      runId: "run-ma-v1",
      sessionId: null,
      experimentId: 42,
      studentId: 20240002,
      modelType: "ma",
      requestHash: "request-v1",
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      body: { ...baseBody, moving_average_window: 6 },
      result: successResult(),
      artifactPath: "/models/run-ma-v1.pkl",
    })).rejects.toThrow("学生账号不存在或身份已变化");

    expect(query).toHaveBeenCalledTimes(1);
    expect(String(query.mock.calls[0]?.[0])).toContain("FROM users");
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
  });

  test("destroys an ambiguous COMMIT connection without rolling back or releasing it", async () => {
    const body: TrainInput = { ...baseBody, moving_average_window: 6 };
    const requestHash = buildModelTrainingRequestHash({
      modelType: "ma",
      body,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      experimentRevision: buildExperimentTrainingRevision("ma", baseExperiment, body),
      baseRunFingerprint: {},
    });
    query.mockImplementation(async (sql: unknown) => {
      const statement = String(sql);
      if (statement.includes("FROM users")) {
        return [[{ user_id: 20240002, role: "Student" }]];
      }
      if (statement.includes("SELECT *") && statement.includes("FROM experiment_status")) {
        return [[baseExperiment]];
      }
      if (statement.includes("SELECT state_version")) {
        return [[{ state_version: 5 }]];
      }
      return [{ affectedRows: 1 }];
    });
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));

    await expect(commitModelTrainingRun({
      runId: "run-ma-ambiguous",
      sessionId: "guided-ma-ambiguous",
      experimentId: 42,
      studentId: 20240002,
      modelType: "ma",
      requestHash,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      body,
      result: successResult(),
      artifactPath: "/models/run-ma-ambiguous.pkl",
      artifactSha256: "a".repeat(64),
    })).rejects.toMatchObject({
      code: "MODEL_COMMIT_OUTCOME_UNKNOWN",
      retryable: true,
    });

    expect(rollback).not.toHaveBeenCalled();
    expect(release).not.toHaveBeenCalled();
    expect(destroy).toHaveBeenCalledTimes(1);
  });

  test("updates and conditionally clears production artifact integrity metadata", async () => {
    poolQuery.mockResolvedValue([{ affectedRows: 1 }]);

    expect(await setProductionArtifactForRun(
      "run-ma-v1",
      "/models/run-ma-v1-production.pkl",
      6,
      "revision-v1",
      "b".repeat(64),
      "observed-revision",
    )).toBe(true);
    expect(await backfillArtifactSha256ForRun("run-ma-v1", "a".repeat(64))).toBe(true);
    expect(await backfillProductionArtifactSha256ForRun(
      "run-ma-v1",
      "revision-v1",
      "b".repeat(64),
    )).toBe(true);
    expect(await clearProductionArtifactForRun("run-ma-v1", "revision-v1")).toBe(true);

    expect(poolQuery.mock.calls[0]?.[1]).toEqual([
      "/models/run-ma-v1-production.pkl",
      "b".repeat(64),
      6,
      "revision-v1",
      "run-ma-v1",
      "observed-revision",
      6,
    ]);
    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("production_revision <=> ?");
    expect(String(poolQuery.mock.calls[0]?.[0])).toContain("production_artifact_forecast_steps < ?");
    expect(String(poolQuery.mock.calls[1]?.[0])).toContain("artifact_sha256 IS NULL");
    expect(String(poolQuery.mock.calls[2]?.[0])).toContain("production_revision = ?");
    expect(String(poolQuery.mock.calls[3]?.[0])).toContain("production_artifact_path = NULL");
    expect(poolQuery.mock.calls[3]?.[1]).toEqual(["run-ma-v1", "revision-v1"]);
  });

  test("publishes the production artifact, attempt, and session in one locked transaction", async () => {
    query.mockImplementation(async (sqlValue: unknown) => {
      const sql = String(sqlValue);
      if (sql.includes("FROM experiment_status")) {
        return [[{ experiment_id: 42, student_id: 20240002, status: "In Progress" }]];
      }
      if (sql.includes("FROM model_training_runs")) {
        return [[{
          run_id: "run-ma-v1",
          experiment_id: 42,
          student_id: 20240002,
          model_type: "ma",
          production_artifact_path: "/models/run-ma-v1-production-old.pkl",
          production_artifact_forecast_steps: 4,
          production_revision: "revision-old",
          is_active: 1,
        }]];
      }
      if (sql.includes("FROM model_production_preparation_sessions")) {
        return [[{
          session_id: "production-session-v1",
          run_id: "run-ma-v1",
          experiment_id: 42,
          student_id: 20240002,
          model_type: "ma",
          status: "running",
          active_attempt_id: "production-attempt-v1",
        }]];
      }
      return [{ affectedRows: 1 }];
    });

    await expect(commitProductionPreparationResult(productionCommitInput()))
      .resolves.toEqual({
        committed: true,
        replacedProductionArtifactPath: "/models/run-ma-v1-production-old.pkl",
      });

    const statements = query.mock.calls.map(([sql]) => String(sql));
    const experimentLockIndex = statements.findIndex((sql) =>
      sql.includes("FROM experiment_status") && sql.includes("FOR UPDATE")
    );
    const runLockIndex = statements.findIndex((sql) =>
      sql.includes("FROM model_training_runs") && sql.includes("FOR UPDATE")
    );
    const sessionLockIndex = statements.findIndex((sql) =>
      sql.includes("FROM model_production_preparation_sessions") && sql.includes("FOR UPDATE")
    );
    const runUpdateIndex = statements.findIndex((sql) =>
      sql.includes("UPDATE model_training_runs")
    );
    const attemptUpdateIndex = statements.findIndex((sql) =>
      sql.includes("UPDATE model_production_preparation_step_attempts")
    );
    const sessionUpdateIndex = statements.findIndex((sql) =>
      sql.includes("UPDATE model_production_preparation_sessions")
    );
    const orderedStatementIndexes = [
      experimentLockIndex,
      runLockIndex,
      sessionLockIndex,
      runUpdateIndex,
      attemptUpdateIndex,
      sessionUpdateIndex,
    ];
    expect(orderedStatementIndexes.every((index) => index >= 0)).toBeTrue();
    expect(orderedStatementIndexes).toEqual(
      [...orderedStatementIndexes].sort((left, right) => left - right),
    );
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledTimes(1);
  });

  test("treats a production publish COMMIT acknowledgement loss as outcome-unknown", async () => {
    query.mockImplementation(async (sqlValue: unknown) => {
      const sql = String(sqlValue);
      if (sql.includes("FROM experiment_status")) {
        return [[{ experiment_id: 42, student_id: 20240002, status: "In Progress" }]];
      }
      if (sql.includes("FROM model_training_runs")) {
        return [[{
          run_id: "run-ma-v1",
          experiment_id: 42,
          student_id: 20240002,
          model_type: "ma",
          production_artifact_path: "/models/run-ma-v1-production-old.pkl",
          production_artifact_forecast_steps: 4,
          production_revision: "revision-old",
          is_active: 1,
        }]];
      }
      if (sql.includes("FROM model_production_preparation_sessions")) {
        return [[{
          session_id: "production-session-v1",
          run_id: "run-ma-v1",
          experiment_id: 42,
          student_id: 20240002,
          model_type: "ma",
          status: "running",
          active_attempt_id: "production-attempt-v1",
        }]];
      }
      return [{ affectedRows: 1 }];
    });
    commit.mockRejectedValueOnce(new Error("production COMMIT acknowledgement lost"));

    await expect(commitProductionPreparationResult(productionCommitInput()))
      .rejects.toMatchObject({
        code: "MODEL_COMMIT_OUTCOME_UNKNOWN",
        retryable: true,
      });

    expect(rollback).not.toHaveBeenCalled();
    expect(release).not.toHaveBeenCalled();
    expect(destroy).toHaveBeenCalledTimes(1);
  });

  test("reuses an artifact under the same experiment, run, session lock order", async () => {
    query.mockImplementation(async (sqlValue: unknown) => {
      const sql = String(sqlValue);
      if (sql.includes("FROM experiment_status")) {
        return [[{ experiment_id: 42, student_id: 20240002, status: "In Progress" }]];
      }
      if (sql.includes("FROM model_training_runs")) {
        return [[{
          run_id: "run-ma-v1",
          experiment_id: 42,
          student_id: 20240002,
          model_type: "ma",
          production_artifact_path: "/models/run-ma-v1-production-old.pkl",
          production_artifact_forecast_steps: 8,
          production_revision: "revision-old",
          is_active: 1,
        }]];
      }
      if (sql.includes("FROM model_production_preparation_sessions")) {
        return [[{
          session_id: "production-session-v1",
          run_id: "run-ma-v1",
          experiment_id: 42,
          student_id: 20240002,
          model_type: "ma",
          status: "ready",
          active_attempt_id: null,
        }]];
      }
      return [{ affectedRows: 1 }];
    });
    const input = productionCommitInput();

    await expect(commitProductionPreparationResult({
      ...input,
      productionRevision: input.expectedProductionRevision,
      attempt: undefined,
    })).resolves.toEqual({
      committed: true,
      replacedProductionArtifactPath: null,
    });

    const statements = query.mock.calls.map(([sql]) => String(sql));
    const orderedLockIndexes = [
      statements.findIndex((sql) =>
        sql.includes("FROM experiment_status") && sql.includes("FOR UPDATE")
      ),
      statements.findIndex((sql) =>
        sql.includes("FROM model_training_runs") && sql.includes("FOR UPDATE")
      ),
      statements.findIndex((sql) =>
        sql.includes("FROM model_production_preparation_sessions") && sql.includes("FOR UPDATE")
      ),
    ];
    expect(orderedLockIndexes.every((index) => index >= 0)).toBeTrue();
    expect(orderedLockIndexes).toEqual(
      [...orderedLockIndexes].sort((left, right) => left - right),
    );
    expect(statements.some((sql) => sql.includes("UPDATE model_training_runs"))).toBeFalse();
    expect(statements.some((sql) =>
      sql.includes("UPDATE model_production_preparation_sessions")
    )).toBeTrue();
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
  });

  test("report-first barrier prevents a pre-existing preparation from publishing after completion", async () => {
    const experimentReadStarted = createBarrier();
    const reportCommitted = createBarrier();
    const state = {
      experimentStatus: "In Progress",
      runRevision: "revision-old",
      runUpdated: false,
    };
    const raceQuery = mock(async (sqlValue: unknown) => {
      const sql = String(sqlValue);
      if (sql.includes("FROM experiment_status")) {
        experimentReadStarted.release();
        await reportCommitted.promise;
        return [[{
          experiment_id: 42,
          student_id: 20240002,
          status: state.experimentStatus,
        }]];
      }
      if (sql.includes("UPDATE model_training_runs")) state.runUpdated = true;
      return [{ affectedRows: 1 }];
    });
    const raceConnection = {
      query: raceQuery,
      beginTransaction: mock(async () => {}),
      commit: mock(async () => {}),
      rollback: mock(async () => {}),
      release: mock(() => {}),
    };
    getConnection.mockResolvedValueOnce(raceConnection as typeof connection);

    const preparation = commitProductionPreparationResult(productionCommitInput());
    await experimentReadStarted.promise;
    state.experimentStatus = "Completed";
    reportCommitted.release();

    await expect(preparation).resolves.toEqual({
      committed: false,
      reason: "experiment_completed",
    });
    expect(state.runUpdated).toBeFalse();
    expect(state.runRevision).toBe("revision-old");
    expect(raceConnection.rollback).toHaveBeenCalledTimes(1);
    expect(raceConnection.commit).not.toHaveBeenCalled();
  });

  test("preparation-first barrier makes a report with old evidence reject instead of completing stale", async () => {
    const preparationLockedExperiment = createBarrier();
    const allowPreparationToContinue = createBarrier();
    const preparationCommitted = createBarrier();
    const state = {
      experimentStatus: "In Progress",
      runRevision: "revision-old",
      pendingRunRevision: null as string | null,
    };
    const raceQuery = mock(async (sqlValue: unknown, params: unknown[] = []) => {
      const sql = String(sqlValue);
      if (sql.includes("FROM experiment_status")) {
        preparationLockedExperiment.release();
        await allowPreparationToContinue.promise;
        return [[{ experiment_id: 42, student_id: 20240002, status: state.experimentStatus }]];
      }
      if (sql.includes("FROM model_training_runs")) {
        return [[{
          run_id: "run-ma-v1",
          experiment_id: 42,
          student_id: 20240002,
          model_type: "ma",
          production_artifact_path: "/models/run-ma-v1-production-old.pkl",
          production_artifact_forecast_steps: 4,
          production_revision: state.runRevision,
          is_active: 1,
        }]];
      }
      if (sql.includes("FROM model_production_preparation_sessions")) {
        return [[{
          session_id: "production-session-v1",
          run_id: "run-ma-v1",
          experiment_id: 42,
          student_id: 20240002,
          model_type: "ma",
          status: "running",
          active_attempt_id: "production-attempt-v1",
        }]];
      }
      if (sql.includes("UPDATE model_training_runs")) {
        state.pendingRunRevision = String(params[3]);
      }
      return [{ affectedRows: 1 }];
    });
    const raceCommit = mock(async () => {
      state.runRevision = state.pendingRunRevision ?? state.runRevision;
      preparationCommitted.release();
    });
    const raceConnection = {
      query: raceQuery,
      beginTransaction: mock(async () => {}),
      commit: raceCommit,
      rollback: mock(async () => {}),
      release: mock(() => {}),
    };
    getConnection.mockResolvedValueOnce(raceConnection as typeof connection);

    const preparation = commitProductionPreparationResult(productionCommitInput());
    await preparationLockedExperiment.promise;
    const reportWithOldEvidence = (async () => {
      await preparationCommitted.promise;
      if (state.runRevision !== "revision-old") return "stale_evidence" as const;
      state.experimentStatus = "Completed";
      return "completed" as const;
    })();
    allowPreparationToContinue.release();

    await expect(preparation).resolves.toMatchObject({ committed: true });
    await expect(reportWithOldEvidence).resolves.toBe("stale_evidence");
    expect(state.runRevision).toBe("revision-new");
    expect(state.experimentStatus).toBe("In Progress");
    expect(raceCommit).toHaveBeenCalledTimes(1);
  });

  test("returns the current authoritative experiment state when a completed session reuses its active run", async () => {
    const experiment = {
      ...baseExperiment,
      moving_average_completed: 1,
      moving_average_metrics_rmse: 1.2,
      moving_average_metrics_mae: 0.8,
      moving_average_metrics_r2: 0.91,
      ensemble_weighted_completed: 0,
      ensemble_weighted_base_models: null,
      selected_best_model: null,
      production_plan_completed: 0,
      production_mps_table: null,
    };
    query.mockImplementation(async (sql: unknown) => {
      const statement = String(sql);
      if (statement.includes("FROM users")) {
        return [[{ user_id: 20240002, role: "Student" }]];
      }
      if (statement.includes("SELECT *") && statement.includes("FROM experiment_status")) {
        return [[experiment]];
      }
      if (statement.includes("SELECT *") && statement.includes("FROM model_training_runs")) {
        return [[{ run_id: "run-ma-v1", is_active: 1 }]];
      }
      return [{ affectedRows: 1 }];
    });
    const resultPayload = successResult();

    const committed = await commitModelTrainingRun({
      runId: "run-ma-v1",
      sessionId: "guided-ma-v1",
      experimentId: 42,
      studentId: 20240002,
      modelType: "ma",
      requestHash: "request-v1",
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      body: { ...baseBody, moving_average_window: 6 },
      result: resultPayload,
      artifactPath: "/models/run-ma-v1.pkl",
      guidedSession: {
        sessionId: "guided-ma-v1",
        currentStepId: null,
        completedStepIds: ["save_artifact"],
        stepOutputs: {},
        stepPlan: [],
        result: resultPayload,
      },
    });

    expect(committed).toEqual({
      runId: "run-ma-v1",
      stateVersion: 4,
      reusedExistingRun: true,
    });
    expect(resultPayload.experiment_state_patch).toMatchObject({
      moving_average_completed: true,
      moving_average_window: 6,
      moving_average_metrics_rmse: 1.2,
      moving_average_metrics_mae: 0.8,
      moving_average_metrics_r2: 0.91,
      ensemble_weighted_completed: false,
      ensemble_weighted_base_models: [],
      selected_best_model: null,
      production_plan_completed: false,
      production_mps_table: [],
      state_version: 4,
    });
    expect(query.mock.calls.some(([sql]) => String(sql).includes("UPDATE experiment_status"))).toBe(false);

    const sessionUpdate = query.mock.calls.find(([sql]) =>
      String(sql).includes("UPDATE model_guided_training_sessions")
    );
    expect(sessionUpdate).toBeDefined();
    const sessionUpdateValues = sessionUpdate?.[1] as unknown[];
    expect(JSON.parse(String(sessionUpdateValues[1])).experiment_state_patch).toEqual(
      resultPayload.experiment_state_patch,
    );
  });

  test("rejects idempotent reuse when the committed artifact checksum differs", async () => {
    query.mockImplementation(async (sql: unknown) => {
      const statement = String(sql);
      if (statement.includes("FROM users")) {
        return [[{ user_id: 20240002, role: "Student" }]];
      }
      if (statement.includes("SELECT *") && statement.includes("FROM experiment_status")) {
        return [[baseExperiment]];
      }
      if (statement.includes("SELECT *") && statement.includes("FROM model_training_runs")) {
        return [[{
          run_id: "run-ma-v1",
          is_active: 1,
          artifact_sha256: "a".repeat(64),
        }]];
      }
      return [{ affectedRows: 1 }];
    });

    await expect(commitModelTrainingRun({
      runId: "run-ma-v1",
      sessionId: "guided-ma-v1",
      experimentId: 42,
      studentId: 20240002,
      modelType: "ma",
      requestHash: "request-v1",
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      body: { ...baseBody, moving_average_window: 6 },
      result: successResult(),
      artifactPath: "/models/run-ma-v1.pkl",
      artifactSha256: "b".repeat(64),
    })).rejects.toThrow("已提交训练版本的模型产物校验和不一致");

    expect(commit).not.toHaveBeenCalled();
    expect(rollback).toHaveBeenCalledTimes(1);
  });

  test("rolls back an ensemble commit if a base run changes during training", async () => {
    const body: TrainInput = { ...baseBody, models: "ma,es" };
    const requestHash = buildModelTrainingRequestHash({
      modelType: "weighted_avg",
      body,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      experimentRevision: buildExperimentTrainingRevision("weighted_avg", baseExperiment, body),
      baseRunFingerprint: { ma: "ma-run-old", es: "es-run-1" },
    });
    query.mockImplementation(async (sql: unknown, values?: unknown[]) => {
      const statement = String(sql);
      if (statement.includes("FROM users")) {
        return [[{ user_id: 20240002, role: "Student" }]];
      }
      if (statement.includes("SELECT *") && statement.includes("FROM experiment_status")) {
        return [[baseExperiment]];
      }
      if (statement.includes("SELECT run_id") && statement.includes("model_type = ?")) {
        const modelType = values?.[2];
        return [[{ run_id: modelType === "ma" ? "ma-run-new" : "es-run-1" }]];
      }
      return [[]];
    });

    await expect(commitModelTrainingRun({
      runId: "ensemble-run-1",
      sessionId: null,
      experimentId: 42,
      studentId: 20240002,
      modelType: "weighted_avg",
      requestHash,
      datasetRevision: "dataset-v1",
      runtimeRevision: "runtime-v1",
      body,
      result: successResult(),
      artifactPath: "/models/ensemble-run-1.pkl",
    })).rejects.toMatchObject({
      message: "训练期间实验配置或基础模型版本已变化，请重新创建训练会话",
    });

    expect(commit).not.toHaveBeenCalled();
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(query.mock.calls.some(([sql]) => String(sql).includes("INSERT INTO model_training_runs"))).toBe(false);
    expect(query.mock.calls.some(([sql]) => String(sql).includes("UPDATE experiment_status"))).toBe(false);
  });
});
