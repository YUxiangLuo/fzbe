import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";
import {
  buildCanonicalProductionMps,
  buildProductionCapacityScenarios,
  updateExperimentSchema,
} from "../../../src/modules/experiments/experiments.schema";
import { issueProductionForecastEvidence } from "../../../src/modules/experiments/production-forecast-evidence";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const experimentsServiceModulePath = path.join(srcRoot, "modules", "experiments", "experiments.service.ts");
const freshExperimentsServiceModulePath = `${experimentsServiceModulePath}?experiments-service-test`;
const validProductionPredictionPoint = {
  prediction: 100,
  std_dev: 12,
  upper_error_p99: 28,
  uncertainty_source: "model",
  calibration_source: "arima_forecast_distribution",
  calibration_mean_error: 1.5,
  calibration_count: 8,
  bias_diagnostic_source: "outer_selection_assessment",
  interval_lower: 76.48,
  interval_upper: 123.52,
  interval_level: 0.95,
  interval_kind: "model_prediction_interval",
  coverage_claim: "model_nominal",
} as const;
const validProductionMpsRow = {
  period: 1,
  period_label: "期 1",
  demand_forecast: 100,
  safety_stock: 0,
  forecast_quantity: 100,
  planned_production: 100,
  beginning_inventory: 0,
  production_output: 100,
  ending_inventory: 0,
  stockout: 0,
  service_level: 1,
} as const;
const legacyProductionPredictionPoint = {
  prediction: 100,
  std_dev: 12,
} as const;

const buildSignedProductionPlan = () => {
  const forecasts = [
    { ...validProductionPredictionPoint },
    { ...validProductionPredictionPoint },
  ];
  const capacity = buildProductionCapacityScenarios(forecasts, 1.65).normal;
  const runId = "experiments-service-run";
  const productionRevision = "experiments-service-revision";
  return {
    runId,
    productionRevision,
    patch: {
      production_plan_completed: true,
      production_forecast_periods: forecasts.length,
      production_initial_inventory: 0,
      production_target_service_level: 0.95,
      production_safety_stock_z_score: 1.65,
      production_forecast_results: forecasts,
      production_forecast_evidence: issueProductionForecastEvidence({
        experiment_id: 990002,
        student_id: 20240002,
        model_type: "arima",
        run_id: runId,
        production_revision: productionRevision,
        forecast_steps: forecasts.length,
        forecasts,
      }),
      production_mps_table: buildCanonicalProductionMps({
        forecasts,
        zScore: 1.65,
        initialInventory: 0,
        capacity,
      }),
      production_capacity_mode: "scenario" as const,
      production_capacity_scenario: "normal" as const,
      production_capacity: capacity,
      production_custom_capacity: null,
    },
  };
};

const connectionQuery = mock(async () => [[]]);
const beginTransaction = mock(async () => {});
const commit = mock(async () => {});
const rollback = mock(async () => {});
const release = mock(() => {});
const destroy = mock(() => {});
const getConnection = mock(async () => ({
  query: connectionQuery,
  beginTransaction,
  commit,
  rollback,
  release,
  destroy,
}));
const poolQuery = mock(async () => [[]]);
const mockedEnv = {
  DB_HOST: process.env.DB_HOST ?? "127.0.0.1",
  DB_USER: process.env.DB_USER ?? "test",
  DB_PASSWORD: process.env.DB_PASSWORD ?? "test",
  DB_DATABASE: process.env.DB_DATABASE ?? "test",
  DB_TIMEZONE: "Z",
  JWT_SECRET: "test-jwt-secret",
  ALLOW_LEGACY_PRODUCTION_PLAN_WRITES: true,
  PORT: Number(process.env.PORT ?? "3001"),
};

mock.module(configModulePath, () => ({
  env: mockedEnv,
  connectionLimit: 64,
  pool: {
    getConnection,
    query: poolQuery,
  },
}));

const {
  createExperiment,
  createExperimentEvent,
  getActiveExperiment,
  retryRejectedExperiment,
  updateExperiment,
  normalizeExperimentRow,
} = await import(freshExperimentsServiceModulePath);

describe("experiments service", () => {
  beforeEach(() => {
    mockedEnv.ALLOW_LEGACY_PRODUCTION_PLAN_WRITES = true;
    connectionQuery.mockReset();
    poolQuery.mockReset();
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    release.mockClear();
    destroy.mockClear();
    getConnection.mockClear();
  });

  test("returns a completed experiment when its report was never created", async () => {
    const completedExperiment = {
      experiment_id: 990002,
      student_id: 20240002,
      status: "Completed",
      current_step: 8,
    };
    poolQuery
      .mockResolvedValueOnce([[completedExperiment]])
      .mockResolvedValueOnce([[]]);

    const result = await getActiveExperiment(20240002);

    expect(result).toMatchObject({
      experiment_id: 990002,
      student_id: 20240002,
      status: "Completed",
    });
    expect(poolQuery.mock.calls[1]?.[1]).toEqual([990002, 20240002]);
  });

  test("does not return a completed experiment after its report exists", async () => {
    poolQuery
      .mockResolvedValueOnce([[
        {
          experiment_id: 990002,
          student_id: 20240002,
          status: "Completed",
        },
      ]])
      .mockResolvedValueOnce([[{ report_id: 77 }]]);

    expect(await getActiveExperiment(20240002)).toBeNull();
  });

  test("creates the first official experiment while holding the student creation lock", async () => {
    connectionQuery
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([[{ user_id: 20240002 }]])
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([{ insertId: 990002 }])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([[
        {
          experiment_id: 990002,
          student_id: 20240002,
          retry_of_experiment_id: null,
          status: "Not Started",
          current_step: 1,
          highest_completed_step: 0,
          state_version: 0,
        },
      ]]);

    const result = await createExperiment(20240002, { selected_industry: "制造业" });

    expect(result).toEqual({
      data: expect.objectContaining({
        experiment_id: 990002,
        student_id: 20240002,
        retry_of_experiment_id: null,
      }),
    });
    expect(connectionQuery.mock.calls[0]?.[0]).toBe("SET TRANSACTION ISOLATION LEVEL READ COMMITTED");
    expect(String(connectionQuery.mock.calls[1]?.[0])).toContain("users");
    expect(String(connectionQuery.mock.calls[1]?.[0])).toContain("FOR UPDATE");
    expect(String(connectionQuery.mock.calls[2]?.[0])).toContain("ORDER BY experiment_id DESC");
    expect(String(connectionQuery.mock.calls[3]?.[0])).toContain("INSERT INTO experiment_status");
    expect(connectionQuery.mock.calls[3]?.[1]).toEqual(["制造业", 20240002]);
    expect(commit).toHaveBeenCalledTimes(1);
  });

  const prepareCreationQueries = (beforeInsert?: () => void) => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM users")) return [[{ user_id: 20240002 }]];
      if (sql.includes("ORDER BY experiment_id")) return [[]];
      if (sql.startsWith("INSERT INTO experiment_status")) {
        beforeInsert?.();
        return [{ insertId: 990002 }];
      }
      if (sql.includes("SELECT * FROM experiment_status")) {
        return [[{ experiment_id: 990002, student_id: 20240002, state_version: 0 }]];
      }
      return [[]];
    });
  };

  test("retries creation only after a transaction has rolled back", async () => {
    let inserts = 0;
    prepareCreationQueries(() => {
      if (++inserts === 1) throw Object.assign(new Error("deadlocked insert"), { code: "ER_LOCK_DEADLOCK" });
    });
    const result = await createExperiment(20240002, {});
    expect(result).toMatchObject({ data: { experiment_id: 990002 } });
    expect(getConnection).toHaveBeenCalledTimes(2);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(release).toHaveBeenCalledTimes(2);
    expect(inserts).toBe(2);
  });

  test("does not retry creation after an ambiguous commit acknowledgement", async () => {
    prepareCreationQueries();
    commit.mockRejectedValueOnce(Object.assign(new Error("commit connection lost"), { code: "ER_LOCK_DEADLOCK" }));
    await expect(createExperiment(20240002, {})).rejects.toMatchObject({ code: "MODEL_COMMIT_OUTCOME_UNKNOWN" });
    expect(getConnection).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(release).not.toHaveBeenCalled();
  });

  test("destroys the connection and stops retrying if rollback fails", async () => {
    prepareCreationQueries();
    beginTransaction.mockRejectedValueOnce(Object.assign(new Error("deadlock"), { code: "ER_LOCK_DEADLOCK" }));
    rollback.mockRejectedValueOnce(new Error("rollback connection lost"));
    await expect(createExperiment(20240002, {})).rejects.toThrow("事务回滚失败");
    expect(getConnection).toHaveBeenCalledTimes(1);
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(release).not.toHaveBeenCalled();
  });

  test("blocks ordinary creation when any official experiment already exists", async () => {
    connectionQuery
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([[{ user_id: 20240002 }]])
      .mockResolvedValueOnce([[
        {
          experiment_id: 990002,
          student_id: 20240002,
          status: "Completed",
        },
      ]]);

    const result = await createExperiment(20240002, {});

    expect(result).toEqual({
      error: "already_exists",
      data: expect.objectContaining({ experiment_id: 990002 }),
    });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(connectionQuery.mock.calls.some(([sql]) =>
      String(sql).includes("INSERT INTO experiment_status")
    )).toBeFalse();
  });

  test("creates a blank retry only for the student's latest rejected report", async () => {
    connectionQuery
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([[{ user_id: 20240002 }]])
      .mockResolvedValueOnce([[
        {
          report_id: 77,
          experiment_id: 990002,
          report_student_id: 20240002,
          report_status: "rejected",
          experiment_student_id: 20240002,
          experiment_status: "Completed",
        },
      ]])
      .mockResolvedValueOnce([[
        {
          experiment_id: 990002,
          student_id: 20240002,
          status: "Completed",
        },
      ]])
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([{ insertId: 990003 }])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([[
        {
          experiment_id: 990003,
          student_id: 20240002,
          retry_of_experiment_id: 990002,
          status: "Not Started",
          current_step: 1,
          highest_completed_step: 0,
          state_version: 0,
          selected_industry: null,
        },
      ]]);

    const result = await retryRejectedExperiment(20240002, {
      rejected_report_id: 77,
    });

    expect(result).toEqual({
      created: true,
      data: expect.objectContaining({
        experiment_id: 990003,
        student_id: 20240002,
        retry_of_experiment_id: 990002,
        selected_industry: null,
      }),
    });
    expect(String(connectionQuery.mock.calls[2]?.[0])).toContain("experiment_reports");
    expect(String(connectionQuery.mock.calls[2]?.[0])).toContain("FOR UPDATE");
    expect(connectionQuery.mock.calls[2]?.[1]).toEqual([77]);
    expect(String(connectionQuery.mock.calls[5]?.[0])).toContain(
      "INSERT INTO experiment_status (student_id, retry_of_experiment_id)",
    );
    expect(connectionQuery.mock.calls[5]?.[1]).toEqual([20240002, 990002]);
    expect(commit).toHaveBeenCalledTimes(1);
  });

  test("returns the existing retry idempotently while it remains the latest attempt", async () => {
    const existingRetry = {
      experiment_id: 990003,
      student_id: 20240002,
      retry_of_experiment_id: 990002,
      status: "In Progress",
      current_step: 2,
    };
    connectionQuery
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([[{ user_id: 20240002 }]])
      .mockResolvedValueOnce([[
        {
          report_id: 77,
          experiment_id: 990002,
          report_student_id: 20240002,
          report_status: "rejected",
          experiment_student_id: 20240002,
          experiment_status: "Completed",
        },
      ]])
      .mockResolvedValueOnce([[existingRetry]])
      .mockResolvedValueOnce([[existingRetry]])
      .mockResolvedValueOnce([[]]);

    const result = await retryRejectedExperiment(20240002, {
      rejected_report_id: 77,
    });

    expect(result).toEqual({
      created: false,
      data: expect.objectContaining({ experiment_id: 990003, retry_of_experiment_id: 990002 }),
    });
    expect(connectionQuery).toHaveBeenCalledTimes(6);
    expect(String(connectionQuery.mock.calls[5]?.[0])).toContain("FROM experiment_reports");
    expect(String(connectionQuery.mock.calls[5]?.[0])).toContain("FOR UPDATE");
    expect(connectionQuery.mock.calls[5]?.[1]).toEqual([990003, 20240002]);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
  });

  test("returns a latest Completed retry idempotently while report recovery is still pending", async () => {
    const completedRetry = {
      experiment_id: 990003,
      student_id: 20240002,
      retry_of_experiment_id: 990002,
      status: "Completed",
      current_step: 7,
    };
    connectionQuery
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([[{ user_id: 20240002 }]])
      .mockResolvedValueOnce([[
        {
          report_id: 77,
          experiment_id: 990002,
          report_student_id: 20240002,
          report_status: "rejected",
          experiment_student_id: 20240002,
          experiment_status: "Completed",
        },
      ]])
      .mockResolvedValueOnce([[completedRetry]])
      .mockResolvedValueOnce([[completedRetry]])
      .mockResolvedValueOnce([[]]);

    await expect(retryRejectedExperiment(20240002, { rejected_report_id: 77 }))
      .resolves.toEqual({
        created: false,
        data: expect.objectContaining({
          experiment_id: 990003,
          retry_of_experiment_id: 990002,
          status: "Completed",
        }),
      });
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
  });

  for (const reportStatus of ["submitted", "graded", "rejected"] as const) {
    test(`rejects the old parent once the latest retry has a ${reportStatus} report`, async () => {
      const completedRetry = {
        experiment_id: 990003,
        student_id: 20240002,
        retry_of_experiment_id: 990002,
        status: "Completed",
        current_step: 7,
      };
      connectionQuery
        .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([[{ user_id: 20240002 }]])
        .mockResolvedValueOnce([[
          {
            report_id: 77,
            experiment_id: 990002,
            report_student_id: 20240002,
            report_status: "rejected",
            experiment_student_id: 20240002,
            experiment_status: "Completed",
          },
        ]])
        .mockResolvedValueOnce([[completedRetry]])
        .mockResolvedValueOnce([[completedRetry]])
        .mockResolvedValueOnce([[
          { report_id: 78, status: reportStatus },
        ]]);

      await expect(retryRejectedExperiment(20240002, { rejected_report_id: 77 }))
        .resolves.toEqual({
          error: "report_not_latest",
          data: expect.objectContaining({
            experiment_id: 990003,
            retry_of_experiment_id: 990002,
            status: "Completed",
          }),
        });
      expect(String(connectionQuery.mock.calls[5]?.[0])).toContain("FROM experiment_reports");
      expect(String(connectionQuery.mock.calls[5]?.[0])).toContain("FOR UPDATE");
      expect(connectionQuery.mock.calls[5]?.[1]).toEqual([990003, 20240002]);
      expect(rollback).toHaveBeenCalledTimes(1);
      expect(commit).not.toHaveBeenCalled();
    });
  }

  test("rejects a stale ancestor retry after a later retry generation becomes latest", async () => {
    const firstRetry = {
      experiment_id: 990003,
      student_id: 20240002,
      retry_of_experiment_id: 990002,
      status: "Completed",
      current_step: 7,
    };
    const latestRetry = {
      experiment_id: 990004,
      student_id: 20240002,
      retry_of_experiment_id: 990003,
      status: "In Progress",
      current_step: 2,
    };
    connectionQuery
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([[{ user_id: 20240002 }]])
      .mockResolvedValueOnce([[
        {
          report_id: 77,
          experiment_id: 990002,
          report_student_id: 20240002,
          report_status: "rejected",
          experiment_student_id: 20240002,
          experiment_status: "Completed",
        },
      ]])
      .mockResolvedValueOnce([[latestRetry]])
      .mockResolvedValueOnce([[firstRetry]]);

    expect(await retryRejectedExperiment(20240002, { rejected_report_id: 77 }))
      .toEqual({
        error: "report_not_latest",
        data: expect.objectContaining({
          experiment_id: 990004,
          retry_of_experiment_id: 990003,
        }),
      });
    // A later descendant remains the decisive stale-request signal; do not
    // query the older child's report after that lineage check has failed.
    expect(connectionQuery).toHaveBeenCalledTimes(5);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
  });

  test("rejects retrying another student's report without leaking its state", async () => {
    connectionQuery
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([[{ user_id: 20240002 }]])
      .mockResolvedValueOnce([[
        {
          report_id: 77,
          experiment_id: 990002,
          report_student_id: 20240003,
          report_status: "rejected",
          experiment_student_id: 20240003,
          experiment_status: "Completed",
        },
      ]]);

    expect(await retryRejectedExperiment(20240002, { rejected_report_id: 77 }))
      .toEqual({ error: "report_not_found" });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
  });

  test("rejects a rejected report once another official attempt is latest", async () => {
    connectionQuery
      .mockResolvedValueOnce([[]])
      .mockResolvedValueOnce([[{ user_id: 20240002 }]])
      .mockResolvedValueOnce([[
        {
          report_id: 77,
          experiment_id: 990002,
          report_student_id: 20240002,
          report_status: "rejected",
          experiment_student_id: 20240002,
          experiment_status: "Completed",
        },
      ]])
      .mockResolvedValueOnce([[
        {
          experiment_id: 990004,
          student_id: 20240002,
          status: "In Progress",
        },
      ]])
      .mockResolvedValueOnce([[]]);

    expect(await retryRejectedExperiment(20240002, { rejected_report_id: 77 }))
      .toEqual({
        error: "report_not_latest",
        data: expect.objectContaining({ experiment_id: 990004 }),
      });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
  });

  test("locks the user before the experiment and blocks events for completed experiments", async () => {
    connectionQuery
      .mockResolvedValueOnce([[{ user_id: 20240002 }]])
      .mockResolvedValueOnce([[
        { student_id: 20240002, status: "Completed" },
      ]]);

    const result = await createExperimentEvent(990002, 20240002, {
      step_order: 5,
      event_type: "STARTED",
    });

    expect(result).toEqual({ error: "completed" });
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledTimes(1);
    expect(String(connectionQuery.mock.calls[0]?.[0])).toContain("FROM users");
    expect(String(connectionQuery.mock.calls[0]?.[0])).toContain("FOR UPDATE");
    expect(connectionQuery.mock.calls[0]?.[1]).toEqual([20240002]);
    expect(String(connectionQuery.mock.calls[1]?.[0])).toContain("FROM experiment_status");
    expect(String(connectionQuery.mock.calls[1]?.[0])).toContain("FOR UPDATE");
  });

  test("stops before locking an experiment when the authenticated user was deleted", async () => {
    connectionQuery.mockResolvedValueOnce([[]]);

    const result = await createExperimentEvent(990002, 20240002, {
      step_order: 1,
      event_type: "STARTED",
    });

    expect(result).toEqual({ error: "not_found" });
    expect(connectionQuery).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
  });

  test("retries a deadlocked step event with a fresh transaction", async () => {
    let experimentLockAttempts = 0;
    connectionQuery.mockImplementation(async (sqlValue: unknown) => {
      const sql = String(sqlValue);
      if (sql.includes("FROM users")) {
        return [[{ user_id: 20240002 }]];
      }
      if (sql.includes("FROM experiment_status")) {
        experimentLockAttempts += 1;
        if (experimentLockAttempts === 1) {
          throw Object.assign(new Error("Deadlock found when trying to get lock"), {
            code: "ER_LOCK_DEADLOCK",
            errno: 1213,
            sqlState: "40001",
          });
        }
        return [[{ student_id: 20240002, status: "In Progress" }]];
      }
      if (sql.includes("INSERT INTO experiment_step_events")) {
        return [{ insertId: 321, affectedRows: 1 }];
      }
      throw new Error(`Unexpected query: ${sql}`);
    });

    await expect(createExperimentEvent(990002, 20240002, {
      step_order: 2,
      event_type: "STARTED",
    })).resolves.toEqual({ data: { event_id: 321 } });

    expect(experimentLockAttempts).toBe(2);
    expect(getConnection).toHaveBeenCalledTimes(2);
    expect(beginTransaction).toHaveBeenCalledTimes(2);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(release).toHaveBeenCalledTimes(2);
    const lockStatements = connectionQuery.mock.calls
      .map(([sql]) => String(sql))
      .filter((sql) => sql.includes("FROM users") || sql.includes("FROM experiment_status"));
    expect(lockStatements).toEqual([
      expect.stringContaining("FROM users"),
      expect.stringContaining("FROM experiment_status"),
      expect.stringContaining("FROM users"),
      expect.stringContaining("FROM experiment_status"),
    ]);
  });

  test("blocks experiment updates for completed experiments after locking the experiment row", async () => {
    connectionQuery.mockResolvedValueOnce([
      [{
        experiment_id: 990002,
        student_id: 20240002,
        status: "Completed",
        start_time: "2026-03-08T09:00:00.000Z",
        last_activity_at: "2026-03-08T09:30:00.000Z",
        total_active_duration_seconds: 120,
      }],
    ]);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 0,
      selected_company: "企业A",
    });

    expect(result).toEqual({ error: "completed" });
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledTimes(1);
  });

  test("defensively rejects lineage-only updates when the controller schema is bypassed", async () => {
    connectionQuery.mockResolvedValueOnce([
      [{
        experiment_id: 990002,
        student_id: 20240002,
        retry_of_experiment_id: 990001,
        status: "In Progress",
        state_version: 0,
        start_time: "2026-03-08T09:00:00.000Z",
        last_activity_at: "2026-03-08T09:30:00.000Z",
        total_active_duration_seconds: 120,
      }],
    ]);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 0,
      retry_of_experiment_id: null,
    } as never);

    expect(result).toEqual({ error: "no_fields" });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(connectionQuery).toHaveBeenCalledTimes(1);
    expect(connectionQuery.mock.calls.some(([sql]) =>
      String(sql).includes("UPDATE experiment_status SET")
    )).toBeFalse();
  });

  test("rejects selecting a best model that has no authoritative completed run", async () => {
    connectionQuery.mockResolvedValueOnce([
      [{
        experiment_id: 990002,
        student_id: 20240002,
        status: "In Progress",
        state_version: 3,
        moving_average_completed: 0,
        start_time: "2026-03-08T09:00:00.000Z",
        last_activity_at: "2026-03-08T09:30:00.000Z",
        total_active_duration_seconds: 120,
      }],
    ]);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 3,
      selected_best_model: "ma",
    });

    expect(result).toEqual({ error: "invalid_best_model" });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(connectionQuery).toHaveBeenCalledTimes(1);
  });

  test("rejects selecting a model in the same patch that invalidates its training result", async () => {
    connectionQuery.mockResolvedValueOnce([
      [{
        experiment_id: 990002,
        student_id: 20240002,
        status: "In Progress",
        state_version: 3,
        moving_average_completed: 1,
        moving_average_window: 6,
        start_time: "2026-03-08T09:00:00.000Z",
        last_activity_at: "2026-03-08T09:30:00.000Z",
        total_active_duration_seconds: 120,
      }],
    ]);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 3,
      moving_average_window: 7,
      selected_best_model: "ma",
    });

    expect(result).toEqual({ error: "invalid_best_model" });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
  });

  test("normalizes legacy model identifiers in experiment rows", () => {
    const normalized = normalizeExperimentRow({
      selected_base_models: JSON.stringify(["ma", "exp", "arima", "lstm", "ma"]),
      selected_ensemble_models: JSON.stringify([
        "ensemble_weighted",
        "boosting_ensemble",
        "stacking",
        "weighted_avg",
      ]),
      ensemble_weighted_base_models: JSON.stringify(["ma", "es"]),
      ensemble_boosting_base_models: JSON.stringify(["exp", "arima"]),
      ensemble_stacking_base_models: JSON.stringify(["moving_average", "lstm"]),
    } as any);

    expect(normalized.selected_base_models).toEqual([
      "moving_average",
      "exponential_smoothing",
      "arima",
      "lstm",
    ]);
    expect(normalized.selected_ensemble_models).toEqual([
      "weighted_ensemble",
      "boosting_ensemble",
      "stacking_ensemble",
    ]);
    expect(normalized.ensemble_weighted_base_models).toEqual([
      "moving_average",
      "exponential_smoothing",
    ]);
    expect(normalized.ensemble_boosting_base_models).toEqual([
      "exponential_smoothing",
      "arima",
    ]);
    expect(normalized.ensemble_stacking_base_models).toEqual([
      "moving_average",
      "lstm",
    ]);
  });

  test("normalizes legacy string model selections in experiment rows", () => {
    const normalized = normalizeExperimentRow({
      selected_base_models: "ma, exp lstm",
      selected_ensemble_models: "ensemble_weighted stacking",
      ensemble_weighted_base_models: "ma,es",
    } as any);

    expect(normalized.selected_base_models).toEqual([
      "moving_average",
      "exponential_smoothing",
      "lstm",
    ]);
    expect(normalized.selected_ensemble_models).toEqual([
      "weighted_ensemble",
      "stacking_ensemble",
    ]);
    expect(normalized.ensemble_weighted_base_models).toEqual([
      "moving_average",
      "exponential_smoothing",
    ]);
  });

  test("normalizes timezone-free database timestamps using DB_TIMEZONE", () => {
    expect(normalizeExperimentRow({
      start_time: "2026-08-04 12:34:56",
      last_activity_at: "2026-08-04 12:35:56.123456",
      completion_time: "0000-00-00 00:00:00",
    } as any)).toMatchObject({
      start_time: "2026-08-04T12:34:56.000Z",
      last_activity_at: "2026-08-04T12:35:56.123Z",
      completion_time: null,
    });
  });

  test("normalizes valid stored forecasts and discards malformed forecast JSON", () => {
    expect(normalizeExperimentRow({
      production_forecast_results: JSON.stringify([validProductionPredictionPoint]),
    } as any).production_forecast_results).toEqual([validProductionPredictionPoint]);
    expect(normalizeExperimentRow({
      status: "Completed",
      production_forecast_results: JSON.stringify([{ prediction: 100 }]),
    } as any).production_forecast_results).toBeNull();
  });

  test("returns active v1 forecasts while the rollout bridge is enabled", () => {
    const v1Json = JSON.stringify([legacyProductionPredictionPoint]);
    expect(normalizeExperimentRow({
      status: "Completed",
      production_forecast_results: v1Json,
      production_mps_table: JSON.stringify([{ period: 1 }]),
    } as any)).toMatchObject({
      production_forecast_results: [legacyProductionPredictionPoint],
      production_mps_table: [{ period: 1 }],
    });
    expect(normalizeExperimentRow({
      status: "In Progress",
      production_forecast_results: v1Json,
    } as any).production_forecast_results).toEqual([legacyProductionPredictionPoint]);
    expect(normalizeExperimentRow({
      status: "Completed",
      production_forecast_results: JSON.stringify([{
        ...legacyProductionPredictionPoint,
        interval_lower: 76.48,
      }]),
    } as any).production_forecast_results).toBeNull();
  });

  test("normalizes valid stored MPS rows and discards malformed legacy MPS JSON", () => {
    expect(normalizeExperimentRow({
      production_mps_table: JSON.stringify([validProductionMpsRow]),
    } as any).production_mps_table).toEqual([validProductionMpsRow]);
    const { forecast_quantity: _omitted, ...rowWithoutForecastQuantity } = validProductionMpsRow;
    expect(normalizeExperimentRow({
      status: "In Progress",
      production_mps_table: JSON.stringify([rowWithoutForecastQuantity]),
    } as any).production_mps_table).toBeNull();
    expect(normalizeExperimentRow({
      production_mps_table: JSON.stringify([{ period: 1 }]),
    } as any).production_mps_table).toBeNull();
  });

  test("canonicalizes legacy model identifiers before saving experiment updates", async () => {
    connectionQuery
      .mockResolvedValueOnce([
        [{
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          start_time: "2026-03-08T09:00:00.000Z",
          last_activity_at: "2026-03-08T09:30:00.000Z",
          total_active_duration_seconds: 120,
          state_version: 0,
        }],
      ])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([
        [{
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          selected_base_models: JSON.stringify(["moving_average", "lstm"]),
          selected_ensemble_models: JSON.stringify(["weighted_ensemble"]),
          ensemble_weighted_base_models: JSON.stringify([
            "moving_average",
            "exponential_smoothing",
          ]),
        }],
      ]);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 0,
      selected_base_models: ["ma", "lstm"],
      selected_ensemble_models: ["ensemble_weighted"],
      ensemble_weighted_base_models: ["ma", "es"],
    });

    const updateCall = connectionQuery.mock.calls.find(([sql]) =>
      String(sql).includes("UPDATE experiment_status SET")
    );
    expect(updateCall?.[0]).toContain("UPDATE experiment_status SET");
    expect(updateCall?.[1]).toEqual(expect.arrayContaining([
      JSON.stringify(["moving_average", "lstm"]),
      JSON.stringify(["weighted_ensemble"]),
      JSON.stringify(["moving_average", "exponential_smoothing"]),
    ]));
    expect(result).toEqual({
      data: expect.objectContaining({
        selected_base_models: ["moving_average", "lstm"],
        selected_ensemble_models: ["weighted_ensemble"],
        ensemble_weighted_base_models: ["moving_average", "exponential_smoothing"],
      }),
    });
  });

  test("canonicalizes legacy string model selections before saving experiment updates", async () => {
    connectionQuery
      .mockResolvedValueOnce([
        [{
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          start_time: "2026-03-08T09:00:00.000Z",
          last_activity_at: "2026-03-08T09:30:00.000Z",
          total_active_duration_seconds: 120,
          state_version: 0,
        }],
      ])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([
        [{
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          selected_base_models: JSON.stringify(["moving_average", "lstm"]),
          selected_ensemble_models: JSON.stringify(["weighted_ensemble"]),
        }],
      ]);

    await updateExperiment(990002, 20240002, {
      expected_version: 0,
      selected_base_models: "ma lstm" as never,
      selected_ensemble_models: "ensemble_weighted" as never,
    });

    const updateCall = connectionQuery.mock.calls.find(([sql]) =>
      String(sql).includes("UPDATE experiment_status SET")
    );
    expect(updateCall?.[1]).toEqual(expect.arrayContaining([
      JSON.stringify(["moving_average", "lstm"]),
      JSON.stringify(["weighted_ensemble"]),
    ]));
  });

  test("persists only schema-validated production forecast points as JSON", async () => {
    connectionQuery
      .mockResolvedValueOnce([
        [{
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          start_time: "2026-03-08T09:00:00.000Z",
          last_activity_at: "2026-03-08T09:30:00.000Z",
          total_active_duration_seconds: 120,
          state_version: 0,
        }],
      ])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([
        [{
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          production_forecast_results: JSON.stringify([validProductionPredictionPoint]),
        }],
      ]);
    const input = updateExperimentSchema.parse({
      expected_version: 0,
      production_forecast_results: [validProductionPredictionPoint],
    });

    const result = await updateExperiment(990002, 20240002, input);

    expect(result).toEqual({
      error: "invalid_production_plan",
      code: "PRODUCTION_PLAN_INVALID",
      message: expect.stringContaining("完整覆盖"),
    });
    const updateCall = connectionQuery.mock.calls.find(([sql]) =>
      String(sql).includes("UPDATE experiment_status SET")
    );
    expect(updateCall).toBeUndefined();
  });

  test("persists a signed incomplete v2 forecast draft bound to the selected model", async () => {
    const forecasts = [{ ...validProductionPredictionPoint }, { ...validProductionPredictionPoint }];
    const evidence = issueProductionForecastEvidence({
      experiment_id: 990002,
      student_id: 20240002,
      model_type: "arima",
      run_id: "draft-run",
      production_revision: "draft-revision",
      forecast_steps: forecasts.length,
      forecasts,
    });
    connectionQuery
      .mockResolvedValueOnce([
        [{
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          selected_best_model: "arima",
          start_time: "2026-03-08T09:00:00.000Z",
          last_activity_at: "2026-03-08T09:30:00.000Z",
          total_active_duration_seconds: 120,
          state_version: 0,
        }],
      ])
      .mockResolvedValueOnce([
        [{
          run_id: "draft-run",
          production_revision: "draft-revision",
        }],
      ])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([
        [{
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          selected_best_model: "arima",
          production_forecast_results: JSON.stringify(forecasts),
          production_forecast_evidence: evidence,
        }],
      ]);
    const input = updateExperimentSchema.parse({
      expected_version: 0,
      production_forecast_periods: 2,
      production_forecast_results: forecasts,
      production_forecast_evidence: evidence,
    });

    const result = await updateExperiment(990002, 20240002, input);

    const updateCall = connectionQuery.mock.calls.find(([sql]) =>
      String(sql).includes("UPDATE experiment_status SET")
    );
    expect(updateCall?.[1]).toContain(JSON.stringify(forecasts));
    expect(updateCall?.[1]).toContain(evidence);
    expect(result).toEqual({
      data: expect.objectContaining({
        production_forecast_results: forecasts,
        production_forecast_evidence: evidence,
      }),
    });
  });

  test("defensively rejects malformed forecasts if a caller bypasses the controller schema", async () => {
    connectionQuery.mockResolvedValueOnce([
      [{
        experiment_id: 990002,
        student_id: 20240002,
        status: "In Progress",
        start_time: "2026-03-08T09:00:00.000Z",
        last_activity_at: "2026-03-08T09:30:00.000Z",
        total_active_duration_seconds: 120,
        state_version: 0,
      }],
    ]);

    await expect(updateExperiment(990002, 20240002, {
      expected_version: 0,
      production_forecast_results: [{ prediction: -1, std_dev: 12 }],
    } as never)).rejects.toThrow("production_forecast_results");
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(connectionQuery.mock.calls.some(([sql]) =>
      String(sql).includes("UPDATE experiment_status SET")
    )).toBeFalse();
  });

  test("rejects a completed production plan whose horizon and persisted outputs disagree", async () => {
    connectionQuery.mockResolvedValueOnce([
      [{
        experiment_id: 990002,
        student_id: 20240002,
        status: "In Progress",
        start_time: "2026-03-08T09:00:00.000Z",
        last_activity_at: "2026-03-08T09:30:00.000Z",
        total_active_duration_seconds: 120,
        state_version: 0,
      }],
    ]);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 0,
      production_plan_completed: true,
      production_forecast_periods: 6,
      production_initial_inventory: 0,
      production_target_service_level: 0.95,
      production_safety_stock_z_score: 1.65,
      production_forecast_results: [validProductionPredictionPoint],
      production_mps_table: [validProductionMpsRow],
      production_capacity_mode: "scenario",
      production_capacity_scenario: "normal",
      production_capacity: 120,
      production_custom_capacity: null,
    });

    expect(result).toEqual({
      error: "invalid_production_plan",
      code: "PRODUCTION_PLAN_INVALID",
      message: expect.stringContaining("完整覆盖"),
    });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(connectionQuery.mock.calls.some(([sql]) =>
      String(sql).includes("UPDATE experiment_status SET")
    )).toBeFalse();
  });

  test("locks and validates the signed forecast run before completing a production plan", async () => {
    const signedPlan = buildSignedProductionPlan();
    const experimentRow = {
      experiment_id: 990002,
      student_id: 20240002,
      status: "In Progress",
      selected_best_model: "arima",
      arima_completed: 1,
      start_time: "2026-03-08T09:00:00.000Z",
      last_activity_at: "2026-03-08T09:30:00.000Z",
      total_active_duration_seconds: 120,
      state_version: 0,
    };
    connectionQuery
      .mockResolvedValueOnce([[experimentRow]])
      .mockResolvedValueOnce([[{ // active model run
        run_id: signedPlan.runId,
        production_revision: signedPlan.productionRevision,
      }]])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([[{
        ...experimentRow,
        ...signedPlan.patch,
        state_version: 1,
      }]]);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 0,
      ...signedPlan.patch,
    });

    expect(result).toHaveProperty("data.production_plan_completed", true);
    const activeRunCall = connectionQuery.mock.calls.find(([sql]) =>
      String(sql).includes("SELECT run_id, production_revision")
    );
    expect(String(activeRunCall?.[0])).toContain("FOR UPDATE");
    expect(activeRunCall?.[1]).toEqual([990002, 20240002, "arima"]);
    expect(commit).toHaveBeenCalledTimes(1);
  });

  test("persists an exact v1 plan without querying evidence during the rollout bridge", async () => {
    const signedPlan = buildSignedProductionPlan();
    const legacyForecasts = signedPlan.patch.production_forecast_results.map((point) => {
      const {
        interval_lower: _lower,
        interval_upper: _upper,
        interval_level: _level,
        interval_kind: _kind,
        coverage_claim: _claim,
        ...legacyPoint
      } = point;
      return legacyPoint;
    });
    const { production_forecast_evidence: _evidence, ...planWithoutEvidence } = signedPlan.patch;
    const legacyPatch = {
      ...planWithoutEvidence,
      production_forecast_results: legacyForecasts,
    };
    const experimentRow = {
      experiment_id: 990002,
      student_id: 20240002,
      status: "In Progress",
      selected_best_model: "arima",
      arima_completed: 1,
      start_time: "2026-03-08T09:00:00.000Z",
      last_activity_at: "2026-03-08T09:30:00.000Z",
      total_active_duration_seconds: 120,
      state_version: 0,
    };
    connectionQuery
      .mockResolvedValueOnce([[experimentRow]])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([[
        {
          ...experimentRow,
          ...legacyPatch,
          production_forecast_results: JSON.stringify(legacyForecasts),
          state_version: 1,
        },
      ]]);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 0,
      ...legacyPatch,
    });

    expect(result).toMatchObject({
      data: {
        production_plan_completed: true,
        production_forecast_results: legacyForecasts,
      },
    });
    expect(connectionQuery.mock.calls.some(([sql]) =>
      String(sql).includes("SELECT run_id, production_revision")
    )).toBeFalse();
    expect(commit).toHaveBeenCalledTimes(1);
  });

  test("returns a dedicated rejection code for an active v1 write when the rollout bridge is off", async () => {
    mockedEnv.ALLOW_LEGACY_PRODUCTION_PLAN_WRITES = false;
    const signedPlan = buildSignedProductionPlan();
    const legacyForecasts = signedPlan.patch.production_forecast_results.map((point) => {
      const {
        interval_lower: _lower,
        interval_upper: _upper,
        interval_level: _level,
        interval_kind: _kind,
        coverage_claim: _claim,
        ...legacyPoint
      } = point;
      return legacyPoint;
    });
    const { production_forecast_evidence: _evidence, ...planWithoutEvidence } = signedPlan.patch;
    const experimentRow = {
      experiment_id: 990002,
      student_id: 20240002,
      status: "In Progress",
      selected_best_model: "arima",
      arima_completed: 1,
      start_time: "2026-03-08T09:00:00.000Z",
      last_activity_at: "2026-03-08T09:30:00.000Z",
      total_active_duration_seconds: 120,
      state_version: 0,
    };
    connectionQuery.mockResolvedValueOnce([[experimentRow]]);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 0,
      ...planWithoutEvidence,
      production_forecast_results: legacyForecasts,
    });

    expect(result).toEqual({
      error: "invalid_production_plan",
      code: "PRODUCTION_FORECAST_LEGACY_REJECTED",
      message: expect.stringContaining("重新生成"),
    });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(connectionQuery.mock.calls.some(([sql]) =>
      String(sql).includes("UPDATE experiment_status SET")
    )).toBeFalse();
  });

  test("rejects a signed plan when its production revision is no longer active", async () => {
    const signedPlan = buildSignedProductionPlan();
    connectionQuery
      .mockResolvedValueOnce([[
        {
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          selected_best_model: "arima",
          arima_completed: 1,
          start_time: "2026-03-08T09:00:00.000Z",
          last_activity_at: "2026-03-08T09:30:00.000Z",
          total_active_duration_seconds: 120,
          state_version: 0,
        },
      ]])
      .mockResolvedValueOnce([[
        {
          run_id: signedPlan.runId,
          production_revision: `${signedPlan.productionRevision}-new`,
        },
      ]]);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 0,
      ...signedPlan.patch,
    });

    expect(result).toMatchObject({
      error: "invalid_production_plan",
      code: "PRODUCTION_FORECAST_EVIDENCE_STALE",
    });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(connectionQuery.mock.calls.some(([sql]) =>
      String(sql).includes("UPDATE experiment_status SET")
    )).toBeFalse();
  });

  test("keeps server invalidation authoritative over dependent values in the same patch", async () => {
    connectionQuery
      .mockResolvedValueOnce([
        [{
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          state_version: 3,
          moving_average_window: 6,
          moving_average_completed: 1,
          production_plan_completed: 1,
          start_time: "2026-03-08T09:00:00.000Z",
          last_activity_at: "2026-03-08T09:30:00.000Z",
          total_active_duration_seconds: 120,
        }],
      ])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([
        [{
          experiment_id: 990002,
          student_id: 20240002,
          status: "In Progress",
          state_version: 4,
          moving_average_window: 7,
          moving_average_completed: 0,
          production_plan_completed: 0,
          production_forecast_results: null,
          production_mps_table: null,
        }],
      ]);

    await updateExperiment(990002, 20240002, {
      expected_version: 3,
      moving_average_window: 7,
      production_plan_completed: true,
    });

    const updateCall = connectionQuery.mock.calls.find(([sql]) =>
      String(sql).includes("UPDATE experiment_status SET")
    );
    const assignments = String(updateCall?.[0])
      .match(/SET (.+), state_version/s)?.[1]
      ?.split(", ")
      .map((assignment) => assignment.split(" = ")[0]);
    const values = (updateCall?.[1] as unknown[]) ?? [];
    const persisted = Object.fromEntries(
      (assignments ?? []).map((field, index) => [field, values[index]]),
    );

    expect(persisted.production_plan_completed).toBe(0);
    expect(persisted.production_forecast_results).toBeNull();
    expect(persisted.production_mps_table).toBeNull();
    expect(persisted.moving_average_completed).toBe(0);
    expect(commit).toHaveBeenCalledTimes(1);
  });

  test("preserves production_forecast_evidence when MySQL 8 returns key-reordered parsed JSON objects", async () => {
    const signedPlan = buildSignedProductionPlan();
    const {
      production_forecast_evidence: storedEvidence,
      ...patchWithoutEvidence
    } = signedPlan.patch;

    // Simulate MySQL 8's native JSON key ordering: shorter keys first, then lexicographical.
    const reorderKeysLikeMySql8 = (value: unknown): unknown => {
      if (Array.isArray(value)) return value.map(reorderKeysLikeMySql8);
      if (value && typeof value === "object") {
        const record = value as Record<string, unknown>;
        return Object.fromEntries(
          Object.keys(record)
            .sort((left, right) => left.length - right.length || left.localeCompare(right))
            .map((key) => [key, reorderKeysLikeMySql8(record[key])]),
        );
      }
      return value;
    };

    const existingRow = {
      experiment_id: 990002,
      student_id: 20240002,
      status: "In Progress",
      selected_best_model: "arima",
      arima_completed: 1,
      start_time: "2026-03-08T09:00:00.000Z",
      last_activity_at: "2026-03-08T09:30:00.000Z",
      total_active_duration_seconds: 120,
      state_version: 2,
      ...patchWithoutEvidence,
      production_forecast_results: reorderKeysLikeMySql8(signedPlan.patch.production_forecast_results),
      production_mps_table: reorderKeysLikeMySql8(signedPlan.patch.production_mps_table),
      production_forecast_evidence: storedEvidence,
    };

    const updatedMps = buildCanonicalProductionMps({
      forecasts: signedPlan.patch.production_forecast_results,
      zScore: signedPlan.patch.production_safety_stock_z_score,
      initialInventory: 25,
      capacity: signedPlan.patch.production_capacity,
    });

    connectionQuery
      .mockResolvedValueOnce([[existingRow]])
      .mockResolvedValueOnce([[{
        run_id: signedPlan.runId,
        production_revision: signedPlan.productionRevision,
      }]])
      .mockResolvedValueOnce([{ affectedRows: 1 }])
      .mockResolvedValueOnce([[{
        ...existingRow,
        production_initial_inventory: 25,
        production_mps_table: reorderKeysLikeMySql8(updatedMps),
        state_version: 3,
      }]]);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 2,
      ...patchWithoutEvidence,
      production_initial_inventory: 25,
      production_mps_table: updatedMps,
    });

    expect(result).toMatchObject({
      data: {
        production_plan_completed: true,
        production_initial_inventory: 25,
      },
    });
    const updateCall = connectionQuery.mock.calls.find(([sql]) =>
      String(sql).includes("UPDATE experiment_status SET")
    );
    expect(String(updateCall?.[0])).not.toContain("production_forecast_evidence = ?");
    expect(commit).toHaveBeenCalledTimes(1);
  });
});
