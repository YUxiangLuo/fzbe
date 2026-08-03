import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

type RunState = {
  model_type: string;
  is_active: boolean;
};

type DatabaseState = {
  experiment: Record<string, unknown>;
  runs: RunState[];
};

const cloneState = (state: DatabaseState): DatabaseState =>
  structuredClone(state);

const initialState = (): DatabaseState => ({
  experiment: {
    experiment_id: 990002,
    student_id: 20240002,
    status: "In Progress",
    state_version: 3,
    start_time: "2026-03-08T09:00:00.000Z",
    last_activity_at: "2026-03-08T09:30:00.000Z",
    total_active_duration_seconds: 120,
    moving_average_window: 6,
    moving_average_completed: 1,
    production_plan_completed: 1,
    production_forecast_periods: 2,
    production_initial_inventory: 0,
    production_target_service_level: 0.95,
    production_safety_stock_z_score: 1.65,
    production_forecast_results: null,
    production_mps_table: null,
    production_capacity_mode: "auto",
    production_capacity_scenario: null,
    production_capacity: 120,
    production_custom_capacity: null,
  },
  runs: [
    "ma",
    "es",
    "arima",
    "lstm",
    "weighted_avg",
    "stacking",
    "boosting",
  ].map((model_type) => ({ model_type, is_active: true })),
});

let durableState = initialState();
let transactionState: DatabaseState | null = null;

const beginTransaction = mock(async () => {
  transactionState = cloneState(durableState);
});
const commit = mock(async () => {
  if (!transactionState) throw new Error("no transaction");
  durableState = cloneState(transactionState);
  transactionState = null;
});
const rollback = mock(async () => {
  transactionState = null;
});
const release = mock(() => {});

const connectionQuery = mock(async (sqlValue: string, params: unknown[] = []) => {
  if (!transactionState) throw new Error("query outside transaction");
  const sql = String(sqlValue);

  if (sql.includes("SELECT * FROM experiment_status")) {
    return [[structuredClone(transactionState.experiment)]];
  }

  if (sql.includes("UPDATE model_training_runs")) {
    const selectedModels = new Set(params.slice(2).map(String));
    for (const run of transactionState.runs) {
      if (selectedModels.has(run.model_type)) run.is_active = false;
    }
    return [{ affectedRows: selectedModels.size }];
  }

  if (sql.includes("UPDATE experiment_status SET")) {
    const assignmentSource = sql.match(/SET (.+), state_version =/s)?.[1];
    if (!assignmentSource) throw new Error(`unsupported experiment update: ${sql}`);
    const fields = assignmentSource
      .split(", ")
      .map((assignment) => assignment.split(" = ")[0]!);
    fields.forEach((field, index) => {
      transactionState!.experiment[field] = params[index];
    });
    transactionState.experiment.state_version =
      Number(transactionState.experiment.state_version) + 1;
    return [{ affectedRows: 1 }];
  }

  throw new Error(`unsupported transactional query: ${sql}`);
});

const connection = {
  query: connectionQuery,
  beginTransaction,
  commit,
  rollback,
  release,
};

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const serviceModulePath = path.join(
  srcRoot,
  "modules",
  "experiments",
  "experiments.service.ts",
);

mock.module(configModulePath, () => ({
  env: {
    DB_HOST: "127.0.0.1",
    DB_USER: "test",
    DB_PASSWORD: "test",
    DB_DATABASE: "test",
    JWT_SECRET: "test-jwt-secret",
    PORT: 3001,
  },
  connectionLimit: 64,
  pool: {
    getConnection: mock(async () => connection),
    query: mock(async () => [[]]),
  },
}));

const { updateExperiment } = await import(
  `${serviceModulePath}?experiments-service-atomicity-test`
);

describe("updateExperiment transaction atomicity", () => {
  beforeEach(() => {
    durableState = initialState();
    transactionState = null;
    connectionQuery.mockClear();
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    release.mockClear();
  });

  test("rolls back an invalid production plan without changing experiment or runs", async () => {
    const before = cloneState(durableState);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 3,
      production_plan_completed: true,
      production_forecast_periods: 6,
      production_initial_inventory: 0,
      production_target_service_level: 0.95,
      production_safety_stock_z_score: 1.65,
      production_forecast_results: [{
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
      }],
      production_mps_table: [{
        period: 1,
        period_label: "期 1",
        demand_forecast: 100,
        safety_stock: 0,
        planned_production: 100,
        beginning_inventory: 0,
        production_output: 100,
        ending_inventory: 0,
        stockout: 0,
        service_level: 1,
      }],
      production_capacity_mode: "auto",
      production_capacity: 120,
      production_custom_capacity: null,
    });

    expect(result).toMatchObject({ error: "invalid_production_plan" });
    expect(durableState).toEqual(before);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
  });

  test("commits configuration invalidation, run deactivation, and one version increment together", async () => {
    const result = await updateExperiment(990002, 20240002, {
      expected_version: 3,
      moving_average_window: 7,
    });

    expect(result).toHaveProperty("data.state_version", 4);
    expect(durableState.experiment).toMatchObject({
      state_version: 4,
      moving_average_window: 7,
      moving_average_completed: 0,
      production_plan_completed: 0,
      production_forecast_results: null,
      production_mps_table: null,
    });
    expect(
      durableState.runs
        .filter((run) => ["ma", "weighted_avg", "stacking", "boosting"]
          .includes(run.model_type))
        .every((run) => !run.is_active),
    ).toBeTrue();
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
  });

  test("keeps a new configuration submitted atomically with a context change", async () => {
    const result = await updateExperiment(990002, 20240002, {
      expected_version: 3,
      selected_product: "新品",
      moving_average_window: 7,
    });

    expect(result).toHaveProperty("data.state_version", 4);
    expect(durableState.experiment).toMatchObject({
      state_version: 4,
      selected_product: "新品",
      moving_average_window: 7,
      moving_average_completed: 0,
      production_plan_completed: 0,
    });
    expect(durableState.runs.every((run) => !run.is_active)).toBeTrue();
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
  });

  test("rejects a stale version without changing experiment or active runs", async () => {
    const before = cloneState(durableState);

    const result = await updateExperiment(990002, 20240002, {
      expected_version: 2,
      moving_average_window: 7,
    });

    expect(result).toMatchObject({ error: "conflict" });
    expect(durableState).toEqual(before);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
  });
});
