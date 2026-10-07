import { buildCanonicalProductionMps as buildServerCanonicalProductionMps } from "../src/modules/experiments/experiments.schema";

const NORMAL_95 = 1.959963984540054;
const NORMAL_99 = 2.3263478740408408;

export interface ProductionForecastSeed {
  prediction: number;
  std_dev: number;
}

export interface CanonicalProductionForecast extends ProductionForecastSeed {
  upper_error_p99: number;
  uncertainty_source: "fallback";
  uncertainty_reason: "synthetic_fixture_normal_approximation";
  calibration_source: "synthetic_fixture";
  calibration_mean_error: null;
  calibration_count: null;
  bias_diagnostic_source: null;
  interval_lower: number;
  interval_upper: number;
  interval_level: 0.95;
  interval_kind: "fallback_normal_approximation";
  coverage_claim: "none";
}

export interface CanonicalProductionMpsRow {
  period: number;
  period_label: string;
  demand_forecast: number;
  safety_stock: number;
  forecast_quantity: number;
  planned_production: number;
  beginning_inventory: number;
  production_output: number;
  ending_inventory: number;
  stockout: number;
  service_level: number;
}

export interface CanonicalProductionPlanSnapshot {
  production_plan_completed: true;
  production_forecast_periods: number;
  production_initial_inventory: number;
  production_target_service_level: number;
  production_safety_stock_z_score: number;
  production_forecast_results: CanonicalProductionForecast[];
  production_mps_table: CanonicalProductionMpsRow[];
  production_capacity_mode: "auto";
  production_capacity_scenario: null;
  production_capacity: number;
  production_custom_capacity: null;
}

const requireNonnegativeFinite = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(`${label} must be a nonnegative finite number`);
  }
  return value;
};

export function buildCanonicalProductionForecasts(
  points: readonly ProductionForecastSeed[],
): CanonicalProductionForecast[] {
  if (points.length === 0) {
    throw new Error("production forecast fixture must contain at least one point");
  }
  return points.map((point, index) => {
    const prediction = requireNonnegativeFinite(
      point.prediction,
      `production forecast ${index + 1} prediction`,
    );
    const stdDev = requireNonnegativeFinite(
      point.std_dev,
      `production forecast ${index + 1} std_dev`,
    );
    return {
      prediction,
      std_dev: stdDev,
      upper_error_p99: NORMAL_99 * stdDev,
      uncertainty_source: "fallback",
      uncertainty_reason: "synthetic_fixture_normal_approximation",
      calibration_source: "synthetic_fixture",
      calibration_mean_error: null,
      calibration_count: null,
      bias_diagnostic_source: null,
      interval_lower: prediction - NORMAL_95 * stdDev,
      interval_upper: prediction + NORMAL_95 * stdDev,
      interval_level: 0.95,
      interval_kind: "fallback_normal_approximation",
      coverage_claim: "none",
    };
  });
}

export function buildCanonicalProductionMps(
  forecasts: readonly ProductionForecastSeed[],
  input: {
    initialInventory: number;
    zScore: number;
    capacity: number;
  },
): CanonicalProductionMpsRow[] {
  const initialInventory = requireNonnegativeFinite(
    input.initialInventory,
    "initial inventory",
  );
  const zScore = requireNonnegativeFinite(input.zScore, "safety-stock z-score");
  const capacity = requireNonnegativeFinite(input.capacity, "production capacity");
  if (!Number.isInteger(capacity)) {
    throw new Error("production capacity must be an integer");
  }

  const validatedForecasts = forecasts.map((forecast, index) => ({
    prediction: requireNonnegativeFinite(
      forecast.prediction,
      `production forecast ${index + 1} prediction`,
    ),
    std_dev: requireNonnegativeFinite(
      forecast.std_dev,
      `production forecast ${index + 1} std_dev`,
    ),
  }));
  // Single source of truth: the server-side canonical formula.  Seeds must
  // never drift from what `validateCompletedProductionPlan` replays.
  return buildServerCanonicalProductionMps({
    forecasts: validatedForecasts,
    zScore,
    initialInventory,
    capacity,
  });
}

export function buildCanonicalProductionPlanSnapshot(
  points: readonly ProductionForecastSeed[],
  input: {
    initialInventory: number;
    targetServiceLevel: number;
    zScore: number;
    capacity: number;
  },
): CanonicalProductionPlanSnapshot {
  const targetServiceLevel = requireNonnegativeFinite(
    input.targetServiceLevel,
    "target service level",
  );
  if (targetServiceLevel > 1) {
    throw new Error("target service level must not exceed 1");
  }
  const forecasts = buildCanonicalProductionForecasts(points);
  const mpsTable = buildCanonicalProductionMps(forecasts, input);

  return {
    production_plan_completed: true,
    production_forecast_periods: forecasts.length,
    production_initial_inventory: input.initialInventory,
    production_target_service_level: targetServiceLevel,
    production_safety_stock_z_score: input.zScore,
    production_forecast_results: forecasts,
    production_mps_table: mpsTable,
    production_capacity_mode: "auto",
    production_capacity_scenario: null,
    production_capacity: input.capacity,
    production_custom_capacity: null,
  };
}
