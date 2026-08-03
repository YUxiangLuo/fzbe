import { describe, expect, test } from "bun:test";
import {
  getCompletedProductionPlanValidationError,
  parseProductionForecastResults,
} from "../../../src/modules/experiments/experiments.schema";
import {
  buildCanonicalProductionForecasts,
  buildCanonicalProductionMps,
  buildCanonicalProductionPlanSnapshot,
} from "../../../scripts/production-plan-fixture";
import { issueProductionForecastEvidence } from "../../../src/modules/experiments/production-forecast-evidence";

describe("production-plan seed fixture", () => {
  test("builds a six-period canonical v2 snapshot with exactly replayable MPS rows", () => {
    const plan = buildCanonicalProductionPlanSnapshot(
      Array.from({ length: 6 }, (_, index) => ({
        prediction: 126 + index * 6,
        std_dev: 8 + index * 0.5,
      })),
      {
        initialInventory: 120,
        targetServiceLevel: 0.95,
        zScore: 1.65,
        capacity: 140,
      },
    );

    expect(plan.production_forecast_periods).toBe(6);
    expect(plan.production_forecast_results).toHaveLength(6);
    expect(plan.production_mps_table).toHaveLength(6);
    expect(parseProductionForecastResults(
      plan.production_forecast_results,
    )).toEqual(plan.production_forecast_results);
    expect(getCompletedProductionPlanValidationError({
      status: "In Progress",
      ...plan,
      experiment_id: 990001,
      student_id: 20240001,
      selected_best_model: "ma",
      production_forecast_evidence: issueProductionForecastEvidence({
        experiment_id: 990001,
        student_id: 20240001,
        model_type: "ma",
        run_id: "fixture-run",
        production_revision: "fixture-revision",
        forecast_steps: plan.production_forecast_periods,
        forecasts: plan.production_forecast_results,
      }),
      production_capacity_mode: "custom",
      production_capacity_scenario: null,
      production_custom_capacity: plan.production_capacity,
    })).toBeNull();
    expect(plan.production_mps_table.map((row) => row.period_label)).toEqual([
      "期 1",
      "期 2",
      "期 3",
      "期 4",
      "期 5",
      "期 6",
    ]);
  });

  test("derives periods from the forecast array instead of a separate constant", () => {
    const plan = buildCanonicalProductionPlanSnapshot([
      { prediction: 126, std_dev: 8 },
      { prediction: 132, std_dev: 9 },
    ], {
      initialInventory: 120,
      targetServiceLevel: 0.95,
      zScore: 1.65,
      capacity: 140,
    });

    expect(plan.production_forecast_periods).toBe(2);
  });

  test("rejects non-finite or structurally invalid fixture inputs", () => {
    expect(() => buildCanonicalProductionForecasts([
      { prediction: Number.NaN, std_dev: 8 },
    ])).toThrow("prediction");
    expect(() => buildCanonicalProductionMps(
      [{ prediction: 100, std_dev: 5 }],
      { initialInventory: 0, zScore: 1.65, capacity: 12.5 },
    )).toThrow("capacity must be an integer");
  });
});
