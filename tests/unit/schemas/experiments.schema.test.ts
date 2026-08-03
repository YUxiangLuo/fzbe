import { describe, test, expect } from "bun:test";
import {
  experimentIdParamSchema,
  createExperimentSchema,
  retryExperimentSchema,
  updateExperimentSchema,
  experimentEventSchema,
  STATUS_VALUES,
  BEST_MODEL_VALUES,
  classifyProductionForecastResults,
  buildCanonicalProductionMps,
  buildProductionCapacityScenarios,
  getCompletedProductionPlanValidationError,
  parseCompatibleProductionForecastResults,
  validateCompletedProductionPlan,
} from "../../../src/modules/experiments/experiments.schema";
import { issueProductionForecastEvidence } from "../../../src/modules/experiments/production-forecast-evidence";

const validProductionPredictionPoint = {
  prediction: 100,
  std_dev: 12,
  upper_error_p99: 28,
  uncertainty_source: "model",
  uncertainty_reason: "model forecast distribution",
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

const buildCurrentProductionSnapshot = () => {
  const forecasts = [
    { ...validProductionPredictionPoint },
    { ...validProductionPredictionPoint },
  ];
  const capacity = buildProductionCapacityScenarios(forecasts, 1.65).normal;
  return {
    experiment_id: 91,
    student_id: 92,
    status: "In Progress",
    selected_best_model: "arima",
    production_plan_completed: true,
    production_forecast_periods: forecasts.length,
    production_initial_inventory: 0,
    production_target_service_level: 0.95,
    production_safety_stock_z_score: 1.65,
    production_forecast_results: forecasts,
    production_forecast_evidence: issueProductionForecastEvidence({
      experiment_id: 91,
      student_id: 92,
      model_type: "arima",
      run_id: "schema-test-run",
      production_revision: "schema-test-revision",
      forecast_steps: forecasts.length,
      forecasts,
    }),
    production_mps_table: buildCanonicalProductionMps({
      forecasts,
      zScore: 1.65,
      initialInventory: 0,
      capacity,
    }),
    production_capacity_mode: "scenario",
    production_capacity_scenario: "normal",
    production_capacity: capacity,
    production_custom_capacity: null,
  };
};

describe("Experiments Schemas", () => {
  describe("experimentIdParamSchema", () => {
    test("should accept valid numeric string", () => {
      const result = experimentIdParamSchema.safeParse({ experimentId: "123" });
      expect(result.success).toBe(true);
    });

    test("should reject non-numeric string", () => {
      const result = experimentIdParamSchema.safeParse({ experimentId: "abc" });
      expect(result.success).toBe(false);
    });

    test("should reject decimal numbers", () => {
      const result = experimentIdParamSchema.safeParse({ experimentId: "12.5" });
      expect(result.success).toBe(false);
    });
  });

  describe("createExperimentSchema", () => {
    test("should accept minimal data", () => {
      const result = createExperimentSchema.safeParse({});
      expect(result.success).toBe(true);
    });

    test("should reject client-controlled status", () => {
      const result = createExperimentSchema.safeParse({
        status: "In Progress",
      });
      expect(result.success).toBe(false);
      expect(createExperimentSchema.safeParse({ retry_of_experiment_id: 91 }).success)
        .toBe(false);
    });

    test("should reject dependent workflow results on creation", () => {
      expect(createExperimentSchema.safeParse({ selected_best_model: "ma" }).success).toBe(false);
      expect(createExperimentSchema.safeParse({ production_plan_completed: true }).success).toBe(false);
      expect(createExperimentSchema.safeParse({
        production_forecast_results: [validProductionPredictionPoint],
      }).success).toBe(false);
      expect(createExperimentSchema.safeParse({ moving_average_completed: true }).success).toBe(false);
    });

    test("should accept current_step", () => {
      const result = createExperimentSchema.safeParse({
        current_step: 3,
      });
      expect(result.success).toBe(true);
    });

    test("should accept selection fields", () => {
      const result = createExperimentSchema.safeParse({
        selected_industry: "Manufacturing",
        selected_company: "ABC Corp",
        selected_product: "Widget A",
      });
      expect(result.success).toBe(true);
    });
  });

  describe("retryExperimentSchema", () => {
    test("accepts only a positive numeric rejected report id", () => {
      expect(retryExperimentSchema.parse({ rejected_report_id: 77 }))
        .toEqual({ rejected_report_id: 77 });
      expect(retryExperimentSchema.safeParse({ rejected_report_id: 0 }).success).toBe(false);
      expect(retryExperimentSchema.safeParse({ rejected_report_id: "77" }).success).toBe(false);
      expect(retryExperimentSchema.safeParse({ rejected_report_id: 77, current_step: 1 }).success)
        .toBe(false);
    });
  });

  describe("updateExperimentSchema", () => {
    test("should accept current_step update", () => {
      const result = updateExperimentSchema.safeParse({
        expected_version: 0,
        current_step: 3,
      });
      expect(result.success).toBe(true);
    });

    test("should reject status update", () => {
      const result = updateExperimentSchema.safeParse({
        expected_version: 0,
        status: "In Progress",
      });
      expect(result.success).toBe(false);
    });

    test("should reject retry lineage updates", () => {
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        retry_of_experiment_id: null,
      }).success).toBe(false);
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        retry_of_experiment_id: 91,
      }).success).toBe(false);
    });

    test("should reject Completed status", () => {
      const result = updateExperimentSchema.safeParse({
        expected_version: 0,
        status: "Completed",
      });
      expect(result.success).toBe(false);
    });

    test("should reject invalid status", () => {
      const result = updateExperimentSchema.safeParse({
        expected_version: 0,
        status: "invalid_status",
      });
      expect(result.success).toBe(false);
    });

    test("should accept selection fields", () => {
      const result = updateExperimentSchema.safeParse({
        expected_version: 0,
        selected_industry: "Manufacturing",
        selected_company: "ABC Corp",
        selected_product: "Widget A",
      });
      expect(result.success).toBe(true);
    });

    test("should accept highest_completed_step", () => {
      const result = updateExperimentSchema.safeParse({
        expected_version: 0,
        highest_completed_step: 5,
      });
      expect(result.success).toBe(true);
    });

    test("should accept multiple fields together", () => {
      const result = updateExperimentSchema.safeParse({
        expected_version: 0,
        current_step: 4,
        selected_industry: "Retail",
        highest_completed_step: 3,
      });
      expect(result.success).toBe(true);
    });

    test("should reject empty object", () => {
      const result = updateExperimentSchema.safeParse({});
      expect(result.success).toBe(false);
    });

    test("should reject client-controlled model results", () => {
      const result = updateExperimentSchema.safeParse({
        expected_version: 0,
        moving_average_completed: true,
        moving_average_window: 5,
        moving_average_metrics_rmse: 0.5,
      });
      expect(result.success).toBe(false);
    });

    test("should reject non-integer moving average windows", () => {
      const result = updateExperimentSchema.safeParse({
        expected_version: 0,
        moving_average_window: 2.5,
      });
      expect(result.success).toBe(false);
    });

    test("should reject moving average windows below 2", () => {
      const result = updateExperimentSchema.safeParse({
        expected_version: 0,
        moving_average_window: 1,
      });
      expect(result.success).toBe(false);
    });

    test("should accept best model selection", () => {
      const result = updateExperimentSchema.safeParse({
        expected_version: 0,
        selected_best_model: "lstm",
      });
      expect(result.success).toBe(true);
    });

    test("should accept complete ADF rows and reject incomplete or duplicate rows", () => {
      const validRow = {
        diff_order: 1,
        statistic: -3.84,
        p_value: 0.018,
        used_lags: 2,
        n_obs: 11,
        stationary: true,
        critical_values: { "1%": -3.75, "5%": -2.99, "10%": -2.64 },
      };
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        arima_adf_stationarity: [validRow],
      }).success).toBe(true);
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        arima_adf_stationarity: [{ diff_order: 1 }],
      }).success).toBe(false);
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        arima_adf_stationarity: [validRow, validRow],
      }).success).toBe(false);
    });

    test("should accept complete production prediction points, null, and empty arrays", () => {
      const direct = updateExperimentSchema.safeParse({
        expected_version: 0,
        production_forecast_results: [validProductionPredictionPoint],
      });
      expect(direct.success).toBe(true);
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        production_forecast_results: null,
      }).success).toBe(true);
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        production_forecast_results: [],
      }).success).toBe(true);

      const encoded = updateExperimentSchema.safeParse({
        expected_version: 0,
        production_forecast_results: JSON.stringify([validProductionPredictionPoint]),
      });
      expect(encoded.success).toBe(true);
      if (encoded.success) {
        expect(encoded.data.production_forecast_results).toEqual([validProductionPredictionPoint]);
      }
    });

    test("should accept a consistent v1 transport payload and reject malformed arrays", () => {
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        production_forecast_results: [legacyProductionPredictionPoint],
      }).success).toBe(true);

      const invalidPoints = [
        { ...validProductionPredictionPoint, calibration_count: null },
        { ...validProductionPredictionPoint, interval_lower: 130 },
        { ...validProductionPredictionPoint, interval_level: 0.9 },
        { ...validProductionPredictionPoint, interval_kind: "custom_interval" },
        { ...validProductionPredictionPoint, coverage_claim: "none" },
        { ...validProductionPredictionPoint, uncertainty_source: "unknown" },
      ];
      for (const invalidPoint of invalidPoints) {
        expect(updateExperimentSchema.safeParse({
          expected_version: 0,
          production_forecast_results: [invalidPoint],
        }).success).toBe(false);
      }
    });

    test("classifies only all-absent or all-present interval tuples at the transport boundary", () => {
      expect(classifyProductionForecastResults([legacyProductionPredictionPoint])).toBe("v1");
      expect(parseCompatibleProductionForecastResults([legacyProductionPredictionPoint]))
        .toEqual({ version: "v1", data: [legacyProductionPredictionPoint] });
      expect(parseCompatibleProductionForecastResults([
        { prediction: 100 },
      ])).toBeNull();
      expect(parseCompatibleProductionForecastResults([
        { prediction: -1, std_dev: 12 },
      ])).toBeNull();
      expect(parseCompatibleProductionForecastResults([
        { prediction: 100, std_dev: Number.POSITIVE_INFINITY },
      ])).toBeNull();
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        production_forecast_results: [legacyProductionPredictionPoint],
      }).success).toBe(true);

      const partialInterval = {
        ...legacyProductionPredictionPoint,
        interval_lower: 76.48,
      };
      expect(classifyProductionForecastResults([partialInterval])).toBe("partial");
      expect(classifyProductionForecastResults([
        legacyProductionPredictionPoint,
        validProductionPredictionPoint,
      ])).toBe("partial");
      expect(parseCompatibleProductionForecastResults([partialInterval])).toBeNull();
    });

    test("grandfathers v1 history while keeping the default active policy strict", () => {
      const snapshot = {
        production_plan_completed: true,
        production_forecast_periods: 2,
        production_forecast_results: [
          legacyProductionPredictionPoint,
          legacyProductionPredictionPoint,
        ],
        production_mps_table: [{ period: 1 }, { period: 2 }],
      };
      expect(getCompletedProductionPlanValidationError({
        ...snapshot,
        status: "Completed",
      })).toBeNull();
      expect(getCompletedProductionPlanValidationError({
        ...snapshot,
        status: "In Progress",
      })).toContain("旧版格式");
      expect(getCompletedProductionPlanValidationError({
        status: "Completed",
        production_plan_completed: true,
        production_forecast_periods: 1,
        production_forecast_results: [legacyProductionPredictionPoint],
        production_mps_table: [{ period: 1 }],
      })).toBeNull();
    });

    test("allows exact active v1 plans only through the rollout bridge", () => {
      const current = buildCurrentProductionSnapshot();
      const legacyForecasts = current.production_forecast_results.map((point) => {
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
      const legacySnapshot = {
        ...current,
        production_forecast_results: legacyForecasts,
        production_forecast_evidence: null,
      };

      expect(validateCompletedProductionPlan(legacySnapshot)).toMatchObject({
        ok: false,
        code: "PRODUCTION_FORECAST_LEGACY_REJECTED",
      });
      expect(validateCompletedProductionPlan({
        ...legacySnapshot,
        production_plan_completed: false,
      })).toMatchObject({
        ok: false,
        code: "PRODUCTION_FORECAST_LEGACY_REJECTED",
      });
      expect(validateCompletedProductionPlan({
        ...legacySnapshot,
        production_plan_completed: false,
      }, {
        allowLegacyV1Write: true,
      })).toEqual({ ok: true, evidenceClaims: null });
      expect(validateCompletedProductionPlan(legacySnapshot, {
        allowLegacyV1Write: true,
      })).toEqual({ ok: true, evidenceClaims: null });
      expect(validateCompletedProductionPlan({
        ...legacySnapshot,
        production_forecast_evidence: "mixed-v2-token",
      }, {
        allowLegacyV1Write: true,
      })).toMatchObject({
        ok: false,
        message: expect.stringContaining("不能携带"),
      });
      expect(validateCompletedProductionPlan({
        ...legacySnapshot,
        production_mps_table: legacySnapshot.production_mps_table.map((row, index) =>
          index === 0 ? { ...row, planned_production: row.planned_production + 1 } : row
        ),
      }, {
        allowLegacyV1Write: true,
      })).toMatchObject({
        ok: false,
        code: "PRODUCTION_PLAN_MISMATCH",
      });
    });

    test("requires signed server forecast evidence for every active completed plan", () => {
      const snapshot = buildCurrentProductionSnapshot();
      expect(validateCompletedProductionPlan(snapshot)).toMatchObject({
        ok: true,
        evidenceClaims: {
          experiment_id: 91,
          student_id: 92,
          model_type: "arima",
          forecast_steps: 2,
        },
      });
      expect(validateCompletedProductionPlan({
        ...snapshot,
        production_forecast_evidence: null,
      })).toMatchObject({
        ok: false,
        code: "PRODUCTION_FORECAST_EVIDENCE_REQUIRED",
      });
      expect(validateCompletedProductionPlan({
        ...snapshot,
        production_forecast_evidence: `${snapshot.production_forecast_evidence}x`,
      })).toMatchObject({
        ok: false,
        code: "PRODUCTION_FORECAST_EVIDENCE_INVALID",
      });
      expect(validateCompletedProductionPlan({
        ...snapshot,
        student_id: 93,
      })).toMatchObject({
        ok: false,
        code: "PRODUCTION_FORECAST_EVIDENCE_STALE",
      });

      for (const mismatchedSnapshot of [
        { ...snapshot, experiment_id: 999 },
        { ...snapshot, selected_best_model: "lstm" },
        {
          ...snapshot,
          production_forecast_periods: 3,
          production_forecast_results: [
            ...snapshot.production_forecast_results,
            { ...validProductionPredictionPoint },
          ],
        },
      ]) {
        expect(validateCompletedProductionPlan(mismatchedSnapshot)).toMatchObject({
          ok: false,
          code: "PRODUCTION_FORECAST_EVIDENCE_STALE",
        });
      }
    });

    test("permits incomplete drafts without treating fabricated outputs as completed", () => {
      expect(validateCompletedProductionPlan({
        production_plan_completed: false,
        production_forecast_results: [{ prediction: -999 }],
        production_mps_table: [{ arbitrary: true }],
      })).toEqual({ ok: true, evidenceClaims: null });
    });

    test("accepts a frontend forecast roundtrip when Python adds runner-only metadata", () => {
      const snapshot = buildCurrentProductionSnapshot();
      const runnerForecasts = snapshot.production_forecast_results.map((point, index) => ({
        ...point,
        future_diagnostic: {
          horizon: index + 1,
          source: "new-python-version",
        },
      }));
      const evidence = issueProductionForecastEvidence({
        experiment_id: snapshot.experiment_id,
        student_id: snapshot.student_id,
        model_type: "arima",
        run_id: "schema-test-run",
        production_revision: "schema-test-revision",
        forecast_steps: runnerForecasts.length,
        forecasts: runnerForecasts,
      });

      // parseModelPredictionResponse persists only the writable point fields,
      // while the evidence was issued over the backend response above.
      expect(validateCompletedProductionPlan({
        ...snapshot,
        production_forecast_evidence: evidence,
      })).toMatchObject({
        ok: true,
        evidenceClaims: {
          forecast_steps: runnerForecasts.length,
        },
      });
    });

    test("recomputes scenario capacity with the frontend formula and rejects client overrides", () => {
      const snapshot = buildCurrentProductionSnapshot();
      expect(buildProductionCapacityScenarios(
        snapshot.production_forecast_results,
        1.65,
      )).toEqual({ tight: 99, normal: 110, abundant: 121 });

      expect(validateCompletedProductionPlan({
        ...snapshot,
        production_capacity: 0,
      })).toMatchObject({
        ok: false,
        code: "PRODUCTION_PLAN_MISMATCH",
      });
      expect(validateCompletedProductionPlan({
        ...snapshot,
        production_capacity_mode: "auto",
        production_capacity_scenario: null,
      })).toMatchObject({
        ok: false,
        code: "PRODUCTION_PLAN_INVALID",
      });
    });

    test("accepts every supported service-level pair and rejects cross-paired values", () => {
      for (const [serviceLevel, zScore] of [
        [0.9, 1.28],
        [0.95, 1.65],
        [0.99, 2.33],
      ] as const) {
        const snapshot = buildCurrentProductionSnapshot();
        const capacity = buildProductionCapacityScenarios(
          snapshot.production_forecast_results,
          zScore,
        ).normal;
        const mps = buildCanonicalProductionMps({
          forecasts: snapshot.production_forecast_results,
          zScore,
          initialInventory: 0,
          capacity,
        });
        expect(validateCompletedProductionPlan({
          ...snapshot,
          production_target_service_level: serviceLevel,
          production_safety_stock_z_score: zScore,
          production_capacity: capacity,
          production_mps_table: mps,
        }).ok).toBeTrue();
      }

      expect(validateCompletedProductionPlan({
        ...buildCurrentProductionSnapshot(),
        production_target_service_level: 0.99,
        production_safety_stock_z_score: 1.65,
      })).toMatchObject({
        ok: false,
        code: "PRODUCTION_PLAN_INVALID",
      });
    });

    test("enforces custom capacity identity and integer bounds", () => {
      const snapshot = buildCurrentProductionSnapshot();
      const customCapacity = 125;
      const customSnapshot = {
        ...snapshot,
        production_capacity_mode: "custom",
        production_capacity_scenario: null,
        production_capacity: customCapacity,
        production_custom_capacity: customCapacity,
        production_mps_table: buildCanonicalProductionMps({
          forecasts: snapshot.production_forecast_results,
          zScore: snapshot.production_safety_stock_z_score,
          initialInventory: snapshot.production_initial_inventory,
          capacity: customCapacity,
        }),
      };

      expect(validateCompletedProductionPlan(customSnapshot).ok).toBeTrue();
      for (const invalidCapacity of [0, -1, 1.5, null]) {
        expect(validateCompletedProductionPlan({
          ...customSnapshot,
          production_capacity: invalidCapacity,
          production_custom_capacity: invalidCapacity,
        })).toMatchObject({ ok: false, code: "PRODUCTION_PLAN_INVALID" });
      }
      expect(validateCompletedProductionPlan({
        ...customSnapshot,
        production_custom_capacity: customCapacity + 1,
      })).toMatchObject({ ok: false, code: "PRODUCTION_PLAN_INVALID" });
    });

    test("keeps zero-demand MPS service at one without producing non-finite values", () => {
      const forecasts = [
        { ...validProductionPredictionPoint, prediction: 0, std_dev: 0 },
        { ...validProductionPredictionPoint, prediction: 0, std_dev: 0 },
      ];
      const rows = buildCanonicalProductionMps({
        forecasts,
        zScore: 1.65,
        initialInventory: 0,
        capacity: 0,
      });

      expect(rows).toHaveLength(2);
      expect(rows.every((row) => row.service_level === 1)).toBeTrue();
      expect(rows.every((row) => Object.values(row).every(
        (value) => typeof value !== "number" || Number.isFinite(value),
      ))).toBeTrue();
    });

    test("rejects a fabricated or stale MPS even when every row passes the shape schema", () => {
      const snapshot = buildCurrentProductionSnapshot();
      const tamperedRows = snapshot.production_mps_table.map((row) => ({ ...row }));
      tamperedRows[1]!.planned_production = 0;
      expect(validateCompletedProductionPlan({
        ...snapshot,
        production_mps_table: tamperedRows,
      })).toMatchObject({
        ok: false,
        code: "PRODUCTION_PLAN_MISMATCH",
      });

      const changedForecasts = snapshot.production_forecast_results.map((point, index) => ({
        ...point,
        prediction: point.prediction + index + 1,
      }));
      expect(validateCompletedProductionPlan({
        ...snapshot,
        production_forecast_results: changedForecasts,
      })).toMatchObject({
        ok: false,
        code: "PRODUCTION_FORECAST_EVIDENCE_STALE",
      });
    });

    test("grandfathers evidence-free v2 only after the report made the experiment immutable", () => {
      const snapshot = buildCurrentProductionSnapshot();
      expect(validateCompletedProductionPlan({
        ...snapshot,
        status: "Completed",
        production_forecast_evidence: null,
      })).toEqual({ ok: true, evidenceClaims: null });
    });

    test("should reject production forecasts longer than the supported 12-period horizon", () => {
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        production_forecast_results: Array.from(
          { length: 12 },
          () => validProductionPredictionPoint,
        ),
      }).success).toBe(true);
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        production_forecast_results: Array.from(
          { length: 13 },
          () => validProductionPredictionPoint,
        ),
      }).success).toBe(false);
    });

    test("should accept complete sequential MPS rows and reject shape-only placeholders", () => {
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        production_mps_table: [validProductionMpsRow],
      }).success).toBe(true);
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        production_mps_table: JSON.stringify([validProductionMpsRow]),
      }).success).toBe(true);

      const invalidRows = [
        { period: 1 },
        { ...validProductionMpsRow, period: 2 },
        { ...validProductionMpsRow, period_label: "第一期" },
        { ...validProductionMpsRow, demand_forecast: Number.NaN },
        { ...validProductionMpsRow, service_level: 1.1 },
        { ...validProductionMpsRow, unexpected: true },
      ];
      for (const invalidRow of invalidRows) {
        expect(updateExperimentSchema.safeParse({
          expected_version: 0,
          production_mps_table: [invalidRow],
        }).success).toBe(false);
      }
    });

    test("should enforce the supported production forecast-period range", () => {
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        production_forecast_periods: 2,
      }).success).toBe(true);
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        production_forecast_periods: 12,
      }).success).toBe(true);
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        production_forecast_periods: 1,
      }).success).toBe(false);
      expect(updateExperimentSchema.safeParse({
        expected_version: 0,
        production_forecast_periods: 13,
      }).success).toBe(false);
    });
  });

  describe("experimentEventSchema", () => {
    test("should accept valid STARTED event", () => {
      const result = experimentEventSchema.safeParse({
        step_order: 1,
        event_type: "STARTED",
      });
      expect(result.success).toBe(true);
    });

    test("should accept valid COMPLETED event", () => {
      const result = experimentEventSchema.safeParse({
        step_order: 3,
        event_type: "COMPLETED",
      });
      expect(result.success).toBe(true);
    });

    test("should reject step_order less than 1", () => {
      const result = experimentEventSchema.safeParse({
        step_order: 0,
        event_type: "STARTED",
      });
      expect(result.success).toBe(false);
    });

    test("should reject step_order greater than 7", () => {
      const result = experimentEventSchema.safeParse({
        step_order: 8,
        event_type: "STARTED",
      });
      expect(result.success).toBe(false);
    });

    test("should reject invalid event_type", () => {
      const result = experimentEventSchema.safeParse({
        step_order: 1,
        event_type: "PAUSED",
      });
      expect(result.success).toBe(false);
    });
  });

  describe("Constants", () => {
    test("STATUS_VALUES should have correct values", () => {
      expect(STATUS_VALUES).toContain("Not Started");
      expect(STATUS_VALUES).toContain("In Progress");
      expect(STATUS_VALUES).toContain("Completed");
    });

    test("BEST_MODEL_VALUES should have all model types", () => {
      expect(BEST_MODEL_VALUES).toContain("ma");
      expect(BEST_MODEL_VALUES).toContain("exp");
      expect(BEST_MODEL_VALUES).toContain("arima");
      expect(BEST_MODEL_VALUES).toContain("lstm");
      expect(BEST_MODEL_VALUES).toContain("ensemble_weighted");
    });
  });
});
