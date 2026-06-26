import { describe, test, expect } from "bun:test";
import {
  experimentIdParamSchema,
  createExperimentSchema,
  updateExperimentSchema,
  experimentEventSchema,
  STATUS_VALUES,
  BEST_MODEL_VALUES,
} from "../../../src/modules/experiments/experiments.schema";

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

    test("should accept status", () => {
      const result = createExperimentSchema.safeParse({
        status: "In Progress",
      });
      expect(result.success).toBe(true);
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

  describe("updateExperimentSchema", () => {
    test("should accept current_step update", () => {
      const result = updateExperimentSchema.safeParse({
        current_step: 3,
      });
      expect(result.success).toBe(true);
    });

    test("should accept status update", () => {
      const result = updateExperimentSchema.safeParse({
        status: "In Progress",
      });
      expect(result.success).toBe(true);
    });

    test("should accept Completed status", () => {
      const result = updateExperimentSchema.safeParse({
        status: "Completed",
      });
      expect(result.success).toBe(true);
    });

    test("should reject invalid status", () => {
      const result = updateExperimentSchema.safeParse({
        status: "invalid_status",
      });
      expect(result.success).toBe(false);
    });

    test("should accept selection fields", () => {
      const result = updateExperimentSchema.safeParse({
        selected_industry: "Manufacturing",
        selected_company: "ABC Corp",
        selected_product: "Widget A",
      });
      expect(result.success).toBe(true);
    });

    test("should accept highest_completed_step", () => {
      const result = updateExperimentSchema.safeParse({
        highest_completed_step: 5,
      });
      expect(result.success).toBe(true);
    });

    test("should accept multiple fields together", () => {
      const result = updateExperimentSchema.safeParse({
        current_step: 4,
        status: "In Progress",
        selected_industry: "Retail",
        highest_completed_step: 3,
      });
      expect(result.success).toBe(true);
    });

    test("should reject empty object", () => {
      const result = updateExperimentSchema.safeParse({});
      expect(result.success).toBe(false);
    });

    test("should accept model metrics", () => {
      const result = updateExperimentSchema.safeParse({
        moving_average_completed: true,
        moving_average_window: 5,
        moving_average_metrics_rmse: 0.5,
      });
      expect(result.success).toBe(true);
    });

    test("should reject non-integer moving average windows", () => {
      const result = updateExperimentSchema.safeParse({
        moving_average_window: 2.5,
      });
      expect(result.success).toBe(false);
    });

    test("should reject moving average windows below 2", () => {
      const result = updateExperimentSchema.safeParse({
        moving_average_window: 1,
      });
      expect(result.success).toBe(false);
    });

    test("should accept best model selection", () => {
      const result = updateExperimentSchema.safeParse({
        selected_best_model: "lstm",
      });
      expect(result.success).toBe(true);
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
