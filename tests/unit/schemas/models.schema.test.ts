import { describe, test, expect } from "bun:test";
import {
  trainSchema,
  predictSchema,
  ALL_MODELS,
  isModelType,
} from "../../../src/modules/models/models.schema";

describe("Models Schemas", () => {
  describe("trainSchema", () => {
    test("should require experiment_id for training requests", () => {
      const result = trainSchema.safeParse({
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
      });

      expect(result.success).toBe(false);
    });

    test("should accept valid training payload with experiment_id", () => {
      const result = trainSchema.safeParse({
        experiment_id: 12,
        moving_average_window: 4,
      });

      expect(result.success).toBe(true);
    });

    test("should still accept legacy training payloads with selection and window fields", () => {
      const result = trainSchema.safeParse({
        experiment_id: 12,
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 12,
        data_window_evaluate_end_index: 17,
        moving_average_window: 4,
      });

      expect(result.success).toBe(true);
    });

    test("should reject reversed evaluation ranges", () => {
      const result = trainSchema.safeParse({
        experiment_id: 12,
        selected_industry: "制造业",
        selected_company: "样本企业",
        selected_product: "样本产品",
        data_window_train_start_index: 0,
        data_window_train_end_index: 11,
        data_window_evaluate_start_index: 18,
        data_window_evaluate_end_index: 17,
      });

      expect(result.success).toBe(false);
    });
  });

  describe("predictSchema", () => {
    test("should require experiment_id for prediction requests", () => {
      const result = predictSchema.safeParse({
        forecast_steps: 6,
      });

      expect(result.success).toBe(false);
    });

    test("should accept valid prediction payload", () => {
      const result = predictSchema.safeParse({
        experiment_id: 12,
        forecast_steps: 6,
      });

      expect(result.success).toBe(true);
    });
  });

  describe("model type helpers", () => {
    test("should expose supported model types", () => {
      expect(ALL_MODELS).toContain("weighted_avg");
      expect(ALL_MODELS).toContain("boosting");
      expect(ALL_MODELS).toContain("lstm");
    });

    test("should validate model type strings", () => {
      expect(isModelType("ma")).toBe(true);
      expect(isModelType("weighted_avg")).toBe(true);
      expect(isModelType("ensemble_weighted")).toBe(false);
    });
  });
});