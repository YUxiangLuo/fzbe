import { describe, test, expect } from "bun:test";
import {
  trainSchema,
  predictSchema,
  prepareProductionSchema,
  ALL_MODELS,
  isModelType,
  getTrainSchemaForModel,
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

    test("should require model-specific params for moving average training", () => {
      const result = getTrainSchemaForModel("ma").safeParse({
        experiment_id: 12,
      });

      expect(result.success).toBe(false);
    });

    test("should reject decimal moving average windows", () => {
      const result = getTrainSchemaForModel("ma").safeParse({
        experiment_id: 12,
        moving_average_window: 2.5,
      });

      expect(result.success).toBe(false);
    });

    test("should require model-specific params for exponential smoothing training", () => {
      const result = getTrainSchemaForModel("es").safeParse({
        experiment_id: 12,
      });

      expect(result.success).toBe(false);
    });

    test("should reject non-positive alpha for exponential smoothing training", () => {
      const result = getTrainSchemaForModel("es").safeParse({
        experiment_id: 12,
        exponential_smoothing_alpha: 0,
      });

      expect(result.success).toBe(false);
    });

    test("should require model-specific params for ARIMA training", () => {
      const result = getTrainSchemaForModel("arima").safeParse({
        experiment_id: 12,
      });

      expect(result.success).toBe(false);
    });

    test("should require model-specific params for LSTM training", () => {
      const result = getTrainSchemaForModel("lstm").safeParse({
        experiment_id: 12,
        lstmFeatures: "价格指数,产能利用率",
        lstmTargetFeature: "销售数量",
      });

      expect(result.success).toBe(false);
    });

    test("should accept LSTM feature arrays and encode them without losing comma-containing field names", () => {
      const result = getTrainSchemaForModel("lstm").safeParse({
        experiment_id: 12,
        lstmFeatures: ["价格指数", "外部,指标"],
        lstmTargetFeature: "销售数量",
        lstmNormalization: "minmax",
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.lstmFeatures).toBe(JSON.stringify(["价格指数", "外部,指标"]));
      }
    });

    test("should require model-specific params for weighted ensemble training", () => {
      const result = getTrainSchemaForModel("weighted_avg").safeParse({
        experiment_id: 12,
      });

      expect(result.success).toBe(false);
    });

    test("should require model-specific params for boosting ensemble training", () => {
      const result = getTrainSchemaForModel("boosting").safeParse({
        experiment_id: 12,
      });

      expect(result.success).toBe(false);
    });

    test("should require model-specific params for stacking ensemble training", () => {
      const result = getTrainSchemaForModel("stacking").safeParse({
        experiment_id: 12,
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

  describe("prepareProductionSchema", () => {
    test("should require experiment_id for production preparation requests", () => {
      const result = prepareProductionSchema.safeParse({});

      expect(result.success).toBe(false);
    });

    test("should accept a valid production preparation payload", () => {
      const result = prepareProductionSchema.safeParse({
        experiment_id: 12,
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
      expect(isModelType(undefined)).toBe(false);
      expect(isModelType(null)).toBe(false);
    });
  });
});
