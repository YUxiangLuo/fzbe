import { describe, test, expect } from "bun:test";
import {
  trainSchema,
  predictSchema,
  prepareProductionSchema,
  ALL_MODELS,
  isModelType,
  getTrainSchemaForModel,
  MAX_LSTM_ADDITIONAL_FEATURES,
} from "../../../src/modules/models/models.schema";

const LSTM_FEATURE_LIMIT_MESSAGE =
  `LSTM 附加输入特征最多可选择 ${MAX_LSTM_ADDITIONAL_FEATURES} 个（目标字段不计入上限），请减少所选特征后重试`;

const buildLstmFeatureLimitPayload = (lstmFeatures: string[] | string) => ({
  experiment_id: 12,
  lstmFeatures,
  lstmTargetFeature: "销售数量",
  lstmNormalization: "minmax" as const,
});

const expectLstmFeatureLimitRejection = (result: ReturnType<ReturnType<typeof getTrainSchemaForModel>["safeParse"]>) => {
  expect(result.success).toBe(false);
  if (!result.success) {
    expect(result.error.issues).toContainEqual(expect.objectContaining({
      path: ["lstmFeatures"],
      message: LSTM_FEATURE_LIMIT_MESSAGE,
    }));
  }
};

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

    test("should accept target-only LSTM training with no additional features", () => {
      const result = getTrainSchemaForModel("lstm").safeParse({
        experiment_id: 12,
        lstmFeatures: [],
        lstmTargetFeature: "销售数量",
        lstmNormalization: "zscore",
      });

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.lstmFeatures).toBe("[]");
      }
    });

    test("should accept 64 and reject 65 LSTM features for JSON arrays without splitting comma-containing names", () => {
      const acceptedFeatures = [
        "外部,指标",
        ...Array.from(
          { length: MAX_LSTM_ADDITIONAL_FEATURES - 1 },
          (_, index) => `特征${index + 1}`,
        ),
      ];
      const rejectedFeatures = [...acceptedFeatures, "超限特征"];

      const accepted = getTrainSchemaForModel("lstm").safeParse(
        buildLstmFeatureLimitPayload(acceptedFeatures),
      );
      expect(accepted.success).toBe(true);
      if (accepted.success) {
        expect(accepted.data.lstmFeatures).toBe(JSON.stringify(acceptedFeatures));
      }

      expectLstmFeatureLimitRejection(
        getTrainSchemaForModel("lstm").safeParse(
          buildLstmFeatureLimitPayload(rejectedFeatures),
        ),
      );
    });

    test("should accept 64 and reject 65 legacy comma-separated LSTM features", () => {
      const acceptedFeatures = Array.from(
        { length: MAX_LSTM_ADDITIONAL_FEATURES },
        (_, index) => `旧特征${index + 1}`,
      );

      expect(getTrainSchemaForModel("lstm").safeParse(
        buildLstmFeatureLimitPayload(acceptedFeatures.join(",")),
      ).success).toBe(true);
      expectLstmFeatureLimitRejection(
        getTrainSchemaForModel("lstm").safeParse(
          buildLstmFeatureLimitPayload([...acceptedFeatures, "超限特征"].join(",")),
        ),
      );
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

    test("should reject duplicate or unsupported ensemble members", () => {
      expect(getTrainSchemaForModel("weighted_avg").safeParse({
        experiment_id: 12,
        models: "ma,ma",
      }).success).toBe(false);
      expect(getTrainSchemaForModel("stacking").safeParse({
        experiment_id: 12,
        models: "ma,unknown",
      }).success).toBe(false);
      expect(getTrainSchemaForModel("boosting").safeParse({
        experiment_id: 12,
        models: "ma,es",
      }).success).toBe(true);
    });

    test("should require explicit LSTM configuration for every ensemble and preserve target-only selection", () => {
      for (const modelType of ["weighted_avg", "stacking", "boosting"] as const) {
        const missing = getTrainSchemaForModel(modelType).safeParse({
          experiment_id: 12,
          models: "ma,lstm",
        });
        expect(missing.success).toBe(false);
        if (!missing.success) {
          expect(missing.error.issues.map((issue) => issue.path[0])).toEqual([
            "lstmFeatures",
            "lstmTargetFeature",
            "lstmNormalization",
          ]);
        }

        const targetOnly = getTrainSchemaForModel(modelType).safeParse({
          experiment_id: 12,
          models: "ma,lstm",
          lstmFeatures: [],
          lstmTargetFeature: "销售数量",
          lstmNormalization: "minmax",
        });
        expect(targetOnly.success).toBe(true);
        if (targetOnly.success) {
          expect(targetOnly.data.lstmFeatures).toBe("[]");
        }
      }
    });

    test("should enforce the 64-feature LSTM boundary for all three ensemble schemas", () => {
      const acceptedFeatures = [
        "外部,指标",
        ...Array.from(
          { length: MAX_LSTM_ADDITIONAL_FEATURES - 1 },
          (_, index) => `特征${index + 1}`,
        ),
      ];
      const rejectedFeatures = [...acceptedFeatures, "超限特征"];

      for (const modelType of ["weighted_avg", "stacking", "boosting"] as const) {
        expect(getTrainSchemaForModel(modelType).safeParse({
          ...buildLstmFeatureLimitPayload(JSON.stringify(acceptedFeatures)),
          models: "ma,lstm",
        }).success).toBe(true);

        expectLstmFeatureLimitRejection(getTrainSchemaForModel(modelType).safeParse({
          ...buildLstmFeatureLimitPayload(JSON.stringify(rejectedFeatures)),
          models: "ma,lstm",
        }));
      }
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
        preparation_token: "123e4567-e89b-42d3-a456-426614174000",
      });

      expect(result.success).toBe(true);
    });

    test("should enforce the persistable 2-12 period production horizon", () => {
      const request = {
        experiment_id: 12,
        preparation_token: "123e4567-e89b-42d3-a456-426614174000",
      };

      expect(predictSchema.safeParse({ ...request, forecast_steps: 1 }).success).toBe(false);
      expect(predictSchema.safeParse({ ...request, forecast_steps: 2 }).success).toBe(true);
      expect(predictSchema.safeParse({ ...request, forecast_steps: 12 }).success).toBe(true);
      expect(predictSchema.safeParse({ ...request, forecast_steps: 13 }).success).toBe(false);
    });

    test("should reject forecasts longer than the production horizon", () => {
      const result = predictSchema.safeParse({
        experiment_id: 12,
        forecast_steps: 13,
      });

      expect(result.success).toBe(false);
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
        forecast_steps: 6,
      });

      expect(result.success).toBe(true);
    });

    test("should enforce the same 2-12 period horizon as prediction and plan persistence", () => {
      const request = { experiment_id: 12 };

      expect(prepareProductionSchema.safeParse({ ...request, forecast_steps: 1 }).success)
        .toBe(false);
      expect(prepareProductionSchema.safeParse({ ...request, forecast_steps: 2 }).success)
        .toBe(true);
      expect(prepareProductionSchema.safeParse({ ...request, forecast_steps: 12 }).success)
        .toBe(true);
      expect(prepareProductionSchema.safeParse({ ...request, forecast_steps: 13 }).success)
        .toBe(false);
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
