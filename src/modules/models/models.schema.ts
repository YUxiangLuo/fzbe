import { z } from "zod";

export const ALL_MODELS = ["ma", "es", "arima", "lstm", "stacking", "weighted_avg", "boosting"] as const;
export type ModelType = (typeof ALL_MODELS)[number];

const baseDataSchema = z.object({
  experiment_id: z.coerce.number().int().min(1),
  selected_industry: z.string().min(1).optional(),
  selected_company: z.string().min(1).optional(),
  selected_product: z.string().min(1).optional(),
});

const trainRangeSchema = z.object({
  data_window_train_start_index: z.coerce.number().int().min(0).optional(),
  data_window_train_end_index: z.coerce.number().int().min(0).optional(),
});

const evalRangeSchema = z.object({
  data_window_evaluate_start_index: z.coerce.number().int().min(0).optional(),
  data_window_evaluate_end_index: z.coerce.number().int().min(0).optional(),
});

const normalizeLstmFeaturesInput = (value: unknown) => {
  if (Array.isArray(value)) {
    const features = value
      .filter((item): item is string => typeof item === "string")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
    return features.length > 0 ? JSON.stringify(features) : "";
  }

  return value;
};

const lstmFeaturesSchema = z.preprocess(normalizeLstmFeaturesInput, z.string().min(1));

const modelParamsSchema = z.object({
  moving_average_window: z.coerce.number().int().min(2).optional(),
  exponential_smoothing_alpha: z.coerce.number().gt(0).max(1).optional(),
  arimaD: z.coerce.number().int().min(0).optional(),
  lstmFeatures: lstmFeaturesSchema.optional(),
  lstmTargetFeature: z.string().min(1).optional(),
  lstmNormalization: z.enum(["minmax", "zscore"]).optional(),
  models: z.string().min(1).optional(),
});

const trainSchemaBase = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .merge(modelParamsSchema);

const withRangeValidation = <TSchema extends z.ZodTypeAny>(schema: TSchema) => {
  return schema
    .refine(
      (data) =>
        typeof data !== "object"
        || data === null
        || (data as Record<string, unknown>).data_window_train_start_index === undefined
        || (data as Record<string, unknown>).data_window_train_end_index === undefined
        || Number((data as Record<string, unknown>).data_window_train_start_index)
          <= Number((data as Record<string, unknown>).data_window_train_end_index),
      { message: "训练起始索引必须小于等于训练结束索引", path: ["data_window_train_end_index"] },
    )
    .refine(
      (data) =>
        typeof data !== "object"
        || data === null
        || (data as Record<string, unknown>).data_window_evaluate_start_index === undefined
        || (data as Record<string, unknown>).data_window_evaluate_end_index === undefined
        || Number((data as Record<string, unknown>).data_window_evaluate_start_index)
          <= Number((data as Record<string, unknown>).data_window_evaluate_end_index),
      { message: "评估起始索引必须小于等于评估结束索引", path: ["data_window_evaluate_end_index"] },
    )
    .refine(
      (data) =>
        typeof data !== "object"
        || data === null
        || (data as Record<string, unknown>).data_window_train_end_index === undefined
        || (data as Record<string, unknown>).data_window_evaluate_start_index === undefined
        || Number((data as Record<string, unknown>).data_window_evaluate_start_index)
          > Number((data as Record<string, unknown>).data_window_train_end_index),
      { message: "评估起始索引必须大于训练结束索引", path: ["data_window_evaluate_start_index"] },
    );
};

export const trainSchema = withRangeValidation(trainSchemaBase);

const movingAverageTrainSchema = withRangeValidation(
  trainSchemaBase.extend({
    moving_average_window: z.coerce.number().int().min(2),
  }),
).refine(
  (data) =>
    data.data_window_train_start_index === undefined
    || data.data_window_train_end_index === undefined
    || data.moving_average_window === undefined
    || data.moving_average_window
      <= data.data_window_train_end_index - data.data_window_train_start_index + 1,
  {
    message: "移动平均时间窗口不能超过训练数据长度",
    path: ["moving_average_window"],
  },
);

const exponentialSmoothingTrainSchema = withRangeValidation(
  trainSchemaBase.extend({
    exponential_smoothing_alpha: z.coerce.number().gt(0).max(1),
  }),
);

const arimaTrainSchema = withRangeValidation(
  trainSchemaBase.extend({
    arimaD: z.coerce.number().int().min(0),
  }),
);

const lstmTrainSchema = withRangeValidation(
  trainSchemaBase.extend({
    lstmFeatures: lstmFeaturesSchema,
    lstmTargetFeature: z.string().min(1),
    lstmNormalization: z.enum(["minmax", "zscore"]),
  }),
);

const ensembleTrainSchema = withRangeValidation(
  trainSchemaBase.extend({
    models: z.string().min(1),
  }),
);

export function getTrainSchemaForModel(modelType: ModelType) {
  switch (modelType) {
    case "ma":
      return movingAverageTrainSchema;
    case "es":
      return exponentialSmoothingTrainSchema;
    case "arima":
      return arimaTrainSchema;
    case "lstm":
      return lstmTrainSchema;
    case "stacking":
    case "weighted_avg":
    case "boosting":
      return ensembleTrainSchema;
    default: {
      const _exhaustive: never = modelType;
      throw new Error(`No train schema defined for model type: ${_exhaustive}`);
    }
  }
}

export const predictSchema = z.object({
  experiment_id: z.coerce.number().int().min(1),
  forecast_steps: z.coerce.number().int().min(1),
});

export const prepareProductionSchema = z.object({
  experiment_id: z.coerce.number().int().min(1),
});

export function isModelType(value: string): value is ModelType {
  return (ALL_MODELS as readonly string[]).includes(value);
}

export type TrainInput = z.infer<typeof trainSchema>;
export type PredictInput = z.infer<typeof predictSchema>;
