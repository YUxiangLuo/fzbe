import { z } from "zod";

const baseDataSchema = z.object({
  selected_industry: z.string().min(1),
  selected_company: z.string().min(1),
  selected_product: z.string().min(1),
});

const trainRangeSchema = z.object({
  data_window_train_start_index: z.coerce.number().int().min(0),
  data_window_train_end_index: z.coerce.number().int().min(0),
});

const evalRangeSchema = z.object({
  data_window_evaluate_start_index: z.coerce.number().int().min(0),
  data_window_evaluate_end_index: z.coerce.number().int().min(0),
});

const modelParamsSchema = z.object({
  moving_average_window: z.coerce.number().int().min(2).optional(),
  exponential_smoothing_alpha: z.coerce.number().min(0).max(1).optional(),
  arimaD: z.coerce.number().int().min(0).optional(),
  lstmFeatures: z.string().min(1).optional(),
  lstmTargetFeature: z.string().min(1).optional(),
  lstmNormalization: z.enum(["minmax", "zscore"]).optional(),
  models: z.string().min(1).optional(),
});

const trainSchemaBase = baseDataSchema
  .merge(trainRangeSchema)
  .merge(evalRangeSchema)
  .merge(modelParamsSchema);

export const trainSchema = trainSchemaBase
  .refine(
    (data) => data.data_window_train_start_index <= data.data_window_train_end_index,
    { message: "训练起始索引必须小于等于训练结束索引", path: ["data_window_train_end_index"] },
  )
  .refine(
    (data) => data.data_window_evaluate_start_index <= data.data_window_evaluate_end_index,
    { message: "评估起始索引必须小于等于评估结束索引", path: ["data_window_evaluate_end_index"] },
  );

export const predictSchema = z.object({
  forecast_steps: z.coerce.number().int().min(1),
});

export const ALL_MODELS = ["ma", "es", "arima", "lstm", "stacking", "weighted_avg", "boosting"] as const;
export type ModelType = (typeof ALL_MODELS)[number];

export function isModelType(value: string): value is ModelType {
  return (ALL_MODELS as readonly string[]).includes(value);
}

export type TrainInput = z.infer<typeof trainSchema>;
export type PredictInput = z.infer<typeof predictSchema>;
