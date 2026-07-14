import type { ModelType } from "./models.schema";

interface TrainingMetricsPayload extends Record<string, unknown> {
  evaluate_indices?: unknown;
  evaluate_months?: unknown;
}

interface PredictionPoint extends Record<string, unknown> {
  prediction?: unknown;
  std_dev?: unknown;
}

interface DemandPredictionNormalizationOptions {
  requireStdDev?: boolean;
}

const METRIC_EPSILON = 1e-12;

export const normalizeDemandPredictionPoints = (
  value: unknown,
  expectedLength?: number,
  options: DemandPredictionNormalizationOptions = {},
): PredictionPoint[] => {
  // Python applies the demand-domain constraint before residuals and metrics.
  // Keep this boundary clamp as defense in depth for malformed/custom runner
  // output; it must not be the primary implementation.
  if (!Array.isArray(value)) {
    throw new Error("[PYTHON_DATA] predictions must be an array");
  }
  if (value.length === 0) {
    throw new Error("[PYTHON_DATA] predictions must contain at least one point");
  }
  if (expectedLength !== undefined && value.length !== expectedLength) {
    throw new Error(
      `[PYTHON_DATA] Expected ${expectedLength} prediction points, received ${value.length}`,
    );
  }
  return value.map((item, index) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new Error(`[PYTHON_DATA] Prediction point ${index} must be an object`);
    }
    const point = item as PredictionPoint;
    const prediction = point.prediction;
    if (typeof prediction !== "number" || !Number.isFinite(prediction)) {
      throw new Error(`[PYTHON_DATA] Prediction point ${index} is not finite`);
    }
    if (
      options.requireStdDev
      && (typeof point.std_dev !== "number" || !Number.isFinite(point.std_dev) || point.std_dev < 0)
    ) {
      throw new Error(`[PYTHON_DATA] Prediction point ${index} std_dev must be a nonnegative finite number`);
    }
    return {
      ...point,
      prediction: Math.max(0, prediction),
    };
  });
};

const normalizeEvaluationActuals = (
  value: unknown,
  expectedLength: number,
): number[] => {
  if (!Array.isArray(value)) {
    throw new Error("[PYTHON_DATA] eval_y_true must be an array");
  }
  if (value.length !== expectedLength) {
    throw new Error(
      `[PYTHON_DATA] Expected ${expectedLength} evaluation values, received ${value.length}`,
    );
  }
  return value.map((item, index) => {
    if (typeof item !== "number" || !Number.isFinite(item)) {
      throw new Error(`[PYTHON_DATA] Evaluation value ${index} is not finite`);
    }
    return item;
  });
};

const recomputeCanonicalMetrics = (
  metrics: TrainingMetricsPayload | undefined,
  actual: number[],
  predictions: PredictionPoint[],
): TrainingMetricsPayload => {
  const predicted = predictions.map((point) => Number(point.prediction));

  const squaredErrors = actual.map((value, index) => (value - predicted[index]!) ** 2);
  const absoluteErrors = actual.map((value, index) => Math.abs(value - predicted[index]!));
  const nonzeroIndices = actual
    .map((value, index) => Math.abs(value) > METRIC_EPSILON ? index : -1)
    .filter((index) => index >= 0);
  const meanActual = actual.reduce((sum, value) => sum + value, 0) / actual.length;
  const residualSumOfSquares = squaredErrors.reduce((sum, value) => sum + value, 0);
  const totalSumOfSquares = actual.reduce(
    (sum, value) => sum + (value - meanActual) ** 2,
    0,
  );

  return {
    ...(metrics ?? {}),
    rmse: Math.sqrt(residualSumOfSquares / actual.length),
    mae: absoluteErrors.reduce((sum, value) => sum + value, 0) / actual.length,
    mape: nonzeroIndices.length === 0
      ? 0
      : 100 * nonzeroIndices.reduce(
        (sum, index) => sum + absoluteErrors[index]! / Math.abs(actual[index]!),
        0,
      ) / nonzeroIndices.length,
    r2: totalSumOfSquares <= METRIC_EPSILON
      ? (residualSumOfSquares <= METRIC_EPSILON ? 1 : 0)
      : 1 - residualSumOfSquares / totalSumOfSquares,
  };
};

const normalizeNumericArray = (value: unknown): number[] | undefined => {
  if (!Array.isArray(value)) return undefined;
  const normalized = value
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item));
  return normalized.length > 0 ? normalized : undefined;
};

const normalizeStringArray = (value: unknown): string[] | undefined => {
  if (!Array.isArray(value)) return undefined;
  const normalized = value
    .map((item) => String(item))
    .filter((item) => item.length > 0);
  return normalized.length > 0 ? normalized : undefined;
};

const resultTransformers: Record<ModelType, (result: Record<string, unknown>) => Record<string, unknown>> = {
  arima: (result) => ({
    best_order: result.best_order,
    search_summary: result.search_summary,
    diagnostics: result.diagnostics
      ? { ...(result.diagnostics as object), y_pred_eval: (result.predictions as { prediction: number }[] | undefined)?.map((p) => p.prediction) || [] }
      : undefined,
  }),
  es: () => ({}),
  ma: () => ({}),
  lstm: (result) => ({
    feature_types: result.feature_types,
    look_back: result.look_back,
    epochs: result.epochs,
    forecast_horizon: result.forecast_horizon,
    diagnostics: result.diagnostics
      ? { ...(result.diagnostics as object), y_pred_eval: (result.predictions as { prediction: number }[] | undefined)?.map((p) => p.prediction) || [] }
      : undefined,
  }),
  stacking: (result) => ({ model_names: result.models, meta_model: result.meta_model }),
  weighted_avg: (result) => ({
    model_names: result.models,
    weights: (result.models as string[] | undefined)?.map((name) => (result.weights as Record<string, number>)?.[name]),
  }),
  boosting: (result) => ({
    model_names: result.model_chain,
    search_summary: result.search_summary,
  }),
};

export function transformTrainingResult(
  modelType: ModelType,
  result: Record<string, unknown>,
  expectedEvaluationLength?: number,
): { status: string; results: Record<string, unknown>; inferred_feature_types?: unknown } {
  const originalMetrics =
    result.metrics && typeof result.metrics === "object"
      ? (result.metrics as TrainingMetricsPayload)
      : undefined;
  const predictions = normalizeDemandPredictionPoints(result.predictions, expectedEvaluationLength);
  const actual = normalizeEvaluationActuals(result.eval_y_true, predictions.length);
  const metrics = recomputeCanonicalMetrics(originalMetrics, actual, predictions);
  const normalizedResult = {
    ...result,
    eval_y_true: actual,
    metrics,
    predictions,
  };

  const baseResults = {
    metrics,
    eval_y_true: actual,
    eval_predictions: predictions.map((point) => point.prediction),
    evaluate_indices: normalizeNumericArray(metrics?.evaluate_indices),
    evaluate_months: normalizeStringArray(metrics?.evaluate_months),
    method_name: result.method_name,
    forecast_strategy: result.forecast_strategy,
    implementation_notes: result.implementation_notes,
    notes: result.notes || [],
    saved_model: result.saved_model,
  };

  const modelSpecificTransformedResult = resultTransformers[modelType]?.(normalizedResult) || {};
  const finalTransformedResult = { ...baseResults, ...modelSpecificTransformedResult };

  const finalResponse: { status: string; results: Record<string, unknown>; inferred_feature_types?: unknown } = {
    status: "success",
    results: finalTransformedResult,
  };

  if (modelType === "lstm" && result.feature_types) {
    finalResponse.inferred_feature_types = result.feature_types;
  }

  return finalResponse;
}
