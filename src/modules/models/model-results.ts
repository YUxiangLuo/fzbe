import type { ModelType } from "./models.schema";

interface TrainingMetricsPayload extends Record<string, unknown> {
  evaluate_indices?: unknown;
  evaluate_months?: unknown;
}

interface PredictionPoint extends Record<string, unknown> {
  prediction?: unknown;
  std_dev?: unknown;
  uncertainty_source?: unknown;
  uncertainty_reason?: unknown;
  calibration_source?: unknown;
  interval_lower?: unknown;
  interval_upper?: unknown;
  interval_level?: unknown;
  interval_kind?: unknown;
  coverage_claim?: unknown;
  upper_error_p99?: unknown;
  calibration_mean_error?: unknown;
  calibration_count?: unknown;
  bias_diagnostic_source?: unknown;
}

interface DemandPredictionNormalizationOptions {
  requireStdDev?: boolean;
  requireUpperErrorP99?: boolean;
  requireCalibrationDiagnostics?: boolean;
  requireIntervalMetadata?: boolean;
}

const METRIC_EPSILON = 1e-12;
const INTERVAL_KINDS = [
  "model_prediction_interval",
  "empirical_residual_quantile",
  "normal_approximation",
  "fallback_normal_approximation",
] as const;
type IntervalKind = (typeof INTERVAL_KINDS)[number];

const isIntervalKind = (value: unknown): value is IntervalKind =>
  typeof value === "string" && INTERVAL_KINDS.includes(value as IntervalKind);

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
    if (
      options.requireStdDev
      && !["model", "empirical", "fallback"].includes(String(point.uncertainty_source))
    ) {
      throw new Error(`[PYTHON_DATA] Prediction point ${index} uncertainty_source is invalid`);
    }
    if (
      options.requireStdDev
      && point.uncertainty_reason !== undefined
      && typeof point.uncertainty_reason !== "string"
    ) {
      throw new Error(`[PYTHON_DATA] Prediction point ${index} uncertainty_reason must be a string`);
    }
    if (
      (options.requireUpperErrorP99 || point.upper_error_p99 !== undefined)
      && (
        typeof point.upper_error_p99 !== "number"
        || !Number.isFinite(point.upper_error_p99)
        || point.upper_error_p99 < 0
      )
    ) {
      throw new Error(`[PYTHON_DATA] Prediction point ${index} upper_error_p99 must be a nonnegative finite number`);
    }
    if (
      point.calibration_source !== undefined
      && (typeof point.calibration_source !== "string" || point.calibration_source.length === 0)
    ) {
      throw new Error(`[PYTHON_DATA] Prediction point ${index} calibration_source must be a non-empty string`);
    }
    const hasCalibrationMeanError = point.calibration_mean_error !== undefined;
    const hasCalibrationCount = point.calibration_count !== undefined;
    if (
      options.requireCalibrationDiagnostics
      && (!hasCalibrationMeanError || !hasCalibrationCount)
    ) {
      throw new Error(`[PYTHON_DATA] Prediction point ${index} calibration diagnostics are required`);
    }
    if (hasCalibrationMeanError || hasCalibrationCount) {
      const bothNull = point.calibration_mean_error === null && point.calibration_count === null;
      const bothValid = (
        typeof point.calibration_mean_error === "number"
        && Number.isFinite(point.calibration_mean_error)
        && typeof point.calibration_count === "number"
        && Number.isInteger(point.calibration_count)
        && point.calibration_count > 0
      );
      if (!bothNull && !bothValid) {
        throw new Error(
          `[PYTHON_DATA] Prediction point ${index} calibration diagnostics must be a finite mean and positive count, or both null`,
        );
      }
    }
    if (
      point.bias_diagnostic_source !== undefined
      && point.bias_diagnostic_source !== null
      && (typeof point.bias_diagnostic_source !== "string" || point.bias_diagnostic_source.length === 0)
    ) {
      throw new Error(`[PYTHON_DATA] Prediction point ${index} bias_diagnostic_source must be a non-empty string or null`);
    }
    const hasIntervalField = options.requireIntervalMetadata || [
      point.interval_lower,
      point.interval_upper,
      point.interval_level,
      point.interval_kind,
      point.coverage_claim,
    ].some((value) => value !== undefined);
    if (hasIntervalField) {
      if (
        typeof point.interval_lower !== "number"
        || !Number.isFinite(point.interval_lower)
        || typeof point.interval_upper !== "number"
        || !Number.isFinite(point.interval_upper)
        || point.interval_lower > point.interval_upper
      ) {
        throw new Error(`[PYTHON_DATA] Prediction point ${index} interval bounds are invalid`);
      }
      if (
        typeof point.interval_level !== "number"
        || !Number.isFinite(point.interval_level)
        || Math.abs(point.interval_level - 0.95) > 1e-9
      ) {
        throw new Error(`[PYTHON_DATA] Prediction point ${index} interval_level must equal 0.95`);
      }
      if (!isIntervalKind(point.interval_kind)) {
        throw new Error(
          `[PYTHON_DATA] Prediction point ${index} interval_kind must be one of ${INTERVAL_KINDS.join(", ")}`,
        );
      }
      const expectedCoverageClaim = point.interval_kind === "model_prediction_interval"
        ? "model_nominal"
        : "none";
      if (point.coverage_claim !== expectedCoverageClaim) {
        throw new Error(
          `[PYTHON_DATA] Prediction point ${index} coverage_claim must equal ${expectedCoverageClaim} for ${point.interval_kind}`,
        );
      }
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
  const {
    // Interval evaluation is derived exclusively from the validated outer
    // evaluation rows below. Never retain a runner-supplied aggregate.
    interval_evaluation: _untrustedIntervalEvaluation,
    ...trustedMetrics
  } = metrics ?? {};
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
  const intervalRows = predictions.flatMap((point, index) => {
    if (
      typeof point.interval_lower !== "number"
      || typeof point.interval_upper !== "number"
      || typeof point.interval_level !== "number"
    ) {
      return [];
    }
    return [{
      actual: actual[index]!,
      lower: point.interval_lower,
      upper: point.interval_upper,
      level: point.interval_level,
      coverageClaim: point.coverage_claim,
    }];
  });
  const intervalEvaluation = intervalRows.length === 0
    ? undefined
    : (() => {
      const includedCount = intervalRows.filter(
        ({ actual: value, lower, upper }) => value >= lower && value <= upper,
      ).length;
      const totalWidth = intervalRows.reduce(
        (sum, { lower, upper }) => sum + (upper - lower),
        0,
      );
      const totalWinkler = intervalRows.reduce(
        (sum, { actual: value, lower, upper, level }) => {
          const alpha = 1 - level;
          const width = upper - lower;
          if (value < lower) return sum + width + (2 / alpha) * (lower - value);
          if (value > upper) return sum + width + (2 / alpha) * (value - upper);
          return sum + width;
        },
        0,
      );
      const levels = [...new Set(intervalRows.map(({ level }) => level))];

      return {
        scope: "outer_evaluation_descriptive",
        count: intervalRows.length,
        observed_inclusion_rate: includedCount / intervalRows.length,
        mean_width: totalWidth / intervalRows.length,
        mean_winkler_score: totalWinkler / intervalRows.length,
        nominal_level: levels.length === 1 ? levels[0] : null,
        coverage_claim: intervalRows.every(
          ({ coverageClaim }) => coverageClaim === "model_nominal",
        )
          ? "model_nominal"
          : "none",
      };
    })();

  return {
    ...trustedMetrics,
    rmse: Math.sqrt(residualSumOfSquares / actual.length),
    mae: absoluteErrors.reduce((sum, value) => sum + value, 0) / actual.length,
    mape: nonzeroIndices.length === 0
      ? null
      : 100 * nonzeroIndices.reduce(
        (sum, index) => sum + absoluteErrors[index]! / Math.abs(actual[index]!),
        0,
      ) / nonzeroIndices.length,
    r2: totalSumOfSquares <= METRIC_EPSILON
      ? (residualSumOfSquares <= METRIC_EPSILON ? 1 : 0)
      : 1 - residualSumOfSquares / totalSumOfSquares,
    ...(intervalEvaluation ? { interval_evaluation: intervalEvaluation } : {}),
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
  es: (result) => ({
    initial_level: result.initial_level,
    alpha_comparison: result.alpha_comparison,
    uncertainty_summary: result.uncertainty_summary,
  }),
  ma: () => ({}),
  lstm: (result) => {
    const trainingMetadata = (
      result.training_metadata
      && typeof result.training_metadata === "object"
      && !Array.isArray(result.training_metadata)
    )
      ? result.training_metadata as Record<string, unknown>
      : undefined;
    return {
      target_key: result.target_key,
      feature_keys: result.feature_keys ?? trainingMetadata?.feature_keys,
      feature_types: result.feature_types,
      normalization: result.normalization,
      forecast_horizon: result.forecast_horizon,
      training_regime: result.training_regime,
      training_metadata: result.training_metadata,
      uncertainty_summary: result.uncertainty_summary,
      diagnostics: result.diagnostics
        ? { ...(result.diagnostics as object), y_pred_eval: (result.predictions as { prediction: number }[] | undefined)?.map((p) => p.prediction) || [] }
        : undefined,
    };
  },
  stacking: (result) => ({
    model_names: result.models,
    meta_model: result.meta_model,
    split_plan: result.split_plan,
  }),
  weighted_avg: (result) => ({
    model_names: result.models,
    weights: (result.models as string[] | undefined)?.map((name) => (result.weights as Record<string, number>)?.[name]),
    weight_diagnostics: result.weight_diagnostics,
    split_plan: result.split_plan,
  }),
  boosting: (result) => ({
    model_names: result.model_chain,
    stage_coefficients: result.stage_coefficients,
    post_line_search_shrinkage: result.post_line_search_shrinkage,
    baseline: result.baseline,
    split_plan: result.split_plan,
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
  const predictions = normalizeDemandPredictionPoints(
    result.predictions,
    expectedEvaluationLength,
    { requireStdDev: true, requireIntervalMetadata: true },
  );
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
    prediction_points: predictions,
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
