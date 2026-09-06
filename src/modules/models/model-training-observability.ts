import { MAX_LSTM_ADDITIONAL_FEATURES } from "./models.schema";
import type { GuidedModelType } from "./model-types";
import { isRecordValue as isRecord } from "./model-json";

// Build bounded diagnostic projections before the run is persisted.
const OBSERVABILITY_STRING_MAX_LENGTH = 512;
// Bound student-facing metadata without applying the frontend's former
// 16-field display limit. feature_keys contains only additional inputs;
// feature_types may also contain the target, hence the +1 check below.
const LSTM_OBSERVABILITY_MAX_FEATURES = MAX_LSTM_ADDITIONAL_FEATURES;
const UNCERTAINTY_SOURCES = ["model", "empirical", "fallback"] as const;
const INTERVAL_KINDS = [
  "model_prediction_interval",
  "empirical_residual_quantile",
  "normal_approximation",
  "fallback_normal_approximation",
] as const;
const COVERAGE_CLAIMS = ["model_nominal", "none"] as const;

const boundedString = (value: unknown): string | undefined => {
  if (typeof value !== "string" || value.length === 0) return undefined;
  return value.slice(0, OBSERVABILITY_STRING_MAX_LENGTH);
};

const LSTM_METADATA_SCALAR_FIELDS = [
  "regime",
  "look_back",
  "units",
  "parameter_count",
  "parameter_budget",
  "batch_size",
  "max_epochs",
  "epochs_ran",
  "windows",
  "validation_windows",
  "purge_windows",
  "selection_train_windows",
  "feature_count",
  "raw_feature_count",
  "random_seed",
  "seed_runs",
  "stability_note",
] as const;

const LSTM_SEED_STABILITY_FIELDS = [
  "scope",
  "run_count",
  "validation_rmse_mean",
  "validation_rmse_std",
  "validation_mae_mean",
  "validation_mae_std",
] as const;

const boundedStringArray = (
  value: unknown,
  maxItems = LSTM_OBSERVABILITY_MAX_FEATURES,
): string[] | undefined => {
  if (!Array.isArray(value) || value.length > maxItems) return undefined;
  return value.every((item) => (
    typeof item === "string"
    && item.length > 0
    && item.length <= OBSERVABILITY_STRING_MAX_LENGTH
  ))
    ? [...value] as string[]
    : undefined;
};

const copyBoundedScalars = (
  source: Record<string, unknown>,
  fields: readonly string[],
): Record<string, unknown> => {
  const bounded: Record<string, unknown> = {};
  for (const field of fields) {
    const value = source[field];
    if (typeof value === "number" && Number.isFinite(value)) {
      bounded[field] = value;
      continue;
    }
    if (typeof value === "boolean") {
      bounded[field] = value;
      continue;
    }
    const text = boundedString(value);
    if (text !== undefined) bounded[field] = text;
  }
  return bounded;
};

const buildBoundedLstmTrainingMetadata = (
  value: unknown,
  canonicalFeatureKeys: string[] | undefined,
): Record<string, unknown> | undefined => {
  if (!isRecord(value)) return undefined;
  const metadata = copyBoundedScalars(value, LSTM_METADATA_SCALAR_FIELDS);
  const featureKeys = canonicalFeatureKeys ?? boundedStringArray(value.feature_keys);
  if (featureKeys !== undefined) metadata.feature_keys = featureKeys;
  if (isRecord(value.seed_stability)) {
    const seedStability = copyBoundedScalars(
      value.seed_stability,
      LSTM_SEED_STABILITY_FIELDS,
    );
    if (Object.keys(seedStability).length > 0) metadata.seed_stability = seedStability;
  }
  return Object.keys(metadata).length > 0 ? metadata : undefined;
};

const buildBoundedLstmFeatureTypes = (
  value: unknown,
): Record<string, "numeric" | "categorical"> | undefined => {
  if (
    !isRecord(value)
    || Object.keys(value).length > LSTM_OBSERVABILITY_MAX_FEATURES + 1
  ) return undefined;
  const entries = Object.entries(value);
  if (entries.some(([field, type]) => (
    field.length === 0
    || field.length > OBSERVABILITY_STRING_MAX_LENGTH
    || (type !== "numeric" && type !== "categorical")
  ))) {
    return undefined;
  }
  return Object.fromEntries(entries) as Record<string, "numeric" | "categorical">;
};

const buildBoundedLstmObservability = (
  results: Record<string, unknown>,
): Record<string, unknown> => {
  const observability: Record<string, unknown> = {};
  const targetKey = (
    typeof results.target_key === "string"
    && results.target_key.length > 0
    && results.target_key.length <= OBSERVABILITY_STRING_MAX_LENGTH
  )
    ? results.target_key
    : undefined;
  if (targetKey !== undefined) observability.target_key = targetKey;

  const metadataSource = isRecord(results.training_metadata)
    ? results.training_metadata
    : undefined;
  const featureKeys = boundedStringArray(results.feature_keys)
    ?? boundedStringArray(metadataSource?.feature_keys);
  if (featureKeys !== undefined) observability.feature_keys = featureKeys;

  const featureTypes = buildBoundedLstmFeatureTypes(results.feature_types);
  if (featureTypes !== undefined) observability.feature_types = featureTypes;
  if (results.normalization === "minmax" || results.normalization === "zscore") {
    observability.normalization = results.normalization;
  }
  if (
    typeof results.forecast_horizon === "number"
    && Number.isSafeInteger(results.forecast_horizon)
    && results.forecast_horizon > 0
  ) {
    observability.forecast_horizon = results.forecast_horizon;
  }
  if (results.training_regime === "standard" || results.training_regime === "teaching_demo") {
    observability.training_regime = results.training_regime;
  }
  const trainingMetadata = buildBoundedLstmTrainingMetadata(
    results.training_metadata,
    featureKeys,
  );
  if (trainingMetadata !== undefined) observability.training_metadata = trainingMetadata;
  return observability;
};

const finiteNumberRange = (
  rows: readonly Record<string, unknown>[],
  field: string,
): { min: number; max: number } | null => {
  const values = rows
    .map((row) => row[field])
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return values.length === 0
    ? null
    : { min: Math.min(...values), max: Math.max(...values) };
};

const collectKnownValues = <T extends string>(
  rows: readonly Record<string, unknown>[],
  field: string,
  allowed: readonly T[],
): T[] => allowed.filter((candidate) =>
  rows.some((row) => row[field] === candidate));

const buildBoundedUncertaintySummary = (
  value: unknown,
): Record<string, unknown> | undefined => {
  if (!isRecord(value)) return undefined;

  const horizons = Array.isArray(value.horizons)
    ? value.horizons.filter(isRecord)
    : [];
  const horizonStart = (
    typeof value.horizon_start === "number"
    && Number.isSafeInteger(value.horizon_start)
    && value.horizon_start > 0
  )
    ? value.horizon_start
    : 1;
  const explicitFallbackHorizons = Array.isArray(value.fallback_horizons)
    ? value.fallback_horizons
      .filter((horizon): horizon is number =>
        typeof horizon === "number"
        && Number.isSafeInteger(horizon)
        && horizon > 0)
      .map((horizon) => horizonStart + horizon - 1)
    : [];
  const fallbackHorizons = explicitFallbackHorizons.length > 0
    ? explicitFallbackHorizons
    : horizons.flatMap((horizon, index) =>
      horizon.source === "fallback" ? [horizonStart + index] : []);
  const bounded: Record<string, unknown> = {
    fallback: typeof value.fallback === "boolean"
      ? value.fallback
      : fallbackHorizons.length > 0,
    horizon_start: horizonStart,
    horizons_total: horizons.length,
    horizons_included: 0,
    horizons_truncated: horizons.length > 0,
    fallback_horizons_total: fallbackHorizons.length,
    fallback_horizons_included: 0,
    fallback_horizons_truncated: fallbackHorizons.length > 0,
    fallback_horizon_start: fallbackHorizons.length > 0
      ? Math.min(...fallbackHorizons)
      : null,
    fallback_horizon_end: fallbackHorizons.length > 0
      ? Math.max(...fallbackHorizons)
      : null,
    sources: collectKnownValues(horizons, "source", UNCERTAINTY_SOURCES),
    interval_kinds: collectKnownValues(horizons, "interval_kind", INTERVAL_KINDS),
    coverage_claims: collectKnownValues(horizons, "coverage_claim", COVERAGE_CLAIMS),
    horizon_statistics: {
      std_dev: finiteNumberRange(horizons, "std_dev"),
      upper_error_p99: finiteNumberRange(horizons, "upper_error_p99"),
      calibration_mean_error: finiteNumberRange(horizons, "calibration_mean_error"),
      calibration_count: finiteNumberRange(horizons, "calibration_count"),
    },
  };
  for (const field of ["method", "source", "calibration_source", "reason"] as const) {
    const normalized = boundedString(value[field]);
    if (normalized !== undefined) bounded[field] = normalized;
  }
  return bounded;
};

const MODEL_OBSERVABILITY_FIELDS: Record<GuidedModelType, readonly string[]> = {
  ma: [],
  es: ["alpha_comparison", "uncertainty_summary"],
  arima: ["best_order", "search_summary", "diagnostics"],
  lstm: ["uncertainty_summary"],
  weighted_avg: ["model_names", "weights", "weight_diagnostics", "split_plan"],
  boosting: [
    "model_names",
    "stage_coefficients",
    "post_line_search_shrinkage",
    "baseline",
    "split_plan",
    "search_summary",
  ],
  stacking: ["model_names", "meta_model", "split_plan"],
};

export const buildTrainingObservability = (
  modelType: GuidedModelType,
  results: Record<string, unknown>,
  metrics: Record<string, unknown>,
): Record<string, unknown> => {
  const predictionPoints = Array.isArray(results.prediction_points)
    ? results.prediction_points.filter(isRecord)
    : [];
  const collectStrings = (field: string) => Array.from(new Set(
    predictionPoints
      .map((point) => point[field])
      .filter((value): value is string => typeof value === "string" && value.length > 0),
  ));
  const calibrationCounts = predictionPoints
    .filter((point) => (
      typeof point.bias_diagnostic_source !== "string"
      || point.bias_diagnostic_source === point.calibration_source
    ))
    .map((point) => point.calibration_count)
    .filter((value): value is number => typeof value === "number" && Number.isInteger(value) && value > 0);
  const biasDiagnosticCounts = predictionPoints
    .filter((point) => typeof point.bias_diagnostic_source === "string")
    .map((point) => point.calibration_count)
    .filter((value): value is number => typeof value === "number" && Number.isInteger(value) && value > 0);
  const modelSpecific = Object.fromEntries(
    MODEL_OBSERVABILITY_FIELDS[modelType]
      .filter((field) => results[field] !== undefined)
      .map((field) => [field, results[field]]),
  );
  if (modelType === "lstm") {
    Object.assign(modelSpecific, buildBoundedLstmObservability(results));
  }
  if (modelType === "arima" && isRecord(modelSpecific.diagnostics)) {
    const { y_pred_eval: _predictionArray, ...boundedDiagnostics } = modelSpecific.diagnostics;
    modelSpecific.diagnostics = boundedDiagnostics;
  }
  if ("uncertainty_summary" in modelSpecific) {
    const boundedSummary = buildBoundedUncertaintySummary(modelSpecific.uncertainty_summary);
    if (boundedSummary === undefined) {
      delete modelSpecific.uncertainty_summary;
    } else {
      modelSpecific.uncertainty_summary = boundedSummary;
    }
  }

  return {
    method_name: results.method_name ?? null,
    forecast_strategy: results.forecast_strategy ?? null,
    implementation_notes: Array.isArray(results.implementation_notes)
      ? results.implementation_notes
      : [],
    outer_interval_evaluation: isRecord(metrics.interval_evaluation)
      ? metrics.interval_evaluation
      : null,
    uncertainty_evidence: {
      sources: collectStrings("uncertainty_source"),
      interval_kinds: collectStrings("interval_kind"),
      coverage_claims: collectStrings("coverage_claim"),
      calibration_sources: collectStrings("calibration_source"),
      bias_diagnostic_sources: collectStrings("bias_diagnostic_source"),
      calibration_count_min: calibrationCounts.length > 0 ? Math.min(...calibrationCounts) : null,
      calibration_count_max: calibrationCounts.length > 0 ? Math.max(...calibrationCounts) : null,
      bias_diagnostic_count_min: biasDiagnosticCounts.length > 0 ? Math.min(...biasDiagnosticCounts) : null,
      bias_diagnostic_count_max: biasDiagnosticCounts.length > 0 ? Math.max(...biasDiagnosticCounts) : null,
      upper_error_p99_points: predictionPoints.filter(
        (point) => typeof point.upper_error_p99 === "number",
      ).length,
    },
    ...modelSpecific,
  };
};
