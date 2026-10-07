import { z } from "zod";
import {
  EXPERIMENT_BEST_MODEL_TO_RUNTIME_MODEL,
  hashProductionForecasts,
  verifyProductionForecastEvidence,
  type ProductionForecastEvidenceClaims,
} from "./production-forecast-evidence";
import {
  PRODUCTION_FORECAST_PERIODS_MAX,
  PRODUCTION_FORECAST_PERIODS_MIN,
} from "../models/production-forecast-contract";

export {
  PRODUCTION_FORECAST_PERIODS_MAX,
  PRODUCTION_FORECAST_PERIODS_MIN,
} from "../models/production-forecast-contract";

export const STATUS_VALUES = ["Not Started", "In Progress", "Completed"] as const;
export const BEST_MODEL_VALUES = ["ma", "exp", "arima", "lstm", "ensemble_weighted", "ensemble_boosting", "ensemble_stacking"] as const;
export const NORMALIZATION_VALUES = ["minmax", "zscore"] as const;
export const CAPACITY_MODE_VALUES = ["scenario", "auto", "custom"] as const;
export const CAPACITY_SCENARIO_VALUES = ["tight", "normal", "abundant"] as const;
export const UNCERTAINTY_SOURCE_VALUES = ["model", "empirical", "fallback"] as const;
export const PREDICTION_INTERVAL_KIND_VALUES = [
  "model_prediction_interval",
  "empirical_residual_quantile",
  "normal_approximation",
  "fallback_normal_approximation",
] as const;
export const PREDICTION_COVERAGE_CLAIM_VALUES = ["model_nominal", "none"] as const;
export const PRODUCTION_SERVICE_LEVEL_PAIRS = [
  { serviceLevel: 0.9, zScore: 1.28 },
  { serviceLevel: 0.95, zScore: 1.65 },
  { serviceLevel: 0.99, zScore: 2.33 },
] as const;

const numberPreprocess = (value: unknown) => {
  if (value === undefined || value === null) return value;
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const parsed = Number(trimmed);
    return Number.isNaN(parsed) ? value : parsed;
  }
  return value;
};

const booleanPreprocess = (value: unknown) => {
  if (value === undefined || value === null) return value;
  if (typeof value === "string") {
    const trimmed = value.trim().toLowerCase();
    if (!trimmed) return null;
    if (trimmed === "true" || trimmed === "1") return true;
    if (trimmed === "false" || trimmed === "0") return false;
  }
  if (typeof value === "number") {
    if (value === 1) return true;
    if (value === 0) return false;
  }
  return value;
};

const stringPreprocess = (value: unknown) => {
  if (value === undefined || value === null) return null;
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length ? trimmed : null;
  }
  return value;
};

const jsonPreprocess = (value: unknown) => {
  if (value === undefined || value === null) return null;
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return null;
    try { return JSON.parse(trimmed); } catch { return trimmed; }
  }
  return value;
};

const buildIntSchema = (min?: number, max?: number) => {
  let schema = z.number().int();
  if (min !== undefined) schema = schema.min(min);
  if (max !== undefined) schema = schema.max(max);
  return schema;
};

const buildFloatSchema = (min?: number) => {
  let schema = z.number();
  if (min !== undefined) schema = schema.min(min);
  return schema;
};

const optionalInt = (min?: number, max?: number) => z.preprocess(numberPreprocess, buildIntSchema(min, max)).nullable().optional();
const optionalFloat = (min?: number) => z.preprocess(numberPreprocess, buildFloatSchema(min)).nullable().optional();
const optionalBoolean = z.preprocess(booleanPreprocess, z.boolean()).nullable().optional();
const optionalString = z.preprocess(stringPreprocess, z.string()).nullable().optional();
const optionalJson = z.preprocess(jsonPreprocess, z.union([z.array(z.any()), z.record(z.string(), z.any()), z.string()])).nullable().optional();
const adfResultRowSchema = z.object({
  diff_order: z.number().int().min(0).max(2),
  statistic: z.number().finite(),
  p_value: z.number().finite().min(0).max(1),
  used_lags: z.number().int().min(0),
  n_obs: z.number().int().min(1),
  stationary: z.boolean(),
  critical_values: z.object({
    "1%": z.number().finite(),
    "5%": z.number().finite(),
    "10%": z.number().finite(),
  }).strict(),
}).strict();
const adfResultRowsSchema = z.array(adfResultRowSchema).max(3).superRefine((rows, context) => {
  const diffOrders = rows.map((row) => row.diff_order);
  if (new Set(diffOrders).size !== diffOrders.length) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "ADF 检验结果不能包含重复的差分阶数",
    });
  }
});
const optionalAdfResults = z.preprocess(jsonPreprocess, adfResultRowsSchema).nullable().optional();

/**
 * Production forecasts written before interval metadata was introduced.  These
 * are deliberately kept separate from the current contract: they can be read
 * to grandfather completed experiments and to support the bounded v1 rollout
 * bridge. Semantic validation decides whether an active v1 write is enabled.
 */
const productionPredictionPointV1Schema = z.object({
  prediction: z.number().finite().min(0),
  std_dev: z.number().finite().min(0),
}).passthrough();

export const productionPredictionPointSchema = z.object({
  prediction: z.number().finite().min(0),
  std_dev: z.number().finite().min(0),
  upper_error_p99: z.number().finite().min(0),
  uncertainty_source: z.enum(UNCERTAINTY_SOURCE_VALUES),
  uncertainty_reason: z.string().optional(),
  calibration_source: z.string().min(1).optional(),
  calibration_mean_error: z.number().finite().nullable(),
  calibration_count: z.number().int().min(1).nullable(),
  bias_diagnostic_source: z.string().min(1).nullable().optional(),
  interval_lower: z.number().finite(),
  interval_upper: z.number().finite(),
  interval_level: z.literal(0.95),
  interval_kind: z.enum(PREDICTION_INTERVAL_KIND_VALUES),
  coverage_claim: z.enum(PREDICTION_COVERAGE_CLAIM_VALUES),
}).passthrough().superRefine((point, context) => {
  const bothCalibrationValuesNull = (
    point.calibration_mean_error === null
    && point.calibration_count === null
  );
  const bothCalibrationValuesPresent = (
    point.calibration_mean_error !== null
    && point.calibration_count !== null
  );
  if (!bothCalibrationValuesNull && !bothCalibrationValuesPresent) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "校准平均误差和校准样本数必须同时提供，或同时为 null",
      path: ["calibration_count"],
    });
  }
  if (point.interval_lower > point.interval_upper) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "预测区间下界不能大于上界",
      path: ["interval_upper"],
    });
  }
  const expectedCoverageClaim = point.interval_kind === "model_prediction_interval"
    ? "model_nominal"
    : "none";
  if (point.coverage_claim !== expectedCoverageClaim) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: `${point.interval_kind} 的 coverage_claim 必须为 ${expectedCoverageClaim}`,
      path: ["coverage_claim"],
    });
  }
});

export const productionForecastResultsSchema = z.preprocess(
  jsonPreprocess,
  z.array(productionPredictionPointSchema).max(PRODUCTION_FORECAST_PERIODS_MAX),
);

const productionForecastResultsV1Schema = z.preprocess(
  jsonPreprocess,
  z.array(productionPredictionPointV1Schema).max(PRODUCTION_FORECAST_PERIODS_MAX),
);

const PRODUCTION_INTERVAL_FIELDS = [
  "interval_lower",
  "interval_upper",
  "interval_level",
  "interval_kind",
  "coverage_claim",
] as const;

export type ProductionForecastResultsVersion = "v1" | "v2";

export type CompatibleProductionForecastResults = {
  version: ProductionForecastResultsVersion;
  data: z.infer<typeof productionPredictionPointV1Schema>[];
};

/**
 * Classify persisted forecast JSON by its interval tuple instead of a database
 * migration flag.  A tuple must be absent from every point (v1) or present on
 * every point (v2); a partially populated or mixed array is never trustworthy.
 */
export const classifyProductionForecastResults = (
  value: unknown,
): ProductionForecastResultsVersion | "partial" | null => {
  const raw = z.preprocess(jsonPreprocess, z.array(z.unknown())
    .max(PRODUCTION_FORECAST_PERIODS_MAX)).safeParse(value);
  if (!raw.success) return null;
  if (raw.data.length === 0) return "v2";

  let version: ProductionForecastResultsVersion | null = null;
  for (const point of raw.data) {
    if (!point || typeof point !== "object" || Array.isArray(point)) return null;
    const intervalFieldCount = PRODUCTION_INTERVAL_FIELDS.filter((field) =>
      Object.prototype.hasOwnProperty.call(point, field),
    ).length;
    if (intervalFieldCount > 0 && intervalFieldCount < PRODUCTION_INTERVAL_FIELDS.length) {
      return "partial";
    }

    const pointVersion: ProductionForecastResultsVersion = intervalFieldCount === 0 ? "v1" : "v2";
    if (version && version !== pointVersion) return "partial";
    version = pointVersion;
  }
  return version;
};

export const parseProductionForecastResults = (
  value: unknown,
): z.infer<typeof productionPredictionPointSchema>[] | null => {
  const parsed = productionForecastResultsSchema.safeParse(value);
  return parsed.success ? parsed.data : null;
};

export const parseCompatibleProductionForecastResults = (
  value: unknown,
): CompatibleProductionForecastResults | null => {
  const version = classifyProductionForecastResults(value);
  if (version === null || version === "partial") return null;
  const schema = version === "v2"
    ? productionForecastResultsSchema
    : productionForecastResultsV1Schema;
  const parsed = schema.safeParse(value);
  return parsed.success ? { version, data: parsed.data } : null;
};

const compatibleProductionForecastResultsWriteSchema = z.preprocess(
  jsonPreprocess,
  z.array(z.unknown()).max(PRODUCTION_FORECAST_PERIODS_MAX),
).transform((value, context) => {
  const parsed = parseCompatibleProductionForecastResults(value);
  if (!parsed) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "生产预测结果必须使用完整且一致的 v1 或 v2 格式",
    });
    return z.NEVER;
  }
  return parsed.data;
});

// Keep the transport schema capable of decoding both adjacent contract
// versions. The service-level rollout flag below is the authorization boundary
// for active v1 writes; partial and mixed-version arrays never reach it.
const optionalProductionForecastResults = compatibleProductionForecastResultsWriteSchema
  .nullable()
  .optional();

/**
 * Current (v3) MPS row.  Field ↔ 需求文档第五章 名词：
 *
 *   demand_forecast      需求量（需求预测结果）
 *   safety_stock         安全库存（首期为 0）
 *   forecast_quantity    预测量 = 需求预测结果 + 安全库存
 *   planned_production   投入量 = 预测量 - 上月库存量 + 上月缺货量
 *   beginning_inventory  上月库存量（首期为期初库存）
 *   production_output    产出量 = 上月投入量（受产能约束；首期 = 需求量）
 *   ending_inventory     库存量 = max(0, 余额)
 *   stockout             缺货量 = max(0, -余额)，连续缺货时含上期未补部分
 *   service_level        服务水平 = max(0, 1 - 缺货量 / 需求量)
 *
 * 余额_t = 余额_{t-1} + 产出量_t - 需求量_t，余额_0 = 期初库存。
 */
export const productionMpsRowSchema = z.object({
  period: z.number().int().min(1).max(PRODUCTION_FORECAST_PERIODS_MAX),
  period_label: z.string().trim().min(1),
  demand_forecast: z.number().finite().min(0),
  safety_stock: z.number().finite().min(0),
  forecast_quantity: z.number().finite().min(0),
  planned_production: z.number().finite().min(0),
  beginning_inventory: z.number().finite().min(0),
  production_output: z.number().finite().min(0),
  ending_inventory: z.number().finite().min(0),
  stockout: z.number().finite().min(0),
  service_level: z.number().finite().min(0).max(1),
}).strict();

export const productionMpsTableSchema = z.preprocess(
    jsonPreprocess,
    z.array(productionMpsRowSchema)
      .max(PRODUCTION_FORECAST_PERIODS_MAX)
      .superRefine((rows, context) => {
        rows.forEach((row, index) => {
          const expectedPeriod = index + 1;
          if (row.period !== expectedPeriod) {
            context.addIssue({
              code: z.ZodIssueCode.custom,
              message: `MPS 第 ${expectedPeriod} 行的 period 必须为 ${expectedPeriod}`,
              path: [index, "period"],
            });
          }
          if (row.period_label !== `期 ${expectedPeriod}`) {
            context.addIssue({
              code: z.ZodIssueCode.custom,
              message: `MPS 第 ${expectedPeriod} 行的 period_label 必须为“期 ${expectedPeriod}”`,
              path: [index, "period_label"],
            });
          }
        });
      }),
  );

export const parseProductionMpsTable = (
  value: unknown,
): z.infer<typeof productionMpsRowSchema>[] | null => {
  const parsed = productionMpsTableSchema.safeParse(value);
  return parsed.success ? parsed.data : null;
};

const optionalProductionMpsTable = productionMpsTableSchema.nullable().optional();
const optionalProductionForecastEvidence = z.preprocess(
  stringPreprocess,
  z.string().max(4096),
).nullable().optional();

const toFiniteNumber = (value: unknown): number | null => {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const toBoolean = (value: unknown): boolean => {
  if (value === true || value === 1) return true;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    return normalized === "true" || normalized === "1";
  }
  return false;
};

const numbersEqual = (actual: number, expected: number): boolean =>
  Math.abs(actual - expected) <= 1e-9;

export type ProductionPlanValidationCode =
  | "PRODUCTION_PLAN_INVALID"
  | "PRODUCTION_PLAN_MISMATCH"
  | "PRODUCTION_FORECAST_LEGACY_REJECTED"
  | "PRODUCTION_FORECAST_EVIDENCE_REQUIRED"
  | "PRODUCTION_FORECAST_EVIDENCE_INVALID"
  | "PRODUCTION_FORECAST_EVIDENCE_STALE";

export type ProductionPlanValidationResult =
  | {
    ok: true;
    evidenceClaims: ProductionForecastEvidenceClaims | null;
  }
  | {
    ok: false;
    code: ProductionPlanValidationCode;
    message: string;
  };

const invalidProductionPlan = (
  message: string,
  code: ProductionPlanValidationCode = "PRODUCTION_PLAN_INVALID",
): ProductionPlanValidationResult => ({ ok: false, code, message });

const hasProductionForecastEvidenceValue = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0;

const bindWritableV2ForecastEvidence = (
  snapshot: Record<string, unknown>,
  forecasts: CompatibleProductionForecastResults["data"],
  periods: number,
): ProductionPlanValidationResult => {
  if (!hasProductionForecastEvidenceValue(snapshot.production_forecast_evidence)) {
    return invalidProductionPlan(
      "生产预测缺少服务端生成的有效凭证，请重新生成预测",
      "PRODUCTION_FORECAST_EVIDENCE_REQUIRED",
    );
  }
  const evidence = verifyProductionForecastEvidence(
    snapshot.production_forecast_evidence,
  );
  if (!evidence.ok) {
    return invalidProductionPlan(
      "生产预测凭证无效，请重新生成预测",
      "PRODUCTION_FORECAST_EVIDENCE_INVALID",
    );
  }

  const experimentId = toFiniteNumber(snapshot.experiment_id);
  const studentId = toFiniteNumber(snapshot.student_id);
  const selectedBestModel = typeof snapshot.selected_best_model === "string"
    ? snapshot.selected_best_model
    : null;
  const runtimeModel = selectedBestModel
    ? EXPERIMENT_BEST_MODEL_TO_RUNTIME_MODEL[
      selectedBestModel as keyof typeof EXPERIMENT_BEST_MODEL_TO_RUNTIME_MODEL
    ]
    : undefined;
  const claimsMatchSnapshot = (
    experimentId !== null
    && Number.isInteger(experimentId)
    && studentId !== null
    && Number.isInteger(studentId)
    && runtimeModel !== undefined
    && evidence.claims.experiment_id === experimentId
    && evidence.claims.student_id === studentId
    && evidence.claims.model_type === runtimeModel
    && evidence.claims.forecast_steps === periods
    && evidence.claims.forecast_sha256 === hashProductionForecasts(forecasts)
  );
  if (!claimsMatchSnapshot) {
    return invalidProductionPlan(
      "生产预测凭证与当前实验、模型或预测结果不匹配，请重新生成预测",
      "PRODUCTION_FORECAST_EVIDENCE_STALE",
    );
  }
  return { ok: true, evidenceClaims: evidence.claims };
};

export interface ProductionPlanValidationOptions {
  /**
   * Temporary compatibility bridge for already-open pages using the immediately
   * previous browser contract. Completed v1 snapshots are independently grandfathered.
   */
  allowLegacyV1Write?: boolean;
}

export const buildProductionCapacityScenarios = (
  forecasts: Array<{ prediction: number; std_dev: number }>,
  zScore: number,
) => {
  const forecastLoads = forecasts.map((forecast, index) => {
    const demand = Math.max(0, Math.round(forecast.prediction));
    const safetyStock = index === 0
      ? 0
      : Math.max(0, Math.round(zScore * forecast.std_dev));
    return demand + safetyStock;
  });
  const averageLoad = Math.round(
    forecastLoads.reduce((sum, load) => sum + load, 0) / forecastLoads.length,
  );
  const peakLoad = Math.max(...forecastLoads);

  return {
    tight: Math.round(averageLoad * 0.9),
    normal: averageLoad,
    abundant: Math.max(peakLoad, Math.round(averageLoad * 1.1)),
  } as const;
};

export interface ProductionMpsInput {
  forecasts: Array<{ prediction: number; std_dev: number }>;
  zScore: number;
  initialInventory: number;
  capacity: number;
}

/**
 * Canonical (v3) MPS formula, following 需求文档第五章 with a backorder
 * inventory balance.  With 期初库存 = 0 and unconstrained capacity every cell
 * matches the document's worked formulas:
 *
 *   第一个月标准化：产出量 = 需求量，安全库存 = 0，库存量 = 缺货量 = 0，
 *                   投入量 = 预测量
 *   预测量 = 需求预测结果 + 安全库存                       （5.4）
 *   投入量 = 预测量 - 上月库存量 + 上月缺货量               （5.5）
 *   产出量 = 上月投入量（提前期 1 个月，受产能约束）          （5.2）
 *   库存量 / 缺货量 来自余额 = 上月余额 + 产出量 - 需求量     （5.2 的通式）
 *   服务水平 = 1 - 缺货量 / 实际需求量                       （5.3）
 */
export const buildCanonicalProductionMps = (input: ProductionMpsInput) => {
  const rows: z.infer<typeof productionMpsRowSchema>[] = [];
  let previousBalance = input.initialInventory;
  input.forecasts.forEach((forecast, index) => {
    const previous = index > 0 ? rows[index - 1]! : null;
    const demand = Math.max(0, Math.round(forecast.prediction));
    const safetyStock = index === 0
      ? 0
      : Math.max(0, Math.round(input.zScore * forecast.std_dev));
    const forecastQuantity = demand + safetyStock;
    const beginningInventory = Math.max(0, previousBalance);
    const productionOutput = previous
      ? Math.max(0, Math.min(previous.planned_production, input.capacity))
      : demand;
    const balance = previousBalance + productionOutput - demand;
    const endingInventory = Math.max(0, balance);
    const stockout = Math.max(0, -balance);
    const serviceLevel = demand > 0 ? Math.max(0, 1 - stockout / demand) : 1;
    // 投入量 = 预测量 - 上月库存量 + 上月缺货量 = 预测量 - 上月余额
    const plannedProduction = Math.max(0, forecastQuantity - previousBalance);

    rows.push({
      period: index + 1,
      period_label: `期 ${index + 1}`,
      demand_forecast: demand,
      safety_stock: safetyStock,
      forecast_quantity: forecastQuantity,
      planned_production: plannedProduction,
      beginning_inventory: beginningInventory,
      production_output: productionOutput,
      ending_inventory: endingInventory,
      stockout,
      service_level: serviceLevel,
    });
    previousBalance = balance;
  });
  return rows;
};

export const resolveProductionServiceLevelPair = (
  serviceLevelValue: unknown,
  zScoreValue: unknown,
) => {
  const serviceLevel = toFiniteNumber(serviceLevelValue);
  const zScore = toFiniteNumber(zScoreValue);
  if (serviceLevel === null || zScore === null) return null;
  return PRODUCTION_SERVICE_LEVEL_PAIRS.find(
    (pair) => numbersEqual(serviceLevel, pair.serviceLevel)
      && numbersEqual(zScore, pair.zScore),
  ) ?? null;
};

/**
 * Validates a persisted production-plan snapshot.
 *
 * Incomplete drafts may omit forecasts entirely. Once a draft carries forecast
 * results or evidence, those fields must bind to the selected model even if
 * the plan is not yet marked completed. A completed snapshot must also replay
 * MPS, capacity, and service-level fields.
 */
export const validateCompletedProductionPlan = (
  snapshot: Record<string, unknown>,
  options: ProductionPlanValidationOptions = {},
): ProductionPlanValidationResult => {
  const compatibleForecasts = parseCompatibleProductionForecastResults(
    snapshot.production_forecast_results,
  );
  if (
    compatibleForecasts?.version === "v1"
    && snapshot.status !== "Completed"
    && !options.allowLegacyV1Write
  ) {
    return invalidProductionPlan(
      "生产预测结果为旧版格式，请重新生成并保存生产计划",
      "PRODUCTION_FORECAST_LEGACY_REJECTED",
    );
  }
  if (
    compatibleForecasts?.version === "v1"
    && snapshot.status !== "Completed"
    && options.allowLegacyV1Write
    && snapshot.production_forecast_evidence !== null
    && snapshot.production_forecast_evidence !== undefined
  ) {
    return invalidProductionPlan("旧版生产预测不能携带新版签名凭证");
  }
  if (!toBoolean(snapshot.production_plan_completed)) {
    const hasEvidence = hasProductionForecastEvidenceValue(
      snapshot.production_forecast_evidence,
    );
    if (compatibleForecasts?.version !== "v2" && !hasEvidence) {
      return { ok: true, evidenceClaims: null };
    }
    if (compatibleForecasts?.version !== "v2") {
      return invalidProductionPlan(
        "生产预测凭证与当前实验、模型或预测结果不匹配，请重新生成预测",
        "PRODUCTION_FORECAST_EVIDENCE_STALE",
      );
    }
    const draftPeriods = toFiniteNumber(snapshot.production_forecast_periods);
    if (
      draftPeriods === null
      || !Number.isInteger(draftPeriods)
      || draftPeriods !== compatibleForecasts.data.length
    ) {
      return invalidProductionPlan("生产预测结果必须完整覆盖全部预测期");
    }
    return bindWritableV2ForecastEvidence(
      snapshot,
      compatibleForecasts.data,
      draftPeriods,
    );
  }
  const isCompletedV1Snapshot = (
    snapshot.status === "Completed"
    && compatibleForecasts?.version === "v1"
  );
  const periods = toFiniteNumber(snapshot.production_forecast_periods);
  const minimumPeriods = isCompletedV1Snapshot ? 1 : PRODUCTION_FORECAST_PERIODS_MIN;
  if (
    periods === null
    || !Number.isInteger(periods)
    || periods < minimumPeriods
    || periods > PRODUCTION_FORECAST_PERIODS_MAX
  ) {
    return invalidProductionPlan(
      `预测期数必须是 ${minimumPeriods}-${PRODUCTION_FORECAST_PERIODS_MAX} 的整数`,
    );
  }

  if (!compatibleForecasts || compatibleForecasts.data.length !== periods) {
    return invalidProductionPlan("生产预测结果必须完整覆盖全部预测期");
  }

  if (compatibleForecasts.version === "v1" && snapshot.status === "Completed") {
    // v1 snapshots predate interval metadata and the exact MPS replay rules.
    // Keep immutable historical records readable without pretending their
    // pre-contract rows can satisfy the current exact replay rules.
    const legacyMpsTable = z.preprocess(
      jsonPreprocess,
      z.array(z.unknown()).max(PRODUCTION_FORECAST_PERIODS_MAX),
    ).safeParse(snapshot.production_mps_table);
    if (!legacyMpsTable.success || legacyMpsTable.data.length !== periods) {
      return invalidProductionPlan("MPS 表必须包含与预测期数一致的完整连续行");
    }
    return { ok: true, evidenceClaims: null };
  }

  const forecasts = compatibleForecasts.data;
  let evidenceClaims: ProductionForecastEvidenceClaims | null = null;
  if (compatibleForecasts.version === "v2" && snapshot.status !== "Completed") {
    const evidenceBinding = bindWritableV2ForecastEvidence(snapshot, forecasts, periods);
    if (!evidenceBinding.ok) return evidenceBinding;
    evidenceClaims = evidenceBinding.evidenceClaims;
  }

  const mpsTable = parseProductionMpsTable(snapshot.production_mps_table);
  if (!mpsTable || mpsTable.length !== periods) {
    return invalidProductionPlan("MPS 表必须包含与预测期数一致的完整连续行");
  }

  const servicePair = resolveProductionServiceLevelPair(
    snapshot.production_target_service_level,
    snapshot.production_safety_stock_z_score,
  );
  if (!servicePair) {
    return invalidProductionPlan("目标服务水平与安全库存 Z 值不是受支持的配对");
  }

  const initialInventory = toFiniteNumber(snapshot.production_initial_inventory);
  if (initialInventory === null || initialInventory < 0) {
    return invalidProductionPlan("期初库存必须是非负有限数字");
  }

  const capacityMode = snapshot.production_capacity_mode;
  const capacity = toFiniteNumber(snapshot.production_capacity);
  const customCapacity = toFiniteNumber(snapshot.production_custom_capacity);
  const capacityScenario = snapshot.production_capacity_scenario;
  if (!CAPACITY_MODE_VALUES.includes(capacityMode as typeof CAPACITY_MODE_VALUES[number])) {
    return invalidProductionPlan("产能模式无效");
  }
  if (capacity === null || !Number.isInteger(capacity) || capacity < 0) {
    return invalidProductionPlan("生产产能必须是非负整数");
  }
  if (capacityMode === "auto" && snapshot.status !== "Completed") {
    return invalidProductionPlan("自动产能模式已停用，请选择场景产能或自定义产能");
  }
  if (capacityMode === "scenario") {
    if (!CAPACITY_SCENARIO_VALUES.includes(
      capacityScenario as typeof CAPACITY_SCENARIO_VALUES[number],
    )) {
      return invalidProductionPlan("场景产能模式必须选择有效的产能场景");
    }
    const capacities = buildProductionCapacityScenarios(forecasts, servicePair.zScore);
    const expectedCapacity = capacities[
      capacityScenario as typeof CAPACITY_SCENARIO_VALUES[number]
    ];
    if (capacity !== expectedCapacity) {
      return invalidProductionPlan(
        `生产产能与 ${String(capacityScenario)} 场景的服务端计算结果不一致`,
        "PRODUCTION_PLAN_MISMATCH",
      );
    }
  }
  if (
    capacityMode === "custom"
    && (
      customCapacity === null
      || !Number.isInteger(customCapacity)
      || customCapacity <= 0
      || customCapacity !== capacity
    )
  ) {
    return invalidProductionPlan("自定义产能必须是正整数并与生产产能一致");
  }

  const expectedMpsTable = buildCanonicalProductionMps({
    forecasts,
    zScore: servicePair.zScore,
    initialInventory,
    capacity,
  });
  for (let index = 0; index < periods; index += 1) {
    const row = mpsTable[index]!;
    const expectedRow = expectedMpsTable[index]!;
    for (const [field, expected] of Object.entries(expectedRow)) {
      const actual = row[field as keyof typeof expectedRow];
      const matches = typeof expected === "number"
        ? typeof actual === "number" && numbersEqual(actual, expected)
        : actual === expected;
      if (!matches) {
        return invalidProductionPlan(
          `MPS 第 ${index + 1} 行的 ${field} 与服务端生产计划公式不一致`,
          "PRODUCTION_PLAN_MISMATCH",
        );
      }
    }
  }

  return { ok: true, evidenceClaims };
};

export const getCompletedProductionPlanValidationError = (
  snapshot: Record<string, unknown>,
  options: ProductionPlanValidationOptions = {},
): string | null => {
  const result = validateCompletedProductionPlan(snapshot, options);
  return result.ok ? null : result.message;
};

const studentWritableFields: Record<string, z.ZodTypeAny> = {
  highest_completed_step: optionalInt(0, 7),
  selected_industry: optionalString, selected_company: optionalString, selected_product: optionalString,
  selected_base_models: optionalJson, selected_ensemble_models: optionalJson,
  data_window_train_start_index: optionalInt(0), data_window_train_end_index: optionalInt(0),
  data_window_evaluate_start_index: optionalInt(0), data_window_evaluate_end_index: optionalInt(0),
  moving_average_window: optionalInt(2),
  exponential_smoothing_alpha: optionalFloat(0),
  arima_d: optionalInt(0, 2), arima_adf_stationarity: optionalAdfResults,
  lstm_normalization: z.preprocess(stringPreprocess, z.enum(NORMALIZATION_VALUES)).nullable().optional(),
  lstm_features: optionalJson, lstm_target_field: optionalString,
  ensemble_weighted_base_models: optionalJson,
  ensemble_boosting_base_models: optionalJson,
  ensemble_stacking_base_models: optionalJson,
  selected_best_model: z.preprocess(stringPreprocess, z.enum(BEST_MODEL_VALUES)).nullable().optional(),
  production_plan_completed: optionalBoolean,
  production_forecast_periods: optionalInt(
    PRODUCTION_FORECAST_PERIODS_MIN,
    PRODUCTION_FORECAST_PERIODS_MAX,
  ),
  production_initial_inventory: optionalFloat(0),
  production_target_service_level: optionalFloat(0),
  production_safety_stock_z_score: optionalFloat(0),
  production_mps_table: optionalProductionMpsTable,
  production_forecast_results: optionalProductionForecastResults,
  production_forecast_evidence: optionalProductionForecastEvidence,
  production_capacity_mode: z.preprocess(stringPreprocess, z.enum(CAPACITY_MODE_VALUES)).nullable().optional(),
  production_capacity_scenario: z.preprocess(stringPreprocess, z.enum(CAPACITY_SCENARIO_VALUES)).nullable().optional(),
  production_capacity: optionalInt(0), production_custom_capacity: optionalInt(0),
};

const studentInitialWritableFields: Record<string, z.ZodTypeAny> = {
  highest_completed_step: studentWritableFields.highest_completed_step!,
  selected_industry: studentWritableFields.selected_industry!,
  selected_company: studentWritableFields.selected_company!,
  selected_product: studentWritableFields.selected_product!,
  selected_base_models: studentWritableFields.selected_base_models!,
  selected_ensemble_models: studentWritableFields.selected_ensemble_models!,
  data_window_train_start_index: studentWritableFields.data_window_train_start_index!,
  data_window_train_end_index: studentWritableFields.data_window_train_end_index!,
  data_window_evaluate_start_index: studentWritableFields.data_window_evaluate_start_index!,
  data_window_evaluate_end_index: studentWritableFields.data_window_evaluate_end_index!,
  moving_average_window: studentWritableFields.moving_average_window!,
  exponential_smoothing_alpha: studentWritableFields.exponential_smoothing_alpha!,
  arima_d: studentWritableFields.arima_d!,
  arima_adf_stationarity: studentWritableFields.arima_adf_stationarity!,
  lstm_normalization: studentWritableFields.lstm_normalization!,
  lstm_features: studentWritableFields.lstm_features!,
  lstm_target_field: studentWritableFields.lstm_target_field!,
  ensemble_weighted_base_models: studentWritableFields.ensemble_weighted_base_models!,
  ensemble_boosting_base_models: studentWritableFields.ensemble_boosting_base_models!,
  ensemble_stacking_base_models: studentWritableFields.ensemble_stacking_base_models!,
};

export const experimentIdParamSchema = z.object({
  experimentId: z.string().regex(/^\d+$/, { message: "experimentId 必须是数字" }),
});

export const createExperimentSchema = z.object({
  current_step: optionalInt(1, 7),
  ...studentInitialWritableFields,
}).strict();

export const retryExperimentSchema = z.object({
  rejected_report_id: z.number().int().positive(),
}).strict();

export const updateExperimentSchema = z.object({
  expected_version: z.coerce.number().int().min(0),
  current_step: optionalInt(1, 7),
  ...studentWritableFields,
}).strict().refine((data) => Object.keys(data).some((key) => key !== "expected_version"), "至少需要提供一个更新字段");

export const experimentEventSchema = z.object({
  step_order: z.number().int().min(1).max(7),
  event_type: z.enum(["STARTED", "COMPLETED"]),
});

export type ExperimentIdParam = z.infer<typeof experimentIdParamSchema>;
export type CreateExperimentInput = z.infer<typeof createExperimentSchema>;
export type RetryExperimentInput = z.infer<typeof retryExperimentSchema>;
export type UpdateExperimentInput = z.infer<typeof updateExperimentSchema>;
export type ExperimentEventInput = z.infer<typeof experimentEventSchema>;
