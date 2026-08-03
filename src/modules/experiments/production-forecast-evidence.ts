import crypto from "crypto";
import { z } from "zod";
import { env } from "../../config/env";
import type { ModelType } from "../models/models.schema";
import { productionForecastPeriodsSchema } from "../models/production-forecast-contract";

const EVIDENCE_KEY_DERIVATION_DOMAIN =
  "fz-production-forecast-evidence-signing-key-v1";
const EVIDENCE_TOKEN_DOMAIN = "fz-production-forecast-evidence-token-v1";
const SHA256_HEX_PATTERN = /^[a-f\d]{64}$/;
const BASE64URL_PATTERN = /^[A-Za-z0-9_-]+$/;

const productionForecastEvidenceCommonClaimsSchema = z.object({
  experiment_id: z.number().int().positive(),
  student_id: z.number().int().positive(),
  model_type: z.enum([
    "ma",
    "es",
    "arima",
    "lstm",
    "stacking",
    "weighted_avg",
    "boosting",
  ]),
  run_id: z.string().min(1),
  production_revision: z.string().min(1),
  forecast_steps: productionForecastPeriodsSchema,
  forecast_sha256: z.string().regex(SHA256_HEX_PATTERN),
});

const productionForecastEvidenceClaimsSchema =
  productionForecastEvidenceCommonClaimsSchema.extend({
    version: z.literal(1),
  }).strict();

export type ProductionForecastEvidenceClaims = z.infer<
  typeof productionForecastEvidenceClaimsSchema
>;

type IssueProductionForecastEvidenceInput = Omit<
  z.infer<typeof productionForecastEvidenceCommonClaimsSchema>,
  "forecast_sha256"
> & {
  forecasts: unknown;
};

/**
 * The evidence covers the persisted v2 forecast contract, not every diagnostic
 * key a Python runner may add to its response.  The browser deliberately
 * rebuilds prediction points from these fields before saving them, so signing
 * arbitrary passthrough keys would make an otherwise valid forecast impossible
 * to save after a forward-compatible runner change.
 *
 * Keep additions explicit: a newly persisted, integrity-relevant field should
 * be added here as part of a coordinated forecast/evidence contract upgrade.
 */
const PRODUCTION_FORECAST_EVIDENCE_POINT_FIELDS = [
  "prediction",
  "std_dev",
  "upper_error_p99",
  "uncertainty_source",
  "uncertainty_reason",
  "calibration_source",
  "calibration_mean_error",
  "calibration_count",
  "bias_diagnostic_source",
  "interval_lower",
  "interval_upper",
  "interval_level",
  "interval_kind",
  "coverage_claim",
] as const;

const projectProductionForecastsForEvidence = (forecasts: unknown): unknown => {
  if (!Array.isArray(forecasts)) return forecasts;

  return forecasts.map((point) => {
    if (!point || typeof point !== "object" || Array.isArray(point)) return point;

    const source = point as Record<string, unknown>;
    const projected: Record<string, unknown> = {};
    for (const field of PRODUCTION_FORECAST_EVIDENCE_POINT_FIELDS) {
      if (
        Object.prototype.hasOwnProperty.call(source, field)
        && source[field] !== undefined
      ) {
        projected[field] = source[field];
      }
    }
    return projected;
  });
};

const stableJsonValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(stableJsonValue);
  if (!value || typeof value !== "object") return value;

  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .filter(([, nestedValue]) => nestedValue !== undefined)
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
      .map(([key, nestedValue]) => [key, stableJsonValue(nestedValue)]),
  );
};

const encodeBase64Url = (value: string | Buffer) =>
  Buffer.from(value).toString("base64url");

const decodeBase64Url = (value: string): Buffer | null => {
  if (!BASE64URL_PATTERN.test(value)) return null;
  try {
    const decoded = Buffer.from(value, "base64url");
    return decoded.toString("base64url") === value ? decoded : null;
  } catch {
    return null;
  }
};

const deriveEvidenceSigningKey = (jwtSecret: string): Buffer => crypto
  .createHmac("sha256", jwtSecret)
  .update(EVIDENCE_KEY_DERIVATION_DOMAIN, "utf8")
  .digest();

const signPayload = (encodedPayload: string, jwtSecret: string) => crypto
  .createHmac("sha256", deriveEvidenceSigningKey(jwtSecret))
  .update(`${EVIDENCE_TOKEN_DOMAIN}\n${encodedPayload}`, "utf8")
  .digest();

const signatureMatches = (
  suppliedSignature: Buffer,
  expectedSignature: Buffer,
) => suppliedSignature.length === expectedSignature.length
  && crypto.timingSafeEqual(suppliedSignature, expectedSignature);

export const hashProductionForecasts = (forecasts: unknown): string => crypto
  .createHash("sha256")
  .update(
    JSON.stringify(stableJsonValue(projectProductionForecastsForEvidence(forecasts))),
    "utf8",
  )
  .digest("hex");

export const issueProductionForecastEvidence = (
  input: IssueProductionForecastEvidenceInput,
  jwtSecretOverride?: string,
): string => {
  const claims: ProductionForecastEvidenceClaims = {
    version: 1,
    experiment_id: input.experiment_id,
    student_id: input.student_id,
    model_type: input.model_type,
    run_id: input.run_id,
    production_revision: input.production_revision,
    forecast_steps: input.forecast_steps,
    forecast_sha256: hashProductionForecasts(input.forecasts),
  };
  const parsedClaims = productionForecastEvidenceClaimsSchema.parse(claims);
  const encodedPayload = encodeBase64Url(JSON.stringify(parsedClaims));
  const signature = signPayload(encodedPayload, jwtSecretOverride ?? env.JWT_SECRET);
  return `${encodedPayload}.${encodeBase64Url(signature)}`;
};

export const verifyProductionForecastEvidence = (
  token: unknown,
  jwtSecretOverride?: string,
): { ok: true; claims: ProductionForecastEvidenceClaims } | { ok: false } => {
  if (typeof token !== "string" || token.length === 0 || token.length > 4096) {
    return { ok: false };
  }

  const segments = token.split(".");
  if (segments.length !== 2) return { ok: false };

  const [encodedPayload, encodedSignature] = segments;
  if (!encodedPayload || !encodedSignature) return { ok: false };

  const payload = decodeBase64Url(encodedPayload);
  const suppliedSignature = decodeBase64Url(encodedSignature);
  if (!payload || !suppliedSignature || suppliedSignature.length !== 32) {
    return { ok: false };
  }

  const expectedSignature = signPayload(
    encodedPayload,
    jwtSecretOverride ?? env.JWT_SECRET,
  );
  if (!signatureMatches(suppliedSignature, expectedSignature)) return { ok: false };

  try {
    const decodedClaims = JSON.parse(payload.toString("utf8"));
    const parsed = productionForecastEvidenceClaimsSchema.safeParse(decodedClaims);
    return parsed.success ? { ok: true, claims: parsed.data } : { ok: false };
  } catch {
    return { ok: false };
  }
};

export const EXPERIMENT_BEST_MODEL_TO_RUNTIME_MODEL = {
  ma: "ma",
  exp: "es",
  arima: "arima",
  lstm: "lstm",
  ensemble_weighted: "weighted_avg",
  ensemble_boosting: "boosting",
  ensemble_stacking: "stacking",
} as const satisfies Record<string, ModelType>;
