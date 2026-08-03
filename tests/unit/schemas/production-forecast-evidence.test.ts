import { describe, expect, test } from "bun:test";
import crypto from "crypto";
import {
  hashProductionForecasts,
  issueProductionForecastEvidence,
  verifyProductionForecastEvidence,
} from "../../../src/modules/experiments/production-forecast-evidence";

const TEST_SECRET = "production-forecast-evidence-unit-test-secret";
const forecasts = [
  {
    prediction: 101.25,
    std_dev: 4,
    upper_error_p99: 9.3,
    uncertainty_source: "model",
    calibration_source: "model_distribution",
    calibration_mean_error: 0.5,
    calibration_count: 8,
    bias_diagnostic_source: "outer_selection_assessment",
    interval_lower: 93.4,
    interval_upper: 109.1,
    interval_level: 0.95,
    interval_kind: "model_prediction_interval",
    coverage_claim: "model_nominal",
  },
  {
    prediction: 98.5,
    std_dev: 5,
    upper_error_p99: 11.7,
    uncertainty_source: "empirical",
    uncertainty_reason: "residual calibration",
    calibration_source: "evaluation_residuals",
    calibration_mean_error: -0.25,
    calibration_count: 8,
    bias_diagnostic_source: "evaluation_residuals",
    interval_lower: 88.7,
    interval_upper: 108.3,
    interval_level: 0.95,
    interval_kind: "empirical_residual_quantile",
    coverage_claim: "none",
  },
];

describe("production forecast evidence", () => {
  test("binds a signed token to the run revision, horizon, and canonical forecasts", () => {
    const token = issueProductionForecastEvidence({
      experiment_id: 19,
      student_id: 23,
      model_type: "arima",
      run_id: "run-19",
      production_revision: "revision-7",
      forecast_steps: 2,
      forecasts,
    }, TEST_SECRET);

    expect(verifyProductionForecastEvidence(token, TEST_SECRET)).toEqual({
      ok: true,
      claims: {
        version: 1,
        experiment_id: 19,
        student_id: 23,
        model_type: "arima",
        run_id: "run-19",
        production_revision: "revision-7",
        forecast_steps: 2,
        forecast_sha256: hashProductionForecasts(forecasts),
      },
    });
    expect(token.split(".")).toHaveLength(2);
  });

  test("uses the configured JWT secret when no override is provided", () => {
    const token = issueProductionForecastEvidence({
      experiment_id: 19,
      student_id: 23,
      model_type: "arima",
      run_id: "run-19",
      production_revision: "revision-7",
      forecast_steps: 2,
      forecasts,
    });

    expect(verifyProductionForecastEvidence(token)).toMatchObject({
      ok: true,
      claims: { version: 1 },
    });
  });

  test("derives a domain-separated evidence key from JWT_SECRET", () => {
    const token = issueProductionForecastEvidence({
      experiment_id: 19,
      student_id: 23,
      model_type: "arima",
      run_id: "run-19",
      production_revision: "revision-7",
      forecast_steps: 2,
      forecasts,
    }, TEST_SECRET);
    const [payload, derivedSignature] = token.split(".") as [string, string];
    const expectedDerivedKey = crypto.createHmac("sha256", TEST_SECRET)
      .update("fz-production-forecast-evidence-signing-key-v1", "utf8")
      .digest();
    const expectedDerivedSignature = crypto.createHmac("sha256", expectedDerivedKey)
      .update(`fz-production-forecast-evidence-token-v1\n${payload}`, "utf8")
      .digest("base64url");
    const rawJwtSignature = crypto.createHmac("sha256", TEST_SECRET)
      .update(`fz-production-forecast-evidence-token-v1\n${payload}`, "utf8")
      .digest("base64url");

    expect(derivedSignature).toBe(expectedDerivedSignature);
    expect(derivedSignature).not.toBe(rawJwtSignature);
    expect(verifyProductionForecastEvidence(
      `${payload}.${rawJwtSignature}`,
      TEST_SECRET,
    )).toEqual({ ok: false });
  });

  test("uses a key-order-independent digest for persisted JSON objects", () => {
    const reordered = forecasts.map((point) => ({
      coverage_claim: point.coverage_claim,
      interval_kind: point.interval_kind,
      std_dev: point.std_dev,
      prediction: point.prediction,
      upper_error_p99: point.upper_error_p99,
      uncertainty_source: point.uncertainty_source,
      ...(point.uncertainty_reason === undefined
        ? {}
        : { uncertainty_reason: point.uncertainty_reason }),
      calibration_source: point.calibration_source,
      calibration_mean_error: point.calibration_mean_error,
      calibration_count: point.calibration_count,
      bias_diagnostic_source: point.bias_diagnostic_source,
      interval_upper: point.interval_upper,
      interval_lower: point.interval_lower,
      interval_level: point.interval_level,
    }));
    expect(hashProductionForecasts(reordered)).toBe(hashProductionForecasts(forecasts));
  });

  test("ignores runner-only point metadata that the frontend does not persist", () => {
    const runnerForecasts = forecasts.map((point, index) => ({
      ...point,
      future_diagnostic: {
        horizon: index + 1,
        source: "new-python-version",
      },
    }));
    // Mirrors parseModelPredictionResponse: the current frontend reconstructs
    // each point from the writable contract and drops unknown runner metadata.
    const frontendRoundTrip = runnerForecasts.map(({
      future_diagnostic: _runnerOnly,
      ...persistedPoint
    }) => persistedPoint);

    const token = issueProductionForecastEvidence({
      experiment_id: 19,
      student_id: 23,
      model_type: "arima",
      run_id: "run-19",
      production_revision: "revision-7",
      forecast_steps: 2,
      forecasts: runnerForecasts,
    }, TEST_SECRET);
    const verified = verifyProductionForecastEvidence(token, TEST_SECRET);

    expect(hashProductionForecasts(runnerForecasts))
      .toBe(hashProductionForecasts(frontendRoundTrip));
    expect(verified).toMatchObject({
      ok: true,
      claims: {
        forecast_sha256: hashProductionForecasts(frontendRoundTrip),
      },
    });
  });

  test("binds every persisted field preserved by the frontend roundtrip", () => {
    const frontendPersistedFields = [
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
    const baselineHash = hashProductionForecasts(forecasts);

    for (const field of frontendPersistedFields) {
      const changed = forecasts.map((point) => ({ ...point })) as Array<Record<string, unknown>>;
      changed[1]![field] = { deliberately: "different" };
      expect(hashProductionForecasts(changed), field).not.toBe(baselineHash);
    }
  });

  test("rejects payload/signature tampering and the wrong secret", () => {
    const token = issueProductionForecastEvidence({
      experiment_id: 19,
      student_id: 23,
      model_type: "lstm",
      run_id: "run-19",
      production_revision: "revision-7",
      forecast_steps: 2,
      forecasts,
    }, TEST_SECRET);
    const segments = token.split(".");
    const payloadTampered = [...segments];
    payloadTampered[0] = `${payloadTampered[0]}a`;
    const signatureTampered = [...segments];
    signatureTampered[1] = `${signatureTampered[1]}a`;

    expect(verifyProductionForecastEvidence(payloadTampered.join("."), TEST_SECRET))
      .toEqual({ ok: false });
    expect(verifyProductionForecastEvidence(signatureTampered.join("."), TEST_SECRET))
      .toEqual({ ok: false });
    expect(verifyProductionForecastEvidence(token, `${TEST_SECRET}-wrong`))
      .toEqual({ ok: false });
    expect(verifyProductionForecastEvidence("not-a-token", TEST_SECRET))
      .toEqual({ ok: false });
  });
});
