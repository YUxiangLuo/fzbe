process.env.DB_HOST ??= "127.0.0.1";
process.env.DB_USER ??= "test";
process.env.DB_PASSWORD ??= "test";
process.env.DB_DATABASE ??= "test";
process.env.JWT_SECRET = "test-jwt-secret";
process.env.PORT ??= "3001";

import { describe, expect, test } from "bun:test";
import * as actualFs from "node:fs";
import path from "node:path";
import { mock } from "bun:test";

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const pythonModulePath = path.join(srcRoot, "utils", "python.ts");

mock.module(configModulePath, () => ({
  env: {
    DB_HOST: process.env.DB_HOST ?? "127.0.0.1",
    DB_USER: process.env.DB_USER ?? "test",
    DB_PASSWORD: process.env.DB_PASSWORD ?? "test",
    DB_DATABASE: process.env.DB_DATABASE ?? "test",
    JWT_SECRET: "test-jwt-secret",
    PORT: Number(process.env.PORT ?? "3001"),
  },
  connectionLimit: 64,
  pool: {
    query: async () => [[]],
    getConnection: async () => ({
      query: async () => [[]],
      release: () => {},
    }),
  },
}));

mock.module("fs", () => ({
  ...actualFs,
  default: {
    ...actualFs.default,
    existsSync: () => false,
    createReadStream: () => {
      throw new Error("createReadStream should not be called in transformer tests");
    },
    readdirSync: () => [],
    unlinkSync: () => {},
  },
  existsSync: () => false,
  createReadStream: () => {
    throw new Error("createReadStream should not be called in transformer tests");
  },
  readdirSync: () => [],
  unlinkSync: () => {},
}));

mock.module(pythonModulePath, () => ({
  PY_DIR: "/tmp/fangzhen-py",
  resolvePythonExecutable: () => "/tmp/fangzhen-py/venv/bin/python",
}));

const modelsServiceModulePath = path.join(srcRoot, "modules", "models", "models.service.ts");
const freshModelsServiceModulePath = `${modelsServiceModulePath}?models-result-transformers-test`;

const { transformTrainingResult } = await import(freshModelsServiceModulePath);
const { normalizeDemandPredictionPoints } = await import(
  `${path.join(srcRoot, "modules", "models", "model-results.ts")}?models-result-normalization-test`
);

const predictionPoint = (prediction: number, stdDev = 1) => ({
  prediction,
  std_dev: stdDev,
  uncertainty_source: "empirical",
});

describe("model training result transformers", () => {
  test("keeps shared fields for simple models", () => {
    const transformed = transformTrainingResult("ma", {
      metrics: { rmse: 1.2, mae: 0.8, r2: 0.91 },
      eval_y_true: [10, 12],
      predictions: [predictionPoint(9.5), predictionPoint(12.4)],
      notes: ["ok"],
      saved_model: "/tmp/ma.pkl",
      logs: ["done"],
    });

    expect(transformed).toMatchObject({
      status: "success",
      results: {
        eval_y_true: [10, 12],
        eval_predictions: [9.5, 12.4],
        notes: ["ok"],
        saved_model: "/tmp/ma.pkl",
      },
    });
    expect(transformed.results).not.toHaveProperty("logs");
    expect((transformed.results.metrics as Record<string, number>).rmse).toBeCloseTo(0.4527692569);
    expect((transformed.results.metrics as Record<string, number>).mae).toBeCloseTo(0.45);
    expect((transformed.results.metrics as Record<string, number>).mape).toBeCloseTo(4.1666666667);
    expect((transformed.results.metrics as Record<string, number>).r2).toBeCloseTo(0.795);
  });

  test("adds ARIMA-specific order and diagnostics fields", () => {
    const transformed = transformTrainingResult("arima", {
      metrics: { rmse: 6.4, mae: 4.8, r2: 0.91 },
      eval_y_true: [100, 120],
      predictions: [predictionPoint(103), predictionPoint(118)],
      best_order: { p: 2, d: 1, q: 1 },
      search_summary: { tried: 8 },
      diagnostics: { aic: 10.5 },
    });

    expect(transformed.results).toMatchObject({
      best_order: { p: 2, d: 1, q: 1 },
      search_summary: { tried: 8 },
      diagnostics: { aic: 10.5, y_pred_eval: [103, 118] },
      eval_predictions: [103, 118],
    });
  });

  test("adds LSTM inferred feature metadata and diagnostics", () => {
    const featureTypes = {
      价格指数: "numerical",
      产能利用率: "numerical",
    };

    const transformed = transformTrainingResult("lstm", {
      metrics: { rmse: 7.6, mae: 5.4, r2: 0.93 },
      eval_y_true: [88, 95],
      predictions: [predictionPoint(90), predictionPoint(96)],
      diagnostics: { loss_curve: [0.5, 0.2] },
      feature_types: featureTypes,
    });

    expect(transformed.inferred_feature_types).toEqual(featureTypes);
    expect(transformed.results).toMatchObject({
      diagnostics: { loss_curve: [0.5, 0.2], y_pred_eval: [90, 96] },
      feature_types: featureTypes,
      eval_predictions: [90, 96],
    });
  });

  test("maps weighted ensemble model names and weights in request order", () => {
    const transformed = transformTrainingResult("weighted_avg", {
      metrics: { rmse: 8.2, mae: 6.3, r2: 0.9 },
      eval_y_true: [50, 60],
      predictions: [predictionPoint(49), predictionPoint(63)],
      models: ["ma", "arima", "lstm"],
      weights: { ma: 0.2, arima: 0.5, lstm: 0.3 },
    });

    expect(transformed.results).toMatchObject({
      model_names: ["ma", "arima", "lstm"],
      weights: [0.2, 0.5, 0.3],
      eval_predictions: [49, 63],
    });
  });

  test("maps boosting model chain to model_names and preserves repeated stages", () => {
    const transformed = transformTrainingResult("boosting", {
      metrics: { rmse: 8.8, mae: 6.9, r2: 0.92 },
      eval_y_true: [71, 73],
      predictions: [predictionPoint(70), predictionPoint(74)],
      model_chain: ["ma", "es", "ma"],
      stage_coefficients: [0.92, 0.27, 0.11],
      search_summary: { rounds: 12 },
    });

    expect(transformed.results).toMatchObject({
      model_names: ["ma", "es", "ma"],
      stage_coefficients: [0.92, 0.27, 0.11],
      search_summary: { rounds: 12 },
      eval_predictions: [70, 74],
    });
  });

  test("maps stacking models from models field", () => {
    const transformed = transformTrainingResult("stacking", {
      metrics: { rmse: 8.1, mae: 6.1, r2: 0.94 },
      eval_y_true: [41, 44],
      predictions: [predictionPoint(42), predictionPoint(43)],
      models: ["arima", "lstm"],
      meta_model: { intercept: 0.1, coefficients: [0.4, 0.6] },
    });

    expect(transformed.results).toMatchObject({
      model_names: ["arima", "lstm"],
      eval_predictions: [42, 43],
      meta_model: { intercept: 0.1, coefficients: [0.4, 0.6] },
    });
  });

  test("floors negative demand predictions and recomputes metrics from displayed values", () => {
    const transformed = transformTrainingResult("ma", {
      metrics: { rmse: 999, mae: 999, mape: 999, r2: 999, custom: "kept" },
      eval_y_true: [0, 10],
      predictions: [predictionPoint(-5), predictionPoint(-2)],
    });

    expect(transformed.results.eval_predictions).toEqual([0, 0]);
    expect(transformed.results.metrics).toMatchObject({
      custom: "kept",
      rmse: Math.sqrt(50),
      mae: 5,
      mape: 100,
      r2: -1,
    });
  });

  test("uses finite conventions for all-zero actual values", () => {
    const perfect = transformTrainingResult("es", {
      metrics: {},
      eval_y_true: [0, 0],
      predictions: [predictionPoint(-1), predictionPoint(0)],
    });
    const imperfect = transformTrainingResult("es", {
      metrics: {},
      eval_y_true: [0, 0],
      predictions: [predictionPoint(1), predictionPoint(0)],
    });

    expect(perfect.results.metrics).toMatchObject({ mape: 0, r2: 1 });
    expect(imperfect.results.metrics).toMatchObject({ mape: 0, r2: 0 });
  });

  test("rejects malformed Python prediction points instead of converting them to demand", () => {
    expect(() => transformTrainingResult("ma", {
      metrics: {},
      eval_y_true: [10],
      predictions: [{ ...predictionPoint(0), prediction: null }],
    })).toThrow("[PYTHON_DATA] Prediction point 0 is not finite");
  });

  test("rejects uncertainty that would violate the frontend prediction contract", () => {
    expect(() => transformTrainingResult("ma", {
      metrics: {},
      eval_y_true: [10],
      predictions: [{ prediction: 9, std_dev: 1 }],
    })).toThrow("[PYTHON_DATA] Prediction point 0 uncertainty_source is invalid");

    expect(() => transformTrainingResult("ma", {
      metrics: {},
      eval_y_true: [10],
      predictions: [{ ...predictionPoint(9), uncertainty_reason: 42 }],
    })).toThrow("[PYTHON_DATA] Prediction point 0 uncertainty_reason must be a string");
  });

  test("preserves calibrated prediction interval metadata", () => {
    const transformed = transformTrainingResult("ma", {
      metrics: {},
      eval_y_true: [10],
      predictions: [{
        ...predictionPoint(9, 1.5),
        calibration_source: "training_rolling_origin",
        interval_lower: 6.06,
        interval_upper: 11.94,
        interval_level: 0.95,
        interval_kind: "normal_approximation",
      }],
    });

    expect(transformed.results.prediction_points).toEqual([{
      prediction: 9,
      std_dev: 1.5,
      uncertainty_source: "empirical",
      calibration_source: "training_rolling_origin",
      interval_lower: 6.06,
      interval_upper: 11.94,
      interval_level: 0.95,
      interval_kind: "normal_approximation",
    }]);
  });

  test("rejects incomplete or malformed prediction interval metadata", () => {
    expect(() => transformTrainingResult("ma", {
      metrics: {},
      eval_y_true: [10],
      predictions: [{ ...predictionPoint(9), interval_lower: 7 }],
    })).toThrow("[PYTHON_DATA] Prediction point 0 interval bounds are invalid");

    expect(() => transformTrainingResult("ma", {
      metrics: {},
      eval_y_true: [10],
      predictions: [{
        ...predictionPoint(9),
        interval_lower: 11,
        interval_upper: 7,
        interval_level: 0.95,
        interval_kind: "normal_approximation",
      }],
    })).toThrow("[PYTHON_DATA] Prediction point 0 interval bounds are invalid");

    expect(() => transformTrainingResult("ma", {
      metrics: {},
      eval_y_true: [10],
      predictions: [{ ...predictionPoint(9), calibration_source: "" }],
    })).toThrow("[PYTHON_DATA] Prediction point 0 calibration_source must be a non-empty string");

    expect(() => normalizeDemandPredictionPoints([{
      prediction: 0,
      interval_lower: 1,
      interval_upper: 2,
      interval_level: 0.95,
      interval_kind: "censored_nonnegative_model_prediction_interval",
    }])).toThrow(
      "[PYTHON_DATA] Prediction point 0 must lie inside its nonnegative interval",
    );
  });

  test("rejects missing or empty Python prediction arrays", () => {
    expect(() => transformTrainingResult("ma", {
      metrics: {},
      eval_y_true: [10],
    })).toThrow("[PYTHON_DATA] predictions must be an array");

    expect(() => transformTrainingResult("ma", {
      metrics: {},
      eval_y_true: [],
      predictions: [],
    })).toThrow("[PYTHON_DATA] predictions must contain at least one point");
  });

  test("rejects evaluation values that do not match the displayed predictions", () => {
    expect(() => transformTrainingResult("ma", {
      metrics: { rmse: 999 },
      eval_y_true: [10],
      predictions: [predictionPoint(9), predictionPoint(11)],
    })).toThrow("[PYTHON_DATA] Expected 2 evaluation values, received 1");

    expect(() => transformTrainingResult("ma", {
      metrics: {},
      eval_y_true: [Number.NaN],
      predictions: [predictionPoint(9)],
    })).toThrow("[PYTHON_DATA] Evaluation value 0 is not finite");
  });

  test("rejects a Python result shorter than the configured evaluation range", () => {
    expect(() => transformTrainingResult("ma", {
      metrics: {},
      eval_y_true: [10],
      predictions: [predictionPoint(9)],
    }, 2)).toThrow("[PYTHON_DATA] Expected 2 prediction points, received 1");
  });
});

describe("production prediction calibration diagnostics", () => {
  const normalizeProductionPoint = (point: Record<string, unknown>) =>
    normalizeDemandPredictionPoints([{
      prediction: 10,
      std_dev: 2,
      upper_error_p99: 5,
      uncertainty_source: "empirical",
      ...point,
    }], 1, {
      requireStdDev: true,
      requireUpperErrorP99: true,
      requireCalibrationDiagnostics: true,
    });

  test("accepts either a finite mean/count pair or an explicit unavailable pair", () => {
    expect(normalizeProductionPoint({
      calibration_mean_error: 3.5,
      calibration_count: 8,
    })[0]).toMatchObject({ calibration_mean_error: 3.5, calibration_count: 8 });
    expect(normalizeProductionPoint({
      calibration_mean_error: null,
      calibration_count: null,
    })[0]).toMatchObject({ calibration_mean_error: null, calibration_count: null });
  });

  test("rejects missing, partial, and malformed diagnostics", () => {
    expect(() => normalizeProductionPoint({})).toThrow("calibration diagnostics are required");
    expect(() => normalizeProductionPoint({
      calibration_mean_error: 3.5,
      calibration_count: null,
    })).toThrow("calibration diagnostics must be");
    expect(() => normalizeProductionPoint({
      calibration_mean_error: Number.NaN,
      calibration_count: 2,
    })).toThrow("calibration diagnostics must be");
    expect(() => normalizeProductionPoint({
      calibration_mean_error: 3.5,
      calibration_count: 0,
    })).toThrow("calibration diagnostics must be");
  });
});
