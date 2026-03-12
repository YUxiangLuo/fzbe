process.env.DB_HOST ??= "127.0.0.1";
process.env.DB_USER ??= "test";
process.env.DB_PASSWORD ??= "test";
process.env.DB_DATABASE ??= "test";
process.env.JWT_SECRET ??= "test-jwt-secret";
process.env.PORT ??= "3001";
process.env.DEFAULT_PASSWORD_HASH ??= "test-default-hash";

import { describe, expect, test } from "bun:test";
import path from "node:path";
import { mock } from "bun:test";

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const pythonModulePath = path.join(srcRoot, "utils", "python.ts");

mock.module(configModulePath, () => ({
  pool: {
    query: async () => [[]],
    getConnection: async () => ({
      query: async () => [[]],
      release: () => {},
    }),
  },
}));

mock.module("fs", () => ({
  default: {
    existsSync: () => false,
    createReadStream: () => {
      throw new Error("createReadStream should not be called in transformer tests");
    },
  },
  existsSync: () => false,
  createReadStream: () => {
    throw new Error("createReadStream should not be called in transformer tests");
  },
}));

mock.module(pythonModulePath, () => ({
  PY_DIR: "/tmp/fangzhen-py",
  resolvePythonExecutable: () => "/tmp/fangzhen-py/venv/bin/python",
}));

const modelsServiceModulePath = path.join(srcRoot, "modules", "models", "models.service.ts");
const freshModelsServiceModulePath = `${modelsServiceModulePath}?models-result-transformers-test`;

const { transformTrainingResult } = await import(freshModelsServiceModulePath);

describe("model training result transformers", () => {
  test("keeps shared fields for simple models", () => {
    const transformed = transformTrainingResult("ma", {
      metrics: { rmse: 1.2, mae: 0.8, r2: 0.91 },
      eval_y_true: [10, 12],
      predictions: [{ prediction: 9.5 }, { prediction: 12.4 }],
      notes: ["ok"],
      saved_model: "/tmp/ma.pkl",
      logs: ["done"],
    });

    expect(transformed).toEqual({
      status: "success",
      results: {
        metrics: { rmse: 1.2, mae: 0.8, r2: 0.91 },
        eval_y_true: [10, 12],
        eval_predictions: [9.5, 12.4],
        notes: ["ok"],
        saved_model: "/tmp/ma.pkl",
        logs: ["done"],
      },
    });
  });

  test("adds ARIMA-specific order and diagnostics fields", () => {
    const transformed = transformTrainingResult("arima", {
      metrics: { rmse: 6.4, mae: 4.8, r2: 0.91 },
      eval_y_true: [100, 120],
      predictions: [{ prediction: 103 }, { prediction: 118 }],
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
      predictions: [{ prediction: 90 }, { prediction: 96 }],
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
      predictions: [{ prediction: 49 }, { prediction: 63 }],
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
      predictions: [{ prediction: 70 }, { prediction: 74 }],
      model_chain: ["ma", "es", "ma"],
      stage_coefficients: [1, 0.3, 0.3],
      search_summary: { rounds: 12 },
    });

    expect(transformed.results).toMatchObject({
      model_names: ["ma", "es", "ma"],
      search_summary: { rounds: 12 },
      eval_predictions: [70, 74],
    });
  });

  test("maps stacking models from models field", () => {
    const transformed = transformTrainingResult("stacking", {
      metrics: { rmse: 8.1, mae: 6.1, r2: 0.94 },
      eval_y_true: [41, 44],
      predictions: [{ prediction: 42 }, { prediction: 43 }],
      models: ["arima", "lstm"],
      meta_model: { intercept: 0.1, coefficients: [0.4, 0.6] },
    });

    expect(transformed.results).toMatchObject({
      model_names: ["arima", "lstm"],
      eval_predictions: [42, 43],
    });
    expect(transformed.results.meta_model).toBeUndefined();
  });
});
