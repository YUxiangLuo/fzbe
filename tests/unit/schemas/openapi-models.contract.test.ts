import { describe, expect, test } from "bun:test";
import path from "node:path";
import { parse } from "yaml";

type SchemaRecord = Record<string, any>;

const openApiPath = path.resolve(import.meta.dir, "../../../docs/openapi.yaml");
const document = parse(await Bun.file(openApiPath).text()) as SchemaRecord;
const schemas = document.components.schemas as SchemaRecord;

describe("OpenAPI model contracts", () => {
  test("documents explicit target-only LSTM features and conditional ensemble requirements", () => {
    const standaloneArray = schemas.LstmTrainRequest.allOf[1].properties.lstmFeatures.oneOf[1];
    expect(standaloneArray.minItems).toBeUndefined();

    const withLstm = schemas.EnsembleWithLstmTrainRequest.allOf[1];
    expect(withLstm.required).toEqual([
      "models",
      "lstmFeatures",
      "lstmTargetFeature",
      "lstmNormalization",
    ]);
    expect(withLstm.properties.lstmFeatures.oneOf[1].minItems).toBeUndefined();

    const containsLstm = new RegExp(withLstm.properties.models.pattern);
    const withoutLstm = schemas.EnsembleWithoutLstmTrainRequest.allOf[1];
    const excludesLstm = new RegExp(withoutLstm.properties.models.not.pattern);
    expect(containsLstm.test("ma, lstm")).toBe(true);
    expect(containsLstm.test("ma,es")).toBe(false);
    expect(excludesLstm.test("ma,lstm")).toBe(true);
  });

  test("documents the wrapped production prediction uncertainty contract", () => {
    const point = schemas.ProductionPredictionPoint;
    expect(point.required).toEqual(expect.arrayContaining([
      "prediction",
      "std_dev",
      "upper_error_p99",
      "uncertainty_source",
      "calibration_mean_error",
      "calibration_count",
    ]));
    expect(point.properties.uncertainty_source.enum).toEqual(["model", "empirical", "fallback"]);
    expect(point.properties.interval_level.enum).toEqual([0.95]);
    expect(
      document.paths["/api/v1/models/{model_type}/predict"].post.responses["200"]
        .content["application/json"].schema.$ref,
    ).toBe("#/components/schemas/ModelPredictionResponse");
    expect(schemas.ModelPredictionResponse.required).toEqual(["data"]);
  });
});
