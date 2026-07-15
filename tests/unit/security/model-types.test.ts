import { describe, expect, test } from "bun:test";
import { GUIDED_TRAINING_STEPS } from "../../../src/modules/models/model-types";

describe("guided model teaching copy", () => {
  test("describes ARIMA interval conversion with the implemented normal critical value", () => {
    const forecastStep = GUIDED_TRAINING_STEPS.arima.find((step) => step.id === "forecast");

    expect(forecastStep?.description).toContain("1.9599639845");
    expect(forecastStep?.description).not.toContain("除以3.92");
  });
});
