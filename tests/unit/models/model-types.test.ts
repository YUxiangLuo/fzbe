import { describe, expect, test } from "bun:test";
import { GUIDED_TRAINING_STEPS } from "../../../src/modules/models/model-types";

describe("guided model teaching copy", () => {
  test("describes ARIMA interval conversion with the student-facing normal critical value", () => {
    const forecastStep = GUIDED_TRAINING_STEPS.arima.find((step) => step.id === "forecast");

    expect(forecastStep?.description).toContain("约 1.96");
    expect(forecastStep?.description).not.toContain("除以3.92");
  });

  test("separates ES training-residual calibration from evaluation-period bias diagnosis", () => {
    const residualStep = GUIDED_TRAINING_STEPS.es.find((step) => step.id === "estimate_residuals");

    expect(residualStep?.description).toContain("训练期一步预测残差估计创新尺度");
    expect(residualStep?.description).toContain("统一评估区间平均残差只作误差诊断");
    expect(residualStep?.description).toContain("不参与范围校准");
  });

  test("keeps guided training descriptions free of unexplained internal English", () => {
    const copy = Object.values(GUIDED_TRAINING_STEPS)
      .flat()
      .flatMap((step) => [step.label, step.description, step.actionLabel])
      .join("\n");

    for (const rawTerm of [
      "rolling-origin",
      "stepwise",
      "look-back",
      "fallback",
      "Level-1",
      "OOF",
      "ridge",
      "naïve",
      "seasonal naïve",
      "checkpoint",
    ]) {
      expect(copy).not.toContain(rawTerm);
    }
  });
});
