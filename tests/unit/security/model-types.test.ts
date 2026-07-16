import { describe, expect, test } from "bun:test";
import { GUIDED_TRAINING_STEPS } from "../../../src/modules/models/model-types";

describe("guided model teaching copy", () => {
  test("describes ARIMA interval conversion with the implemented normal critical value", () => {
    const forecastStep = GUIDED_TRAINING_STEPS.arima.find((step) => step.id === "forecast");

    expect(forecastStep?.description).toContain("1.9599639845");
    expect(forecastStep?.description).not.toContain("除以3.92");
  });

  test("keeps LSTM configuration separate from network construction", () => {
    const configStep = GUIDED_TRAINING_STEPS.lstm.find((step) => step.id === "build_model");

    expect(configStep).toMatchObject({
      label: "确认网络配置",
      actionLabel: "确认配置",
    });
    expect(configStep?.description).toContain("网络会在下一阶段按该配置创建并训练");
  });

  test("describes ES uncertainty as training-residual-only", () => {
    const residualStep = GUIDED_TRAINING_STEPS.es.find((step) => step.id === "estimate_residuals");

    expect(residualStep?.description).toContain("只用训练期一步预测残差");
    expect(residualStep?.description).toContain("评估区间真实值不参与不确定性校准");
  });

  test("describes Boosting deployment preparation without promising universal refits", () => {
    const deploymentStep = GUIDED_TRAINING_STEPS.boosting.find((step) => step.id === "retrain_chain");

    expect(deploymentStep).toMatchObject({
      label: "准备部署模型链",
      actionLabel: "准备模型链",
    });
    expect(deploymentStep?.description).toContain("首阶段可复用");
    expect(deploymentStep?.description).toContain("保留时间留出验证学到的非负系数");
    expect(deploymentStep?.description).not.toContain("逐级重训");
  });
});
