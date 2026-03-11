import { describe, expect, test } from "bun:test";
import type { ZodIssue } from "zod";
import {
  buildLocalizedPredictValidationDetails,
  buildLocalizedTrainValidationDetails,
  localizePredictValidationIssue,
  localizeTrainValidationIssue,
} from "../../../src/modules/models/models.validation";

const buildIssue = (issue: Partial<ZodIssue> & { code: ZodIssue["code"] }): ZodIssue => {
  return {
    path: [],
    message: "",
    ...issue,
  } as ZodIssue;
};

describe("models validation helpers", () => {
  describe("localizeTrainValidationIssue", () => {
    test("localizes missing required training fields", () => {
      const issue = buildIssue({
        code: "invalid_type",
        path: ["moving_average_window"],
        input: undefined,
      });

      expect(localizeTrainValidationIssue(issue)).toBe("移动平均时间窗口不能为空");
    });

    test("localizes invalid LSTM normalization values", () => {
      const issue = buildIssue({
        code: "invalid_value",
        path: ["lstmNormalization"],
        message: "Invalid option",
      });

      expect(localizeTrainValidationIssue(issue)).toBe("LSTM 归一化方式必须是 minmax 或 zscore");
    });

    test("keeps explicit schema messages for range validation", () => {
      const issue = buildIssue({
        code: "custom",
        path: ["data_window_evaluate_end_index"],
        message: "评估起始索引必须小于等于评估结束索引",
      });

      expect(localizeTrainValidationIssue(issue)).toBe("评估起始索引必须小于等于评估结束索引");
    });

    test("localizes numeric lower bound violations with the actual minimum", () => {
      const issue = buildIssue({
        code: "too_small",
        path: ["moving_average_window"],
        minimum: 2,
        inclusive: true,
        origin: "number",
      });

      expect(localizeTrainValidationIssue(issue)).toBe("移动平均时间窗口必须大于等于 2");
    });

    test("localizes invalid type errors with natural spacing", () => {
      const issue = buildIssue({
        code: "invalid_type",
        path: ["experiment_id"],
        input: "abc",
      });

      expect(localizeTrainValidationIssue(issue)).toBe("实验 ID 类型无效");
    });
  });

  describe("buildLocalizedTrainValidationDetails", () => {
    test("groups form and field errors separately", () => {
      const details = buildLocalizedTrainValidationDetails([
        buildIssue({
          code: "invalid_type",
          path: ["models"],
          input: undefined,
        }),
        buildIssue({
          code: "custom",
          path: [],
          message: "训练请求参数无效",
        }),
      ]);

      expect(details).toEqual({
        formErrors: ["训练请求参数无效"],
        fieldErrors: {
          models: ["基础模型列表不能为空"],
        },
      });
    });
  });

  describe("localizePredictValidationIssue", () => {
    test("localizes missing forecast steps", () => {
      const issue = buildIssue({
        code: "invalid_type",
        path: ["forecast_steps"],
        input: undefined,
      });

      expect(localizePredictValidationIssue(issue)).toBe("预测步数不能为空");
    });

    test("localizes forecast step lower bound violations", () => {
      const issue = buildIssue({
        code: "too_small",
        path: ["forecast_steps"],
        message: "Too small",
      });

      expect(localizePredictValidationIssue(issue)).toBe("预测步数必须大于等于 1");
    });
  });

  describe("buildLocalizedPredictValidationDetails", () => {
    test("collects localized predict field errors", () => {
      const details = buildLocalizedPredictValidationDetails([
        buildIssue({
          code: "invalid_type",
          path: ["forecast_steps"],
          input: undefined,
        }),
        buildIssue({
          code: "invalid_type",
          path: ["experiment_id"],
          input: "abc",
        }),
      ]);

      expect(details).toEqual({
        formErrors: [],
        fieldErrors: {
          forecast_steps: ["预测步数不能为空"],
          experiment_id: ["实验 ID 类型无效"],
        },
      });
    });
  });
});