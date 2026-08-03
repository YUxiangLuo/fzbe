import { describe, expect, test } from "bun:test";
import {
  buildEnsemblePythonParams,
  type ExperimentModelContextRow,
} from "../../../src/modules/models/model-context";
import {
  getTrainSchemaForModel,
  type ModelType,
  type TrainInput,
} from "../../../src/modules/models/models.schema";

const selectedFeatures = ["价格,指数", "促销投入"];

const buildExperiment = (
  overrides: Partial<ExperimentModelContextRow> = {},
): ExperimentModelContextRow => ({
  student_id: 20240002,
  status: "In Progress",
  selected_industry: "制造业",
  selected_company: "样本企业",
  selected_product: "家用空调",
  data_window_train_start_index: 0,
  data_window_train_end_index: 71,
  data_window_evaluate_start_index: 72,
  data_window_evaluate_end_index: 83,
  moving_average_completed: 1,
  moving_average_window: 9,
  exponential_smoothing_completed: 1,
  exponential_smoothing_alpha: 0.27,
  arima_completed: 1,
  arima_d: 2,
  lstm_completed: 1,
  lstm_normalization: "zscore",
  lstm_features: JSON.stringify(selectedFeatures),
  lstm_target_field: "销售数量",
  ensemble_weighted_completed: 0,
  ensemble_weighted_base_models: null,
  ensemble_boosting_completed: 0,
  ensemble_boosting_base_models: null,
  ensemble_stacking_completed: 0,
  ensemble_stacking_base_models: null,
  ...overrides,
} as ExperimentModelContextRow);

const exactRequest = (): TrainInput => ({
  experiment_id: 990002,
  models: "ma,es,arima,lstm",
  moving_average_window: 9,
  exponential_smoothing_alpha: 0.27,
  arimaD: 2,
  lstmNormalization: "zscore",
  lstmTargetFeature: "销售数量",
  lstmFeatures: JSON.stringify(selectedFeatures),
});

const ensembleTypes = [
  "weighted_avg",
  "boosting",
  "stacking",
] as const satisfies readonly ModelType[];

describe("ensemble manual parameter inheritance contract", () => {
  test("all three ensemble schemas forward the persisted base-model choices", () => {
    const experiment = buildExperiment();
    const expected = {
      models: "ma,es,arima,lstm",
      moving_average_window: 9,
      exponential_smoothing_alpha: 0.27,
      arimaD: 2,
      lstmNormalization: "zscore",
      lstmTargetFeature: "销售数量",
      lstmFeatures: JSON.stringify(selectedFeatures),
    };

    for (const ensembleType of ensembleTypes) {
      const parsedRequest = getTrainSchemaForModel(ensembleType).parse(exactRequest());
      expect(buildEnsemblePythonParams(experiment, parsedRequest)).toEqual(expected);
    }
  });

  test("a stale client cannot override any completed base-model parameter", () => {
    const experiment = buildExperiment();
    const mismatches: Array<{
      field: keyof TrainInput;
      value: unknown;
      message: string;
    }> = [
      {
        field: "moving_average_window",
        value: 8,
        message: "移动平均时间窗口",
      },
      {
        field: "exponential_smoothing_alpha",
        value: 0.5,
        message: "指数平滑系数",
      },
      {
        field: "arimaD",
        value: 1,
        message: "ARIMA 差分阶数",
      },
      {
        field: "lstmNormalization",
        value: "minmax",
        message: "LSTM 归一化方式",
      },
      {
        field: "lstmTargetFeature",
        value: "销售额",
        message: "LSTM 目标特征",
      },
      {
        field: "lstmFeatures",
        value: JSON.stringify(["价格", "指数,促销投入"]),
        message: "LSTM 输入特征",
      },
    ];

    for (const mismatch of mismatches) {
      const request = {
        ...exactRequest(),
        [mismatch.field]: mismatch.value,
      } as TrainInput;
      expect(() => buildEnsemblePythonParams(experiment, request))
        .toThrow(mismatch.message);
    }
  });

  test("an explicitly empty target-only LSTM selection remains empty", () => {
    const experiment = buildExperiment({
      lstm_features: "[]",
    });
    const request = {
      ...exactRequest(),
      models: "ma,lstm",
      lstmFeatures: "[]",
    };

    expect(buildEnsemblePythonParams(experiment, request)).toEqual({
      models: "ma,lstm",
      moving_average_window: 9,
      lstmNormalization: "zscore",
      lstmTargetFeature: "销售数量",
      lstmFeatures: "[]",
    });
  });

  test("missing persisted configuration is never replaced by request fallbacks", () => {
    const cases: Array<{
      override: Partial<ExperimentModelContextRow>;
      message: string;
    }> = [
      {
        override: { moving_average_window: null },
        message: "缺少时间窗口配置",
      },
      {
        override: { exponential_smoothing_alpha: null },
        message: "缺少平滑系数配置",
      },
      {
        override: { arima_d: null },
        message: "缺少差分阶数配置",
      },
      {
        override: { lstm_normalization: null },
        message: "缺少特征配置、目标字段或归一化配置",
      },
    ];

    for (const testCase of cases) {
      expect(() => buildEnsemblePythonParams(
        buildExperiment(testCase.override),
        exactRequest(),
      )).toThrow(testCase.message);
    }
  });
});
