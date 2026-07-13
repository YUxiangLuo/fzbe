import type { ModelType } from "./models.schema";

export type BaseModelType = "ma" | "es" | "arima" | "lstm";
export type EnsembleModelType = Extract<ModelType, "weighted_avg" | "boosting" | "stacking">;
export type GuidedModelType = BaseModelType | EnsembleModelType;
export type GuidedTrainingStatus = "ready" | "running" | "failed" | "completed" | "superseded";

export const BASE_MODEL_TYPES = new Set<ModelType>(["ma", "es", "arima", "lstm"]);
export const ENSEMBLE_MODEL_TYPES = new Set<ModelType>(["weighted_avg", "boosting", "stacking"]);
export const GUIDED_MODEL_TYPES = new Set<ModelType>(["ma", "es", "arima", "lstm", "weighted_avg", "boosting", "stacking"]);
export const ENSEMBLE_DERIVED_BODY_KEYS = new Set([
  "models",
  "moving_average_window",
  "exponential_smoothing_alpha",
  "arimaD",
  "lstmFeatures",
  "lstmTargetFeature",
  "lstmNormalization",
]);

export interface GuidedTrainingStepDefinition {
  id: string;
  label: string;
  description: string;
  actionLabel: string;
}

export interface GuidedTrainingSessionPayload {
  session_id: string;
  experiment_id: number;
  model_type: GuidedModelType;
  status: GuidedTrainingStatus;
  current_step_id: string | null;
  next_step_id: string | null;
  steps: Array<GuidedTrainingStepDefinition & {
    status: "pending" | "active" | "completed" | "failed";
    output?: unknown;
  }>;
  step_outputs: Record<string, unknown>;
  result: {
    status: string;
    results: Record<string, unknown>;
    inferred_feature_types?: unknown;
    experiment_state_patch?: Record<string, unknown>;
  } | null;
  error_message: string | null;
  error_code: "GUIDED_SESSION_SUPERSEDED" | null;
  experiment_state_version: number | null;
}

export const GUIDED_TRAINING_STEPS: Record<GuidedModelType, GuidedTrainingStepDefinition[]> = {
  ma: [
    {
      id: "prepare_data",
      label: "确认训练材料",
      description: "读取实验数据，检查训练区间、评估区间和移动平均窗口，确保后续计算有足够历史样本。",
      actionLabel: "校验数据",
    },
    {
      id: "build_windows",
      label: "准备初始窗口",
      description: "取训练集末尾的连续销量作为第一个滚动窗口，明确每次预测会参考哪些最近历史值。",
      actionLabel: "准备窗口",
    },
    {
      id: "forecast",
      label: "滚动生成预测",
      description: "按窗口均值逐期外推；每得到一个预测值，就把它放回窗口中作为下一期的历史输入。",
      actionLabel: "生成预测",
    },
    {
      id: "estimate_residuals",
      label: "估计预测波动",
      description: "比较评估区间真实值和预测值，估计残差波动，用于给预测结果附上不确定性范围。",
      actionLabel: "估计残差",
    },
    {
      id: "evaluate_metrics",
      label: "计算误差指标",
      description: "汇总 RMSE、MAE、MAPE、R² 等指标，把模型误差转化为可比较的评估结果。",
      actionLabel: "计算指标",
    },
    {
      id: "save_artifact",
      label: "保存模型产物",
      description: "保存移动平均窗口、最近历史值和残差波动，供后续预测和报告复用。",
      actionLabel: "保存结果",
    },
  ],
  es: [
    {
      id: "prepare_data",
      label: "确认平滑设置",
      description: "读取实验数据，检查训练区间和评估区间，并确认 α（alpha，平滑系数）处在有效范围内。",
      actionLabel: "校验数据",
    },
    {
      id: "fit_model",
      label: "拟合平滑水平",
      description: "用训练集销量和固定 α 更新指数平滑水平项，让近期观测对模型产生更高权重。",
      actionLabel: "拟合模型",
    },
    {
      id: "forecast",
      label: "外推评估区间",
      description: "使用拟合后的平滑水平，对评估区间进行多期预测，形成待检验的销量序列。",
      actionLabel: "生成预测",
    },
    {
      id: "estimate_residuals",
      label: "估计残差波动",
      description: "比较预测值和真实值，估计指数平滑模型在评估区间的误差离散程度。",
      actionLabel: "估计残差",
    },
    {
      id: "evaluate_metrics",
      label: "计算误差指标",
      description: "计算 RMSE、MAE、MAPE、R²，判断固定 α 的预测效果是否稳定。",
      actionLabel: "计算指标",
    },
    {
      id: "save_artifact",
      label: "保存模型产物",
      description: "保存平滑模型、α 和残差波动，供后续预测、生产准备和实验报告复用。",
      actionLabel: "保存结果",
    },
  ],
  arima: [
    {
      id: "prepare_data",
      label: "确认差分和搜索范围",
      description: "读取训练序列，检查 d 阶差分后是否仍有足够样本，并确定 p/q 搜索上限。",
      actionLabel: "准备搜索",
    },
    {
      id: "aic_search",
      label: "执行 AIC 搜索",
      description: "按 AIC 信息准则搜索候选 ARIMA(p,d,q) 模型，优先寻找拟合能力更强的阶数组合。",
      actionLabel: "运行 AIC",
    },
    {
      id: "bic_search",
      label: "执行 BIC 搜索",
      description: "按 BIC 信息准则搜索候选模型，给复杂模型更强惩罚，降低小样本过拟合风险。",
      actionLabel: "运行 BIC",
    },
    {
      id: "select_model",
      label: "选择最优阶数",
      description: "比较 AIC 与 BIC 的候选结果，结合样本量选择最终用于预测的 p、d、q。",
      actionLabel: "选择模型",
    },
    {
      id: "forecast",
      label: "预测评估区间",
      description: "使用选定阶数的 ARIMA 模型预测评估区间，并生成置信区间作为不确定性参考。",
      actionLabel: "生成预测",
    },
    {
      id: "evaluate_metrics",
      label: "计算误差指标",
      description: "把 ARIMA 预测与真实销量对齐，计算误差指标并记录模型收敛状态。",
      actionLabel: "计算指标",
    },
    {
      id: "save_artifact",
      label: "保存模型产物",
      description: "保存最终 ARIMA 模型、阶数选择摘要和评估结果，供后续预测和报告复用。",
      actionLabel: "保存结果",
    },
  ],
  lstm: [
    {
      id: "prepare_data",
      label: "解析字段并校验窗口",
      description: "读取多变量数据，确认目标字段、特征字段、字段类型、归一化方式和训练窗口有效。",
      actionLabel: "解析数据",
    },
    {
      id: "preprocess",
      label: "预处理并构造样本",
      description: "仅用训练集拟合归一化和编码器，再把时间序列切成 LSTM 输入窗口与多步预测标签。",
      actionLabel: "构造样本",
    },
    {
      id: "build_model",
      label: "确定网络结构",
      description: "根据特征维度、样本量和预测跨度确定 LSTM 层、输出层、批大小和学习率计划。",
      actionLabel: "构建网络",
    },
    {
      id: "train_model",
      label: "训练神经网络",
      description: "执行真实神经网络训练，使用学习率衰减和早停策略，让网络学习历史窗口与未来销量的关系。",
      actionLabel: "训练模型",
    },
    {
      id: "forecast",
      label: "生成评估预测",
      description: "加载训练后的网络，对评估区间进行多步预测，并把预测结果还原到原始销量尺度。",
      actionLabel: "生成预测",
    },
    {
      id: "evaluate_metrics",
      label: "计算误差指标",
      description: "比较 LSTM 预测和真实销量，计算 RMSE、MAE、MAPE、R²，并估计残差波动。",
      actionLabel: "计算指标",
    },
    {
      id: "save_artifact",
      label: "保存模型产物",
      description: "保存预处理器、历史窗口、神经网络文件和评估结果，供后续预测和生产准备复用。",
      actionLabel: "保存结果",
    },
  ],
  weighted_avg: [
    {
      id: "prepare_data",
      label: "确认融合材料",
      description: "读取实验数据，校验训练区间、评估区间和已选择的基础模型，确认可复用的基础模型产物。",
      actionLabel: "校验数据",
    },
    {
      id: "split_validation",
      label: "划分权重验证段",
      description: "按时间顺序从训练区间末尾留出一段验证数据，专门用于比较各基础模型的残差大小。",
      actionLabel: "划分数据",
    },
    {
      id: "validation_predictions",
      label: "生成验证段预测",
      description: "让每个基础模型只用验证段之前的训练数据学习，再预测留出的验证段，形成可比较的误差证据。",
      actionLabel: "训练并预测",
    },
    {
      id: "compute_weights",
      label: "计算融合权重",
      description: "根据验证段残差 MSE 的倒数分配权重，误差越小的基础模型获得越高的组合权重。",
      actionLabel: "计算权重",
    },
    {
      id: "eval_predictions",
      label: "完整训练后评估预测",
      description: "在完整训练区间重训或复用基础模型，并生成独立评估区间的预测值。",
      actionLabel: "生成评估预测",
    },
    {
      id: "blend_evaluate",
      label: "加权组合并评估",
      description: "按固定权重组合基础模型预测，计算最终 RMSE、MAE、MAPE、R² 和残差波动。",
      actionLabel: "组合评估",
    },
    {
      id: "save_artifact",
      label: "保存融合产物",
      description: "保存融合权重、成员模型产物和评估结果，供后续预测和报告复用。",
      actionLabel: "保存结果",
    },
  ],
  stacking: [
    {
      id: "prepare_data",
      label: "确认 Stacking 材料",
      description: "读取实验数据，校验训练区间、评估区间和基础模型选择，准备二层学习所需的输入。",
      actionLabel: "校验数据",
    },
    {
      id: "split_levels",
      label: "划分 Level-0/Level-1",
      description: "按时间顺序把训练区间拆成基础模型训练段和元模型训练段，避免时间序列信息泄漏。",
      actionLabel: "划分层级",
    },
    {
      id: "level1_predictions",
      label: "生成元模型训练特征",
      description: "基础模型在 Level-0 学习后预测 Level-1，预测矩阵将作为元模型的输入特征。",
      actionLabel: "生成二层特征",
    },
    {
      id: "fit_meta_model",
      label: "拟合元模型",
      description: "用 Level-1 的基础模型预测和真实销量训练非负线性元模型，必要时回退到稳定权重。",
      actionLabel: "拟合元模型",
    },
    {
      id: "eval_predictions",
      label: "完整训练后评估预测",
      description: "基础模型在完整训练区间重训或复用产物，为独立评估区间生成二层输入。",
      actionLabel: "生成评估预测",
    },
    {
      id: "meta_evaluate",
      label: "元模型组合并评估",
      description: "把评估区间的基础模型预测交给元模型组合，计算最终误差指标和残差波动。",
      actionLabel: "组合评估",
    },
    {
      id: "save_artifact",
      label: "保存融合产物",
      description: "保存元模型、成员模型产物和评估结果，供后续预测和报告复用。",
      actionLabel: "保存结果",
    },
  ],
  boosting: [
    {
      id: "prepare_data",
      label: "确认 Boosting 材料",
      description: "读取实验数据，校验训练区间、评估区间、基础模型候选库、学习率和最大迭代轮数。",
      actionLabel: "校验数据",
    },
    {
      id: "split_residual_sets",
      label: "初始化残差训练集",
      description: "把训练区间拆成残差训练段和残差验证段，并把原始销量作为第一轮要解释的残差。",
      actionLabel: "初始化残差",
    },
    {
      id: "retrain_chain",
      label: "重训已选模型链",
      description: "按前面选出的模型链，在完整训练区间逐级拟合残差，并生成评估区间阶段预测。",
      actionLabel: "重训模型链",
    },
    {
      id: "combine_evaluate",
      label: "累加阶段预测并评估",
      description: "把各轮模型的预测按学习率累加，计算 Boosting 融合的最终误差指标。",
      actionLabel: "组合评估",
    },
    {
      id: "save_artifact",
      label: "保存融合产物",
      description: "保存模型链、阶段系数、成员模型产物和评估结果，供后续预测和报告复用。",
      actionLabel: "保存结果",
    },
  ],
};
