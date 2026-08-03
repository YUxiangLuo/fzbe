import type { ModelType } from "./models.schema";

export type BaseModelType = "ma" | "es" | "arima" | "lstm";
export type EnsembleModelType = Extract<ModelType, "weighted_avg" | "boosting" | "stacking">;
export type GuidedModelType = BaseModelType | EnsembleModelType;
export type GuidedTrainingStatus = "ready" | "running" | "failed" | "completed" | "superseded";

export const BASE_MODEL_TYPES = new Set<ModelType>(["ma", "es", "arima", "lstm"]);
export const ENSEMBLE_MODEL_TYPES = new Set<ModelType>(["weighted_avg", "boosting", "stacking"]);
export const GUIDED_MODEL_TYPES = new Set<ModelType>(["ma", "es", "arima", "lstm", "weighted_avg", "boosting", "stacking"]);

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
      description: "读取实验数据，检查训练区间、评估区间，并确认窗口 n 为不超过训练长度的整数且 n ≥ 2；n = 1 只是上一期值延续，不体现均值平滑，因此本系统不开放。",
      actionLabel: "校验数据",
    },
    {
      id: "build_windows",
      label: "准备初始窗口",
      description: "取训练集末尾 n 个连续销量作为第一个滚动窗口，明确下一期预测所使用的全部历史值。",
      actionLabel: "准备窗口",
    },
    {
      id: "forecast",
      label: "滚动生成预测",
      description: "按窗口均值逐期外推；每得到一个预测值，就把它放回窗口中作为下一期的历史输入。最终作为销量使用的点预测按 max(0, ŷ) 做非负兜底。",
      actionLabel: "生成预测",
    },
    {
      id: "estimate_residuals",
      label: "估计预测波动",
      description: "只用训练段执行扩展窗口滚动起点多步回测，逐预测步估计预测波动；样本不足的预测步使用明确标记的回退估计，评估区间实际值不参与校准。",
      actionLabel: "估计波动",
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
      description: "保存移动平均窗口、最近历史值和训练段扩展窗口回测得到的预测波动，供后续预测和报告复用。",
      actionLabel: "保存结果",
    },
  ],
  es: [
    {
      id: "prepare_data",
      label: "确认平滑设置",
      description: "读取实验数据，检查训练区间和评估区间，并确认用户给定的平滑系数 α 满足 0 < α ≤ 1；本系统不会自动优化 α。",
      actionLabel: "校验数据",
    },
    {
      id: "fit_model",
      label: "拟合平滑水平",
      description: "在固定用户给定 α 的前提下，用训练数据估计 ETS(A,N,N) 的初始水平，再逐期更新水平项；α 本身不会被自动改变。",
      actionLabel: "拟合模型",
    },
    {
      id: "forecast",
      label: "外推评估区间",
      description: "一次指数平滑没有趋势项或季节项，因此将训练末尾的最新平滑水平作为评估区间每一期的相同预测值；最终销量点预测按 max(0, ŷ) 做非负兜底。",
      actionLabel: "生成预测",
    },
    {
      id: "estimate_residuals",
      label: "估计残差波动",
      description: "用训练期一步预测残差估计创新尺度，并按 ETS(A,N,N) 解析公式让多步预测标准差随预测步数展宽；统一评估区间平均残差只作误差诊断，不参与范围校准。",
      actionLabel: "估计残差",
    },
    {
      id: "evaluate_metrics",
      label: "计算误差指标",
      description: "计算 RMSE、MAE、MAPE、R²，描述固定 α 在当前评估区间的预测误差。",
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
      description: "读取训练序列，检查固定差分阶数 d ≥ 0 且差分后仍有足够样本，并按样本量确定非季节 ARIMA 的 p、q 搜索上限。",
      actionLabel: "准备搜索",
    },
    {
      id: "aic_search",
      label: "执行 AICc 搜索",
      description: "在固定 d 和随样本量调整的 p、q 边界内运行一次 AICc 逐步搜索，并检查胜出模型是否收敛；必要时用更高迭代上限重试。",
      actionLabel: "运行 AICc",
    },
    {
      id: "bic_search",
      label: "检查残差诊断",
      description: "对 AICc 优胜模型执行带模型自由度修正的 Ljung–Box 白噪声检验，记录诊断结果而不再用第二套准则重复搜索。",
      actionLabel: "检查残差",
    },
    {
      id: "select_model",
      label: "确认最终阶数",
      description: "确认一次 AICc 搜索得到的非季节 ARIMA(p, d, q)、截距或漂移设定和收敛信息；残差诊断作为教学解释，而不是事后换模依据。",
      actionLabel: "确认模型",
    },
    {
      id: "forecast",
      label: "预测评估区间",
      description: "使用选定的非季节 ARIMA(p, d, q) 直接生成多步预测；点预测按 max(0, ŷ) 截断。名义 95% 区间仍来自未截断的 ARIMA 分布，并按标准正态分布约 1.96 的 95% 临界值由区间宽度反推标准差。",
      actionLabel: "生成预测",
    },
    {
      id: "evaluate_metrics",
      label: "计算误差指标",
      description: "把非负截断后的 ARIMA 销量点预测与实际值对齐，计算误差指标并记录模型收敛状态。",
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
      description: "读取历史数据，确认数值或类别特征、数值目标、数值缩放方式和训练窗口有效；目标销量会自动作为历史输入的一部分，附加特征可以不选。",
      actionLabel: "解析数据",
    },
    {
      id: "preprocess",
      label: "预处理并构造样本",
      description: "仅用训练区间拟合数值缩放器和类别编码器，再把时间序列切成 LSTM 输入窗口与直接多步预测标签。",
      actionLabel: "构造样本",
    },
    {
      id: "build_model",
      label: "确定网络结构",
      description: "构建单层 LSTM 和长度等于预测跨度的线性全连接输出层；回看窗口在 2 到 12 之间动态确定，若无法保留至少 2 个历史步和 4 个监督窗口则拒绝训练；隐藏单元和批大小按监督窗口数、特征维度及参数预算确定。",
      actionLabel: "构建网络",
    },
    {
      id: "train_model",
      label: "训练神经网络",
      description: "标准数据量下按时间顺序划分并隔离相邻窗口，用验证损失执行早停并选择轮数，再以该轮数在全部窗口重拟合；样本太少时进入明确标注的教学演示模式。",
      actionLabel: "训练模型",
    },
    {
      id: "forecast",
      label: "生成评估预测",
      description: "仅使用训练截止点前由当前数据规模动态确定的最近历史窗口，一次直接输出完整评估跨度并还原销量尺度，再按 max(0, ŷ) 截断；不读取未来特征轨迹。",
      actionLabel: "生成预测",
    },
    {
      id: "evaluate_metrics",
      label: "计算误差指标",
      description: "比较非负截断后的预测值和实际值并计算指标；标准模式按验证窗口逐预测步估计波动，教学演示模式明确使用回退估计。",
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
      label: "划分选择与校准段",
      description: "按时间依次划分成员训练前缀、权重选择段和选择后校准尾段；样本不足时明确标记无独立校准。",
      actionLabel: "划分数据",
    },
    {
      id: "validation_predictions",
      label: "生成选择段预测",
      description: "只继承用户选择的 MA 窗口、ES 的 α、ARIMA 的 d 与 LSTM 特征和数值缩放方式；ARIMA 阶数和 LSTM 隐藏参数都在选择段之前重新推导。",
      actionLabel: "训练并预测",
    },
    {
      id: "compute_weights",
      label: "计算融合权重",
      description: "先按带数据尺度稳定项 ε 的 1 / MSE 得到候选权重，再依据选择样本数向等权组合收缩，减轻短选择段产生极端权重的风险。",
      actionLabel: "计算权重",
    },
    {
      id: "eval_predictions",
      label: "完整训练后生成评估预测",
      description: "在完整训练区间重训或复用基础模型，并为统一评估区间生成预测值。",
      actionLabel: "生成评估预测",
    },
    {
      id: "blend_evaluate",
      label: "加权组合并评估",
      description: "冻结收缩权重后，用更晚且不重叠的时间尾段校准经验范围，再计算统一评估区间的指标；信息不足会明确使用回退估计。",
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
      label: "划分元模型训练与校准段",
      description: "按时间顺序拆分初始训练前缀、元模型训练段和选择后校准段。",
      actionLabel: "划分层级",
    },
    {
      id: "level1_predictions",
      label: "生成元模型训练特征",
      description: "用多个扩展窗口滚动起点折生成元模型训练特征；每个折外预测点的成员只看到该点以前的数据。",
      actionLabel: "生成元模型特征",
    },
    {
      id: "fit_meta_model",
      label: "拟合元模型",
      description: "对折外预测矩阵拟合无截距非负最小二乘（NNLS）；高度共线或近奇异时加入轻度岭式稳定项，系数保持非负且不强制归一化。",
      actionLabel: "拟合元模型",
    },
    {
      id: "eval_predictions",
      label: "完整训练后评估预测",
      description: "基础模型在完整训练区间重训或复用产物，为统一评估区间生成元模型输入。",
      actionLabel: "生成评估预测",
    },
    {
      id: "meta_evaluate",
      label: "元模型组合并评估",
      description: "冻结 NNLS 元模型后，用更晚且不重叠的校准段估计经验范围，再在统一评估区间上计算指标。",
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
      description: "读取实验数据，校验训练区间、评估区间和基础模型候选库；轮数与阶段系数均由数据动态决定。",
      actionLabel: "校验数据",
    },
    {
      id: "split_residual_sets",
      label: "选择第 0 阶段并初始化残差",
      description: "先按时间拆分训练前缀、残差选择段与更晚的校准段，再在选择段比较训练均值、上一期延续基线和可用时的同季延续基线，选出第 0 阶段。",
      actionLabel: "初始化残差",
    },
    {
      id: "retrain_chain",
      label: "重训已选模型链",
      description: "先冻结选择链，并用选择后时间留出段之前的历史重拟合候选链、生成留出残差；随后把该留出段纳入完整训练并重新执行非负平方损失线搜索。两次最终链结构不一致时弃用残差。",
      actionLabel: "重训模型链",
    },
    {
      id: "combine_evaluate",
      label: "累加阶段预测并评估",
      description: "按第 0 阶段和残差阶段累加预测，最终销量做非负兜底；确认留出重拟合与完整重拟合的链结构一致后，用无名义覆盖保证的经验范围计算统一评估区间指标。",
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
