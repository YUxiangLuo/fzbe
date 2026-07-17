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
      description: "读取实验数据，检查训练区间、评估区间，并确认窗口 n 为不超过训练长度的整数且 n≥2；n=1 只是上一期值延续，不体现均值平滑，因此本系统不开放。",
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
      description: "只用训练段执行 rolling-origin 多步回测，逐 horizon 估计预测波动；样本不足的 horizon 使用明确标记的 fallback，评估区间真实值不参与校准。",
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
      description: "保存移动平均窗口、最近历史值和训练段 rolling-origin 预测波动，供后续预测和报告复用。",
      actionLabel: "保存结果",
    },
  ],
  es: [
    {
      id: "prepare_data",
      label: "确认平滑设置",
      description: "读取实验数据，检查训练区间和评估区间，并确认用户给定的 α（alpha，平滑系数）满足 0<α≤1；本系统不会自动优化 α。",
      actionLabel: "校验数据",
    },
    {
      id: "fit_model",
      label: "拟合平滑水平",
      description: "在固定用户 α 的前提下，用训练数据估计 ETS(A,N,N) 的初始水平，再逐期更新水平项；α 本身不会被自动改变。",
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
      description: "只用训练期一步预测残差估计创新尺度，并按 ETS(A,N,N) 解析公式让多步预测标准差随 horizon 展宽；评估区间真实值不参与不确定性校准。",
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
      description: "读取训练序列，检查固定差分阶数 d≥0 且差分后仍有足够样本，并按样本量确定非季节 ARIMA 的 p/q 搜索上限。",
      actionLabel: "准备搜索",
    },
    {
      id: "aic_search",
      label: "执行 AICc 搜索",
      description: "在固定 d 和样本量自适应的 p/q 边界内运行 AICc stepwise 搜索，每个候选拟合使用 maxiter=100；检查优胜模型是否收敛，必要时按完整规格以 maxiter=200 重试。",
      actionLabel: "运行 AICc",
    },
    {
      id: "bic_search",
      label: "检查残差诊断",
      description: "对 AICc 优胜模型在注明的单个非季节滞后执行带模型自由度修正的 Ljung–Box 检验；未发现残差自相关不等于已经证明完整残差过程为白噪声。",
      actionLabel: "检查残差",
    },
    {
      id: "select_model",
      label: "确认最终阶数",
      description: "确认一次 AICc 搜索得到的非季节 ARIMA(p,d,q)、截距/漂移设定和收敛信息；残差诊断作为教学解释而非事后换模依据。",
      actionLabel: "确认模型",
    },
    {
      id: "forecast",
      label: "预测评估区间",
      description: "使用选定的非季节 ARIMA(p,d,q) 直接生成多步预测；点预测和 95% 区间分位数在销量域同步按 max(0, ·) 映射，标准差仍按未截断分布及精确的 95% 正态临界值 1.9599639845 计算。",
      actionLabel: "生成预测",
    },
    {
      id: "evaluate_metrics",
      label: "计算误差指标",
      description: "把非负截断后的 ARIMA 销量点预测与真实值对齐，计算误差指标并记录模型收敛状态。",
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
      description: "读取多变量历史数据，确认数值/类别特征、数值目标、归一化方式和训练窗口有效；目标销量会自动作为历史输入的一部分。",
      actionLabel: "解析数据",
    },
    {
      id: "preprocess",
      label: "预处理并构造样本",
      description: "仅用训练集拟合归一化和编码器，再把预测起点之前的目标历史与历史协变量切成 LSTM 输入窗口；不接收任何已知未来特征轨迹。",
      actionLabel: "构造样本",
    },
    {
      id: "build_model",
      label: "确认网络配置",
      description: "确认后续训练将使用单层 LSTM 和长度等于预测跨度的线性 Dense 输出层；look-back 在2到12之间动态确定，若无法保留至少两个历史步和四个监督窗口则拒绝训练；隐藏单元和批大小按样本窗数、特征维度及参数预算确定。网络会在下一阶段按该配置创建并训练。",
      actionLabel: "确认配置",
    },
    {
      id: "train_model",
      label: "训练神经网络",
      description: "监督窗口足够时进入有限时间验证模式：按时间顺序划分并隔离相邻窗口，用验证 loss 早停选择轮数，再以该轮数在全部窗口重拟合；样本太少时进入明确标注的教学演示模式。两种模式都不代表生产级泛化验证。",
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
      description: "比较非负截断后的预测和真实值并计算指标；有限时间验证模式复用 EarlyStopping 验证窗口逐 horizon 形成名义误差范围，教学演示模式明确使用 fallback。两者均不宣称 95%/99% 覆盖率已校准，并应另行对比朴素基线。",
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
      description: "按模型数和样本量确定期望验证段，再在保证所有成员均可训练的前提下动态留出时间末段。",
      actionLabel: "划分数据",
    },
    {
      id: "validation_predictions",
      label: "生成验证段预测",
      description: "只继承用户选择的 MA 窗口、ES α、ARIMA d 与 LSTM 特征/归一化；ARIMA 阶数和 LSTM 隐藏参数都在留出段之前重新推导。",
      actionLabel: "训练并预测",
    },
    {
      id: "compute_weights",
      label: "计算融合权重",
      description: "先按带尺度 epsilon 的 1/MSE 得到候选权重，再依据验证样本数向等权组合收缩，减轻短验证段极端权重。",
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
      description: "按收缩权重组合非负成员预测并计算独立评估指标；名义误差范围复用同一个权重拟合留出段的一次固定原点残差，并明确标记为未校准、无覆盖率保证。",
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
      description: "按时间顺序把训练区间拆成基础模型训练段和元模型训练段，保持预测的时间因果顺序并降低信息泄漏风险。",
      actionLabel: "划分层级",
    },
    {
      id: "level1_predictions",
      label: "生成元模型训练特征",
      description: "基础模型只在 Level-0 重新拟合后预测 Level-1；只继承用户参数，ARIMA 阶数与 LSTM 隐藏配置都按 Level-0 重算。",
      actionLabel: "生成二层特征",
    },
    {
      id: "fit_meta_model",
      label: "拟合元模型",
      description: "对 Level-1 预测矩阵拟合无截距非负最小二乘（NNLS）；保留原始非负系数，不强制归一化。",
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
      description: "把非负成员预测交给 NNLS 元模型并对最终销量做非负兜底；名义误差范围复用同一个元模型拟合 Level-1 留出段的一次固定原点残差，明确标记为未校准、无覆盖率保证。",
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
      label: "初始化残差训练集",
      description: "动态拆分时间顺序训练/验证段，把原始销量作为第一轮目标，并由模型数与验证点数确定最多轮数。",
      actionLabel: "初始化残差",
    },
    {
      id: "retrain_chain",
      label: "准备部署模型链",
      description: "按已选模型链准备完整训练区间的部署产物：首阶段可复用配置和训练区间匹配的已完成基础模型，必要时重新拟合；后续阶段按当前完整训练残差拟合。所有阶段都保留时间留出验证学到的非负系数，完整训练拟合值只用于构造下一阶段残差。",
      actionLabel: "准备模型链",
    },
    {
      id: "combine_evaluate",
      label: "累加阶段预测并评估",
      description: "按验证阶段系数累加预测，把最终销量点预测和误差范围限制为非负并计算独立评估指标；名义误差范围复用同一个 Boosting 选模留出段的一次固定原点跨 horizon 残差，明确标记为未校准、无覆盖率保证。",
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
