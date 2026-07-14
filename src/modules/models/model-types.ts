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
      description: "比较评估区间真实值和已经非负截断的销量预测值，估计残差波动，用于给预测结果附上不确定性范围。",
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
      description: "读取实验数据，检查训练区间和评估区间，并确认用户给定的 α（alpha，平滑系数）满足 0<α≤1；本系统不会自动优化 α。",
      actionLabel: "校验数据",
    },
    {
      id: "fit_model",
      label: "拟合平滑水平",
      description: "按教科书递推式，以第一个训练观测作为初始水平 S₁=Y₁，再使用固定 α 逐期更新水平项。",
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
      description: "比较已经非负截断的销量预测值和真实值，估计指数平滑模型在评估区间的误差离散程度。",
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
      description: "读取训练序列，检查固定差分阶数 d≥0 且差分后仍有足够样本，并按样本量确定非季节 ARIMA 的 p/q 搜索上限。",
      actionLabel: "准备搜索",
    },
    {
      id: "aic_search",
      label: "执行 AIC 搜索",
      description: "在固定 d 和样本量自适应的 p/q 边界内，运行一次以 AIC 为目标的 stepwise 搜索；它会逐步探索候选，而不是穷举每个组合。",
      actionLabel: "运行 AIC",
    },
    {
      id: "bic_search",
      label: "执行 BIC 搜索",
      description: "在相同边界内再运行一次以 BIC 为目标的 stepwise 搜索，利用更强的复杂度惩罚得到第二个候选优胜者。",
      actionLabel: "运行 BIC",
    },
    {
      id: "select_model",
      label: "选择最终阶数",
      description: "只比较两次 stepwise 搜索各自的优胜模型：训练点少于30时按 BIC 优先，至少30点时按 AIC 优先，并用另一准则处理并列。",
      actionLabel: "选择模型",
    },
    {
      id: "forecast",
      label: "预测评估区间",
      description: "使用选定的非季节 ARIMA(p,d,q) 直接生成多步预测；点预测按 max(0, ŷ) 截断。95%区间仍来自未截断的 ARIMA 分布，其宽度除以3.92得到近似标准差。",
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
      description: "仅用训练集拟合归一化和编码器，再把时间序列切成 LSTM 输入窗口与多步预测标签。",
      actionLabel: "构造样本",
    },
    {
      id: "build_model",
      label: "确定网络结构",
      description: "构建两层 LSTM、ReLU Dense 隐藏层和长度等于预测跨度的线性 Dense 输出层；隐藏单元与批大小按特征维度和样本量动态确定。",
      actionLabel: "构建网络",
    },
    {
      id: "train_model",
      label: "训练神经网络",
      description: "用 MSE、Adam、逐轮学习率衰减和不打乱的批训练拟合网络；为保留小样本，EarlyStopping 监控训练 loss 而非验证 loss，并恢复最佳训练损失权重。",
      actionLabel: "训练模型",
    },
    {
      id: "forecast",
      label: "生成评估预测",
      description: "仅使用训练截止点之前最后 look_back 行历史特征，一次直接输出完整评估跨度并还原销量尺度，再按 max(0, ŷ) 截断销量点预测；本实现不读取未来特征轨迹。",
      actionLabel: "生成预测",
    },
    {
      id: "evaluate_metrics",
      label: "计算误差指标",
      description: "比较非负截断后的 LSTM 销量点预测和真实值，计算 RMSE、MAE、MAPE、R²，并用同一预测口径估计残差波动。",
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
      description: "按时间顺序从训练区间末尾留出验证段：8–15个训练点时通常留2–3点，更长序列约留20%，专门估计成员误差。",
      actionLabel: "划分数据",
    },
    {
      id: "validation_predictions",
      label: "生成验证段预测",
      description: "只用验证段之前的数据重新拟合：MA窗口、ES的α及LSTM单模型配置沿用已完成产物，LSTM动态结构按当前数据段重算；ARIMA固定d并在前段重新搜索p、q，避免完整训练段的定阶结果进入留出预测。",
      actionLabel: "训练并预测",
    },
    {
      id: "compute_weights",
      label: "计算融合权重",
      description: "按 1/(MSE+10⁻⁹) 归一化为非负且和为1的权重；该小样本折中不估计成员误差之间的协方差，权重可能随短验证段波动。",
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
      description: "先把各成员销量预测截断到不小于0，再按固定权重组合；组合结果也做非负兜底，然后计算指标和残差波动。评估点不足2个时使用内部验证残差。",
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
      description: "基础模型只在 Level-0 重新拟合后预测 Level-1：MA、ES及LSTM沿用单模型配置，LSTM动态结构按当前数据段重算；ARIMA固定d并在Level-0重新搜索p、q，预测矩阵作为元模型输入。",
      actionLabel: "生成二层特征",
    },
    {
      id: "fit_meta_model",
      label: "拟合元模型",
      description: "先拟合无截距非负线性回归，再把系数归一化为和为1的权重；这是稳定、可解释的凸组合近似，并非一般 Stacking 对元学习器的唯一规定。",
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
      description: "把截断到不小于0的成员销量预测交给元模型，最终组合结果也做非负兜底后再评估；Level-1样本不多于模型数+1或矩阵病态时改用 inverse-MAE。",
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
      description: "按与其他融合模型相同的小样本时间留出规则拆分训练/验证段，并把原始销量作为第一轮要拟合的残差目标。",
      actionLabel: "初始化残差",
    },
    {
      id: "retrain_chain",
      label: "重训已选模型链",
      description: "按已选模型链重训：沿用成员核心配置，但第一阶段拟合原始销量并截断为非负，系数为1；后续阶段改训有符号残差且不截断，系数为学习率0.3。",
      actionLabel: "重训模型链",
    },
    {
      id: "combine_evaluate",
      label: "累加阶段预测并评估",
      description: "按各阶段系数累加预测，最后仅把物理销量结果截断到不小于0，再计算误差指标和残差波动。",
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
