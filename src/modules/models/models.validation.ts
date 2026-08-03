import type { ZodIssue } from "zod";

const TRAIN_FIELD_LABELS: Partial<Record<string, string>> = {
  experiment_id: "实验 ID",
  selected_industry: "行业",
  selected_company: "企业",
  selected_product: "产品",
  data_window_train_start_index: "训练起始索引",
  data_window_train_end_index: "训练结束索引",
  data_window_evaluate_start_index: "评估起始索引",
  data_window_evaluate_end_index: "评估结束索引",
  moving_average_window: "移动平均时间窗口",
  exponential_smoothing_alpha: "指数平滑系数",
  arimaD: "ARIMA 差分阶数",
  lstmFeatures: "LSTM 输入特征",
  lstmTargetFeature: "LSTM 目标特征",
  lstmNormalization: "LSTM 归一化方式",
  models: "基础模型列表",
};

const PREDICT_FIELD_LABELS: Partial<Record<string, string>> = {
  experiment_id: "实验 ID",
  forecast_steps: "预测步数",
};

const getTrainFieldLabel = (field: string) => TRAIN_FIELD_LABELS[field] ?? field;
const getPredictFieldLabel = (field: string) => PREDICT_FIELD_LABELS[field] ?? field;

type IssueWithInput = ZodIssue & { input?: unknown; expected?: string };
type IssueWithMinimum = ZodIssue & { minimum?: number | bigint; inclusive?: boolean; origin?: string };

const formatInvalidTypeMessage = (label: string) => `${label} 类型无效`;

const formatMinimumMessage = (
  label: string,
  minimum: number | bigint | undefined,
  inclusive: boolean | undefined,
) => {
  if (minimum === undefined) {
    return `${label} 值过小`;
  }

  const comparator = inclusive === false ? "大于" : "大于等于";
  return `${label}必须${comparator} ${String(minimum)}`;
};

export type LocalizedValidationDetails = {
  formErrors: string[];
  fieldErrors: Record<string, string[]>;
};

export const localizeTrainValidationIssue = (issue: ZodIssue): string => {
  const field = String(issue.path[0] ?? "");
  const label = getTrainFieldLabel(field);

  if (field === "lstmNormalization") {
    return "LSTM 归一化方式必须是 minmax 或 zscore";
  }

  if (issue.code === "invalid_type") {
    const issueWithInput = issue as IssueWithInput;
    if (issueWithInput.expected === "int") {
      return `${label}必须为整数`;
    }
    if (issueWithInput.input === undefined) {
      return `${label}不能为空`;
    }
    return formatInvalidTypeMessage(label);
  }

  if (issue.code === "too_small") {
    const issueWithMinimum = issue as IssueWithMinimum;
    if (issueWithMinimum.origin === "string" && issueWithMinimum.minimum === 1) {
      return `${label}不能为空`;
    }

    return formatMinimumMessage(label, issueWithMinimum.minimum, issueWithMinimum.inclusive);
  }

  if (issue.code === "too_big") {
    if (field === "exponential_smoothing_alpha") {
      return "指数平滑系数必须小于等于 1";
    }
  }

  if (issue.message) {
    return issue.message;
  }

  return "训练请求参数无效";
};

export const buildLocalizedTrainValidationDetails = (issues: ZodIssue[]): LocalizedValidationDetails => {
  const fieldErrors: Record<string, string[]> = {};
  const formErrors: string[] = [];

  for (const issue of issues) {
    const message = localizeTrainValidationIssue(issue);
    const field = String(issue.path[0] ?? "");
    if (!field) {
      formErrors.push(message);
      continue;
    }

    fieldErrors[field] = [...(fieldErrors[field] ?? []), message];
  }

  return { formErrors, fieldErrors };
};

export const localizePredictValidationIssue = (issue: ZodIssue): string => {
  const field = String(issue.path[0] ?? "");
  const label = getPredictFieldLabel(field);

  if (issue.code === "invalid_type") {
    const issueWithInput = issue as IssueWithInput;
    if (issueWithInput.expected === "int") {
      return `${label}必须为整数`;
    }
    if (issueWithInput.input === undefined) {
      return `${label}不能为空`;
    }
    return formatInvalidTypeMessage(label);
  }

  if (issue.code === "too_small") {
    if (field === "forecast_steps") {
      return "预测步数必须大于等于 2";
    }

    const issueWithMinimum = issue as IssueWithMinimum;
    if (issueWithMinimum.origin === "string" && issueWithMinimum.minimum === 1) {
      return `${label}不能为空`;
    }

    return formatMinimumMessage(label, issueWithMinimum.minimum, issueWithMinimum.inclusive);
  }

  if (issue.code === "too_big" && field === "forecast_steps") {
    return "预测步数不能超过 12";
  }

  if (issue.code === "invalid_format") {
    return `${label}格式无效`;
  }

  if (issue.message) {
    return issue.message;
  }

  return "预测请求参数无效";
};

export const buildLocalizedPredictValidationDetails = (issues: ZodIssue[]): LocalizedValidationDetails => {
  const fieldErrors: Record<string, string[]> = {};
  const formErrors: string[] = [];

  for (const issue of issues) {
    const message = localizePredictValidationIssue(issue);
    const field = String(issue.path[0] ?? "");
    if (!field) {
      formErrors.push(message);
      continue;
    }

    fieldErrors[field] = [...(fieldErrors[field] ?? []), message];
  }

  return { formErrors, fieldErrors };
};
