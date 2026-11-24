import type { RowDataPacket } from "mysql2/promise";
import type { ModelType } from "./models.schema";
import { ModelValidationError } from "./model-errors";
import { parseJsonColumn, isRecordValue } from "./model-json";
import { stableJsonStringify } from "../../utils/stable-json";

// Step protocol and response projections; no database, filesystem or Python work.
export type ProductionPreparationStatus = "ready" | "running" | "failed" | "completed" | "superseded";

export interface ProductionPreparationStepDefinition {
  id: string;
  label: string;
}

export interface ProductionPreparationSessionRow extends RowDataPacket {
  session_id: string;
  experiment_id: number;
  student_id: number;
  run_id: string;
  model_type: ModelType;
  request_hash: string;
  dataset_revision: string;
  runtime_revision: string;
  protocol_version: number;
  forecast_steps: number;
  status: ProductionPreparationStatus;
  current_step_id: string | null;
  completed_step_ids: unknown;
  step_plan: unknown;
  artifact_dir: string;
  candidate_model_path: string;
  committed_state_path: string | null;
  committed_state_sha256: string | null;
  active_attempt_id: string | null;
  observed_production_revision: string | null;
  production_revision: string | null;
  error_message: string | null;
  superseded_reason: string | null;
  expires_at: Date | string | null;
  updated_at: Date | string;
  /** Computed by MySQL to avoid application timezone interpretation. */
  session_is_live?: number | boolean | string | null;
  active_attempt_lease_is_live?: number | boolean | string | null;
}

export interface ProductionPreparationAttemptRow extends RowDataPacket {
  attempt_id: string;
  session_id: string;
  step_id: string;
  status: "running" | "committed" | "failed";
  input_state_path: string | null;
  input_state_sha256: string | null;
  candidate_state_path: string;
  candidate_state_sha256: string | null;
  receipt_path: string;
  output_payload: unknown;
  error_message: string | null;
  lease_expires_at: Date | string | null;
  /** Computed by MySQL so DB_DATE_STRINGS and application timezone cannot skew lease ownership. */
  lease_is_active?: number | boolean | string | null;
}

export interface ProductionPreparationSessionPayload {
  status: "success";
  results: {
    session_id: string;
    session_status: ProductionPreparationStatus;
    completed_step_ids: string[];
    total_steps: number;
    next_step: ProductionPreparationStepDefinition | null;
    error_message: string | null;
    preparation_token?: string;
    prepared_forecast_steps?: number;
    reused_existing?: boolean;
  };
}

const PRODUCTION_WORK_ITEM_ID = /^(?:fit_model|fit_member_(?:ma|es|arima|lstm)|fit_stage_\d+_(?:ma|es|arima|lstm))$/;

export const parseProductionCompletedStepIds = (value: unknown): string[] | null => {
  const parsed = parseJsonColumn<unknown>(value, null);
  if (
    !Array.isArray(parsed)
    || !parsed.every((item) => typeof item === "string" && item.length > 0)
  ) {
    return null;
  }
  return parsed;
};

export const parseProductionPreparationStepPlan = (
  value: unknown,
): ProductionPreparationStepDefinition[] | null => {
  const parsed = parseJsonColumn<unknown>(value, null);
  if (!Array.isArray(parsed)) return null;
  const plan: ProductionPreparationStepDefinition[] = [];
  for (const item of parsed) {
    if (
      !isRecordValue(item)
      || typeof item.id !== "string"
      || item.id.length === 0
      || typeof item.label !== "string"
      || item.label.length === 0
    ) {
      return null;
    }
    plan.push({ id: item.id, label: item.label });
  }
  return plan;
};

export const hasValidProductionStepProgress = (
  row: Pick<ProductionPreparationSessionRow, "completed_step_ids" | "step_plan" | "status">,
) => {
  const stepPlan = parseProductionPreparationStepPlan(row.step_plan);
  const completedStepIds = parseProductionCompletedStepIds(row.completed_step_ids);
  if (!stepPlan || !completedStepIds) return false;
  const planIds = stepPlan.map((step) => step.id);
  const hasValidPlan = planIds.length >= 2
    && planIds[0] === "initialize"
    && planIds.at(-1) === "finalize"
    && planIds.slice(1, -1).every((id) => PRODUCTION_WORK_ITEM_ID.test(id))
    && new Set(planIds).size === planIds.length;
  const hasValidPrefix = completedStepIds.every(
    (stepId, index) => stepId === planIds[index],
  ) && (
    row.status === "completed"
      ? completedStepIds.length === planIds.length
      : completedStepIds.length < planIds.length
  );
  return hasValidPlan && hasValidPrefix;
};

export const buildProductionPreparationPayload = (
  row: ProductionPreparationSessionRow,
  reusedExisting = false,
): ProductionPreparationSessionPayload => {
  const completedStepIds = parseProductionCompletedStepIds(row.completed_step_ids) ?? [];
  const stepPlan = parseProductionPreparationStepPlan(row.step_plan) ?? [];
  const completed = new Set(completedStepIds);
  const nextStep = row.status === "completed" || row.status === "superseded"
    ? null
    : stepPlan.find((step) => !completed.has(step.id)) ?? null;
  return {
    status: "success",
    results: {
      session_id: row.session_id,
      session_status: row.status,
      completed_step_ids: completedStepIds,
      total_steps: stepPlan.length,
      next_step: nextStep,
      error_message: row.error_message,
      ...(row.status === "completed" && row.production_revision
        ? {
          preparation_token: row.production_revision,
          prepared_forecast_steps: Number(row.forecast_steps),
          reused_existing: reusedExisting,
        }
        : {}),
    },
  };
};

export const buildProductionStepLabel = (stepId: string) => {
  const modelLabels: Record<string, string> = {
    ma: "移动平均",
    es: "指数平滑",
    arima: "ARIMA",
    lstm: "LSTM",
  };
  if (stepId === "initialize") return "确认生产准备材料";
  if (stepId === "fit_model") return "重拟合生产模型";
  if (stepId === "finalize") return "组装并校验生产产物";
  const member = /^fit_member_(ma|es|arima|lstm)$/.exec(stepId);
  if (member) return `重拟合成员：${modelLabels[member[1]!]}`;
  const stage = /^fit_stage_(\d+)_(ma|es|arima|lstm)$/.exec(stepId);
  if (stage) return `重拟合第 ${stage[1]} 阶段：${modelLabels[stage[2]!]}`;
  throw new ModelValidationError("生产准备返回了未知步骤");
};

export const expandProductionPreparationPlan = (
  currentPlan: ProductionPreparationStepDefinition[],
  output: unknown,
  modelType: ModelType,
) => {
  if (!isRecordValue(output) || !Array.isArray(output.work_items)) {
    throw new ModelValidationError("生产准备初始化未返回有效工作项");
  }
  const workItems = output.work_items.map((item) => {
    if (!isRecordValue(item) || typeof item.id !== "string") {
      throw new ModelValidationError("生产准备初始化返回了无效工作项");
    }
    if (!/^(?:fit_model|fit_member_(?:ma|es|arima|lstm)|fit_stage_\d+_(?:ma|es|arima|lstm))$/.test(item.id)) {
      throw new ModelValidationError("生产准备初始化返回了未知工作项");
    }
    return { id: item.id, label: buildProductionStepLabel(item.id) };
  });
  const deduped = new Set(workItems.map((item) => item.id));
  const directFinalize = modelType === "boosting" && output.direct_finalize === true;
  if (
    deduped.size !== workItems.length
    || (workItems.length === 0 && !directFinalize)
    || (workItems.length > 0 && output.direct_finalize === true)
  ) {
    throw new ModelValidationError("生产准备初始化返回了重复或空工作项");
  }
  return [currentPlan[0]!, ...workItems, currentPlan[currentPlan.length - 1]!];
};

export const jsonColumnCasValue = (value: unknown) => (
  typeof value === "string" ? value : JSON.stringify(value)
);

/**
 * Dual-engine JSON equality predicate compatible with both MySQL 8 (native JSON
 * column type, where `col = ?` always evaluates false against a string parameter)
 * and MariaDB (LONGTEXT JSON alias, where `CAST(? AS JSON)` is a syntax error and
 * one-sided `col = JSON_EXTRACT(?, '$')` fails due to whitespace formatting).
 */
export const jsonColumnEqualsSql = (columnExpr: string) => (
  `JSON_EXTRACT(${columnExpr}, '$') = JSON_EXTRACT(?, '$')`
);

const normalizeCasTimestamp = (value: Date | string | null | undefined): string | null => {
  if (value === null || value === undefined) return null;
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value.toISOString();
  }
  return String(value);
};

const normalizeCasJson = (value: unknown): string => (
  stableJsonStringify(parseJsonColumn<unknown>(value, value))
);

type ProductionPreparationCasSnapshot = Pick<
  ProductionPreparationSessionRow,
  | "status"
  | "active_attempt_id"
  | "current_step_id"
  | "expires_at"
  | "completed_step_ids"
  | "step_plan"
  | "committed_state_path"
  | "committed_state_sha256"
  | "production_revision"
>;

/**
 * Compares only persisted CAS columns (never clock-derived projections such as
 * `session_is_live` or `active_attempt_lease_is_live`) so wall-clock expiration
 * between reads cannot masquerade as concurrent progress and mask a broken CAS.
 */
export const hasProductionPreparationCasRowChanged = (
  before: ProductionPreparationCasSnapshot,
  after: ProductionPreparationCasSnapshot,
): boolean => (
  before.status !== after.status
  || (before.active_attempt_id ?? null) !== (after.active_attempt_id ?? null)
  || (before.current_step_id ?? null) !== (after.current_step_id ?? null)
  || normalizeCasTimestamp(before.expires_at) !== normalizeCasTimestamp(after.expires_at)
  || normalizeCasJson(before.completed_step_ids) !== normalizeCasJson(after.completed_step_ids)
  || normalizeCasJson(before.step_plan) !== normalizeCasJson(after.step_plan)
  || (before.committed_state_path ?? null) !== (after.committed_state_path ?? null)
  || (before.committed_state_sha256 ?? null) !== (after.committed_state_sha256 ?? null)
  || (before.production_revision ?? null) !== (after.production_revision ?? null)
);
