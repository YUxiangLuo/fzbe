type ExperimentStatus = "Not Started" | "In Progress" | "Completed" | string;

export interface ExperimentUpdateRowSnapshot {
  status: ExperimentStatus;
  start_time: string | Date | null;
  last_activity_at: string | Date | null;
  total_active_duration_seconds: number | null;
}

export interface StudentExperimentUpdatePreparationSuccess {
  data: Record<string, unknown>;
}

export interface StudentExperimentUpdatePreparationFailure {
  error: "completed" | "no_fields" | "invalid_current_step" | "invalid_highest_step";
}

export type StudentExperimentUpdatePreparationResult =
  | StudentExperimentUpdatePreparationSuccess
  | StudentExperimentUpdatePreparationFailure;

export const MANAGED_EXPERIMENT_UPDATE_FIELDS = new Set([
  "expected_version",
  "state_version",
  "experiment_id",
  "student_id",
  "status",
  "start_time",
  "completion_time",
  "last_activity_at",
  "total_active_duration_seconds",
  "quiz_about_model_completed",
  "quiz_about_plan_completed",
  "moving_average_completed",
  "moving_average_metrics_rmse",
  "moving_average_metrics_mae",
  "moving_average_metrics_r2",
  "exponential_smoothing_completed",
  "exponential_smoothing_metrics_rmse",
  "exponential_smoothing_metrics_mae",
  "exponential_smoothing_metrics_r2",
  "arima_completed",
  "arima_p",
  "arima_q",
  "arima_metrics_rmse",
  "arima_metrics_mae",
  "arima_metrics_r2",
  "lstm_completed",
  "lstm_metrics_rmse",
  "lstm_metrics_mae",
  "lstm_metrics_r2",
  "ensemble_weighted_completed",
  "ensemble_weighted_metrics_rmse",
  "ensemble_weighted_metrics_mae",
  "ensemble_weighted_metrics_r2",
  "ensemble_boosting_completed",
  "ensemble_boosting_metrics_rmse",
  "ensemble_boosting_metrics_mae",
  "ensemble_boosting_metrics_r2",
  "ensemble_stacking_completed",
  "ensemble_stacking_metrics_rmse",
  "ensemble_stacking_metrics_mae",
  "ensemble_stacking_metrics_r2",
]);

const toDate = (value: string | Date | null): Date | null => {
  if (!value) return null;
  if (value instanceof Date) return value;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

export const prepareStudentExperimentUpdate = (
  row: ExperimentUpdateRowSnapshot,
  sanitizedInput: Record<string, unknown>,
  now: Date = new Date(),
): StudentExperimentUpdatePreparationResult => {
  if (row.status === "Completed") {
    return { error: "completed" };
  }

  const nextData: Record<string, unknown> = { ...sanitizedInput };
  for (const field of MANAGED_EXPERIMENT_UPDATE_FIELDS) {
    delete nextData[field];
  }

  if (Object.keys(nextData).length === 0) {
    return { error: "no_fields" };
  }

  if (nextData.current_step === null) {
    return { error: "invalid_current_step" };
  }

  if (nextData.highest_completed_step === null) {
    return { error: "invalid_highest_step" };
  }

  const previousLast = toDate(row.last_activity_at) ?? toDate(row.start_time);
  nextData.last_activity_at = now;

  if (row.status === "Not Started") {
    nextData.status = "In Progress";
    if (!row.start_time) {
      nextData.start_time = now;
    }
  }

  if (previousLast) {
    const diffMs = now.getTime() - previousLast.getTime();
    const maxGapMs = 60 * 60 * 1000;
    if (diffMs > 0 && diffMs <= maxGapMs) {
      const previousTotal = Number(row.total_active_duration_seconds ?? 0);
      nextData.total_active_duration_seconds = previousTotal + Math.max(1, Math.floor(diffMs / 1000));
    }
  }

  return { data: nextData };
};
