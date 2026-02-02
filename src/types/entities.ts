export type UserRole = "Student" | "Teacher" | "Assistant" | "Admin";

export interface User {
  user_id: number;
  username: string;
  password_hash: string;
  full_name: string;
  email: string;
  phone_number: string | null;
  role: UserRole;
  created_at: Date;
}

export interface Class {
  class_id: number;
  class_name: string;
  class_code: string | null;
  teacher_id: number | null;
  created_at: Date;
}

export interface Enrollment {
  enrollment_id: number;
  student_id: number;
  class_id: number;
}

export interface ClassAssistant {
  assignment_id: number;
  assistant_id: number;
  class_id: number;
}

export type ExperimentStatusEnum = "Not Started" | "In Progress" | "Completed";
export type NormalizationType = "minmax" | "zscore";
export type CapacityMode = "scenario" | "auto" | "custom";
export type CapacityScenario = "tight" | "normal" | "abundant";
export type BestModelType =
  | "ma"
  | "exp"
  | "arima"
  | "lstm"
  | "ensemble_weighted"
  | "ensemble_boosting"
  | "ensemble_stacking";

export interface ExperimentStatus {
  experiment_id: number;
  student_id: number;
  status: ExperimentStatusEnum;
  highest_completed_step: number;
  current_step: number;
  total_active_duration_seconds: number;
  selected_industry: string | null;
  selected_company: string | null;
  selected_product: string | null;

  data_window_train_start_index: number | null;
  data_window_train_end_index: number | null;
  data_window_evaluate_start_index: number | null;
  data_window_evaluate_end_index: number | null;

  moving_average_completed: boolean;
  moving_average_window: number | null;
  moving_average_metrics_rmse: number | null;
  moving_average_metrics_mae: number | null;
  moving_average_metrics_r2: number | null;

  exponential_smoothing_completed: boolean;
  exponential_smoothing_alpha: number | null;
  exponential_smoothing_metrics_rmse: number | null;
  exponential_smoothing_metrics_mae: number | null;
  exponential_smoothing_metrics_r2: number | null;

  arima_completed: boolean;
  arima_p: number | null;
  arima_d: number | null;
  arima_q: number | null;
  arima_metrics_rmse: number | null;
  arima_metrics_mae: number | null;
  arima_metrics_r2: number | null;
  arima_adf_stationarity: unknown;

  lstm_completed: boolean;
  lstm_normalization: NormalizationType | null;
  lstm_features: string[] | null;
  lstm_target_field: string | null;
  lstm_metrics_rmse: number | null;
  lstm_metrics_mae: number | null;
  lstm_metrics_r2: number | null;

  ensemble_weighted_completed: boolean;
  ensemble_weighted_base_models: string[] | null;
  ensemble_weighted_metrics_rmse: number | null;
  ensemble_weighted_metrics_mae: number | null;
  ensemble_weighted_metrics_r2: number | null;

  ensemble_boosting_completed: boolean;
  ensemble_boosting_base_models: string[] | null;
  ensemble_boosting_metrics_rmse: number | null;
  ensemble_boosting_metrics_mae: number | null;
  ensemble_boosting_metrics_r2: number | null;

  ensemble_stacking_completed: boolean;
  ensemble_stacking_base_models: string[] | null;
  ensemble_stacking_metrics_rmse: number | null;
  ensemble_stacking_metrics_mae: number | null;
  ensemble_stacking_metrics_r2: number | null;

  selected_best_model: BestModelType | null;

  quiz_about_model_completed: boolean;
  quiz_about_plan_completed: boolean;

  production_plan_completed: boolean;
  production_forecast_periods: number | null;
  production_initial_inventory: number | null;
  production_target_service_level: number | null;
  production_safety_stock_z_score: number | null;
  production_mps_table: unknown;
  production_forecast_results: unknown;
  production_capacity_mode: CapacityMode | null;
  production_capacity_scenario: CapacityScenario | null;
  production_capacity: number | null;
  production_custom_capacity: number | null;

  start_time: Date;
  last_activity_at: Date;
  completion_time: Date | null;
}

export type QuestionType = "Single Choice" | "Multiple Choice" | "True/False";

export interface QuestionBankItem {
  question_id: number;
  knowledge_point: string;
  question_type: QuestionType;
  question_text: string;
  options: Record<string, string> | string[] | null;
  correct_answers: string[];
  creator_id: number | null;
}

export type QuizType = "quiz_about_model" | "quiz_about_plan";

export interface QuizAnswer {
  answer_id: number;
  experiment_id: number;
  student_id: number;
  question_id: number;
  quiz_type: QuizType;
  submitted_answer: string[];
  is_correct: boolean;
  answered_at: Date;
}

export type ReportStatus = "submitted" | "graded" | "rejected";

export interface ExperimentReport {
  report_id: number;
  experiment_id: number;
  student_id: number;
  report_content: string | null;
  pdf_file_path: string | null;
  status: ReportStatus;
  submitted_at: Date | null;
  grade: number | null;
  feedback: string | null;
  graded_by: number | null;
}

export interface GradeWeights {
  weight_id: number;
  class_id: number;
  exp_flow_demand_data_preparation: number;
  exp_flow_demand_descriptive_stats: number;
  exp_flow_demand_model_selection: number;
  exp_flow_demand_generate_results: number;
  exp_flow_production_inventory_calc: number;
  exp_flow_production_service_level: number;
  exp_flow_production_variable_calc: number;
  exp_flow_production_plan_creation: number;
  exp_flow_report_submission: number;
  exp_flow_weight: number;
  knowledge_test_weight: number;
  model_quality_weight: number;
  report_quality_weight: number;
}

export interface ExperimentGrade {
  grade_id: number;
  experiment_id: number;
  student_id: number;
  exp_flow_demand_data_preparation: number;
  exp_flow_demand_descriptive_stats: number;
  exp_flow_demand_model_selection: number;
  exp_flow_demand_generate_results: number;
  exp_flow_production_inventory_calc: number;
  exp_flow_production_service_level: number;
  exp_flow_production_variable_calc: number;
  exp_flow_production_plan_creation: number;
  exp_flow_report_submission: number;
  knowledge_test: number;
  model_quality: number;
  report_quality: number;
  created_at: Date;
}

export interface ExperimentManual {
  manual_id: number;
  file_name: string;
  file_path: string;
  description: string | null;
  is_active: boolean;
  uploader_id: number | null;
  uploaded_at: Date;
}

export interface Dataset {
  dataset_id: number;
  data_name: string;
  file_path: string;
  description: string | null;
  uploaded_at: Date;
}

export interface DatasetSegment {
  segment_id: number;
  dataset_id: number;
  industry_name: string;
  company_name: string;
  product_name: string;
  file_path: string;
}
