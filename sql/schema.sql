-- --------------------------------------------------------
-- 虚拟仿真系统数据库结构文件 (Schema)
-- 适用于 MariaDB / MySQL
-- --------------------------------------------------------

--
-- 1. 用户与班级管理
--

CREATE TABLE IF NOT EXISTS `users` (
  `user_id` INT AUTO_INCREMENT PRIMARY KEY,
  `username` VARCHAR(255) NOT NULL UNIQUE,
  -- 所有用户的密码固定为$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG
  `password_hash` VARCHAR(255) NOT NULL,
  `full_name` VARCHAR(255) NOT NULL,
  `email` VARCHAR(255) NOT NULL UNIQUE,
  `phone_number` VARCHAR(50),
  `role` ENUM('Student', 'Teacher', 'Assistant', 'Admin') NOT NULL,
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `classes` (
  `class_id` INT AUTO_INCREMENT PRIMARY KEY,
  `class_name` VARCHAR(255) NOT NULL,
  `class_code` VARCHAR(255) UNIQUE,
  `teacher_id` INT NOT NULL,
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (`teacher_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS `enrollments` (
  `enrollment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `student_id` INT NOT NULL,
  `class_id` INT NOT NULL,
  FOREIGN KEY (`student_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,
  FOREIGN KEY (`class_id`) REFERENCES `classes`(`class_id`) ON DELETE CASCADE,
  UNIQUE KEY `student_class_unique` (`student_id`, `class_id`)
);

CREATE TABLE IF NOT EXISTS `class_assistants` (
  `assignment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `assistant_id` INT NOT NULL,
  `class_id` INT NOT NULL,
  FOREIGN KEY (`assistant_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,
  FOREIGN KEY (`class_id`) REFERENCES `classes`(`class_id`) ON DELETE CASCADE,
  UNIQUE KEY `assistant_class_unique` (`assistant_id`, `class_id`)
);

--
-- 2. 实验与生产计划管理
--

CREATE TABLE IF NOT EXISTS `experiment_status` (
  `experiment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `student_id` INT NOT NULL,
  `start_time` TIMESTAMP NULL,
  `last_activity_at` TIMESTAMP NULL,
  `completion_time` TIMESTAMP NULL,
  `status` ENUM('Not Started', 'In Progress', 'Completed') DEFAULT 'Not Started',

  -- 默认值为0，代表还没有完成任何步骤
  `highest_completed_step` INT UNSIGNED NOT NULL DEFAULT 0,

  -- 取值范围为1-7，代表共有七个步骤
  `current_step` INT UNSIGNED NOT NULL DEFAULT 1,
  `selected_industry` VARCHAR(255),
  `selected_company` VARCHAR(255),
  `selected_product` VARCHAR(255),
  `moving_average_completed` BOOLEAN DEFAULT NULL,
  `moving_average_window` INT,
  `moving_average_metrics_rmse` FLOAT,
  `moving_average_metrics_mae` FLOAT,
  `moving_average_metrics_r2` FLOAT,
  `exponential_smoothing_completed` BOOLEAN DEFAULT NULL,
  `exponential_smoothing_alpha` FLOAT,
  `exponential_smoothing_metrics_rmse` FLOAT,
  `exponential_smoothing_metrics_mae` FLOAT,
  `exponential_smoothing_metrics_r2` FLOAT,
  `arima_completed` BOOLEAN DEFAULT NULL,

  -- Tip for frontend: d is user-provided; p and q are auto-selected during training
  `arima_p` TINYINT,
  `arima_d` TINYINT,
  `arima_q` TINYINT,
  `arima_metrics_rmse` FLOAT,
  `arima_metrics_mae` FLOAT,
  `arima_metrics_r2` FLOAT,
  -- ADF stationarity diagnostics JSON array
  -- [
  --   { diff_order: 0, statistic: -1.876, p_value: 0.640, stationary: false,
  --     critical_values: { "1%": -3.500, "5%": -2.890, "10%": -2.580 } },
  --   { diff_order: 1, statistic: -3.954, p_value: 0.002, stationary: true,
  --     critical_values: { "1%": -3.500, "5%": -2.890, "10%": -2.580 } },
  --   { diff_order: 2, statistic: -5.102, p_value: 0.000, stationary: true,
  --     critical_values: { "1%": -3.500, "5%": -2.890, "10%": -2.580 } }
  -- ]
  `arima_adf_stationarity` JSON,


  `lstm_completed` BOOLEAN DEFAULT NULL,
  `lstm_normalization` ENUM('minmax', 'zscore'),
  -- csv文件列名组成的数组：['月份', '销售金额', '是否促销']
  `lstm_features` JSON,
  `lstm_metrics_rmse` FLOAT,
  `lstm_metrics_mae` FLOAT,
  `lstm_metrics_r2` FLOAT,

  `ensemble_weighted_completed` BOOLEAN DEFAULT NULL,
  -- 基础模型的缩写组成的数组：['ma', 'exp', 'arima', 'lstm']
  `ensemble_weighted_base_models` JSON,
  `ensemble_weighted_metrics_rmse` FLOAT,
  `ensemble_weighted_metrics_mae` FLOAT,
  `ensemble_weighted_metrics_r2` FLOAT,

  `ensemble_boosting_completed` BOOLEAN DEFAULT NULL,
  -- 基础模型的缩写组成的数组：['ma', 'exp', 'arima', 'lstm']
  `ensemble_boosting_base_models` JSON,
  `ensemble_boosting_metrics_rmse` FLOAT,
  `ensemble_boosting_metrics_mae` FLOAT,
  `ensemble_boosting_metrics_r2` FLOAT,


  `ensemble_stacking_completed` BOOLEAN DEFAULT NULL,
  -- 基础模型的缩写组成的数组：['ma', 'exp', 'arima', 'lstm']
  `ensemble_stacking_base_models` JSON,
  `ensemble_stacking_metrics_rmse` FLOAT,
  `ensemble_stacking_metrics_mae` FLOAT,
  `ensemble_stacking_metrics_r2` FLOAT,

  --
  `best_model` ENUM('ma', 'exp', 'arima', 'lstm', 'ensemble_weighted', 'ensemble_boosting', 'ensemble_stacking') DEFAULT NULL,
  FOREIGN KEY (`student_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS `experiment_step_events` (
  `event_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `student_id` INT NOT NULL,
  -- 取值范围为1-7
  `step_order` INT NOT NULL,
  `event_type` ENUM('STARTED', 'COMPLETED') NOT NULL,
  `event_timestamp` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (`experiment_id`) REFERENCES `experiment_status`(`experiment_id`) ON DELETE CASCADE,
  FOREIGN KEY (`student_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,
  INDEX `idx_experiment_step_order` (`experiment_id`, `step_order`, `event_timestamp`)
);

--
-- 3. 考核与评分
--

CREATE TABLE IF NOT EXISTS `question_bank` (
  `question_id` INT AUTO_INCREMENT PRIMARY KEY,
  `knowledge_point` VARCHAR(255),
  `question_type` ENUM('Single Choice', 'Multiple Choice', 'True/False') NOT NULL,
  `question_text` TEXT NOT NULL,

  -- {"A":"自回归项数","B":"差分阶数","C":"移动平均项数","D":"季节周期"}/["正确","错误"]
  `options` JSON,
  --["A","B","D"]/["正确"]
  `correct_answers` JSON NOT NULL,
  `creator_id` INT,
  FOREIGN KEY (`creator_id`) REFERENCES `users`(`user_id`) ON DELETE SET NULL
);


CREATE TABLE IF NOT EXISTS `experiment_reports` (
  `report_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `student_id` INT NOT NULL,
  `report_content` TEXT,
  -- absolute path on linux that end's with .pdf
  `pdf_file_path` VARCHAR(255),
  `submitted_at` TIMESTAMP NULL,
  -- 0-100
  `grade` FLOAT,
  `feedback` TEXT,
  `graded_by` INT,
  FOREIGN KEY (`experiment_id`) REFERENCES `experiment_status`(`experiment_id`) ON DELETE CASCADE,
  FOREIGN KEY (`student_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,
  FOREIGN KEY (`graded_by`) REFERENCES `users`(`user_id`) ON DELETE SET NULL
);

DROP TABLE IF EXISTS `grade_weights`;
CREATE TABLE `grade_weights` (
  `weight_id` INT AUTO_INCREMENT PRIMARY KEY,
  `teacher_id` INT NOT NULL UNIQUE,

  -- exp_flow_开头的字段和必须是100
  `exp_flow_demand_data_preparation` FLOAT DEFAULT 0,
  `exp_flow_demand_descriptive_stats` FLOAT DEFAULT 0,
  `exp_flow_demand_model_selection` FLOAT DEFAULT 0,
  `exp_flow_demand_generate_results` FLOAT DEFAULT 0,
  `exp_flow_production_inventory_calc` FLOAT DEFAULT 0,
  `exp_flow_production_service_level` FLOAT DEFAULT 0,
  `exp_flow_production_variable_calc` FLOAT DEFAULT 0,
  `exp_flow_production_plan_creation` FLOAT DEFAULT 0,
  `exp_flow_report_submission` FLOAT DEFAULT 0,

  -- exp_flow+knowledge_test+model_quality+report_quality的和必须是100
  `exp_flow` FLOAT DEFAULT 0,
  `knowledge_test` FLOAT DEFAULT 0,
  `model_quality` FLOAT DEFAULT 0,
  `report_quality` FLOAT DEFAULT 0,
  FOREIGN KEY (`teacher_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE
);

DROP TABLE IF EXISTS `experiment_grade`;
CREATE TABLE `experiment_grade` (
  `grade_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `student_id` INT NOT NULL,
  `grader_id` INT,

  -- 0-100
  `exp_flow` FLOAT DEFAULT 0,

  -- 以下所有字段的取值都是0-100，暂时先不考虑exp_flow_开头的字段的和加权平均得到exp_flow的值这个约束
  `exp_flow_demand_data_preparation` FLOAT DEFAULT 0,
  `exp_flow_demand_descriptive_stats` FLOAT DEFAULT 0,
  `exp_flow_demand_model_selection` FLOAT DEFAULT 0,
  `exp_flow_demand_generate_results` FLOAT DEFAULT 0,
  `exp_flow_production_inventory_calc` FLOAT DEFAULT 0,
  `exp_flow_production_service_level` FLOAT DEFAULT 0,
  `exp_flow_production_variable_calc` FLOAT DEFAULT 0,
  `exp_flow_production_plan_creation` FLOAT DEFAULT 0,
  `exp_flow_report_submission` FLOAT DEFAULT 0,
  `knowledge_test` FLOAT DEFAULT 0,
  `model_quality` FLOAT DEFAULT 0,
  `report_quality` FLOAT DEFAULT 0,
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (`experiment_id`) REFERENCES `experiment_status`(`experiment_id`) ON DELETE CASCADE,
  FOREIGN KEY (`student_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,
  FOREIGN KEY (`grader_id`) REFERENCES `users`(`user_id`) ON DELETE SET NULL
);

--
-- 4. 系统管理
--

CREATE TABLE IF NOT EXISTS `experiment_manuals` (
  `manual_id` INT AUTO_INCREMENT PRIMARY KEY,
  `file_name` VARCHAR(255) NOT NULL,

  -- absolute linux path ends with .pdf
  `file_path` VARCHAR(255) NOT NULL,
  `description` TEXT,
  `is_active` BOOLEAN DEFAULT false,
  `uploader_id` INT,
  `uploaded_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (`uploader_id`) REFERENCES `users`(`user_id`) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS `datasets` (
  `dataset_id` INT AUTO_INCREMENT PRIMARY KEY,
  `data_name` VARCHAR(255) NOT NULL,

  -- absolute linux path ends with .csv
  `file_path` VARCHAR(255) NOT NULL,
  `description` TEXT,
  `uploaded_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `dataset_segments` (
  `segment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `dataset_id` INT NOT NULL,
  `industry_name` VARCHAR(255) NOT NULL,
  `company_name` VARCHAR(255) NOT NULL,
  `product_name` VARCHAR(255) NOT NULL,
  `start_row` INT NOT NULL,
  `end_row` INT NOT NULL,

  -- absolute linux path ends with .csv
  `file_path` VARCHAR(512) NOT NULL,
  FOREIGN KEY (`dataset_id`) REFERENCES `datasets`(`dataset_id`) ON DELETE CASCADE,
  INDEX `idx_dataset_segments_dataset_id` (`dataset_id`),
  UNIQUE KEY `uniq_dataset_segments_product` (`industry_name`, `company_name`, `product_name`)
);
