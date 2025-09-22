-- --------------------------------------------------------
-- Virtual Simulation System Database Schema File
-- Compatible with MariaDB / MySQL
-- --------------------------------------------------------

SET NAMES utf8mb4;

-- ========================================================
-- 1. User and Class Management
-- ========================================================

CREATE TABLE IF NOT EXISTS `users` (
  `user_id` INT AUTO_INCREMENT PRIMARY KEY,
  `username` VARCHAR(255) NOT NULL UNIQUE,
  -- All users' passwords are fixed to $2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG
  `password_hash` VARCHAR(255) NOT NULL,
  `full_name` VARCHAR(255) NOT NULL,
  `email` VARCHAR(255) NOT NULL UNIQUE,
  `phone_number` VARCHAR(50),
  `role` ENUM('Student', 'Teacher', 'Assistant', 'Admin') NOT NULL,
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `classes` (
  `class_id` INT AUTO_INCREMENT PRIMARY KEY,
  `class_name` VARCHAR(255) NOT NULL,
  `class_code` VARCHAR(255) UNIQUE,
  `teacher_id` INT NULL,
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  -- 设计选择：避免 users → classes → enrollments/assistants 与 users → enrollments/assistants 的多路径级联
  -- 因此这里使用 SET NULL（或在你的业务中先改派再删老师）
  FOREIGN KEY (`teacher_id`) REFERENCES `users`(`user_id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `enrollments` (
  `enrollment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `student_id` INT NOT NULL,
  `class_id` INT NOT NULL,
  FOREIGN KEY (`student_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,
  FOREIGN KEY (`class_id`) REFERENCES `classes`(`class_id`) ON DELETE CASCADE,
  UNIQUE KEY `student_class_unique` (`student_id`, `class_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `class_assistants` (
  `assignment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `assistant_id` INT NOT NULL,
  `class_id` INT NOT NULL,
  FOREIGN KEY (`assistant_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,
  FOREIGN KEY (`class_id`) REFERENCES `classes`(`class_id`) ON DELETE CASCADE,
  UNIQUE KEY `assistant_class_unique` (`assistant_id`, `class_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ========================================================
-- 2. Experiment and Production Planning Management
-- ========================================================

CREATE TABLE IF NOT EXISTS `experiment_status` (
  `experiment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `student_id` INT NOT NULL,
  `start_time` TIMESTAMP NULL,
  `last_activity_at` TIMESTAMP NULL,
  `completion_time` TIMESTAMP NULL,
  `status` ENUM('Not Started', 'In Progress', 'Completed') DEFAULT 'Not Started',

  -- Default value is 0, indicating no steps have been completed yet
  `highest_completed_step` INT UNSIGNED NOT NULL DEFAULT 0,

  -- Values range from 1-7, representing a total of seven steps
  `current_step` INT UNSIGNED NOT NULL DEFAULT 1,
  `selected_industry` VARCHAR(255),
  `selected_company` VARCHAR(255),
  `selected_product` VARCHAR(255),

  -- {"train_start_index": 0, "train_end_index": 100, "evaluate_start_index": 101, "evaluate_end_index": 120}
  `data_window` JSON DEFAULT NULL,

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
  -- CSV file column names array: ['Month', 'Sales Amount', 'Is Promotion']
  `lstm_features` JSON,
  `lstm_metrics_rmse` FLOAT,
  `lstm_metrics_mae` FLOAT,
  `lstm_metrics_r2` FLOAT,

  `ensemble_weighted_completed` BOOLEAN DEFAULT NULL,
  -- Array of base model abbreviations: ['ma', 'exp', 'arima', 'lstm']
  `ensemble_weighted_base_models` JSON,
  `ensemble_weighted_metrics_rmse` FLOAT,
  `ensemble_weighted_metrics_mae` FLOAT,
  `ensemble_weighted_metrics_r2` FLOAT,

  `ensemble_boosting_completed` BOOLEAN DEFAULT NULL,
  -- Array of base model abbreviations: ['ma', 'exp', 'arima', 'lstm']
  `ensemble_boosting_base_models` JSON,
  `ensemble_boosting_metrics_rmse` FLOAT,
  `ensemble_boosting_metrics_mae` FLOAT,
  `ensemble_boosting_metrics_r2` FLOAT,

  `ensemble_stacking_completed` BOOLEAN DEFAULT NULL,
  -- Array of base model abbreviations: ['ma', 'exp', 'arima', 'lstm']
  `ensemble_stacking_base_models` JSON,
  `ensemble_stacking_metrics_rmse` FLOAT,
  `ensemble_stacking_metrics_mae` FLOAT,
  `ensemble_stacking_metrics_r2` FLOAT,

  `selected_best_model` ENUM('ma', 'exp', 'arima', 'lstm', 'ensemble_weighted', 'ensemble_boosting', 'ensemble_stacking') DEFAULT NULL,
  -- 1-6
  `predict_steps` INT DEFAULT NULL,
  -- [100], [100, 110, 120, 130], [100, 110, 120, 130, 140, 150]
  `predict_values` JSON DEFAULT NULL,

  -- quiz_1_completed → quiz_about_model_completed
  -- quiz_2_completed → quiz_about_plan_completed
  `quiz_about_model_completed` BOOLEAN DEFAULT NULL,
  `quiz_about_plan_completed` BOOLEAN DEFAULT NULL,

  `experiment_report_submitted` BOOLEAN DEFAULT NULL,

  FOREIGN KEY (`student_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,

  -- 为子表复合外键提供可引用的唯一性
  UNIQUE KEY `uq_experiment_status_exp_stu` (`experiment_id`, `student_id`),
  INDEX `idx_experiment_status_student` (`student_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `experiment_step_events` (
  `event_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `student_id` INT NOT NULL,

  -- Values range from 1-7
  `step_order` INT NOT NULL,
  `event_type` ENUM('STARTED', 'COMPLETED') NOT NULL,
  `event_timestamp` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

  -- 复合外键锚定到 experiment_status，避免 users → 子表 的多路径级联
  FOREIGN KEY (`experiment_id`, `student_id`)
    REFERENCES `experiment_status`(`experiment_id`, `student_id`)
    ON DELETE CASCADE,

  INDEX `idx_experiment_step_order` (`experiment_id`, `step_order`, `event_timestamp`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ========================================================
-- 3. Assessment and Grading
-- ========================================================

CREATE TABLE IF NOT EXISTS `question_bank` (
  `question_id` INT AUTO_INCREMENT PRIMARY KEY,
  `knowledge_point` ENUM('ARIMA基础', 'LSTM基础', '融合模型原理', '生产变量计算公式', '服务水平计算公式') NOT NULL,
  `question_type` ENUM('Single Choice', 'Multiple Choice', 'True/False') NOT NULL,
  `question_text` TEXT NOT NULL,

  -- {"A":"Autoregressive order","B":"Differencing order","C":"Moving average order","D":"Seasonal period"} / ["True","False"]
  `options` JSON,

  -- ["A","B","D"] / ["True"]
  `correct_answers` JSON NOT NULL,

  `creator_id` INT,
  FOREIGN KEY (`creator_id`) REFERENCES `users`(`user_id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `quiz_answers` (
  `answer_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `student_id` INT NOT NULL,
  `question_id` INT NOT NULL,
  `quiz_type` ENUM('quiz_about_model', 'quiz_about_plan') NOT NULL,
  -- Submitted answers, stored in the same JSON array format as the question bank's correct_answers
  `submitted_answer` JSON NOT NULL,
  `is_correct` BOOLEAN, -- Can be calculated and stored upon submission
  `answered_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

  -- Composite foreign key to a specific student's experiment
  FOREIGN KEY (`experiment_id`, `student_id`)
    REFERENCES `experiment_status`(`experiment_id`, `student_id`)
    ON DELETE CASCADE,

  -- Foreign key to the question
  FOREIGN KEY (`question_id`) REFERENCES `question_bank`(`question_id`) ON DELETE CASCADE,

  -- Index to efficiently look up the latest answers for a given quiz
  INDEX `idx_quiz_answers_lookup` (`student_id`, `experiment_id`, `quiz_type`, `answered_at` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `experiment_reports` (
  `report_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `student_id` INT NOT NULL,
  `report_content` TEXT,

  -- Absolute path on Linux that ends with .pdf
  `pdf_file_path` VARCHAR(255),

  `submitted_at` TIMESTAMP NULL,
  -- 0-100
  `grade` FLOAT,
  `feedback` TEXT,
  `graded_by` INT,

  -- 复合外键到 experiment_status（级联删除）
  FOREIGN KEY (`experiment_id`, `student_id`)
    REFERENCES `experiment_status`(`experiment_id`, `student_id`)
    ON DELETE CASCADE,

  -- 打分老师删除时置空
  FOREIGN KEY (`graded_by`) REFERENCES `users`(`user_id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Drop table if exists to ensure clean creation
DROP TABLE IF EXISTS `grade_weights`;
CREATE TABLE `grade_weights` (
  `weight_id` INT AUTO_INCREMENT PRIMARY KEY,
  `teacher_id` INT NOT NULL UNIQUE,

  -- Fields starting with exp_flow_ and must sum to 100
  `exp_flow_demand_data_preparation` FLOAT DEFAULT 0,
  `exp_flow_demand_descriptive_stats` FLOAT DEFAULT 0,
  `exp_flow_demand_model_selection` FLOAT DEFAULT 0,
  `exp_flow_demand_generate_results` FLOAT DEFAULT 0,
  `exp_flow_production_inventory_calc` FLOAT DEFAULT 0,
  `exp_flow_production_service_level` FLOAT DEFAULT 0,
  `exp_flow_production_variable_calc` FLOAT DEFAULT 0,
  `exp_flow_production_plan_creation` FLOAT DEFAULT 0,
  `exp_flow_report_submission` FLOAT DEFAULT 0,

  -- Sum of exp_flow + knowledge_test + model_quality + report_quality must be 100
  `exp_flow` FLOAT DEFAULT 0,
  `knowledge_test` FLOAT DEFAULT 0,
  `model_quality` FLOAT DEFAULT 0,
  `report_quality` FLOAT DEFAULT 0,

  FOREIGN KEY (`teacher_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Drop table if exists to ensure clean creation
DROP TABLE IF EXISTS `experiment_grade`;
CREATE TABLE `experiment_grade` (
  `grade_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `student_id` INT NOT NULL,
  `grader_id` INT,

  -- 0-100
  `exp_flow` FLOAT DEFAULT 0,

  -- All following fields have values from 0-100, temporarily not considering the constraint that exp_flow_ fields' weighted average equals exp_flow value
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

  -- 复合外键（级联删除）
  FOREIGN KEY (`experiment_id`, `student_id`)
    REFERENCES `experiment_status`(`experiment_id`, `student_id`)
    ON DELETE CASCADE,

  -- 打分老师删除时置空
  FOREIGN KEY (`grader_id`) REFERENCES `users`(`user_id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ========================================================
-- 4. System Management
-- ========================================================

CREATE TABLE IF NOT EXISTS `experiment_manuals` (
  `manual_id` INT AUTO_INCREMENT PRIMARY KEY,
  `file_name` VARCHAR(255) NOT NULL,

  -- Absolute Linux path that ends with .pdf
  `file_path` VARCHAR(255) NOT NULL,

  `description` TEXT,
  `is_active` BOOLEAN DEFAULT false,
  `uploader_id` INT,
  `uploaded_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (`uploader_id`) REFERENCES `users`(`user_id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `datasets` (
  `dataset_id` INT AUTO_INCREMENT PRIMARY KEY,
  `data_name` VARCHAR(255) NOT NULL,

  -- Absolute Linux path that ends with .csv
  `file_path` VARCHAR(255) NOT NULL,

  `description` TEXT,
  `uploaded_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `dataset_segments` (
  `segment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `dataset_id` INT NOT NULL,
  `industry_name` VARCHAR(255) NOT NULL,
  `company_name` VARCHAR(255) NOT NULL,
  `product_name` VARCHAR(255) NOT NULL,

  -- Absolute Linux path that ends with .csv
  `file_path` VARCHAR(512) NOT NULL,

  FOREIGN KEY (`dataset_id`) REFERENCES `datasets`(`dataset_id`) ON DELETE CASCADE,
  INDEX `idx_dataset_segments_dataset_id` (`dataset_id`),

  -- 原版唯一约束：不包含 dataset_id
  UNIQUE KEY `uniq_dataset_segments_product` (`industry_name`, `company_name`, `product_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
