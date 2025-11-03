-- --------------------------------------------------------
-- Virtual Simulation System Database Schema File
-- Compatible with MariaDB / MySQL
-- Fixed Version with Improvements (No DELIMITER)
--
-- Performance Optimizations Included (2025-11-03):
-- - Added composite indexes for faster queries
-- - Added FULLTEXT index for user search
-- - Optimized experiment status and grade queries
-- --------------------------------------------------------

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
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  -- Performance optimization: fulltext index for user search
  FULLTEXT INDEX `ft_user_search` (`full_name`, `email`, `phone_number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `classes` (
  `class_id` INT AUTO_INCREMENT PRIMARY KEY,
  `class_name` VARCHAR(255) NOT NULL,
  `class_code` VARCHAR(255) UNIQUE,
  `teacher_id` INT NULL,
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (`teacher_id`) REFERENCES `users`(`user_id`) ON DELETE SET NULL,
  INDEX `idx_teacher_id` (`teacher_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `enrollments` (
  `enrollment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `student_id` INT NOT NULL,
  `class_id` INT NOT NULL,
  FOREIGN KEY (`student_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,
  FOREIGN KEY (`class_id`) REFERENCES `classes`(`class_id`) ON DELETE CASCADE,
  UNIQUE KEY `student_class_unique` (`student_id`, `class_id`),
  INDEX `idx_class_student` (`class_id`, `student_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `class_assistants` (
  `assignment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `assistant_id` INT NOT NULL,
  `class_id` INT NOT NULL,
  FOREIGN KEY (`assistant_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,
  FOREIGN KEY (`class_id`) REFERENCES `classes`(`class_id`) ON DELETE CASCADE,
  UNIQUE KEY `assistant_class_unique` (`assistant_id`, `class_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ========================================================
-- 2. Experiment and Production Planning Management
-- ========================================================

CREATE TABLE IF NOT EXISTS `experiment_status` (
  `experiment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `student_id` INT NOT NULL,
  `status` ENUM('Not Started', 'In Progress', 'Completed') DEFAULT 'Not Started',

  -- Default value is 0, indicating no steps have been completed yet
  `highest_completed_step` INT UNSIGNED NOT NULL DEFAULT 0,

  -- Values range from 1-7, representing a total of seven steps
  `current_step` INT UNSIGNED NOT NULL DEFAULT 1,
  `total_active_duration_seconds` INT UNSIGNED NOT NULL DEFAULT 0,
  `selected_industry` VARCHAR(255) DEFAULT NULL,
  `selected_company` VARCHAR(255) DEFAULT NULL,
  `selected_product` VARCHAR(255) DEFAULT NULL,

  `data_window_train_start_index` INT UNSIGNED DEFAULT NULL,
  `data_window_train_end_index` INT UNSIGNED DEFAULT NULL,
  `data_window_evaluate_start_index` INT UNSIGNED DEFAULT NULL,
  `data_window_evaluate_end_index` INT UNSIGNED DEFAULT NULL,

  `moving_average_completed` BOOLEAN DEFAULT false,
  `moving_average_window` INT DEFAULT 2,
  `moving_average_metrics_rmse` FLOAT DEFAULT NULL,
  `moving_average_metrics_mae` FLOAT DEFAULT NULL,
  `moving_average_metrics_r2` FLOAT DEFAULT NULL,

  `exponential_smoothing_completed` BOOLEAN DEFAULT false,
  `exponential_smoothing_alpha` FLOAT DEFAULT NULL,
  `exponential_smoothing_metrics_rmse` FLOAT DEFAULT NULL,
  `exponential_smoothing_metrics_mae` FLOAT DEFAULT NULL,
  `exponential_smoothing_metrics_r2` FLOAT DEFAULT NULL,

  `arima_completed` BOOLEAN DEFAULT false,
  -- Tip for frontend: d is user-provided; p and q are auto-selected during training
  `arima_p` TINYINT DEFAULT NULL,
  `arima_d` TINYINT DEFAULT NULL,
  `arima_q` TINYINT DEFAULT NULL,
  `arima_metrics_rmse` FLOAT DEFAULT NULL,
  `arima_metrics_mae` FLOAT DEFAULT NULL,
  `arima_metrics_r2` FLOAT DEFAULT NULL,

  -- ADF stationarity diagnostics JSON array
  `arima_adf_stationarity` JSON DEFAULT NULL,

  `lstm_completed` BOOLEAN DEFAULT false,
  `lstm_normalization` ENUM('minmax', 'zscore') DEFAULT NULL,
  -- CSV file column names array: ['Month', 'Sales Amount', 'Is Promotion']
  `lstm_features` JSON DEFAULT NULL,
  `lstm_target_field` VARCHAR(255) DEFAULT NULL,
  `lstm_metrics_rmse` FLOAT DEFAULT NULL,
  `lstm_metrics_mae` FLOAT DEFAULT NULL,
  `lstm_metrics_r2` FLOAT DEFAULT NULL,

  `ensemble_weighted_completed` BOOLEAN DEFAULT false,
  -- Array of base model abbreviations: ['ma', 'exp', 'arima', 'lstm']
  `ensemble_weighted_base_models` JSON DEFAULT NULL,
  `ensemble_weighted_metrics_rmse` FLOAT DEFAULT NULL,
  `ensemble_weighted_metrics_mae` FLOAT DEFAULT NULL,
  `ensemble_weighted_metrics_r2` FLOAT DEFAULT NULL,

  `ensemble_boosting_completed` BOOLEAN DEFAULT false,
  -- Array of base model abbreviations: ['ma', 'exp', 'arima', 'lstm']
  `ensemble_boosting_base_models` JSON DEFAULT NULL,
  `ensemble_boosting_metrics_rmse` FLOAT DEFAULT NULL,
  `ensemble_boosting_metrics_mae` FLOAT DEFAULT NULL,
  `ensemble_boosting_metrics_r2` FLOAT DEFAULT NULL,

  `ensemble_stacking_completed` BOOLEAN DEFAULT false,
  -- Array of base model abbreviations: ['ma', 'exp', 'arima', 'lstm']
  `ensemble_stacking_base_models` JSON DEFAULT NULL,
  `ensemble_stacking_metrics_rmse` FLOAT DEFAULT NULL,
  `ensemble_stacking_metrics_mae` FLOAT DEFAULT NULL,
  `ensemble_stacking_metrics_r2` FLOAT DEFAULT NULL,

  `selected_best_model` ENUM('ma', 'exp', 'arima', 'lstm', 'ensemble_weighted', 'ensemble_boosting', 'ensemble_stacking') DEFAULT NULL,

  `quiz_about_model_completed` BOOLEAN DEFAULT false,
  `quiz_about_plan_completed` BOOLEAN DEFAULT false,

  `production_plan_completed` BOOLEAN DEFAULT false,
  `production_forecast_periods` INT DEFAULT NULL,
  `production_initial_inventory` FLOAT DEFAULT NULL,
  `production_target_service_level` FLOAT DEFAULT NULL,
  `production_safety_stock_z_score` FLOAT DEFAULT NULL,
  `production_mps_table` JSON DEFAULT NULL,
  `production_forecast_results` JSON DEFAULT NULL,
  `production_capacity_mode` ENUM('scenario', 'auto', 'custom') DEFAULT NUll,
  `production_capacity_scenario` ENUM('tight', 'normal', 'abundant') DEFAULT NUll,
  `production_capacity` INT DEFAULT NULL,
  `production_custom_capacity` INT DEFAULT NULL,

  `start_time` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  `last_activity_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  `completion_time` TIMESTAMP DEFAULT NULL,

  FOREIGN KEY (`student_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,
  INDEX `idx_experiment_status_student` (`student_id`),
  
  -- Add constraint to ensure step values are valid
  CONSTRAINT `chk_current_step` CHECK (`current_step` BETWEEN 1 AND 7),
  CONSTRAINT `chk_highest_step` CHECK (`highest_completed_step` <= 7),
  INDEX `idx_student_status_completion` (`student_id`, `status`, `completion_time` DESC),
  -- Performance optimization: composite index for fetching latest non-completed experiments
  INDEX `idx_student_status_id` (`student_id`, `status`, `experiment_id` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `experiment_step_events` (
  `event_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `student_id` INT NOT NULL,

  -- Values range from 1-7
  `step_order` INT NOT NULL,
  `event_type` ENUM('STARTED', 'COMPLETED') NOT NULL,
  `event_timestamp` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

  -- Fixed: Use simple foreign key instead of composite
  FOREIGN KEY (`experiment_id`)
    REFERENCES `experiment_status`(`experiment_id`)
    ON DELETE CASCADE,
  
  -- Add separate foreign key for student_id to maintain referential integrity
  FOREIGN KEY (`student_id`)
    REFERENCES `users`(`user_id`)
    ON DELETE CASCADE,

  -- Existing index, useful for some lookups
  INDEX `idx_experiment_step_order` (`experiment_id`, `step_order`, `event_timestamp`),
  
  -- === NEW INDEXES FOR PERFORMANCE ===
  -- Optimizes fetching all events for a student, ordered by time (for class progress view)
  INDEX `idx_student_timestamp` (`student_id`, `event_timestamp`),
  
  -- Optimizes fetching the latest event for a specific experiment (for FOR UPDATE lock)
  INDEX `idx_experiment_timestamp` (`experiment_id`, `event_timestamp`),
  
  -- Add constraint to ensure step_order is valid
  CONSTRAINT `chk_step_order` CHECK (`step_order` BETWEEN 1 AND 7)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ========================================================
-- 3. Assessment and Grading
-- ========================================================

CREATE TABLE IF NOT EXISTS `question_bank` (
  `question_id` INT AUTO_INCREMENT PRIMARY KEY,
  -- Fixed: Use English codes instead of Chinese characters
  `knowledge_point` VARCHAR(255) NOT NULL,
  `question_type` ENUM('Single Choice', 'Multiple Choice', 'True/False') NOT NULL,
  `question_text` TEXT NOT NULL,

  -- {"A":"Autoregressive order","B":"Differencing order","C":"Moving average order","D":"Seasonal period"} / ["True","False"]
  `options` JSON,

  -- ["A","B","D"] / ["True"]
  `correct_answers` JSON NOT NULL,

  `creator_id` INT,
  FOREIGN KEY (`creator_id`) REFERENCES `users`(`user_id`) ON DELETE SET NULL,
  -- Performance optimization: indexes for filtering and creator queries
  INDEX `idx_knowledge_point` (`knowledge_point`),
  INDEX `idx_creator` (`creator_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `quiz_answers` (
  `answer_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `student_id` INT NOT NULL,
  `question_id` INT NOT NULL,
  `quiz_type` ENUM('quiz_about_model', 'quiz_about_plan') NOT NULL,
  -- Submitted answers, stored in the same JSON array format as the question bank's correct_answers
  `submitted_answer` JSON NOT NULL,
  -- Fixed: Add default value
  `is_correct` BOOLEAN DEFAULT false,
  `answered_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

  -- Fixed: Use simple foreign key
  FOREIGN KEY (`experiment_id`)
    REFERENCES `experiment_status`(`experiment_id`)
    ON DELETE CASCADE,
  
  FOREIGN KEY (`student_id`)
    REFERENCES `users`(`user_id`)
    ON DELETE CASCADE,

  -- Foreign key to the question
  FOREIGN KEY (`question_id`) 
    REFERENCES `question_bank`(`question_id`) 
    ON DELETE CASCADE,

  -- Index to efficiently look up the latest answers for a given quiz
  INDEX `idx_quiz_answers_lookup` (`student_id`, `experiment_id`, `quiz_type`, `answered_at` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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

  -- Fixed: Use simple foreign key
  FOREIGN KEY (`experiment_id`)
    REFERENCES `experiment_status`(`experiment_id`)
    ON DELETE CASCADE,
  
  FOREIGN KEY (`student_id`)
    REFERENCES `users`(`user_id`)
    ON DELETE CASCADE,

  FOREIGN KEY (`graded_by`) 
    REFERENCES `users`(`user_id`) 
    ON DELETE SET NULL,
  
  -- Add constraint to ensure grade is in valid range
  CONSTRAINT `chk_report_grade` CHECK (`grade` IS NULL OR (`grade` >= 0 AND `grade` <= 100)),

  -- Ensure that each student can only submit one report per experiment
  UNIQUE KEY `student_experiment_unique` (`student_id`, `experiment_id`),

  -- Performance optimization: index for fetching latest reports
  INDEX `idx_student_submitted` (`student_id`, `submitted_at` DESC, `report_id` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `grade_weights` (
  `weight_id` INT AUTO_INCREMENT PRIMARY KEY,
  `class_id` INT NOT NULL UNIQUE,

  -- Fields starting with exp_flow_ must sum to 100
  `exp_flow_demand_data_preparation` FLOAT DEFAULT 5,
  `exp_flow_demand_descriptive_stats` FLOAT DEFAULT 5,
  `exp_flow_demand_model_selection` FLOAT DEFAULT 10,
  `exp_flow_demand_generate_results` FLOAT DEFAULT 10,
  `exp_flow_production_inventory_calc` FLOAT DEFAULT 10,
  `exp_flow_production_service_level` FLOAT DEFAULT 5,
  `exp_flow_production_variable_calc` FLOAT DEFAULT 5,
  `exp_flow_production_plan_creation` FLOAT DEFAULT 15,
  `exp_flow_report_submission` FLOAT DEFAULT 35,

  -- Sum of exp_flow_weight + knowledge_test_weight + model_quality_weight + report_quality_weight must be 100
  `exp_flow_weight` FLOAT DEFAULT 40,
  `knowledge_test_weight` FLOAT DEFAULT 20,
  `model_quality_weight` FLOAT DEFAULT 20,
  `report_quality_weight` FLOAT DEFAULT 20,

  FOREIGN KEY (`class_id`) REFERENCES `classes`(`class_id`) ON DELETE CASCADE,

  CONSTRAINT `chk_weights_range` CHECK (
    `exp_flow_demand_data_preparation` BETWEEN 0 AND 100 AND
    `exp_flow_demand_descriptive_stats` BETWEEN 0 AND 100 AND
    `exp_flow_demand_model_selection` BETWEEN 0 AND 100 AND
    `exp_flow_demand_generate_results` BETWEEN 0 AND 100 AND
    `exp_flow_production_inventory_calc` BETWEEN 0 AND 100 AND
    `exp_flow_production_service_level` BETWEEN 0 AND 100 AND
    `exp_flow_production_variable_calc` BETWEEN 0 AND 100 AND
    `exp_flow_production_plan_creation` BETWEEN 0 AND 100 AND
    `exp_flow_report_submission` BETWEEN 0 AND 100 AND
    `exp_flow_weight` BETWEEN 0 AND 100 AND
    `knowledge_test_weight` BETWEEN 0 AND 100 AND
    `model_quality_weight` BETWEEN 0 AND 100 AND
    `report_quality_weight` BETWEEN 0 AND 100
  )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `experiment_grade` (
  `grade_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `student_id` INT NOT NULL,
  -- All following fields have values from 0-100
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

  -- Fixed: Use simple foreign key
  FOREIGN KEY (`experiment_id`)
    REFERENCES `experiment_status`(`experiment_id`)
    ON DELETE CASCADE,

  FOREIGN KEY (`student_id`)
    REFERENCES `users`(`user_id`)
    ON DELETE CASCADE,

  INDEX `idx_experiment_grade_experiment_id` (`experiment_id`),
  -- Performance optimization: composite index for grade queries
  INDEX `idx_student_experiment` (`student_id`, `experiment_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `datasets` (
  `dataset_id` INT AUTO_INCREMENT PRIMARY KEY,
  `data_name` VARCHAR(255) NOT NULL,

  -- Absolute Linux path that ends with .csv
  `file_path` VARCHAR(255) NOT NULL,

  `description` TEXT,
  `uploaded_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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

  UNIQUE KEY `uniq_dataset_segments_product` (`industry_name`, `company_name`, `product_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
