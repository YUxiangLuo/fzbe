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

-- 新表: 用于将助教直接分配给班级
CREATE TABLE IF NOT EXISTS `class_assistants` (
  `assignment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `assistant_id` INT NOT NULL,
  `class_id` INT NOT NULL,
  FOREIGN KEY (`assistant_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,
  FOREIGN KEY (`class_id`) REFERENCES `classes`(`class_id`) ON DELETE CASCADE,
  UNIQUE KEY `assistant_class_unique` (`assistant_id`, `class_id`)
);

/*
-- 旧的助教分配表 (已弃用)
CREATE TABLE IF NOT EXISTS `assistant_assignments` (
  `assignment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `assistant_id` INT NOT NULL,
  `teacher_id` INT NOT NULL,
  FOREIGN KEY (`assistant_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE,
  FOREIGN KEY (`teacher_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE
);
*/

--
-- 2. 实验与生产计划管理
--

CREATE TABLE IF NOT EXISTS `experiment_status` (
  `experiment_id` INT AUTO_INCREMENT PRIMARY KEY,
  `student_id` INT NOT NULL,
  `start_time` TIMESTAMP NULL,
  `completion_time` TIMESTAMP NULL,
  `status` ENUM('Not Started', 'In Progress', 'Completed') DEFAULT 'Not Started',
  `selected_industry` VARCHAR(255),
  `selected_company` VARCHAR(255),
  `selected_product` VARCHAR(255),
  `best_model` VARCHAR(255),
  FOREIGN KEY (`student_id`) REFERENCES `users`(`user_id`) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS `model_results` (
  `result_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `model_name` VARCHAR(255) NOT NULL,
  `model_parameters` JSON,
  `rmse` FLOAT,
  `mae` FLOAT,
  `r2` FLOAT,
  `mape` FLOAT,
  `prediction_output` JSON,
  FOREIGN KEY (`experiment_id`) REFERENCES `experiment_status`(`experiment_id`) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS `production_plans` (
  `plan_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `plan_details` JSON,
  FOREIGN KEY (`experiment_id`) REFERENCES `experiment_status`(`experiment_id`) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS `experiment_step_events` (
  `event_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `student_id` INT NOT NULL,
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
  `options` JSON,
  `correct_answer` VARCHAR(255) NOT NULL,
  `creator_id` INT,
  FOREIGN KEY (`creator_id`) REFERENCES `users`(`user_id`) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS `student_responses` (
  `response_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `question_id` INT NOT NULL,
  `answer` TEXT,
  `is_correct` BOOLEAN,
  FOREIGN KEY (`experiment_id`) REFERENCES `experiment_status`(`experiment_id`) ON DELETE CASCADE,
  FOREIGN KEY (`question_id`) REFERENCES `question_bank`(`question_id`) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS `experiment_reports` (
  `report_id` INT AUTO_INCREMENT PRIMARY KEY,
  `experiment_id` INT NOT NULL,
  `report_content` TEXT,
  `pdf_file_path` VARCHAR(255),
  `submitted_at` TIMESTAMP NULL,
  `grade` FLOAT,
  `feedback` TEXT,
  `graded_by` INT,
  FOREIGN KEY (`experiment_id`) REFERENCES `experiment_status`(`experiment_id`) ON DELETE CASCADE,
  FOREIGN KEY (`graded_by`) REFERENCES `users`(`user_id`) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS `grade_weights` (
  `weight_id` INT AUTO_INCREMENT PRIMARY KEY,
  `class_id` INT NOT NULL UNIQUE,
  `experiment_flow` FLOAT DEFAULT 0,
  `knowledge_tests` FLOAT DEFAULT 0,
  `model_selection` FLOAT DEFAULT 0,
  `experiment_report` FLOAT DEFAULT 0,
  `detailed_step_weights` JSON,
  FOREIGN KEY (`class_id`) REFERENCES `classes`(`class_id`) ON DELETE CASCADE
);

--
-- 4. 系统管理
--

CREATE TABLE IF NOT EXISTS `experiment_manuals` (
  `manual_id` INT AUTO_INCREMENT PRIMARY KEY,
  `file_name` VARCHAR(255) NOT NULL,
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
  `file_path` VARCHAR(255) NOT NULL,
  `description` TEXT,
  `uploaded_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
