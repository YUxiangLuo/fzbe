-- --------------------------------------------------------
-- Virtual Simulation System Seed Data File
-- Test data for development and testing
-- All passwords: $2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG
-- --------------------------------------------------------

-- Clear existing data (in reverse order of dependencies)
SET FOREIGN_KEY_CHECKS = 0;

TRUNCATE TABLE `experiment_step_events`;
TRUNCATE TABLE `quiz_answers`;
TRUNCATE TABLE `experiment_reports`;
TRUNCATE TABLE `experiment_grade`;
TRUNCATE TABLE `experiment_status`;
TRUNCATE TABLE `question_bank`;
TRUNCATE TABLE `grade_weights`;
TRUNCATE TABLE `dataset_segments`;
TRUNCATE TABLE `datasets`;
TRUNCATE TABLE `experiment_manuals`;
TRUNCATE TABLE `class_assistants`;
TRUNCATE TABLE `enrollments`;
TRUNCATE TABLE `classes`;
TRUNCATE TABLE `users`;

SET FOREIGN_KEY_CHECKS = 1;

-- ========================================================
-- 1. Users
-- ========================================================

INSERT INTO `users` (`user_id`, `username`, `password_hash`, `full_name`, `email`, `phone_number`, `role`, `created_at`) VALUES
(1, 'admin', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'System Administrator', 'admin@university.edu', '010-1234-5678', 'Admin', '2024-01-01 09:00:00'),
(2, 'prof_wang', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Wang Li', 'wang.li@university.edu', '010-2345-6789', 'Teacher', '2024-01-05 10:00:00'),
(3, 'prof_zhang', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Zhang Wei', 'zhang.wei@university.edu', '010-3456-7890', 'Teacher', '2024-01-05 10:30:00'),
(4, 'ta_liu', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Liu Fang', 'liu.fang@university.edu', '010-4567-8901', 'Assistant', '2024-01-10 11:00:00'),
(5, 'ta_chen', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Chen Ming', 'chen.ming@university.edu', '010-5678-9012', 'Assistant', '2024-01-10 11:30:00'),
(6, 'student_001', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Li Xiaoming', 'li.xiaoming@student.edu', '138-0001-0001', 'Student', '2024-01-15 08:00:00'),
(7, 'student_002', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Wang Xiaohong', 'wang.xiaohong@student.edu', '138-0002-0002', 'Student', '2024-01-15 08:15:00'),
(8, 'student_003', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Zhang San', 'zhang.san@student.edu', '138-0003-0003', 'Student', '2024-01-15 08:30:00'),
(9, 'student_004', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Li Si', 'li.si@student.edu', '138-0004-0004', 'Student', '2024-01-15 08:45:00'),
(10, 'student_005', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Zhao Wu', 'zhao.wu@student.edu', '138-0005-0005', 'Student', '2024-01-15 09:00:00');

-- ========================================================
-- 2. Classes
-- ========================================================

INSERT INTO `classes` (`class_id`, `class_name`, `class_code`, `teacher_id`, `created_at`) VALUES
(1, '企业需求预测与生产计划 - 2024春季', 'DFPP-2024-S1', 2, '2024-01-20 10:00:00'),
(2, '供应链管理与需求分析 - 2024春季', 'SCDA-2024-S1', 3, '2024-01-20 10:30:00'),
(3, '时间序列预测方法 - 2024春季', 'TSFM-2024-S1', 2, '2024-01-20 11:00:00');

-- ========================================================
-- 3. Enrollments
-- ========================================================

INSERT INTO `enrollments` (`student_id`, `class_id`) VALUES
-- Class 1: 企业需求预测与生产计划
(6, 1),
(7, 1),
(8, 1),
(9, 1),
-- Class 2: 供应链管理与需求分析
(7, 2),
(8, 2),
(10, 2),
-- Class 3: 时间序列预测方法
(6, 3),
(9, 3),
(10, 3);

-- ========================================================
-- 4. Class Assistants
-- ========================================================

INSERT INTO `class_assistants` (`assistant_id`, `class_id`) VALUES
(4, 1),  -- Liu Fang assists 企业需求预测与生产计划
(5, 2),  -- Chen Ming assists 供应链管理与需求分析
(4, 3);  -- Liu Fang assists 时间序列预测方法

-- ========================================================
-- 5. Datasets
-- ========================================================

INSERT INTO `datasets` (`dataset_id`, `data_name`, `file_path`, `description`, `uploaded_at`) VALUES
(1, 'Electronics Industry Sales Data 2020-2024', '/var/data/electronics_sales_2020_2024.csv', 'Comprehensive sales data for electronics products', '2024-01-25 09:00:00'),
(2, 'Consumer Goods Demand Data 2019-2024', '/var/data/consumer_goods_2019_2024.csv', 'Monthly demand data for various consumer products', '2024-01-25 09:30:00');

-- ========================================================
-- 6. Dataset Segments
-- ========================================================

INSERT INTO `dataset_segments` (`dataset_id`, `industry_name`, `company_name`, `product_name`, `file_path`) VALUES
-- Electronics Industry
(1, 'Electronics', 'TechCorp', 'Smartphone X1', '/var/data/segments/electronics_techcorp_smartphone_x1.csv'),
(1, 'Electronics', 'TechCorp', 'Laptop Pro 15', '/var/data/segments/electronics_techcorp_laptop_pro15.csv'),
(1, 'Electronics', 'GadgetInc', 'Tablet S2', '/var/data/segments/electronics_gadgetinc_tablet_s2.csv'),
(1, 'Electronics', 'GadgetInc', 'Smartwatch Elite', '/var/data/segments/electronics_gadgetinc_smartwatch_elite.csv'),
-- Consumer Goods
(2, 'Consumer Goods', 'HomeEssentials', 'Vacuum Cleaner V100', '/var/data/segments/consumer_homeessentials_vacuum_v100.csv'),
(2, 'Consumer Goods', 'HomeEssentials', 'Air Purifier AP200', '/var/data/segments/consumer_homeessentials_airpurifier_ap200.csv'),
(2, 'Consumer Goods', 'KitchenMaster', 'Coffee Maker CM500', '/var/data/segments/consumer_kitchenmaster_coffeemaker_cm500.csv'),
(2, 'Consumer Goods', 'KitchenMaster', 'Blender BL300', '/var/data/segments/consumer_kitchenmaster_blender_bl300.csv');

-- ========================================================
-- 7. Experiment Status
-- ========================================================

-- Student 6: Completed experiment
INSERT INTO `experiment_status` (`experiment_id`, `student_id`, `status`, `highest_completed_step`, `current_step`, 
  `selected_industry`, `selected_company`, `selected_product`,
  `data_window_train_start_index`, `data_window_train_end_index`, 
  `data_window_evaluate_start_index`, `data_window_evaluate_end_index`,
  `moving_average_completed`, `moving_average_window`, `moving_average_metrics_rmse`, `moving_average_metrics_mae`, `moving_average_metrics_r2`,
  `exponential_smoothing_completed`, `exponential_smoothing_alpha`, `exponential_smoothing_metrics_rmse`, `exponential_smoothing_metrics_mae`, `exponential_smoothing_metrics_r2`,
  `arima_completed`, `arima_p`, `arima_d`, `arima_q`, `arima_metrics_rmse`, `arima_metrics_mae`, `arima_metrics_r2`,
  `arima_adf_stationarity`,
  `lstm_completed`, `lstm_normalization`, `lstm_features`, `lstm_target_field`, `lstm_metrics_rmse`, `lstm_metrics_mae`, `lstm_metrics_r2`,
  `ensemble_weighted_completed`, `ensemble_weighted_base_models`, `ensemble_weighted_metrics_rmse`, `ensemble_weighted_metrics_mae`, `ensemble_weighted_metrics_r2`,
  `selected_best_model`, `quiz_about_model_completed`, `quiz_about_plan_completed`,
  `start_time`, `last_activity_at`, `completion_time`) 
VALUES 
(1, 6, 'Completed', 7, 7, 
  'Electronics', 'TechCorp', 'Smartphone X1',
  0, 36, 37, 48,
  true, 3, 1250.5, 980.3, 0.92,
  true, 0.3, 1180.2, 920.1, 0.94,
  true, 2, 1, 2, 1050.8, 850.4, 0.96,
  '[{"diff_order":0,"statistic":-1.876,"p_value":0.640,"stationary":false,"critical_values":{"1%":-3.500,"5%":-2.890,"10%":-2.580}},{"diff_order":1,"statistic":-3.954,"p_value":0.002,"stationary":true,"critical_values":{"1%":-3.500,"5%":-2.890,"10%":-2.580}}]',
  true, 'minmax', '["Month", "Sales_Amount", "Is_Promotion", "Holiday_Flag"]', 'Sales_Amount', 980.5, 780.2, 0.97,
  true, '["arima", "lstm"]', 920.3, 720.5, 0.98,
  'ensemble_weighted', true, true,
  '2024-02-01 09:00:00', '2024-02-15 16:30:00', '2024-02-15 16:30:00');

-- Student 7: In progress - Step 4
INSERT INTO `experiment_status` (`experiment_id`, `student_id`, `status`, `highest_completed_step`, `current_step`,
  `selected_industry`, `selected_company`, `selected_product`,
  `data_window_train_start_index`, `data_window_train_end_index`,
  `data_window_evaluate_start_index`, `data_window_evaluate_end_index`,
  `moving_average_completed`, `moving_average_window`, `moving_average_metrics_rmse`, `moving_average_metrics_mae`, `moving_average_metrics_r2`,
  `exponential_smoothing_completed`, `exponential_smoothing_alpha`, `exponential_smoothing_metrics_rmse`, `exponential_smoothing_metrics_mae`, `exponential_smoothing_metrics_r2`,
  `arima_completed`, `arima_p`, `arima_d`, `arima_q`, `arima_metrics_rmse`, `arima_metrics_mae`, `arima_metrics_r2`,
  `start_time`, `last_activity_at`)
VALUES
(2, 7, 'In Progress', 3, 4,
  'Electronics', 'GadgetInc', 'Tablet S2',
  0, 40, 41, 48,
  true, 2, 1450.3, 1120.5, 0.89,
  true, 0.25, 1380.7, 1080.2, 0.91,
  true, 1, 1, 1, 1280.4, 1000.3, 0.93,
  '2024-02-10 10:00:00', '2024-02-20 14:30:00');

-- Student 8: In progress - Step 2
INSERT INTO `experiment_status` (`experiment_id`, `student_id`, `status`, `highest_completed_step`, `current_step`,
  `selected_industry`, `selected_company`, `selected_product`,
  `data_window_train_start_index`, `data_window_train_end_index`,
  `data_window_evaluate_start_index`, `data_window_evaluate_end_index`,
  `moving_average_completed`, `moving_average_window`, `moving_average_metrics_rmse`, `moving_average_metrics_mae`, `moving_average_metrics_r2`,
  `start_time`, `last_activity_at`)
VALUES
(3, 8, 'In Progress', 1, 2,
  'Consumer Goods', 'HomeEssentials', 'Vacuum Cleaner V100',
  0, 38, 39, 48,
  true, 4, 890.2, 720.5, 0.88,
  '2024-02-12 11:00:00', '2024-02-22 10:15:00');

-- Student 9: Just started
INSERT INTO `experiment_status` (`experiment_id`, `student_id`, `status`, `highest_completed_step`, `current_step`,
  `selected_industry`, `selected_company`, `selected_product`,
  `start_time`, `last_activity_at`)
VALUES
(4, 9, 'In Progress', 0, 1,
  'Consumer Goods', 'KitchenMaster', 'Coffee Maker CM500',
  '2024-02-25 09:00:00', '2024-02-25 09:30:00');

-- Student 10: Not started
INSERT INTO `experiment_status` (`experiment_id`, `student_id`, `status`, `highest_completed_step`, `current_step`)
VALUES
(5, 10, 'Not Started', 0, 1);

-- ========================================================
-- 8. Experiment Step Events
-- ========================================================

-- Events for Student 6 (Completed all steps)
INSERT INTO `experiment_step_events` (`experiment_id`, `student_id`, `step_order`, `event_type`, `event_timestamp`) VALUES
(1, 6, 1, 'STARTED', '2024-02-01 09:00:00'),
(1, 6, 1, 'COMPLETED', '2024-02-01 10:30:00'),
(1, 6, 2, 'STARTED', '2024-02-02 09:00:00'),
(1, 6, 2, 'COMPLETED', '2024-02-02 11:00:00'),
(1, 6, 3, 'STARTED', '2024-02-05 09:00:00'),
(1, 6, 3, 'COMPLETED', '2024-02-06 16:00:00'),
(1, 6, 4, 'STARTED', '2024-02-08 09:00:00'),
(1, 6, 4, 'COMPLETED', '2024-02-09 15:00:00'),
(1, 6, 5, 'STARTED', '2024-02-12 09:00:00'),
(1, 6, 5, 'COMPLETED', '2024-02-12 12:00:00'),
(1, 6, 6, 'STARTED', '2024-02-13 09:00:00'),
(1, 6, 6, 'COMPLETED', '2024-02-13 11:00:00'),
(1, 6, 7, 'STARTED', '2024-02-15 09:00:00'),
(1, 6, 7, 'COMPLETED', '2024-02-15 16:30:00');

-- Events for Student 7 (In progress - Step 4)
INSERT INTO `experiment_step_events` (`experiment_id`, `student_id`, `step_order`, `event_type`, `event_timestamp`) VALUES
(2, 7, 1, 'STARTED', '2024-02-10 10:00:00'),
(2, 7, 1, 'COMPLETED', '2024-02-10 11:30:00'),
(2, 7, 2, 'STARTED', '2024-02-12 10:00:00'),
(2, 7, 2, 'COMPLETED', '2024-02-12 12:00:00'),
(2, 7, 3, 'STARTED', '2024-02-15 10:00:00'),
(2, 7, 3, 'COMPLETED', '2024-02-16 15:00:00'),
(2, 7, 4, 'STARTED', '2024-02-20 10:00:00');

-- Events for Student 8 (In progress - Step 2)
INSERT INTO `experiment_step_events` (`experiment_id`, `student_id`, `step_order`, `event_type`, `event_timestamp`) VALUES
(3, 8, 1, 'STARTED', '2024-02-12 11:00:00'),
(3, 8, 1, 'COMPLETED', '2024-02-12 13:00:00'),
(3, 8, 2, 'STARTED', '2024-02-22 10:00:00');

-- Events for Student 9 (Just started)
INSERT INTO `experiment_step_events` (`experiment_id`, `student_id`, `step_order`, `event_type`, `event_timestamp`) VALUES
(4, 9, 1, 'STARTED', '2024-02-25 09:00:00');

-- ========================================================
-- 9. Question Bank
-- ========================================================

INSERT INTO `question_bank` (`question_id`, `knowledge_point`, `question_type`, `question_text`, `options`, `correct_answers`, `creator_id`) VALUES
-- ARIMA Basics
(1, 'ARIMA模型', 'Single Choice', 'In ARIMA(p,d,q) model, what does the parameter "p" represent?', 
  '{"A":"Autoregressive order","B":"Differencing order","C":"Moving average order","D":"Seasonal period"}', 
  '["A"]', 2),
(2, 'ARIMA模型', 'Single Choice', 'What is the purpose of the differencing parameter "d" in ARIMA?', 
  '{"A":"To remove trend","B":"To make series stationary","C":"To remove seasonality","D":"Both A and B"}', 
  '["D"]', 2),
(3, 'ARIMA模型', 'True/False', 'The ADF test is used to check if a time series is stationary.', 
  '["正确","错误"]', 
  '["正确"]', 2),

-- LSTM Basics
(4, 'LSTM模型', 'Single Choice', 'What type of neural network is LSTM?', 
  '{"A":"Feedforward Neural Network","B":"Convolutional Neural Network","C":"Recurrent Neural Network","D":"Generative Adversarial Network"}', 
  '["C"]', 2),
(5, 'LSTM模型', 'Multiple Choice', 'Which of the following are components of an LSTM cell?', 
  '{"A":"Forget gate","B":"Input gate","C":"Output gate","D":"Attention mechanism"}', 
  '["A","B","C"]', 2),
(6, 'LSTM模型', 'Single Choice', 'What is the main advantage of LSTM over traditional RNN?', 
  '{"A":"Faster training","B":"Solves vanishing gradient problem","C":"Requires less data","D":"Simpler architecture"}', 
  '["B"]', 3),

-- Ensemble Principles
(7, '融合模型', 'Multiple Choice', 'Which of the following are ensemble learning methods?', 
  '{"A":"Bagging","B":"Boosting","C":"Stacking","D":"Dropout"}', 
  '["A","B","C"]', 2),
(8, '融合模型', 'Single Choice', 'In weighted ensemble, how are individual model predictions combined?', 
  '{"A":"Simple average","B":"Weighted average","C":"Maximum value","D":"Minimum value"}', 
  '["B"]', 2),
(9, '融合模型', 'True/False', 'Ensemble methods always perform better than individual models.', 
  '["正确","错误"]', 
  '["错误"]', 3),

-- Production Calculation
(10, '生产计划', 'Single Choice', 'If forecasted demand is 1000 units and safety stock is 200 units, what is the total production requirement?', 
  '{"A":"800 units","B":"1000 units","C":"1200 units","D":"Depends on current inventory"}', 
  '["D"]', 2),
(11, '生产计划', 'Single Choice', 'Safety stock is primarily used to:', 
  '{"A":"Reduce costs","B":"Buffer against demand uncertainty","C":"Increase production efficiency","D":"Minimize waste"}', 
  '["B"]', 3),

-- Service Level Calculation
(12, '生产计划', 'Single Choice', 'A 95% service level means:', 
  '{"A":"95% of products are high quality","B":"95% of customer demands are met from stock","C":"95% production efficiency","D":"95% cost reduction"}', 
  '["B"]', 2),
(13, '生产计划', 'True/False', 'Higher service levels always require higher safety stock.', 
  '["正确","错误"]', 
  '["正确"]', 2);

-- ========================================================
-- 10. Quiz Answers (for completed student)
-- ========================================================

-- Student 6's quiz answers
INSERT INTO `quiz_answers` (`experiment_id`, `student_id`, `question_id`, `quiz_type`, `submitted_answer`, `is_correct`, `answered_at`) VALUES
(1, 6, 1, 'quiz_about_model', '["A"]', true, '2024-02-13 10:00:00'),
(1, 6, 2, 'quiz_about_model', '["D"]', true, '2024-02-13 10:01:00'),
(1, 6, 3, 'quiz_about_model', '["True"]', true, '2024-02-13 10:02:00'),
(1, 6, 4, 'quiz_about_model', '["C"]', true, '2024-02-13 10:03:00'),
(1, 6, 5, 'quiz_about_model', '["A","B","C"]', true, '2024-02-13 10:04:00'),
(1, 6, 10, 'quiz_about_plan', '["D"]', true, '2024-02-15 15:00:00'),
(1, 6, 11, 'quiz_about_plan', '["B"]', true, '2024-02-15 15:01:00'),
(1, 6, 12, 'quiz_about_plan', '["B"]', true, '2024-02-15 15:02:00'),
(1, 6, 13, 'quiz_about_plan', '["True"]', true, '2024-02-15 15:03:00');

-- Student 7's partial quiz answers
INSERT INTO `quiz_answers` (`experiment_id`, `student_id`, `question_id`, `quiz_type`, `submitted_answer`, `is_correct`, `answered_at`) VALUES
(2, 7, 1, 'quiz_about_model', '["A"]', true, '2024-02-20 11:00:00'),
(2, 7, 2, 'quiz_about_model', '["B"]', false, '2024-02-20 11:01:00'),
(2, 7, 3, 'quiz_about_model', '["True"]', true, '2024-02-20 11:02:00');

-- ========================================================
-- 11. Experiment Reports
-- ========================================================

INSERT INTO `experiment_reports` (`report_id`, `experiment_id`, `student_id`, `report_content`, `pdf_file_path`, `submitted_at`, `grade`, `feedback`, `graded_by`) VALUES
(1, 1, 6, 'Complete experimental report for demand forecasting and production planning using ensemble methods...', 
  '/var/reports/student_6_experiment_1_report.pdf', 
  '2024-02-15 17:00:00', 92.0, 
  'Excellent work! Your analysis is thorough and the model selection is well-justified. Consider exploring more feature engineering techniques in future experiments.', 
  2);

-- ========================================================
-- 12. Grade Weights (for teachers)
-- ========================================================

INSERT INTO `grade_weights` (`class_id`, 
  `exp_flow_demand_data_preparation`, `exp_flow_demand_descriptive_stats`, `exp_flow_demand_model_selection`, `exp_flow_demand_generate_results`,
  `exp_flow_production_inventory_calc`, `exp_flow_production_service_level`, `exp_flow_production_variable_calc`, `exp_flow_production_plan_creation`,
  `exp_flow_report_submission`,
  `exp_flow_weight`, `knowledge_test_weight`, `model_quality_weight`, `report_quality_weight`) 
VALUES
(1, 5, 5, 10, 10, 10, 5, 5, 15, 35, 40, 20, 20, 20),
(2, 8, 7, 10, 10, 8, 7, 10, 15, 25, 45, 15, 20, 20);

-- ========================================================
-- 13. Experiment Grades
-- ========================================================

INSERT INTO `experiment_grade` (`experiment_id`, `student_id`) VALUES (1, 6);

-- ========================================================
-- 14. Experiment Manuals
-- ========================================================

INSERT INTO `experiment_manuals` (`manual_id`, `file_name`, `file_path`, `description`, `is_active`, `uploader_id`, `uploaded_at`) VALUES
(1, 'Experiment_Manual_v2.3.pdf', '/var/manuals/experiment_manual_v2_3.pdf', 
  'Complete guide for demand forecasting and production planning experiment - Version 2.3', 
  true, 1, '2024-01-20 09:00:00'),
(2, 'Quick_Start_Guide.pdf', '/var/manuals/quick_start_guide.pdf', 
  'Quick start guide for new students', 
  false, 2, '2024-01-22 10:00:00'),
(3, 'Model_Selection_Guidelines.pdf', '/var/manuals/model_selection_guidelines.pdf', 
  'Detailed guidelines for selecting appropriate forecasting models', 
  false, 2, '2024-01-25 11:00:00');

-- ========================================================
-- Summary Statistics Query
-- ========================================================

-- Uncomment to see summary after importing
/*
SELECT 'Users' as Table_Name, COUNT(*) as Count FROM users
UNION ALL SELECT 'Classes', COUNT(*) FROM classes
UNION ALL SELECT 'Enrollments', COUNT(*) FROM enrollments
UNION ALL SELECT 'Class Assistants', COUNT(*) FROM class_assistants
UNION ALL SELECT 'Datasets', COUNT(*) FROM datasets
UNION ALL SELECT 'Dataset Segments', COUNT(*) FROM dataset_segments
UNION ALL SELECT 'Experiment Status', COUNT(*) FROM experiment_status
UNION ALL SELECT 'Experiment Step Events', COUNT(*) FROM experiment_step_events
UNION ALL SELECT 'Question Bank', COUNT(*) FROM question_bank
UNION ALL SELECT 'Quiz Answers', COUNT(*) FROM quiz_answers
UNION ALL SELECT 'Experiment Reports', COUNT(*) FROM experiment_reports
UNION ALL SELECT 'Grade Weights', COUNT(*) FROM grade_weights
UNION ALL SELECT 'Experiment Grades', COUNT(*) FROM experiment_grade
UNION ALL SELECT 'Experiment Manuals', COUNT(*) FROM experiment_manuals;
*/
