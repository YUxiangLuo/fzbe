-- =======================================================
-- Seed data for Virtual Simulation System (MySQL/MariaDB)
-- 严格遵循 schema 注释：路径后缀、取值范围、JSON 结构等
-- =======================================================

SET NAMES utf8mb4;
START TRANSACTION;

SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE dataset_segments;
TRUNCATE TABLE datasets;
TRUNCATE TABLE experiment_manuals;
TRUNCATE TABLE experiment_grade;
TRUNCATE TABLE grade_weights;
TRUNCATE TABLE experiment_reports;
TRUNCATE TABLE question_bank;
TRUNCATE TABLE experiment_step_events;
TRUNCATE TABLE experiment_status;
TRUNCATE TABLE class_assistants;
TRUNCATE TABLE enrollments;
TRUNCATE TABLE classes;
TRUNCATE TABLE users;
SET FOREIGN_KEY_CHECKS = 1;

-- 1) users（密码固定为你原始的那一串）
INSERT INTO users (user_id, username, password_hash, full_name, email, phone_number, role, created_at) VALUES
(1, 'teacher_anna', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Anna Smith', 'anna@example.com', '+1-555-0101', 'Teacher', NOW()),
(2, 'teacher_bob',  '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Bob Lee',    'bob@example.com',  '+1-555-0102', 'Teacher', NOW()),
(3, 'student_chen', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Chen Yang',  'chen@example.com', '+1-555-0201', 'Student', NOW()),
(4, 'student_dai',  '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Dai Wei',    'dai@example.com',  '+1-555-0202', 'Student', NOW()),
(5, 'assistant_eva','$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Eva Park',   'eva@example.com',  '+1-555-0301', 'Assistant', NOW()),
(6, 'admin_fox',    '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Frank Fox',  'admin@example.com','+1-555-0401', 'Admin', NOW());

-- 2) classes（teacher 可为空；此处给 1/2 两位老师）
INSERT INTO classes (class_id, class_name, class_code, teacher_id, created_at) VALUES
(1, 'Time Series 101', 'TS101-2025F', 1, NOW()),
(2, 'Production Planning', 'PP201-2025F', 2, NOW());

-- 3) enrollments
INSERT INTO enrollments (enrollment_id, student_id, class_id) VALUES
(1, 3, 1),
(2, 3, 2),
(3, 4, 1);

-- 4) class_assistants
INSERT INTO class_assistants (assignment_id, assistant_id, class_id) VALUES
(1, 5, 1),
(2, 5, 2);

-- 5) experiment_status
-- 注意：JSON 结构与注释一致；预测步数 1-6；路径/名称示例与注释一致
INSERT INTO experiment_status (
  experiment_id, student_id, start_time, last_activity_at, completion_time, status,
  highest_completed_step, current_step, selected_industry, selected_company, selected_product,
  data_window,
  moving_average_completed, moving_average_window, moving_average_metrics_rmse, moving_average_metrics_mae, moving_average_metrics_r2,
  exponential_smoothing_completed, exponential_smoothing_alpha, exponential_smoothing_metrics_rmse, exponential_smoothing_metrics_mae, exponential_smoothing_metrics_r2,
  arima_completed, arima_p, arima_d, arima_q, arima_metrics_rmse, arima_metrics_mae, arima_metrics_r2, arima_adf_stationarity,
  lstm_completed, lstm_normalization, lstm_features, lstm_metrics_rmse, lstm_metrics_mae, lstm_metrics_r2,
  ensemble_weighted_completed, ensemble_weighted_base_models, ensemble_weighted_metrics_rmse, ensemble_weighted_metrics_mae, ensemble_weighted_metrics_r2,
  ensemble_boosting_completed, ensemble_boosting_base_models, ensemble_boosting_metrics_rmse, ensemble_boosting_metrics_mae, ensemble_boosting_metrics_r2,
  ensemble_stacking_completed, ensemble_stacking_base_models, ensemble_stacking_metrics_rmse, ensemble_stacking_metrics_mae, ensemble_stacking_metrics_r2,
  selected_best_model, predict_steps, predict_values,
  quiz_about_model_completed, quiz_about_plan_completed, experiment_report_submitted
) VALUES
-- Exp #1: 学生 Chen(3)
(1, 3, NOW() - INTERVAL 10 DAY, NOW() - INTERVAL 1 DAY, NULL, 'In Progress',
  3, 4, 'Electronics', 'Acme Corp', 'Phone X',
  JSON_OBJECT('train_start_index',0,'train_end_index',100,'evaluate_start_index',101,'evaluate_end_index',120),
  TRUE, 5, 12.3, 9.7, 0.81,
  TRUE, 0.3, 10.9, 8.5, 0.84,
  TRUE, 2, 1, 2, 9.4, 7.1, 0.87,
  JSON_ARRAY(
    JSON_OBJECT('diff_order',0,'statistic',-1.876,'p_value',0.64,'stationary',FALSE,'critical_values',JSON_OBJECT('1%',-3.5,'5%',-2.89,'10%',-2.58)),
    JSON_OBJECT('diff_order',1,'statistic',-3.954,'p_value',0.002,'stationary',TRUE,'critical_values',JSON_OBJECT('1%',-3.5,'5%',-2.89,'10%',-2.58))
  ),
  TRUE, 'minmax', JSON_ARRAY('Month','Sales','Is Promotion'), 8.8, 6.6, 0.90,
  TRUE, JSON_ARRAY('ma','exp','arima','lstm'), 8.1, 6.2, 0.91,
  TRUE, JSON_ARRAY('ma','exp','arima','lstm'), 7.9, 6.0, 0.92,
  TRUE, JSON_ARRAY('ma','exp','arima','lstm'), 7.7, 5.9, 0.93,
  'ensemble_stacking', 4, JSON_ARRAY(110,120,130,140),
  TRUE, FALSE, FALSE
),
-- Exp #2: 学生 Chen(3)
(2, 3, NOW() - INTERVAL 20 DAY, NOW() - INTERVAL 3 DAY, NOW() - INTERVAL 2 DAY, 'Completed',
  7, 7, 'Textiles', 'Luthai', 'Cotton Shirt',
  JSON_OBJECT('train_start_index',0,'train_end_index',80,'evaluate_start_index',81,'evaluate_end_index',100),
  TRUE, 3, 14.1, 10.5, 0.78,
  TRUE, 0.5, 12.2, 9.2, 0.82,
  TRUE, 1, 1, 1, 10.5, 8.1, 0.85,
  JSON_ARRAY(
    JSON_OBJECT('diff_order',0,'statistic',-2.1,'p_value',0.53,'stationary',FALSE,'critical_values',JSON_OBJECT('1%',-3.5,'5%',-2.89,'10%',-2.58)),
    JSON_OBJECT('diff_order',1,'statistic',-4.12,'p_value',0.001,'stationary',TRUE,'critical_values',JSON_OBJECT('1%',-3.5,'5%',-2.89,'10%',-2.58))
  ),
  TRUE, 'zscore', JSON_ARRAY('Month','Sales Amount'), 9.6, 7.3, 0.88,
  TRUE, JSON_ARRAY('ma','arima'), 9.0, 6.9, 0.89,
  TRUE, JSON_ARRAY('ma','arima'), 8.6, 6.7, 0.90,
  TRUE, JSON_ARRAY('ma','arima'), 8.2, 6.4, 0.91,
  'ensemble_stacking', 6, JSON_ARRAY(105,110,115,120,125,130),
  TRUE, TRUE, TRUE
),
-- Exp #3: 学生 Dai(4)
(3, 4, NOW() - INTERVAL 7 DAY, NOW() - INTERVAL 1 DAY, NULL, 'In Progress',
  2, 3, 'Automotive', 'Vroom', 'Brake Pads',
  JSON_OBJECT('train_start_index',10,'train_end_index',90,'evaluate_start_index',91,'evaluate_end_index',110),
  TRUE, 4, 13.2, 10.1, 0.79,
  NULL, NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL, NULL, NULL, NULL,
  NULL,
  FALSE, 'minmax', JSON_ARRAY('Month','Units'), NULL, NULL, NULL,
  NULL, NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL, NULL,
  NULL, NULL, NULL, NULL, NULL,
  NULL, NULL, NULL,
  FALSE, FALSE, NULL
);

-- 6) experiment_step_events（按注释：1-7 步，事件类型 STARTED/COMPLETED）
INSERT INTO experiment_step_events (event_id, experiment_id, student_id, step_order, event_type, event_timestamp) VALUES
(1, 1, 3, 1, 'STARTED',   NOW() - INTERVAL 9 DAY),
(2, 1, 3, 1, 'COMPLETED', NOW() - INTERVAL 9 DAY + INTERVAL 2 HOUR),
(3, 1, 3, 2, 'STARTED',   NOW() - INTERVAL 8 DAY),
(4, 1, 3, 2, 'COMPLETED', NOW() - INTERVAL 8 DAY + INTERVAL 3 HOUR),
(5, 1, 3, 3, 'STARTED',   NOW() - INTERVAL 7 DAY),
(6, 1, 3, 3, 'COMPLETED', NOW() - INTERVAL 7 DAY + INTERVAL 4 HOUR),
(7, 1, 3, 4, 'STARTED',   NOW() - INTERVAL 1 DAY),

(8,  2, 3, 1, 'STARTED',   NOW() - INTERVAL 19 DAY),
(9,  2, 3, 1, 'COMPLETED', NOW() - INTERVAL 19 DAY + INTERVAL 2 HOUR),
(10, 2, 3, 2, 'STARTED',   NOW() - INTERVAL 18 DAY),
(11, 2, 3, 2, 'COMPLETED', NOW() - INTERVAL 18 DAY + INTERVAL 2 HOUR),
(12, 2, 3, 3, 'STARTED',   NOW() - INTERVAL 17 DAY),
(13, 2, 3, 3, 'COMPLETED', NOW() - INTERVAL 17 DAY + INTERVAL 3 HOUR),
(14, 2, 3, 4, 'STARTED',   NOW() - INTERVAL 16 DAY),
(15, 2, 3, 4, 'COMPLETED', NOW() - INTERVAL 16 DAY + INTERVAL 5 HOUR),
(16, 2, 3, 5, 'STARTED',   NOW() - INTERVAL 15 DAY),
(17, 2, 3, 5, 'COMPLETED', NOW() - INTERVAL 15 DAY + INTERVAL 4 HOUR),
(18, 2, 3, 6, 'STARTED',   NOW() - INTERVAL 14 DAY),
(19, 2, 3, 6, 'COMPLETED', NOW() - INTERVAL 14 DAY + INTERVAL 4 HOUR),
(20, 2, 3, 7, 'STARTED',   NOW() - INTERVAL 13 DAY),
(21, 2, 3, 7, 'COMPLETED', NOW() - INTERVAL 12 DAY),

(22, 3, 4, 1, 'STARTED',   NOW() - INTERVAL 6 DAY),
(23, 3, 4, 1, 'COMPLETED', NOW() - INTERVAL 6 DAY + INTERVAL 1 HOUR),
(24, 3, 4, 2, 'STARTED',   NOW() - INTERVAL 5 DAY),
(25, 3, 4, 2, 'COMPLETED', NOW() - INTERVAL 5 DAY + INTERVAL 2 HOUR),
(26, 3, 4, 3, 'STARTED',   NOW() - INTERVAL 1 DAY);

-- 7) question_bank（options/correct_answers 结构遵循注释示例）
INSERT INTO question_bank (question_id, knowledge_point, question_type, question_text, options, correct_answers, creator_id) VALUES
(1, 'ARIMA basics', 'Single Choice',
 'In ARIMA(p,d,q), what does d stand for?',
 JSON_OBJECT('A','Autoregressive order','B','Differencing order','C','Moving average order','D','Seasonal period'),
 JSON_ARRAY('B'), 1),
(2, 'Stationarity', 'True/False',
 'A stationary series has constant mean and variance over time.',
 JSON_ARRAY('True','False'),
 JSON_ARRAY('True'), 2),
(3, 'Inventory Planning', 'Multiple Choice',
 'Which factors influence safety stock? (Select all that apply)',
 JSON_OBJECT('A','Demand variability','B','Lead time variability','C','Selling price','D','Service level target'),
 JSON_ARRAY('A','B','D'), 1);

-- 8) experiment_reports（pdf_file_path 必须是 .pdf 结尾）
INSERT INTO experiment_reports (
  report_id, experiment_id, student_id, report_content, pdf_file_path, submitted_at, grade, feedback, graded_by
) VALUES
(1, 2, 3, 'Completed report on textiles forecasting.', '/data/reports/exp2_chen.pdf', NOW() - INTERVAL 2 DAY, 92.5, 'Good analysis and clear visualization.', 1);

-- 9) grade_weights（exp_flow_* 须相加为 100；总权重四项之和为 100）
INSERT INTO grade_weights (
  weight_id, teacher_id,
  exp_flow_demand_data_preparation, exp_flow_demand_descriptive_stats, exp_flow_demand_model_selection, exp_flow_demand_generate_results,
  exp_flow_production_inventory_calc, exp_flow_production_service_level, exp_flow_production_variable_calc, exp_flow_production_plan_creation, exp_flow_report_submission,
  exp_flow, knowledge_test, model_quality, report_quality
) VALUES
(1, 1, 8,8,8,8, 8,8,8,8,8, 40, 20, 20, 20),
(2, 2, 10,6,8,6, 6,8,6,6,4, 35, 25, 20, 20);

-- 10) experiment_grade（0-100 评分；复合外键对应 exp_status）
INSERT INTO experiment_grade (
  grade_id, experiment_id, student_id, grader_id,
  exp_flow,
  exp_flow_demand_data_preparation, exp_flow_demand_descriptive_stats, exp_flow_demand_model_selection, exp_flow_demand_generate_results,
  exp_flow_production_inventory_calc, exp_flow_production_service_level, exp_flow_production_variable_calc, exp_flow_production_plan_creation, exp_flow_report_submission,
  knowledge_test, model_quality, report_quality, created_at
) VALUES
(1, 2, 3, 1,
  88,
  90,85,88,87,
  86,84,85,90,89,
  92, 90, 93, NOW() - INTERVAL 2 DAY
);

-- 11) experiment_manuals（file_path 必须 .pdf 结尾）
INSERT INTO experiment_manuals (manual_id, file_name, file_path, description, is_active, uploader_id, uploaded_at) VALUES
(1, 'TS101_LabGuide.pdf', '/data/manuals/TS101_LabGuide.pdf', 'Guide for Time Series 101 labs', TRUE, 1, NOW()),
(2, 'PP201_Manual.pdf',   '/data/manuals/PP201_Manual.pdf',   'Production Planning manual',    TRUE, 2, NOW());

-- 12) datasets（file_path 必须 .csv 结尾）
INSERT INTO datasets (dataset_id, data_name, file_path, description, uploaded_at) VALUES
(1, 'Retail_Sales_Monthly', '/data/datasets/retail_sales_monthly.csv', 'Monthly retail sales with promotions flag', NOW()),
(2, 'Textile_Factory',      '/data/datasets/textile_factory.csv',       'Textile production & demand',              NOW());

-- 13) dataset_segments（唯一约束为 (industry_name, company_name, product_name)）
INSERT INTO dataset_segments (segment_id, dataset_id, industry_name, company_name, product_name, file_path) VALUES
(1, 1, 'Electronics', 'Acme Corp', 'Phone X', '/data/segments/retail/acme_phone_x.csv'),
(2, 2, 'Textiles',    'Luthai',    'Cotton Shirt', '/data/segments/textile/luthai_cotton_shirt.csv'),
(3, 1, 'Automotive',  'Vroom',     'Brake Pads', '/data/segments/auto/vroom_brake_pads.csv');

COMMIT;
