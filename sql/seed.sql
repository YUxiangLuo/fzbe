-- --------------------------------------------------------
-- 虚拟仿真系统数据库初始化数据 (Seed Data)
-- 适用于 MariaDB / MySQL
-- --------------------------------------------------------

-- 清空所有表并重置自增ID（按外键依赖顺序）
SET FOREIGN_KEY_CHECKS = 0;

TRUNCATE TABLE `experiment_reports`;
TRUNCATE TABLE `experiment_grade`;
TRUNCATE TABLE `grade_weights`;
TRUNCATE TABLE `experiment_step_events`;
TRUNCATE TABLE `experiment_status`;
TRUNCATE TABLE `dataset_segments`;
TRUNCATE TABLE `datasets`;
TRUNCATE TABLE `experiment_manuals`;
TRUNCATE TABLE `question_bank`;
TRUNCATE TABLE `class_assistants`;
TRUNCATE TABLE `enrollments`;
TRUNCATE TABLE `classes`;
TRUNCATE TABLE `users`;

ALTER TABLE `users` AUTO_INCREMENT = 1;
ALTER TABLE `classes` AUTO_INCREMENT = 1;
ALTER TABLE `enrollments` AUTO_INCREMENT = 1;
ALTER TABLE `class_assistants` AUTO_INCREMENT = 1;
ALTER TABLE `experiment_status` AUTO_INCREMENT = 1;
ALTER TABLE `experiment_step_events` AUTO_INCREMENT = 1;
ALTER TABLE `question_bank` AUTO_INCREMENT = 1;
ALTER TABLE `experiment_reports` AUTO_INCREMENT = 1;
ALTER TABLE `experiment_grade` AUTO_INCREMENT = 1;
ALTER TABLE `experiment_manuals` AUTO_INCREMENT = 1;
ALTER TABLE `datasets` AUTO_INCREMENT = 1;
ALTER TABLE `dataset_segments` AUTO_INCREMENT = 1;

SET FOREIGN_KEY_CHECKS = 1;

--
-- 1. 用户与班级管理 - 插入测试用户
--

INSERT INTO `users` (`username`, `password_hash`, `full_name`, `email`, `phone_number`, `role`) VALUES
('admin', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '系统管理员', 'admin@system.com', '13800138000', 'Admin'),
('teacher1', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '张老师', 'zhang.teacher@university.edu', '13800138001', 'Teacher'),
('teacher2', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '李老师', 'li.teacher@university.edu', '13800138002', 'Teacher'),
('assistant1', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '王助教', 'wang.assistant@university.edu', '13800138003', 'Assistant'),
('student1', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '赵明', 'zhao.ming@student.university.edu', '13800138004', 'Student'),
('student2', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '钱丽', 'qian.li@student.university.edu', '13800138005', 'Student'),
('student3', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '孙伟', 'sun.wei@student.university.edu', '13800138006', 'Student'),
('student4', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '周芳', 'zhou.fang@student.university.edu', '13800138007', 'Student');

-- 插入班级数据
INSERT INTO `classes` (`class_name`, `class_code`, `teacher_id`) VALUES
('数据分析与预测2023秋', 'DAF2023A', 2),
('数据分析与预测2023春', 'DAF2023S', 3),
('高级时间序列分析', 'ATSA2023', 2);

-- 插入学生选课数据
INSERT INTO `enrollments` (`student_id`, `class_id`) VALUES
(5, 1), (6, 1), (7, 1), (8, 1),
(5, 2), (6, 2),
(7, 3);

-- 插入助教分配数据
INSERT INTO `class_assistants` (`assistant_id`, `class_id`) VALUES
(4, 1),
(4, 2);

--
-- 2. 实验与生产计划管理
--

-- 插入实验状态数据（学生1在第一门课中的实验进度）
INSERT INTO `experiment_status` (
  `student_id`, `start_time`, `last_activity_at`, `completion_time`,
  `status`, `highest_completed_step`, `current_step`,
  `selected_industry`, `selected_company`, `selected_product`,
  `moving_average_completed`, `moving_average_window`,
  `moving_average_metrics_rmse`, `moving_average_metrics_mae`, `moving_average_metrics_r2`,
  `exponential_smoothing_completed`, `exponential_smoothing_alpha`,
  `exponential_smoothing_metrics_rmse`, `exponential_smoothing_metrics_mae`, `exponential_smoothing_metrics_r2`,
  `arima_completed`, `arima_p`, `arima_d`, `arima_q`,
  `arima_metrics_rmse`, `arima_metrics_mae`, `arima_metrics_r2`,
  `arima_adf_stationarity`,
  `lstm_completed`, `lstm_normalization`, `lstm_features`,
  `lstm_metrics_rmse`, `lstm_metrics_mae`, `lstm_metrics_r2`,
  `ensemble_weighted_completed`, `ensemble_weighted_base_models`,
  `ensemble_weighted_metrics_rmse`, `ensemble_weighted_metrics_mae`, `ensemble_weighted_metrics_r2`,
  `ensemble_boosting_completed`, `ensemble_boosting_base_models`,
  `ensemble_boosting_metrics_rmse`, `ensemble_boosting_metrics_mae`, `ensemble_boosting_metrics_r2`,
  `ensemble_stacking_completed`, `ensemble_stacking_base_models`,
  `ensemble_stacking_metrics_rmse`, `ensemble_stacking_metrics_mae`, `ensemble_stacking_metrics_r2`,
  `selected_best_model`,
  `predict_steps`,
  `predict_values`
) VALUES (
  5,
  '2023-09-01 09:00:00',
  '2023-09-15 16:30:00',
  '2023-09-15 17:00:00',
  'Completed',
  7,
  7,
  '电子消费品',
  'XYZ科技公司',
  '智能手机',
  TRUE, 7,
  125.50, 98.20, 0.785,
  TRUE, 0.3,
  110.25, 85.60, 0.823,
  TRUE, 1, 1, 1,
  95.80, 72.40, 0.891,
  '[{"diff_order": 0, "statistic": -1.876, "p_value": 0.640, "stationary": false, "critical_values": {"1%": -3.500, "5%": -2.890, "10%": -2.580}}, {"diff_order": 1, "statistic": -3.954, "p_value": 0.002, "stationary": true, "critical_values": {"1%": -3.500, "5%": -2.890, "10%": -2.580}}, {"diff_order": 2, "statistic": -5.102, "p_value": 0.000, "stationary": true, "critical_values": {"1%": -3.500, "5%": -2.890, "10%": -2.580}}]',
  TRUE, 'minmax',
  '["月份", "销售金额", "是否促销"]',
  88.45, 65.30, 0.912,
  TRUE, '["ma", "exp", "arima", "lstm"]',
  75.20, 58.90, 0.945,
  TRUE, '["ma", "exp", "arima"]',
  82.10, 64.50, 0.928,
  TRUE, '["ma", "exp", "arima", "lstm"]',
  70.85, 55.20, 0.956,
  'ensemble_stacking',
  3,
  '[100, 101, 102]'
);

-- 插入学生2的实验状态（进行中）
INSERT INTO `experiment_status` (
  `student_id`, `start_time`, `last_activity_at`,
  `status`, `highest_completed_step`, `current_step`,
  `selected_industry`, `selected_company`, `selected_product`,
  `moving_average_completed`, `moving_average_window`,
  `moving_average_metrics_rmse`, `moving_average_metrics_mae`, `moving_average_metrics_r2`,
  `exponential_smoothing_completed`, `exponential_smoothing_alpha`
) VALUES (
  6,
  '2023-09-02 10:15:00',
  '2023-09-14 14:20:00',
  'In Progress',
  4,
  5,
  '服装纺织',
  'ABC服饰有限公司',
  '休闲装',
  TRUE, 5,
  210.75, 165.30, 0.623,
  TRUE, 0.2
);

-- 插入学生3的实验状态（未开始）
INSERT INTO `experiment_status` (
  `student_id`, `status`, `highest_completed_step`, `current_step`
) VALUES (
  7,
  'Not Started',
  0,
  1
);

-- 插入实验步骤事件记录
INSERT INTO `experiment_step_events` (`experiment_id`, `student_id`, `step_order`, `event_type`, `event_timestamp`) VALUES
(1, 5, 1, 'STARTED', '2023-09-01 09:00:00'),
(1, 5, 1, 'COMPLETED', '2023-09-01 09:30:00'),
(1, 5, 2, 'STARTED', '2023-09-01 09:35:00'),
(1, 5, 2, 'COMPLETED', '2023-09-01 10:15:00'),
(1, 5, 3, 'STARTED', '2023-09-01 10:20:00'),
(1, 5, 3, 'COMPLETED', '2023-09-01 11:00:00'),
(1, 5, 4, 'STARTED', '2023-09-03 14:00:00'),
(1, 5, 4, 'COMPLETED', '2023-09-03 15:30:00'),
(1, 5, 5, 'STARTED', '2023-09-05 09:00:00'),
(1, 5, 5, 'COMPLETED', '2023-09-05 11:45:00'),
(1, 5, 6, 'STARTED', '2023-09-10 13:30:00'),
(1, 5, 6, 'COMPLETED', '2023-09-10 16:20:00'),
(1, 5, 7, 'STARTED', '2023-09-12 09:15:00'),
(1, 5, 7, 'COMPLETED', '2023-09-15 17:00:00'),
(2, 6, 1, 'STARTED', '2023-09-02 10:15:00'),
(2, 6, 1, 'COMPLETED', '2023-09-02 10:45:00'),
(2, 6, 2, 'STARTED', '2023-09-02 11:00:00'),
(2, 6, 2, 'COMPLETED', '2023-09-02 11:40:00'),
(2, 6, 3, 'STARTED', '2023-09-04 14:30:00'),
(2, 6, 3, 'COMPLETED', '2023-09-04 15:20:00'),
(2, 6, 4, 'STARTED', '2023-09-06 09:00:00'),
(2, 6, 4, 'COMPLETED', '2023-09-06 10:30:00'),
(2, 6, 5, 'STARTED', '2023-09-14 14:20:00');

--
-- 3. 考核与评分
--

-- 插入题库数据
INSERT INTO `question_bank` (
  `knowledge_point`, `question_type`, `question_text`,
  `options`, `correct_answers`, `creator_id`
) VALUES
-- 单选题1
('时间序列平稳性', 'Single Choice', '在ARIMA模型中，d参数代表什么？',
 '{"A": "自回归项数", "B": "差分阶数", "C": "移动平均项数", "D": "季节周期"}',
 '["B"]',
 2),

-- 单选题2
('模型评估指标', 'Single Choice', 'RMSE指标越小说明什么？',
 '{"A": "模型过拟合", "B": "预测精度越高", "C": "数据噪声越大", "D": "计算复杂度越高"}',
 '["B"]',
 2),

-- 多选题
('集成学习', 'Multiple Choice', '以下哪些是常见的集成学习方法？',
 '{"A": "Bagging", "B": "Boosting", "C": "Stacking", "D": "K-Means"}',
 '["A", "B", "C"]',
 3),

-- 判断题1
('时间序列分析', 'True/False', '所有时间序列数据都必须经过一阶差分才能用于ARIMA建模。',
 '{"正确": "一阶差分后序列一定平稳", "错误": "需要根据ADF检验确定差分阶数"}',
 '["错误"]',
 2),

-- 判断题2
('模型选择', 'True/False', 'R²值为0.9表示模型解释了90%的方差。',
 '{"正确": "模型效果很好", "错误": "R²值不代表因果关系"}',
 '["正确"]',
 3);

-- 插入实验报告数据
INSERT INTO `experiment_reports` (
  `experiment_id`, `student_id`, `report_content`,
  `pdf_file_path`, `submitted_at`, `grade`,
  `feedback`, `graded_by`
) VALUES
-- 学生1的完整报告
(1, 5,
 '本实验通过对XYZ科技公司智能手机销售数据的分析，完成了从数据预处理到生产计划制定的全流程。通过对比多种预测模型，最终选择了堆叠集成模型作为最佳方案，该模型在测试集上的RMSE为70.85，R²值为0.956，表现最佳。基于预测结果，制定了详细的生产库存计划。',
 '/var/uploads/reports/student1_xyz_smartphone_report_20230915.pdf',
 '2023-09-15 17:00:00',
 92.5,
 '报告结构清晰，模型选择合理，生产计划制定科学。建议在未来工作中进一步考虑不确定性分析。',
 2),

-- 学生2的报告（待评分）
(2, 6,
 '本次实验分析了ABC服饰休闲装的销售数据，完成了前四个步骤的模型训练。目前已完成移动平均和指数平滑模型的训练，RMSE分别为210.75和未完成。后续将完成剩余模型对比和生产计划制定。',
 '/var/uploads/reports/student2_abc_casualwear_report_20230914.pdf',
 '2023-09-14 14:20:00',
 NULL,
 NULL,
 NULL);

-- 插入评分权重配置
INSERT INTO `grade_weights` (
  `teacher_id`,
  `exp_flow_demand_data_preparation`, `exp_flow_demand_descriptive_stats`,
  `exp_flow_demand_model_selection`, `exp_flow_demand_generate_results`,
  `exp_flow_production_inventory_calc`, `exp_flow_production_service_level`,
  `exp_flow_production_variable_calc`, `exp_flow_production_plan_creation`,
  `exp_flow_report_submission`,
  `exp_flow`, `knowledge_test`, `model_quality`, `report_quality`
) VALUES
-- 老师1的评分权重配置
(2,
 8, 7,
 10, 10,
 5, 2,
 2, 1,
 0,
 45,
 20,
 20,
 15),

-- 老师2的评分权重配置
(3,
 10, 8,
 12, 10,
 4, 3,
 2, 1,
 0,
 50,
 15,
 25,
 10);

-- 插入实验评分数据
INSERT INTO `experiment_grade` (
  `experiment_id`, `student_id`, `grader_id`,
  `exp_flow`,
  `exp_flow_demand_data_preparation`, `exp_flow_demand_descriptive_stats`,
  `exp_flow_demand_model_selection`, `exp_flow_demand_generate_results`,
  `exp_flow_production_inventory_calc`, `exp_flow_production_service_level`,
  `exp_flow_production_variable_calc`, `exp_flow_production_plan_creation`,
  `exp_flow_report_submission`,
  `knowledge_test`, `model_quality`, `report_quality`
) VALUES
-- 学生1的详细评分
(1, 5, 2,
 90,
 95, 92,
 88, 92,
 85, 88,
 90, 95,
 90,
 95,
 92,
 90);

--
-- 4. 系统管理
--

-- 插入实验手册数据
INSERT INTO `experiment_manuals` (
  `file_name`, `file_path`, `description`,
  `is_active`, `uploader_id`
) VALUES
-- 实验操作手册
('虚拟仿真实验手册.pdf',
 '/var/uploads/manuals/shiyanshouce.pdf',
 '详细介绍时间序列预测实验的完整操作流程，包括数据预处理、模型训练、结果分析和生产计划制定等所有步骤。',
 TRUE,
 2),

('生产计划制定指南_v1.0.pdf',
 '/var/uploads/manuals/shiyanshouce.pdf',
 '指导学生如何基于预测结果制定科学的生产库存计划，包括安全库存计算、服务水平评估等内容。',
 FALSE,
 3),

('数据分析基础教程.pdf',
 '/var/uploads/manuals/shiyanshouce.pdf',
 '时间序列数据分析基础知识，包括平稳性检验、模型评估指标等内容。',
 FALSE,
 2);

-- 插入数据集数据
INSERT INTO `datasets` (`data_name`, `file_path`, `description`) VALUES
('消费电子行业销售数据_2020-2023.csv',
 '/var/uploads/datasets/ceshishuju.csv',
 '包含2020-2023年消费电子行业多家企业的月度销售数据，包含智能手机、平板电脑等产品的销售金额和促销信息。'),

('服装纺织行业销售数据_2019-2023.csv',
 '/var/uploads/datasets/ceshishuju.csv',
 '包含2019-2023年服装纺织行业销售数据，涵盖休闲装、正装等多个品类，包含季节性和促销影响因素。'),

('汽车行业销售数据_2021-2023.csv',
 '/var/uploads/datasets/ceshishuju.csv',
 '包含2021-2023年汽车行业销售数据，包括乘用车和商用车销售数据，考虑经济周期影响。');

-- 插入数据集分段数据
INSERT INTO `dataset_segments` (
  `dataset_id`, `industry_name`, `company_name`,
  `product_name`, `file_path`
) VALUES
-- XYZ科技公司智能手机数据 (第一数据集)
(1, '电子消费品', 'XYZ科技公司', '智能手机', '/var/uploads/dataset_segments/electronics_xyz_smartphone_2020-2023.csv'),

-- ABC服饰休闲装数据 (第二数据集)
(2, '服装纺织', 'ABC服饰有限公司', '休闲装', '/var/uploads/dataset_segments/apparel_abc_casualwear_2019-2023.csv'),

-- DEF汽车SUV数据 (第三数据集)
(3, '汽车制造', 'DEF汽车集团', 'SUV车型', '/var/uploads/dataset_segments/automotive_def_suv_2021-2023.csv'),

-- GHI科技平板电脑数据 (第一数据集)
(1, '电子消费品', 'GHI科技公司', '平板电脑', '/var/uploads/dataset_segments/electronics_ghi_tablet_2020-2023.csv'),

-- JKL服饰正装数据 (第二数据集)
(2, '服装纺织', 'JKL服饰公司', '正装', '/var/uploads/dataset_segments/apparel_jkl_formalwear_2019-2023.csv');

-- 插入一条额外的评分记录用于测试
INSERT INTO `experiment_grade` (
  `experiment_id`, `student_id`, `grader_id`,
  `exp_flow`,
  `exp_flow_demand_data_preparation`, `exp_flow_demand_descriptive_stats`,
  `exp_flow_demand_model_selection`, `exp_flow_demand_generate_results`
) VALUES
(2, 6, 4,
 75,
 80, 78,
 72, 75);

-- 恢复外键检查
SET FOREIGN_KEY_CHECKS = 1;

-- 显示插入的数据统计
SELECT '用户数据' as table_name, COUNT(*) as record_count FROM `users`
UNION ALL
SELECT '班级数据', COUNT(*) FROM `classes`
UNION ALL
SELECT '选课数据', COUNT(*) FROM `enrollments`
UNION ALL
SELECT '助教分配', COUNT(*) FROM `class_assistants`
UNION ALL
SELECT '实验状态', COUNT(*) FROM `experiment_status`
UNION ALL
SELECT '步骤事件', COUNT(*) FROM `experiment_step_events`
UNION ALL
SELECT '题库数据', COUNT(*) FROM `question_bank`
UNION ALL
SELECT '实验报告', COUNT(*) FROM `experiment_reports`
UNION ALL
SELECT '评分权重', COUNT(*) FROM `grade_weights`
UNION ALL
SELECT '实验评分', COUNT(*) FROM `experiment_grade`
UNION ALL
SELECT '实验手册', COUNT(*) FROM `experiment_manuals`
UNION ALL
SELECT '数据集', COUNT(*) FROM `datasets`
UNION ALL
SELECT '数据分段', COUNT(*) FROM `dataset_segments`;
