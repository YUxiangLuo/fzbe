-- --------------------------------------------------------
-- 虚拟仿真系统数据库种子文件 (Seed File)
-- 适用于 MariaDB / MySQL
-- --------------------------------------------------------
-- 说明:
--   1. 通过 INSERT IGNORE 保障重复执行脚本时不会报错。
--   2. 所有示例账号的默认密码均为 password123 (bcrypt:
--      $2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG)。
--   3. 数据覆盖了典型业务流程: 用户/班级、助教、实验过程、题库、评分
--      以及资料/数据集上传，方便做端到端联调测试。
-- --------------------------------------------------------

-- 1. 核心用户 (Users)
INSERT IGNORE INTO `users` (`user_id`, `username`, `password_hash`, `full_name`, `email`, `phone_number`, `role`) VALUES
(1,  'admin',         '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '系统管理员', 'admin@example.com',       '10000000000', 'Admin'),
(2,  'admin_yuxiang', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '余翔',       'yuxiang@example.com',    '10000000001', 'Admin'),
(3,  'teacher_liu',   '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '刘老师',     'teacher.liu@example.com','13800138000', 'Teacher'),
(4,  'teacher_chen',  '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '陈老师',     'teacher.chen@example.com','13800138001','Teacher'),
(5,  'teacher_zhao',  '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '赵老师',     'teacher.zhao@example.com','13800138002','Teacher'),
(6,  'teacher_wang',  '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '王老师',     'teacher.wang@example.com','13800138003','Teacher'),
(7,  'assistant_li',  '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '李助教',     'assistant.li@example.com','13900139000','Assistant'),
(8,  'assistant_sun', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '孙助教',     'assistant.sun@example.com','13900139001','Assistant'),
(9,  'student_zhang', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '张小明',     'zhangxm@example.com',    NULL,           'Student'),
(10, 'student_li',    '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '李晓丽',     'lixl@example.com',       NULL,           'Student'),
(11, 'student_wu',    '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '吴嘉禾',     'wujh@example.com',       NULL,           'Student'),
(12, 'student_zhao',  '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '赵梓涵',     'zhaozh@example.com',     NULL,           'Student'),
(13, 'student_chen',  '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '陈语桐',     'chenyt@example.com',     NULL,           'Student'),
(14, 'student_liu',   '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '刘博文',     'liubw@example.com',      NULL,           'Student'),
(15, 'student_sun',   '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '孙晨曦',     'suncx@example.com',      NULL,           'Student');

-- 2. 班级 (Classes)
INSERT IGNORE INTO `classes` (`class_id`, `class_name`, `class_code`, `teacher_id`) VALUES
(1, '工业工程综合实验2025',   'IE2025', 3),
(2, '供应链协同仿真班',       'SCM2025',4),
(3, '智能制造实践小班',       'IM2025', 5),
(4, '数据驱动决策班',         'DDD2025',6),
(5, 'ERP实施实训班',         'ERP2025',3),
(6, '物流系统优化班',         'LSO2025',4);

-- 3. 助教与班级关联 (ClassAssistants)
INSERT IGNORE INTO `class_assistants` (`assistant_id`, `class_id`) VALUES
(7, 1),
(7, 5),
(8, 2),
(8, 6);

-- 4. 学生选课 (Enrollments)
INSERT IGNORE INTO `enrollments` (`student_id`, `class_id`) VALUES
(9, 1),
(10,1),
(11,1),
(12,2),
(13,2),
(14,3),
(15,3),
(9, 4),
(12,4),
(10,5),
(13,5),
(11,6),
(14,6);

-- 5. 题库 (QuestionBank)
INSERT IGNORE INTO `question_bank` (`question_id`, `knowledge_point`, `question_type`, `question_text`, `options`, `correct_answer`, `creator_id`) VALUES
(1, '需求预测基础', 'Single Choice', '在 ARIMA 模型中，"d" 参数的含义是?', '{"A":"自回归项数","B":"差分阶数","C":"移动平均项数","D":"季节周期"}', 'B', 3),
(2, '时间序列预处理', 'Multiple Choice', '以下哪些步骤常用于时间序列数据预处理?', '{"A":"缺失值填补","B":"标准化","C":"图像增强","D":"滞后特征构造"}', '["A","B","D"]', 3),
(3, '生产计划', 'True/False', '安全库存可以帮助降低缺货风险。', NULL, 'True', 4),
(4, '融合模型', 'Single Choice', '使用多个模型结果加权得到最终预测值的方式称为？', '{"A":"合成预测","B":"模型选择","C":"模型集成","D":"模型融合"}', 'C', 4);

-- 6. 成绩权重 (GradeWeights)
INSERT IGNORE INTO `grade_weights` (`class_id`, `experiment_flow`, `knowledge_tests`, `model_selection`, `experiment_report`, `detailed_step_weights`) VALUES
(1, 35, 25, 10, 30, '{"data_preparation":10,"model_training":10,"reporting":10}'),
(2, 30, 20, 20, 30, '{"data_preparation":8,"scenario_design":12,"presentation":10}'),
(4, 40, 15, 15, 30, '{"data_preparation":12,"feature_engineering":13,"deployment":5}');

-- 7. 实验状态 (ExperimentStatus)
INSERT IGNORE INTO `experiment_status` (`experiment_id`, `student_id`, `start_time`, `completion_time`, `status`, `selected_industry`, `selected_company`, `selected_product`, `best_model`) VALUES
(1, 9,  '2025-03-01 09:00:00', '2025-03-01 10:30:00', 'Completed',   '电子制造', '华夏电子',   '智能手环',     'ARIMA(2,1,1)'),
(2, 10, '2025-03-02 13:00:00', '2025-03-02 15:10:00', 'Completed',   '消费品',   '星光家居',   '空气净化器', 'LSTM'),
(3, 12, '2025-03-03 09:30:00', NULL,                   'In Progress', '物流运输', '联合物流',   '冷链配送',   NULL),
(4, 14, '2025-03-04 14:00:00', '2025-03-04 16:00:00', 'Completed',   '食品饮料', '味来食品',   '果汁饮料',   'MovingAverage'),
(5, 11, '2025-03-05 09:15:00', NULL,                   'In Progress', '智能制造', '未来工厂',   '协作机器人', NULL),
(6, 13, '2025-03-06 10:00:00', NULL,                   'In Progress', '新能源',   '澜光能源',   '储能系统',   NULL),
(7, 15, '2025-03-06 15:30:00', NULL,                   'Not Started', NULL,        NULL,        NULL,          NULL);

-- 9. 实验步骤事件 (ExperimentStepEvents)
INSERT IGNORE INTO `experiment_step_events` (`event_id`, `experiment_id`, `student_id`, `step_order`, `event_type`, `event_timestamp`) VALUES
(1,  1, 9,  1, 'STARTED',   '2025-03-01 09:00:00'),
(2,  1, 9,  1, 'COMPLETED', '2025-03-01 09:02:00'),
(3,  1, 9,  2, 'STARTED',   '2025-03-01 09:02:00'),
(4,  1, 9,  2, 'COMPLETED', '2025-03-01 09:05:00'),
(5,  1, 9,  3, 'STARTED',   '2025-03-01 09:05:00'),
(6,  1, 9,  3, 'COMPLETED', '2025-03-01 09:10:00'),
(7,  1, 9,  4, 'STARTED',   '2025-03-01 09:10:00'),
(8,  1, 9,  4, 'COMPLETED', '2025-03-01 09:25:00'),
(9,  1, 9,  5, 'STARTED',   '2025-03-01 09:25:00'),
(10, 1, 9,  5, 'COMPLETED', '2025-03-01 10:05:00'),
(11, 1, 9,  6, 'STARTED',   '2025-03-01 10:05:00'),
(12, 1, 9,  6, 'COMPLETED', '2025-03-01 10:20:00'),
(13, 1, 9,  7, 'STARTED',   '2025-03-01 10:20:00'),
(14, 1, 9,  7, 'COMPLETED', '2025-03-01 10:30:00'),
(15, 2, 10, 1, 'STARTED',   '2025-03-02 13:00:00'),
(16, 2, 10, 1, 'COMPLETED', '2025-03-02 13:03:00'),
(17, 2, 10, 2, 'STARTED',   '2025-03-02 13:03:00'),
(18, 2, 10, 2, 'COMPLETED', '2025-03-02 13:08:00'),
(19, 2, 10, 3, 'STARTED',   '2025-03-02 13:08:00'),
(20, 2, 10, 3, 'COMPLETED', '2025-03-02 13:12:00'),
(21, 2, 10, 4, 'STARTED',   '2025-03-02 13:12:00'),
(22, 2, 10, 4, 'COMPLETED', '2025-03-02 13:35:00'),
(23, 2, 10, 5, 'STARTED',   '2025-03-02 13:35:00'),
(24, 2, 10, 5, 'COMPLETED', '2025-03-02 14:40:00'),
(25, 2, 10, 6, 'STARTED',   '2025-03-02 14:40:00'),
(26, 2, 10, 6, 'COMPLETED', '2025-03-02 14:55:00'),
(27, 2, 10, 7, 'STARTED',   '2025-03-02 14:55:00'),
(28, 2, 10, 7, 'COMPLETED', '2025-03-02 15:10:00'),
(29, 3, 12, 1, 'STARTED',   '2025-03-03 09:30:00'),
(30, 3, 12, 1, 'COMPLETED', '2025-03-03 09:32:00'),
(31, 3, 12, 2, 'STARTED',   '2025-03-03 09:32:00'),
(32, 3, 12, 2, 'COMPLETED', '2025-03-03 09:34:00'),
(33, 3, 12, 3, 'STARTED',   '2025-03-03 09:34:00'),
(34, 3, 12, 3, 'COMPLETED', '2025-03-03 09:38:00'),
(35, 3, 12, 4, 'STARTED',   '2025-03-03 09:38:00'),
(36, 3, 12, 4, 'COMPLETED', '2025-03-03 09:48:00'),
(37, 3, 12, 5, 'STARTED',   '2025-03-03 09:50:00'),
(38, 4, 14, 1, 'STARTED',   '2025-03-04 14:00:00'),
(39, 4, 14, 1, 'COMPLETED', '2025-03-04 14:02:00'),
(40, 4, 14, 2, 'STARTED',   '2025-03-04 14:02:00'),
(41, 4, 14, 2, 'COMPLETED', '2025-03-04 14:05:00'),
(42, 4, 14, 3, 'STARTED',   '2025-03-04 14:05:00'),
(43, 4, 14, 3, 'COMPLETED', '2025-03-04 14:08:00'),
(44, 4, 14, 4, 'STARTED',   '2025-03-04 14:08:00'),
(45, 4, 14, 4, 'COMPLETED', '2025-03-04 14:18:00'),
(46, 4, 14, 5, 'STARTED',   '2025-03-04 14:18:00'),
(47, 4, 14, 5, 'COMPLETED', '2025-03-04 15:20:00'),
(48, 4, 14, 6, 'STARTED',   '2025-03-04 15:20:00'),
(49, 4, 14, 6, 'COMPLETED', '2025-03-04 15:35:00'),
(50, 4, 14, 7, 'STARTED',   '2025-03-04 15:35:00'),
(51, 4, 14, 7, 'COMPLETED', '2025-03-04 16:00:00'),
(52, 5, 11, 1, 'STARTED',   '2025-03-05 09:15:00'),
(53, 5, 11, 1, 'COMPLETED', '2025-03-05 09:17:00'),
(54, 5, 11, 2, 'STARTED',   '2025-03-05 09:17:00'),
(55, 5, 11, 2, 'COMPLETED', '2025-03-05 09:22:00'),
(56, 5, 11, 3, 'STARTED',   '2025-03-05 09:22:00'),
(57, 5, 11, 3, 'COMPLETED', '2025-03-05 09:28:00'),
(58, 5, 11, 4, 'STARTED',   '2025-03-05 09:28:00'),
(59, 5, 11, 4, 'COMPLETED', '2025-03-05 09:45:00'),
(60, 5, 11, 5, 'STARTED',   '2025-03-05 09:50:00'),
(61, 6, 13, 1, 'STARTED',   '2025-03-06 10:00:00'),
(62, 6, 13, 1, 'COMPLETED', '2025-03-06 10:03:00'),
(63, 6, 13, 2, 'STARTED',   '2025-03-06 10:03:00'),
(64, 6, 13, 2, 'COMPLETED', '2025-03-06 10:08:00'),
(65, 6, 13, 3, 'STARTED',   '2025-03-06 10:08:00'),
(66, 6, 13, 3, 'COMPLETED', '2025-03-06 10:15:00'),
(67, 6, 13, 4, 'STARTED',   '2025-03-06 10:15:00'),
(68, 7, 15, 1, 'STARTED',   '2025-03-06 15:30:00');

INSERT IGNORE INTO `model_results` (`result_id`, `experiment_id`, `model_name`, `model_parameters`, `rmse`, `mae`, `r2`, `mape`, `prediction_output`) VALUES
(1, 1, 'ARIMA',               '{"p":2,"d":1,"q":1}',        8.52, 6.30, 0.87, 4.10, '{"forecast":[128,133,139,145]}'),
(2, 1, 'ExponentialSmoothing','{"alpha":0.3}',               10.40, 7.90, 0.79, 5.35, '{"forecast":[126,131,135,140]}'),
(3, 2, 'LSTM',                '{"epochs":50,"layers":2}',    6.12, 4.85, 0.91, 3.72, '{"forecast":[215,221,229,236]}'),
(4, 4, 'MovingAverage',       '{"window":3}',                 9.35, 7.21, 0.83, 4.85, '{"forecast":[182,185,189,192]}');

INSERT IGNORE INTO `production_plans` (`plan_id`, `experiment_id`, `plan_details`) VALUES
(1, 1, '{"horizon_weeks":4,"lines":2,"safety_stock":320,"notes":"结合 ARIMA 预测调整排产节奏"}'),
(2, 2, '{"horizon_weeks":6,"lines":3,"safety_stock":180,"notes":"LSTM 预测显示节假日前需求激增"}'),
(3, 4, '{"horizon_weeks":3,"lines":1,"safety_stock":90,"notes":"采用移动平均模型稳定生产"}');
-- 11. 学生答题记录 (StudentResponses)
INSERT IGNORE INTO `student_responses` (`response_id`, `experiment_id`, `question_id`, `answer`, `is_correct`) VALUES
(1, 1, 1, 'B',    TRUE),
(2, 1, 2, '["A","B","D"]', TRUE),
(3, 2, 1, 'A',    FALSE),
(4, 2, 3, 'True', TRUE),
(5, 4, 4, 'C',    TRUE);

-- 12. 实验报告 (ExperimentReports)
INSERT IGNORE INTO `experiment_reports` (`report_id`, `experiment_id`, `report_content`, `pdf_file_path`, `submitted_at`, `grade`, `feedback`, `graded_by`) VALUES
(1, 1, '基于 ARIMA 模型的智能手环需求预测分析。', '/uploads/reports/exp1-report.pdf', '2025-03-01 21:30:00', 92.5, '模型选择合理，报告结构清晰。', 3),
(2, 2, 'LSTM 模型捕捉到季节特征并提升预测准确率。', '/uploads/reports/exp2-report.pdf', '2025-03-02 22:15:00', 88.0, '建议增加模型可解释性讨论。', 4),
(3, 4, '移动平均模型用于饮料需求预测的快速实践。', '/uploads/reports/exp4-report.pdf', '2025-03-04 20:05:00', 85.0, '适合入门演示，可尝试更多模型比较。', 5);

-- 13. 实验手册 (ExperimentManuals)
INSERT IGNORE INTO `experiment_manuals` (`manual_id`, `file_name`, `file_path`, `description`, `is_active`, `uploader_id`) VALUES
(1, '虚拟仿真实验手册 V2.0.pdf', '/uploads/static/manuals/virtual-lab-v2.pdf', '2025 春季学期使用，包含实验流程与评分标准。', TRUE, 1),
(2, '教师操作速查表.pdf',       '/uploads/static/manuals/teacher-quickstart.pdf', '教师端重点功能速览。', FALSE, 2);

-- 14. 数据集 (Datasets)
INSERT IGNORE INTO `datasets` (`dataset_id`, `data_name`, `file_path`, `description`) VALUES
(1, '电子产品销售月度数据', '/uploads/static/datasets/electronics-monthly.csv', '包含 2022-2024 年月度销量、收入、均价。'),
(2, '家电行业节假日销量', '/uploads/static/datasets/home-appliance-holidays.csv', '节假日前后 60 天的日销量及促销活动标签。'),
(3, '冷链物流温控数据', '/uploads/static/datasets/cold-chain-temp.csv', '多线路冷链配送的温度与到货时效监控数据。');

-- --------------------------------------------------------
-- 结束
-- --------------------------------------------------------
