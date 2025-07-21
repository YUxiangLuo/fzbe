-- --------------------------------------------------------
-- 虚拟仿真系统数据库种子文件 (Seed File)
-- 适用于 MariaDB / MySQL
-- --------------------------------------------------------

-- 提示：在执行此文件前，您可能希望先清空相关表中的数据。
-- 在MariaDB中，可以这样做：
-- SET FOREIGN_KEY_CHECKS = 0;
-- TRUNCATE TABLE `users`;
-- TRUNCATE TABLE `classes`;
-- TRUNCATE TABLE `enrollments`;
-- -- ...对其他表执行TRUNCATE...
-- SET FOREIGN_KEY_CHECKS = 1;


-- 1. 插入用户数据 (Users)
-- 使用 INSERT IGNORE 来避免重复插入时出错。
INSERT IGNORE INTO `users` (`user_id`, `username`, `password_hash`, `full_name`, `email`, `phone_number`, `role`) VALUES
(1, 'admin', '$2b$10$your_secure_hash_here', '系统管理员', 'admin@example.com', '10000000000', 'Admin'),
(2, 'teacher01', '$2b$10$your_secure_hash_here', '张老师', 'teacher01@example.com', '10000000001', 'Teacher'),
(3, 'assistant01', '$2b$10$your_secure_hash_here', '李助教', 'assistant01@example.com', '10000000002', 'Assistant'),
(4, 'student01', '$2b$10$your_secure_hash_here', '王小明', 'student01@example.com', '10000000003', 'Student'),
(5, 'yuxiang', '$2b$10$IugK3IDtW44SFJ2SWFePEe1ZEZlw9ytIbX81UOx/bgEMjGU9LEVeG', 'YUxiangLuo', 'yuxiang@example.com', NULL, 'Admin');

-- 2. 插入班级数据 (Classes)
INSERT IGNORE INTO `classes` (`class_id`, `class_name`, `class_code`, `teacher_id`) VALUES
(1, '生产决策仿真2025春季班', 'PM2025S', 2);

-- 3. 插入学生分班数据 (Enrollments)
INSERT IGNORE INTO `enrollments` (`enrollment_id`, `student_id`, `class_id`) VALUES
(1, 4, 1);

-- 4. 插入助教分配数据 (AssistantAssignments)
INSERT IGNORE INTO `assistant_assignments` (`assignment_id`, `assistant_id`, `teacher_id`) VALUES
(1, 3, 2);

-- 5. 插入题库数据 (QuestionBank)
INSERT IGNORE INTO `question_bank` (`question_id`, `knowledge_point`, `question_type`, `question_text`, `options`, `correct_answer`, `creator_id`) VALUES
(1, '需求预测', 'Single Choice', '在ARIMA模型中，"d"参数代表什么？', '{"A": "自回归项数", "B": "移动平均项数", "C": "差分阶数", "D": "季节性周期"}', 'C', 2),
(2, '生产计划', 'True/False', '安全库存的主要目的是为了应对市场需求的不确定性。', NULL, 'True', 2),
(3, '融合模型', 'Multiple Choice', '以下哪些属于融合预测模型？', '{"A": "移动平均法", "B": "Boosting融合模型", "C": "Stacking融合模型", "D": "指数平滑法"}', '["B", "C"]', 2);

-- 6. 插入默认成绩权重 (GradeWeights)
INSERT IGNORE INTO `grade_weights` (`weight_id`, `class_id`, `experiment_flow`, `knowledge_tests`, `model_selection`, `experiment_report`, `detailed_step_weights`) VALUES
(1, 1, 40, 20, 10, 30, '{"data_preparation": 5, "descriptive_statistics": 5, "model_choice": 10, "prediction_generation": 20}');

-- 7. 插入默认实验手册和数据 (ExperimentManuals & SystemData)
INSERT IGNORE INTO `experiment_manuals` (`manual_id`, `file_name`, `file_path`, `is_active`) VALUES
(1, '虚拟仿真实验手册V1.0.pdf', '/manuals/manual_v1.pdf', 1); -- MariaDB/MySQL 中布尔值通常用 1/0

INSERT IGNORE INTO `system_data` (`data_id`, `data_name`, `file_path`, `description`) VALUES
(1, '某产品历史销售数据', '/data/sales_data_product_a.csv', '包含2022-2024年某产品的月度销售量、销售额和平均价格等信息。');


-- 重置 AUTO_INCREMENT 值 (可选)
-- 如果你需要确保下次插入时ID从最大值+1开始，可以执行以下命令。
-- ALTER TABLE `users` AUTO_INCREMENT = 5;
-- ALTER TABLE `classes` AUTO_INCREMENT = 2;
-- ... 对所有自增主键执行此操作
