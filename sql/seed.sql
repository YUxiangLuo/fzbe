-- --------------------------------------------------------
-- 虚拟仿真系统数据库种子文件 (Seed File)
-- 适用于 MariaDB / MySQL
-- --------------------------------------------------------
-- 注意: 此文件用于提供基础数据。
-- 如需大规模生成用户和班级，请使用 `scripts/` 目录下的 bun 脚本。
-- --------------------------------------------------------

-- 1. 插入核心用户数据 (Users)
-- 默认密码均为 'password123'，哈希值由 `Bun.password.hash('password123', { algorithm: 'bcrypt', cost: 10 })` 生成。
-- 使用 INSERT IGNORE 来避免重复插入时出错。
INSERT IGNORE INTO `users` (`user_id`, `username`, `password_hash`, `full_name`, `email`, `phone_number`, `role`) VALUES
(1, 'admin', '$2a$10$g.5b.1o1b.2i3c.4d.5e.u2q/e3w4r5t6y7u8i9o0p1q2', '系统管理员', 'admin@example.com', '10000000000', 'Admin'),
(2, 'teacher01', '$2a$10$g.5b.1o1b.2i3c.4d.5e.u2q/e3w4r5t6y7u8i9o0p1q2', '张老师', 'teacher01@example.com', '10000000001', 'Teacher'),
(3, 'assistant01', '$2a$10$g.5b.1o1b.2i3c.4d.5e.u2q/e3w4r5t6y7u8i9o0p1q2', '李助教', 'assistant01@example.com', '10000000002', 'Assistant'),
(4, 'student01', '$2a$10$g.5b.1o1b.2i3c.4d.5e.u2q/e3w4r5t6y7u8i9o0p1q2', '王小明', 'student01@example.com', '10000000003', 'Student'),
-- 额外添加的教师
(5, 'teacher02', '$2a$10$g.5b.1o1b.2i3c.4d.5e.u2q/e3w4r5t6y7u8i9o0p1q2', '刘老师', 'teacher02@example.com', NULL, 'Teacher'),
(6, 'teacher03', '$2a$10$g.5b.1o1b.2i3c.4d.5e.u2q/e3w4r5t6y7u8i9o0p1q2', '陈老师', 'teacher03@example.com', NULL, 'Teacher'),
(7, 'teacher04', '$2a$10$g.5b.1o1b.2i3c.4d.5e.u2q/e3w4r5t6y7u8i9o0p1q2', '赵老师', 'teacher04@example.com', NULL, 'Teacher'),
-- 额外添加的学生 (代表)
(8, 'student002', '$2a$10$g.5b.1o1b.2i3c.4d.5e.u2q/e3w4r5t6y7u8i9o0p1q2', '学生002', 'student002@example.com', NULL, 'Student'),
(9, 'student003', '$2a$10$g.5b.1o1b.2i3c.4d.5e.u2q/e3w4r5t6y7u8i9o0p1q2', '学生003', 'student003@example.com', NULL, 'Student'),
(10, 'student004', '$2a$10$g.5b.1o1b.2i3c.4d.5e.u2q/e3w4r5t6y7u8i9o0p1q2', '学生004', 'student004@example.com', NULL, 'Student'),
(11, 'student005', '$2a$10$g.5b.1o1b.2i3c.4d.5e.u2q/e3w4r5t6y7u8i9o0p1q2', '学生005', 'student005@example.com', NULL, 'Student'),
(12, 'student006', '$2a$10$g.5b.1o1b.2i3c.4d.5e.u2q/e3w4r5t6y7u8i9o0p1q2', '学生006', 'student006@example.com', NULL, 'Student'),
(13, 'yuxiang', '$2b$10$quYMFH1j5M4kjmiRXBdLMuozJUPI9OFBfGUk3tnitMTkqUwzdtyQK', 'Yuxiang', 'yuxiang@example.com', NULL, 'Admin'),
(14, 'zhangsan', '$2b$10$54KLAJSweUD3Hw.u.nleKeXj9TgHXVT0QypclSG3/MY8yf7DVhfkO', '张三', 'zhangsan@example.com', NULL, 'Teacher');


-- 2. 插入班级数据 (Classes)
INSERT IGNORE INTO `classes` (`class_id`, `class_name`, `class_code`, `teacher_id`) VALUES
(1, '生产决策仿真2025春季班', 'PM2025S', 2),
(2, '先进制造仿真实验班', 'AM2025A', 5),
(3, '供应链管理虚拟仿真', 'SCM2025', 6),
(4, '工业工程综合实验2025', 'IE2025', 7),
(5, '数据驱动决策模拟', 'DDS2025', 2),
(6, '智能生产系统规划', 'IPSP2025', 5),
(7, '物流与仓储管理仿真', 'LWM2025', 6),
(8, '企业资源计划(ERP)模拟', 'ERP2025', 7),
(9, '市场营销决策仿真', 'MDM2025', 2),
(10, '项目管理虚拟实践', 'PMVP2025', 5),
(11, '金融投资与风险模拟', 'FIRM2025', 6);

-- 3. 插入学生分班数据 (Enrollments)
INSERT IGNORE INTO `enrollments` (`student_id`, `class_id`) VALUES
(4, 1), -- 原始数据
(8, 1), -- 代表学生加入班级1
(9, 1), -- 代表学生加入班级1
(10, 2), -- 代表学生加入班级2
(11, 2), -- 代表学生加入班级2
(12, 3); -- 代表学生加入班级3

-- 4. 插入助教分配数据 (ClassAssistants) - 新结构
INSERT IGNORE INTO `class_assistants` (`assistant_id`, `class_id`) VALUES
(3, 1), -- 将助教1分配给班级1
(3, 2); -- 将助教1分配给班级2


-- 5. 插入题库数据 (QuestionBank)
INSERT IGNORE INTO `question_bank` (`question_id`, `knowledge_point`, `question_type`, `question_text`, `options`, `correct_answer`, `creator_id`) VALUES
(1, '需求预测', 'Single Choice', '在ARIMA模型中，"d"参数代表什么？', '{"A": "自回归项数", "B": "移动平均项数", "C": "差分阶数", "D": "季节性周期"}', 'C', 2),
(2, '生产计划', 'True/False', '安全库存的主要目的是为了应对市场需求的不确定性。', NULL, 'True', 2),
(3, '融合模型', 'Multiple Choice', '以下哪些属于融合预测模型？', '{"A": "移动平均法", "B": "Boosting融合模型", "C": "Stacking融合模型", "D": "指数平滑法"}', '["B", "C"]', 2);

-- 6. 插入默认成绩权重 (GradeWeights)
INSERT IGNORE INTO `grade_weights` (`class_id`, `experiment_flow`, `knowledge_tests`, `model_selection`, `experiment_report`, `detailed_step_weights`) VALUES
(1, 40, 20, 10, 30, '{"data_preparation": 5, "descriptive_statistics": 5, "model_choice": 10, "prediction_generation": 20}'),
(2, 35, 25, 15, 25, '{"data_preparation": 5, "descriptive_statistics": 5, "model_choice": 15, "prediction_generation": 15}');

-- 7. 插入默认实验手册和数据集 (ExperimentManuals & Datasets)
INSERT IGNORE INTO `experiment_manuals` (`manual_id`, `file_name`, `file_path`, `description`, `is_active`, `uploader_id`) VALUES
(1, '虚拟仿真实验手册V2.0.pdf', '/uploads/manuals/placeholder-manual.pdf', '这是2025年春季学期使用的最新版实验手册。', 1, 1);

INSERT IGNORE INTO `datasets` (`dataset_id`, `data_name`, `file_path`, `description`) VALUES
(1, '某产品历史销售数据', '/data/sales_data_product_a.csv', '包含2022-2024年某产品的月度销售量、销售额和平均价格等信息。');
