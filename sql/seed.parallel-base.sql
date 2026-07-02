-- --------------------------------------------------------
-- Parallel Shiyan Clean Seed
-- Dedicated fixture for multi-student concurrent experiment testing
--
-- Credentials:
--   admin      parallel_admin      / AdminParallel!123
--   teacher    parallel_teacher    / TeacherParallel!123
--   assistant  parallel_assistant  / AssistantParallel!123
--   students   20246001-20246050   / StudentParallel!123
-- --------------------------------------------------------

-- ========================================================
-- 1. Users
-- ========================================================
INSERT INTO `users` (`user_id`, `username`, `password_hash`, `full_name`, `email`, `phone_number`, `role`) VALUES
(900001, 'parallel_admin', '$2b$10$R7qJFdI8QRjRTNZJNIuCMudSLuWb2OgIM96rvrntKjcVjUpLkiiba', '并发测试管理员', 'parallel.admin@test.local', NULL, 'Admin'),
(900101, 'parallel_teacher', '$2b$10$06aGdWFJ8rGc13DByQNKvOgCDGY/ja4pKe9sY.IeoE/fseoqIwhAi', '并发测试教师', 'parallel.teacher@test.local', '13800009901', 'Teacher'),
(900201, 'parallel_assistant', '$2b$10$egwcxCPUYFbJuv0U5SgI0ugoT6mQ/YMpGJukgZxcxlAiWVinzN0/2', '并发测试助教', 'parallel.assistant@test.local', '13900009901', 'Assistant'),
(20246001, '20246001', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生01', '20246001@parallel.test.local', NULL, 'Student'),
(20246002, '20246002', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生02', '20246002@parallel.test.local', NULL, 'Student'),
(20246003, '20246003', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生03', '20246003@parallel.test.local', NULL, 'Student'),
(20246004, '20246004', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生04', '20246004@parallel.test.local', NULL, 'Student'),
(20246005, '20246005', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生05', '20246005@parallel.test.local', NULL, 'Student'),
(20246006, '20246006', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生06', '20246006@parallel.test.local', NULL, 'Student'),
(20246007, '20246007', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生07', '20246007@parallel.test.local', NULL, 'Student'),
(20246008, '20246008', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生08', '20246008@parallel.test.local', NULL, 'Student'),
(20246009, '20246009', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生09', '20246009@parallel.test.local', NULL, 'Student'),
(20246010, '20246010', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生10', '20246010@parallel.test.local', NULL, 'Student'),
(20246011, '20246011', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生11', '20246011@parallel.test.local', NULL, 'Student'),
(20246012, '20246012', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生12', '20246012@parallel.test.local', NULL, 'Student'),
(20246013, '20246013', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生13', '20246013@parallel.test.local', NULL, 'Student'),
(20246014, '20246014', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生14', '20246014@parallel.test.local', NULL, 'Student'),
(20246015, '20246015', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生15', '20246015@parallel.test.local', NULL, 'Student'),
(20246016, '20246016', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生16', '20246016@parallel.test.local', NULL, 'Student'),
(20246017, '20246017', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生17', '20246017@parallel.test.local', NULL, 'Student'),
(20246018, '20246018', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生18', '20246018@parallel.test.local', NULL, 'Student'),
(20246019, '20246019', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生19', '20246019@parallel.test.local', NULL, 'Student'),
(20246020, '20246020', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生20', '20246020@parallel.test.local', NULL, 'Student'),
(20246021, '20246021', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生21', '20246021@parallel.test.local', NULL, 'Student'),
(20246022, '20246022', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生22', '20246022@parallel.test.local', NULL, 'Student'),
(20246023, '20246023', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生23', '20246023@parallel.test.local', NULL, 'Student'),
(20246024, '20246024', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生24', '20246024@parallel.test.local', NULL, 'Student'),
(20246025, '20246025', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生25', '20246025@parallel.test.local', NULL, 'Student'),
(20246026, '20246026', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生26', '20246026@parallel.test.local', NULL, 'Student'),
(20246027, '20246027', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生27', '20246027@parallel.test.local', NULL, 'Student'),
(20246028, '20246028', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生28', '20246028@parallel.test.local', NULL, 'Student'),
(20246029, '20246029', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生29', '20246029@parallel.test.local', NULL, 'Student'),
(20246030, '20246030', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生30', '20246030@parallel.test.local', NULL, 'Student'),
(20246031, '20246031', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生31', '20246031@parallel.test.local', NULL, 'Student'),
(20246032, '20246032', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生32', '20246032@parallel.test.local', NULL, 'Student'),
(20246033, '20246033', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生33', '20246033@parallel.test.local', NULL, 'Student'),
(20246034, '20246034', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生34', '20246034@parallel.test.local', NULL, 'Student'),
(20246035, '20246035', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生35', '20246035@parallel.test.local', NULL, 'Student'),
(20246036, '20246036', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生36', '20246036@parallel.test.local', NULL, 'Student'),
(20246037, '20246037', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生37', '20246037@parallel.test.local', NULL, 'Student'),
(20246038, '20246038', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生38', '20246038@parallel.test.local', NULL, 'Student'),
(20246039, '20246039', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生39', '20246039@parallel.test.local', NULL, 'Student'),
(20246040, '20246040', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生40', '20246040@parallel.test.local', NULL, 'Student'),
(20246041, '20246041', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生41', '20246041@parallel.test.local', NULL, 'Student'),
(20246042, '20246042', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生42', '20246042@parallel.test.local', NULL, 'Student'),
(20246043, '20246043', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生43', '20246043@parallel.test.local', NULL, 'Student'),
(20246044, '20246044', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生44', '20246044@parallel.test.local', NULL, 'Student'),
(20246045, '20246045', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生45', '20246045@parallel.test.local', NULL, 'Student'),
(20246046, '20246046', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生46', '20246046@parallel.test.local', NULL, 'Student'),
(20246047, '20246047', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生47', '20246047@parallel.test.local', NULL, 'Student'),
(20246048, '20246048', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生48', '20246048@parallel.test.local', NULL, 'Student'),
(20246049, '20246049', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生49', '20246049@parallel.test.local', NULL, 'Student'),
(20246050, '20246050', '$2b$10$YBOx56IP5bmtGA8i6PxHWOI/gLZIQmBrbPvP5X1jwus4lzBuSUkIy', '并发学生50', '20246050@parallel.test.local', NULL, 'Student');

-- ========================================================
-- 2. Class / Enrollment / Grade Weights
-- ========================================================
INSERT INTO `classes` (`class_id`, `class_name`, `class_code`, `academic_year_start`, `semester`, `teacher_id`) VALUES
(8401, '并发实验测试班', 'PARALLEL-SHIYAN-2026', 2025, 1, 900101);

INSERT INTO `enrollments` (`student_id`, `class_id`) VALUES
(20246001, 8401),
(20246002, 8401),
(20246003, 8401),
(20246004, 8401),
(20246005, 8401),
(20246006, 8401),
(20246007, 8401),
(20246008, 8401),
(20246009, 8401),
(20246010, 8401),
(20246011, 8401),
(20246012, 8401),
(20246013, 8401),
(20246014, 8401),
(20246015, 8401),
(20246016, 8401),
(20246017, 8401),
(20246018, 8401),
(20246019, 8401),
(20246020, 8401),
(20246021, 8401),
(20246022, 8401),
(20246023, 8401),
(20246024, 8401),
(20246025, 8401),
(20246026, 8401),
(20246027, 8401),
(20246028, 8401),
(20246029, 8401),
(20246030, 8401),
(20246031, 8401),
(20246032, 8401),
(20246033, 8401),
(20246034, 8401),
(20246035, 8401),
(20246036, 8401),
(20246037, 8401),
(20246038, 8401),
(20246039, 8401),
(20246040, 8401),
(20246041, 8401),
(20246042, 8401),
(20246043, 8401),
(20246044, 8401),
(20246045, 8401),
(20246046, 8401),
(20246047, 8401),
(20246048, 8401),
(20246049, 8401),
(20246050, 8401);

INSERT INTO `class_assistants` (`assistant_id`, `class_id`) VALUES
(900201, 8401);

INSERT INTO `grade_weights` (
  `class_id`,
  `exp_flow_demand_data_preparation`,
  `exp_flow_demand_descriptive_stats`,
  `exp_flow_demand_model_selection`,
  `exp_flow_demand_generate_results`,
  `exp_flow_production_inventory_calc`,
  `exp_flow_production_service_level`,
  `exp_flow_production_variable_calc`,
  `exp_flow_production_plan_creation`,
  `exp_flow_report_submission`,
  `exp_flow_weight`,
  `knowledge_test_weight`,
  `model_quality_weight`,
  `report_quality_weight`
) VALUES
(8401, 5, 5, 10, 10, 10, 5, 5, 15, 35, 40, 20, 20, 20);

-- ========================================================
-- 3. Question Bank
-- ========================================================
INSERT INTO `question_bank` (`question_id`, `knowledge_point`, `question_type`, `question_text`, `options`, `correct_answers`, `creator_id`) VALUES
(1, '预测模型-时间序列基础', 'Single Choice', '以下哪项不是时间序列的经典组成部分？', '{"A":"趋势","B":"季节性","C":"周期性","D":"回归"}', '["D"]', 900001),
(2, '预测模型-时间序列基础', 'True/False', '平稳时间序列的均值和方差不随时间变化。', '["正确","错误"]', '["正确"]', 900001),
(3, '预测模型-时间序列基础', 'Single Choice', 'ADF 检验的主要用途是什么？', '{"A":"检验自相关性","B":"检验平稳性","C":"检验正态性","D":"检验样本量"}', '["B"]', 900001),
(4, '预测模型-模型评估指标', 'Multiple Choice', '以下哪些是常见的预测评估指标？', '{"A":"RMSE","B":"MAE","C":"R-squared","D":"AUC"}', '["A","B","C"]', 900001),
(5, '预测模型-指数平滑法', 'Single Choice', '在简单指数平滑中，较高的 alpha 表示什么？', '{"A":"更重视较早数据","B":"更重视近期数据","C":"预测更平滑","D":"模型无法更新"}', '["B"]', 900001),
(6, '预测模型-ARIMA模型', 'Single Choice', 'ARIMA 模型中的 d 参数表示什么？', '{"A":"自回归阶数","B":"差分阶数","C":"移动平均阶数","D":"季节周期"}', '["B"]', 900001),
(7, '预测模型-LSTM深度学习', 'True/False', 'LSTM 是一种可以缓解长期依赖问题的循环神经网络。', '["正确","错误"]', '["正确"]', 900001),
(8, '预测模型-模型融合技术', 'True/False', 'Stacking 的思想是训练一个元模型来组合多个基模型的输出。', '["正确","错误"]', '["正确"]', 900001),
(9, '生产计划-MPS基本原理', 'Single Choice', '主生产计划（MPS）的主要作用是什么？', '{"A":"规划最终产品在各时期的产出","B":"只管理原材料采购","C":"记录教师评分","D":"生成财务报表"}', '["A"]', 900001),
(10, '生产计划-安全库存策略', 'True/False', '安全库存用于缓冲需求或供应提前期的不确定性。', '["正确","错误"]', '["正确"]', 900001),
(11, '生产计划-服务水平评估', 'Single Choice', '95% 的客户服务水平通常表示什么？', '{"A":"95% 的客户满意度","B":"95% 的需求可以直接由库存满足","C":"95% 的设备开工率","D":"95% 的订单利润率"}', '["B"]', 900001),
(12, '生产计划-库存与缺货', 'Single Choice', 'EOQ 模型的目标是什么？', '{"A":"最大化库存周转率","B":"最小化缺货次数","C":"最小化订购成本与持有成本之和","D":"最大化销售额"}', '["C"]', 900001),
(13, '生产计划-产能规划', 'Single Choice', '粗能力计划（RCCP）用于验证哪项计划的可行性？', '{"A":"物料清单","B":"主生产计划","C":"工资预算","D":"销售回款计划"}', '["B"]', 900001),
(14, '生产计划-MPS基本原理', 'Single Choice', '净需求通常如何计算？', '{"A":"毛需求 - 现有库存","B":"毛需求 - （现有库存 + 计划接收量）","C":"毛需求 + 安全库存","D":"毛需求 - 安全库存"}', '["B"]', 900001),
(15, '生产计划-MPS基本原理', 'True/False', '物料清单（BOM）描述了最终产品与各层零部件之间的结构关系。', '["正确","错误"]', '["正确"]', 900001),
(16, '生产计划-安全库存策略', 'Single Choice', '再订货点通常应覆盖什么？', '{"A":"仅覆盖一天需求","B":"提前期内需求 + 安全库存","C":"仅覆盖安全库存","D":"最大库存水平"}', '["B"]', 900001);
