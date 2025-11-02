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
(10, 'student_005', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Zhao Wu', 'zhao.wu@student.edu', '138-0005-0005', 'Student', '2024-01-15 09:00:00'),
(11, 'student_011', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 11', 'student011@student.edu', '138-0011-0011', 'Student', '2024-01-16 08:00:00'),
(12, 'student_012', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 12', 'student012@student.edu', '138-0012-0012', 'Student', '2024-01-16 09:00:00'),
(13, 'student_013', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 13', 'student013@student.edu', '138-0013-0013', 'Student', '2024-01-16 10:00:00'),
(14, 'student_014', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 14', 'student014@student.edu', '138-0014-0014', 'Student', '2024-01-16 11:00:00'),
(15, 'student_015', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 15', 'student015@student.edu', '138-0015-0015', 'Student', '2024-01-16 12:00:00'),
(16, 'student_016', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 16', 'student016@student.edu', '138-0016-0016', 'Student', '2024-01-17 08:00:00'),
(17, 'student_017', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 17', 'student017@student.edu', '138-0017-0017', 'Student', '2024-01-17 09:00:00'),
(18, 'student_018', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 18', 'student018@student.edu', '138-0018-0018', 'Student', '2024-01-17 10:00:00'),
(19, 'student_019', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 19', 'student019@student.edu', '138-0019-0019', 'Student', '2024-01-17 11:00:00'),
(20, 'student_020', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 20', 'student020@student.edu', '138-0020-0020', 'Student', '2024-01-17 12:00:00'),
(21, 'student_021', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 21', 'student021@student.edu', '138-0021-0021', 'Student', '2024-01-18 08:00:00'),
(22, 'student_022', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 22', 'student022@student.edu', '138-0022-0022', 'Student', '2024-01-18 09:00:00'),
(23, 'student_023', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 23', 'student023@student.edu', '138-0023-0023', 'Student', '2024-01-18 10:00:00'),
(24, 'student_024', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 24', 'student024@student.edu', '138-0024-0024', 'Student', '2024-01-18 11:00:00'),
(25, 'student_025', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 25', 'student025@student.edu', '138-0025-0025', 'Student', '2024-01-18 12:00:00'),
(26, 'student_026', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 26', 'student026@student.edu', '138-0026-0026', 'Student', '2024-01-19 08:00:00'),
(27, 'student_027', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 27', 'student027@student.edu', '138-0027-0027', 'Student', '2024-01-19 09:00:00'),
(28, 'student_028', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 28', 'student028@student.edu', '138-0028-0028', 'Student', '2024-01-19 10:00:00'),
(29, 'student_029', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 29', 'student029@student.edu', '138-0029-0029', 'Student', '2024-01-19 11:00:00'),
(30, 'student_030', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 30', 'student030@student.edu', '138-0030-0030', 'Student', '2024-01-19 12:00:00'),
(31, 'student_031', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 31', 'student031@student.edu', '138-0031-0031', 'Student', '2024-01-20 08:00:00'),
(32, 'student_032', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 32', 'student032@student.edu', '138-0032-0032', 'Student', '2024-01-20 09:00:00'),
(33, 'student_033', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 33', 'student033@student.edu', '138-0033-0033', 'Student', '2024-01-20 10:00:00'),
(34, 'student_034', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 34', 'student034@student.edu', '138-0034-0034', 'Student', '2024-01-20 11:00:00'),
(35, 'student_035', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 35', 'student035@student.edu', '138-0035-0035', 'Student', '2024-01-20 12:00:00'),
(36, 'student_036', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 36', 'student036@student.edu', '138-0036-0036', 'Student', '2024-01-21 08:00:00'),
(37, 'student_037', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 37', 'student037@student.edu', '138-0037-0037', 'Student', '2024-01-21 09:00:00'),
(38, 'student_038', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 38', 'student038@student.edu', '138-0038-0038', 'Student', '2024-01-21 10:00:00'),
(39, 'student_039', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 39', 'student039@student.edu', '138-0039-0039', 'Student', '2024-01-21 11:00:00'),
(40, 'student_040', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 40', 'student040@student.edu', '138-0040-0040', 'Student', '2024-01-21 12:00:00'),
(41, 'student_041', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 41', 'student041@student.edu', '138-0041-0041', 'Student', '2024-01-22 08:00:00'),
(42, 'student_042', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 42', 'student042@student.edu', '138-0042-0042', 'Student', '2024-01-22 09:00:00'),
(43, 'student_043', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 43', 'student043@student.edu', '138-0043-0043', 'Student', '2024-01-22 10:00:00'),
(44, 'student_044', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 44', 'student044@student.edu', '138-0044-0044', 'Student', '2024-01-22 11:00:00'),
(45, 'student_045', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 45', 'student045@student.edu', '138-0045-0045', 'Student', '2024-01-22 12:00:00'),
(46, 'student_046', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 46', 'student046@student.edu', '138-0046-0046', 'Student', '2024-01-23 08:00:00'),
(47, 'student_047', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 47', 'student047@student.edu', '138-0047-0047', 'Student', '2024-01-23 09:00:00'),
(48, 'student_048', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 48', 'student048@student.edu', '138-0048-0048', 'Student', '2024-01-23 10:00:00'),
(49, 'student_049', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 49', 'student049@student.edu', '138-0049-0049', 'Student', '2024-01-23 11:00:00'),
(50, 'student_050', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 50', 'student050@student.edu', '138-0050-0050', 'Student', '2024-01-23 12:00:00'),
(51, 'student_051', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 51', 'student051@student.edu', '138-0051-0051', 'Student', '2024-01-24 08:00:00'),
(52, 'student_052', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 52', 'student052@student.edu', '138-0052-0052', 'Student', '2024-01-24 09:00:00'),
(53, 'student_053', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 53', 'student053@student.edu', '138-0053-0053', 'Student', '2024-01-24 10:00:00'),
(54, 'student_054', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 54', 'student054@student.edu', '138-0054-0054', 'Student', '2024-01-24 11:00:00'),
(55, 'student_055', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 55', 'student055@student.edu', '138-0055-0055', 'Student', '2024-01-24 12:00:00'),
(56, 'student_056', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 56', 'student056@student.edu', '138-0056-0056', 'Student', '2024-01-25 08:00:00'),
(57, 'student_057', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 57', 'student057@student.edu', '138-0057-0057', 'Student', '2024-01-25 09:00:00'),
(58, 'student_058', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 58', 'student058@student.edu', '138-0058-0058', 'Student', '2024-01-25 10:00:00'),
(59, 'student_059', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 59', 'student059@student.edu', '138-0059-0059', 'Student', '2024-01-25 11:00:00'),
(60, 'student_060', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'Student 60', 'student060@student.edu', '138-0060-0060', 'Student', '2024-01-25 12:00:00');

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
(11, 1),
(12, 1),
(13, 1),
(14, 1),
(15, 1),
(16, 1),
(17, 1),
(18, 1),
(19, 1),
(20, 1),
(21, 1),
(22, 1),
(23, 1),
(24, 1),
(25, 1),
(26, 1),
(27, 1),
(28, 1),
(29, 1),
(30, 1),
(31, 1),
(32, 1),
(33, 1),
(34, 1),
(35, 1),
(36, 1),
(37, 1),
(38, 1),
(39, 1),
(40, 1),
(41, 1),
(42, 1),
(43, 1),
(44, 1),
(45, 1),
(46, 1),
(47, 1),
(48, 1),
(49, 1),
(50, 1),
(51, 1),
(52, 1),
(53, 1),
(54, 1),
(55, 1),
(56, 1),
(57, 1),
(58, 1),
(59, 1),
(60, 1),
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

-- Additional experiment status records for bulk testing
INSERT INTO `experiment_status` (
  `experiment_id`,
  `student_id`,
  `status`,
  `highest_completed_step`,
  `current_step`,
  `start_time`,
  `last_activity_at`,
  `completion_time`
) VALUES
  (1001, 11, 'Completed', 7, 7, '2024-02-01 09:00:00', '2024-02-01 11:00:00', '2024-02-01 11:00:00'),
  (1002, 12, 'Completed', 7, 7, '2024-02-01 12:00:00', '2024-02-01 14:00:00', '2024-02-01 14:00:00'),
  (1003, 13, 'Completed', 7, 7, '2024-02-01 15:00:00', '2024-02-01 17:00:00', '2024-02-01 17:00:00'),
  (1004, 14, 'Completed', 7, 7, '2024-02-01 18:00:00', '2024-02-01 20:00:00', '2024-02-01 20:00:00'),
  (1005, 15, 'Completed', 7, 7, '2024-02-01 21:00:00', '2024-02-01 23:00:00', '2024-02-01 23:00:00'),
  (1006, 16, 'Completed', 7, 7, '2024-02-02 00:00:00', '2024-02-02 02:00:00', '2024-02-02 02:00:00'),
  (1007, 17, 'Completed', 7, 7, '2024-02-02 03:00:00', '2024-02-02 05:00:00', '2024-02-02 05:00:00'),
  (1008, 18, 'Completed', 7, 7, '2024-02-02 06:00:00', '2024-02-02 08:00:00', '2024-02-02 08:00:00'),
  (1009, 19, 'Completed', 7, 7, '2024-02-02 09:00:00', '2024-02-02 11:00:00', '2024-02-02 11:00:00'),
  (1010, 20, 'Completed', 7, 7, '2024-02-02 12:00:00', '2024-02-02 14:00:00', '2024-02-02 14:00:00'),
  (1011, 21, 'Completed', 7, 7, '2024-02-02 15:00:00', '2024-02-02 17:00:00', '2024-02-02 17:00:00'),
  (1012, 22, 'Completed', 7, 7, '2024-02-02 18:00:00', '2024-02-02 20:00:00', '2024-02-02 20:00:00'),
  (1013, 23, 'Completed', 7, 7, '2024-02-02 21:00:00', '2024-02-02 23:00:00', '2024-02-02 23:00:00'),
  (1014, 24, 'Completed', 7, 7, '2024-02-03 00:00:00', '2024-02-03 02:00:00', '2024-02-03 02:00:00'),
  (1015, 25, 'Completed', 7, 7, '2024-02-03 03:00:00', '2024-02-03 05:00:00', '2024-02-03 05:00:00'),
  (1016, 26, 'Completed', 7, 7, '2024-02-03 06:00:00', '2024-02-03 08:00:00', '2024-02-03 08:00:00'),
  (1017, 27, 'Completed', 7, 7, '2024-02-03 09:00:00', '2024-02-03 11:00:00', '2024-02-03 11:00:00'),
  (1018, 28, 'Completed', 7, 7, '2024-02-03 12:00:00', '2024-02-03 14:00:00', '2024-02-03 14:00:00'),
  (1019, 29, 'Completed', 7, 7, '2024-02-03 15:00:00', '2024-02-03 17:00:00', '2024-02-03 17:00:00'),
  (1020, 30, 'Completed', 7, 7, '2024-02-03 18:00:00', '2024-02-03 20:00:00', '2024-02-03 20:00:00'),
  (1021, 31, 'Completed', 7, 7, '2024-02-03 21:00:00', '2024-02-03 23:00:00', '2024-02-03 23:00:00'),
  (1022, 32, 'Completed', 7, 7, '2024-02-04 00:00:00', '2024-02-04 02:00:00', '2024-02-04 02:00:00'),
  (1023, 33, 'Completed', 7, 7, '2024-02-04 03:00:00', '2024-02-04 05:00:00', '2024-02-04 05:00:00'),
  (1024, 34, 'Completed', 7, 7, '2024-02-04 06:00:00', '2024-02-04 08:00:00', '2024-02-04 08:00:00'),
  (1025, 35, 'Completed', 7, 7, '2024-02-04 09:00:00', '2024-02-04 11:00:00', '2024-02-04 11:00:00'),
  (1026, 36, 'Completed', 7, 7, '2024-02-04 12:00:00', '2024-02-04 14:00:00', '2024-02-04 14:00:00'),
  (1027, 37, 'Completed', 7, 7, '2024-02-04 15:00:00', '2024-02-04 17:00:00', '2024-02-04 17:00:00'),
  (1028, 38, 'Completed', 7, 7, '2024-02-04 18:00:00', '2024-02-04 20:00:00', '2024-02-04 20:00:00'),
  (1029, 39, 'Completed', 7, 7, '2024-02-04 21:00:00', '2024-02-04 23:00:00', '2024-02-04 23:00:00'),
  (1030, 40, 'Completed', 7, 7, '2024-02-05 00:00:00', '2024-02-05 02:00:00', '2024-02-05 02:00:00'),
  (1031, 41, 'Completed', 7, 7, '2024-02-05 03:00:00', '2024-02-05 05:00:00', '2024-02-05 05:00:00'),
  (1032, 42, 'Completed', 7, 7, '2024-02-05 06:00:00', '2024-02-05 08:00:00', '2024-02-05 08:00:00'),
  (1033, 43, 'Completed', 7, 7, '2024-02-05 09:00:00', '2024-02-05 11:00:00', '2024-02-05 11:00:00'),
  (1034, 44, 'Completed', 7, 7, '2024-02-05 12:00:00', '2024-02-05 14:00:00', '2024-02-05 14:00:00'),
  (1035, 45, 'Completed', 7, 7, '2024-02-05 15:00:00', '2024-02-05 17:00:00', '2024-02-05 17:00:00'),
  (1036, 46, 'Completed', 7, 7, '2024-02-05 18:00:00', '2024-02-05 20:00:00', '2024-02-05 20:00:00'),
  (1037, 47, 'Completed', 7, 7, '2024-02-05 21:00:00', '2024-02-05 23:00:00', '2024-02-05 23:00:00'),
  (1038, 48, 'Completed', 7, 7, '2024-02-06 00:00:00', '2024-02-06 02:00:00', '2024-02-06 02:00:00'),
  (1039, 49, 'Completed', 7, 7, '2024-02-06 03:00:00', '2024-02-06 05:00:00', '2024-02-06 05:00:00'),
  (1040, 50, 'Completed', 7, 7, '2024-02-06 06:00:00', '2024-02-06 08:00:00', '2024-02-06 08:00:00'),
  (1041, 51, 'Completed', 7, 7, '2024-02-06 09:00:00', '2024-02-06 11:00:00', '2024-02-06 11:00:00'),
  (1042, 52, 'Completed', 7, 7, '2024-02-06 12:00:00', '2024-02-06 14:00:00', '2024-02-06 14:00:00'),
  (1043, 53, 'Completed', 7, 7, '2024-02-06 15:00:00', '2024-02-06 17:00:00', '2024-02-06 17:00:00'),
  (1044, 54, 'Completed', 7, 7, '2024-02-06 18:00:00', '2024-02-06 20:00:00', '2024-02-06 20:00:00'),
  (1045, 55, 'Completed', 7, 7, '2024-02-06 21:00:00', '2024-02-06 23:00:00', '2024-02-06 23:00:00'),
  (1046, 56, 'Completed', 7, 7, '2024-02-07 00:00:00', '2024-02-07 02:00:00', '2024-02-07 02:00:00'),
  (1047, 57, 'Completed', 7, 7, '2024-02-07 03:00:00', '2024-02-07 05:00:00', '2024-02-07 05:00:00'),
  (1048, 58, 'Completed', 7, 7, '2024-02-07 06:00:00', '2024-02-07 08:00:00', '2024-02-07 08:00:00'),
  (1049, 59, 'Completed', 7, 7, '2024-02-07 09:00:00', '2024-02-07 11:00:00', '2024-02-07 11:00:00'),
  (1050, 60, 'Completed', 7, 7, '2024-02-07 12:00:00', '2024-02-07 14:00:00', '2024-02-07 14:00:00');

INSERT INTO `experiment_reports` (`report_id`, `experiment_id`, `student_id`, `report_content`, `pdf_file_path`, `submitted_at`, `grade`, `feedback`, `graded_by`) VALUES
(1, 1, 6, 'Complete experimental report for demand forecasting and production planning using ensemble methods...', 
  '/var/reports/student_6_experiment_1_report.pdf', 
  '2024-02-15 17:00:00', 92.0, 
  'Excellent work! Your analysis is thorough and the model selection is well-justified. Consider exploring more feature engineering techniques in future experiments.', 
  2);

-- Bulk experiment reports for automated testing
INSERT INTO `experiment_reports` (`report_id`, `experiment_id`, `student_id`, `report_content`, `pdf_file_path`, `submitted_at`, `grade`, `feedback`, `graded_by`) VALUES
  (2001, 1001, 11, 'Generated report content for student 11.', '/var/reports/student_11_experiment_1001.pdf', '2024-02-10 09:00:00', 72.0, 'Automated feedback tier 1.', 2),
  (2002, 1002, 12, 'Generated report content for student 12.', '/var/reports/student_12_experiment_1002.pdf', '2024-02-10 11:00:00', 76.0, 'Automated feedback tier 2.', 2),
  (2003, 1003, 13, 'Generated report content for student 13.', '/var/reports/student_13_experiment_1003.pdf', '2024-02-10 13:00:00', 80.0, 'Automated feedback tier 3.', 2),
  (2004, 1004, 14, 'Generated report content for student 14.', '/var/reports/student_14_experiment_1004.pdf', '2024-02-10 15:00:00', 84.0, 'Automated feedback tier 4.', 2),
  (2005, 1005, 15, 'Generated report content for student 15.', '/var/reports/student_15_experiment_1005.pdf', '2024-02-10 17:00:00', 88.0, 'Automated feedback tier 1.', 2),
  (2006, 1006, 16, 'Generated report content for student 16.', '/var/reports/student_16_experiment_1006.pdf', '2024-02-11 09:00:00', 92.0, 'Automated feedback tier 2.', 2),
  (2007, 1007, 17, 'Generated report content for student 17.', '/var/reports/student_17_experiment_1007.pdf', '2024-02-11 11:00:00', 96.0, 'Automated feedback tier 3.', 2),
  (2008, 1008, 18, 'Generated report content for student 18.', '/var/reports/student_18_experiment_1008.pdf', '2024-02-11 13:00:00', 72.0, 'Automated feedback tier 4.', 2),
  (2009, 1009, 19, 'Generated report content for student 19.', '/var/reports/student_19_experiment_1009.pdf', '2024-02-11 15:00:00', 76.0, 'Automated feedback tier 1.', 2),
  (2010, 1010, 20, 'Generated report content for student 20.', '/var/reports/student_20_experiment_1010.pdf', '2024-02-11 17:00:00', 80.0, 'Automated feedback tier 2.', 2),
  (2011, 1011, 21, 'Generated report content for student 21.', '/var/reports/student_21_experiment_1011.pdf', '2024-02-12 09:00:00', 84.0, 'Automated feedback tier 3.', 2),
  (2012, 1012, 22, 'Generated report content for student 22.', '/var/reports/student_22_experiment_1012.pdf', '2024-02-12 11:00:00', 88.0, 'Automated feedback tier 4.', 2),
  (2013, 1013, 23, 'Generated report content for student 23.', '/var/reports/student_23_experiment_1013.pdf', '2024-02-12 13:00:00', 92.0, 'Automated feedback tier 1.', 2),
  (2014, 1014, 24, 'Generated report content for student 24.', '/var/reports/student_24_experiment_1014.pdf', '2024-02-12 15:00:00', 96.0, 'Automated feedback tier 2.', 2),
  (2015, 1015, 25, 'Generated report content for student 25.', '/var/reports/student_25_experiment_1015.pdf', '2024-02-12 17:00:00', 72.0, 'Automated feedback tier 3.', 2),
  (2016, 1016, 26, 'Generated report content for student 26.', '/var/reports/student_26_experiment_1016.pdf', '2024-02-13 09:00:00', 76.0, 'Automated feedback tier 4.', 2),
  (2017, 1017, 27, 'Generated report content for student 27.', '/var/reports/student_27_experiment_1017.pdf', '2024-02-13 11:00:00', 80.0, 'Automated feedback tier 1.', 2),
  (2018, 1018, 28, 'Generated report content for student 28.', '/var/reports/student_28_experiment_1018.pdf', '2024-02-13 13:00:00', 84.0, 'Automated feedback tier 2.', 2),
  (2019, 1019, 29, 'Generated report content for student 29.', '/var/reports/student_29_experiment_1019.pdf', '2024-02-13 15:00:00', 88.0, 'Automated feedback tier 3.', 2),
  (2020, 1020, 30, 'Generated report content for student 30.', '/var/reports/student_30_experiment_1020.pdf', '2024-02-13 17:00:00', 92.0, 'Automated feedback tier 4.', 2),
  (2021, 1021, 31, 'Generated report content for student 31.', '/var/reports/student_31_experiment_1021.pdf', '2024-02-14 09:00:00', 96.0, 'Automated feedback tier 1.', 2),
  (2022, 1022, 32, 'Generated report content for student 32.', '/var/reports/student_32_experiment_1022.pdf', '2024-02-14 11:00:00', 72.0, 'Automated feedback tier 2.', 2),
  (2023, 1023, 33, 'Generated report content for student 33.', '/var/reports/student_33_experiment_1023.pdf', '2024-02-14 13:00:00', 76.0, 'Automated feedback tier 3.', 2),
  (2024, 1024, 34, 'Generated report content for student 34.', '/var/reports/student_34_experiment_1024.pdf', '2024-02-14 15:00:00', 80.0, 'Automated feedback tier 4.', 2),
  (2025, 1025, 35, 'Generated report content for student 35.', '/var/reports/student_35_experiment_1025.pdf', '2024-02-14 17:00:00', 84.0, 'Automated feedback tier 1.', 2),
  (2026, 1026, 36, 'Generated report content for student 36.', '/var/reports/student_36_experiment_1026.pdf', '2024-02-15 09:00:00', 88.0, 'Automated feedback tier 2.', 2),
  (2027, 1027, 37, 'Generated report content for student 37.', '/var/reports/student_37_experiment_1027.pdf', '2024-02-15 11:00:00', 92.0, 'Automated feedback tier 3.', 2),
  (2028, 1028, 38, 'Generated report content for student 38.', '/var/reports/student_38_experiment_1028.pdf', '2024-02-15 13:00:00', 96.0, 'Automated feedback tier 4.', 2),
  (2029, 1029, 39, 'Generated report content for student 39.', '/var/reports/student_39_experiment_1029.pdf', '2024-02-15 15:00:00', 72.0, 'Automated feedback tier 1.', 2),
  (2030, 1030, 40, 'Generated report content for student 40.', '/var/reports/student_40_experiment_1030.pdf', '2024-02-15 17:00:00', 76.0, 'Automated feedback tier 2.', 2),
  (2031, 1031, 41, 'Generated report content for student 41.', '/var/reports/student_41_experiment_1031.pdf', '2024-02-16 09:00:00', 80.0, 'Automated feedback tier 3.', 2),
  (2032, 1032, 42, 'Generated report content for student 42.', '/var/reports/student_42_experiment_1032.pdf', '2024-02-16 11:00:00', 84.0, 'Automated feedback tier 4.', 2),
  (2033, 1033, 43, 'Generated report content for student 43.', '/var/reports/student_43_experiment_1033.pdf', '2024-02-16 13:00:00', 88.0, 'Automated feedback tier 1.', 2),
  (2034, 1034, 44, 'Generated report content for student 44.', '/var/reports/student_44_experiment_1034.pdf', '2024-02-16 15:00:00', 92.0, 'Automated feedback tier 2.', 2),
  (2035, 1035, 45, 'Generated report content for student 45.', '/var/reports/student_45_experiment_1035.pdf', '2024-02-16 17:00:00', 96.0, 'Automated feedback tier 3.', 2),
  (2036, 1036, 46, 'Generated report content for student 46.', '/var/reports/student_46_experiment_1036.pdf', '2024-02-17 09:00:00', 72.0, 'Automated feedback tier 4.', 2),
  (2037, 1037, 47, 'Generated report content for student 47.', '/var/reports/student_47_experiment_1037.pdf', '2024-02-17 11:00:00', 76.0, 'Automated feedback tier 1.', 2),
  (2038, 1038, 48, 'Generated report content for student 48.', '/var/reports/student_48_experiment_1038.pdf', '2024-02-17 13:00:00', 80.0, 'Automated feedback tier 2.', 2),
  (2039, 1039, 49, 'Generated report content for student 49.', '/var/reports/student_49_experiment_1039.pdf', '2024-02-17 15:00:00', 84.0, 'Automated feedback tier 3.', 2),
  (2040, 1040, 50, 'Generated report content for student 50.', '/var/reports/student_50_experiment_1040.pdf', '2024-02-17 17:00:00', 88.0, 'Automated feedback tier 4.', 2),
  (2041, 1041, 51, 'Generated report content for student 51.', '/var/reports/student_51_experiment_1041.pdf', '2024-02-18 09:00:00', 92.0, 'Automated feedback tier 1.', 2),
  (2042, 1042, 52, 'Generated report content for student 52.', '/var/reports/student_52_experiment_1042.pdf', '2024-02-18 11:00:00', 96.0, 'Automated feedback tier 2.', 2),
  (2043, 1043, 53, 'Generated report content for student 53.', '/var/reports/student_53_experiment_1043.pdf', '2024-02-18 13:00:00', 72.0, 'Automated feedback tier 3.', 2),
  (2044, 1044, 54, 'Generated report content for student 54.', '/var/reports/student_54_experiment_1044.pdf', '2024-02-18 15:00:00', 76.0, 'Automated feedback tier 4.', 2),
  (2045, 1045, 55, 'Generated report content for student 55.', '/var/reports/student_55_experiment_1045.pdf', '2024-02-18 17:00:00', 80.0, 'Automated feedback tier 1.', 2),
  (2046, 1046, 56, 'Generated report content for student 56.', '/var/reports/student_56_experiment_1046.pdf', '2024-02-19 09:00:00', 84.0, 'Automated feedback tier 2.', 2),
  (2047, 1047, 57, 'Generated report content for student 57.', '/var/reports/student_57_experiment_1047.pdf', '2024-02-19 11:00:00', 88.0, 'Automated feedback tier 3.', 2),
  (2048, 1048, 58, 'Generated report content for student 58.', '/var/reports/student_58_experiment_1048.pdf', '2024-02-19 13:00:00', 92.0, 'Automated feedback tier 4.', 2),
  (2049, 1049, 59, 'Generated report content for student 59.', '/var/reports/student_59_experiment_1049.pdf', '2024-02-19 15:00:00', 96.0, 'Automated feedback tier 1.', 2),
  (2050, 1050, 60, 'Generated report content for student 60.', '/var/reports/student_60_experiment_1050.pdf', '2024-02-19 17:00:00', 72.0, 'Automated feedback tier 2.', 2);

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

-- Bulk experiment grade records for automated testing
INSERT INTO `experiment_grade` (
  `experiment_id`,
  `student_id`,
  `exp_flow_demand_data_preparation`,
  `exp_flow_demand_descriptive_stats`,
  `exp_flow_demand_model_selection`,
  `exp_flow_demand_generate_results`,
  `exp_flow_production_inventory_calc`,
  `exp_flow_production_service_level`,
  `exp_flow_production_variable_calc`,
  `exp_flow_production_plan_creation`,
  `exp_flow_report_submission`,
  `knowledge_test`,
  `model_quality`,
  `report_quality`,
  `created_at`
) VALUES
  (1001, 11, 78.0, 79.0, 80.0, 81.0, 77.0, 78.0, 79.0, 80.0, 81.0, 70.0, 72.0, 72.0, '2024-02-10 18:00:00'),
  (1002, 12, 81.0, 82.0, 83.0, 84.0, 80.0, 81.0, 82.0, 83.0, 84.0, 74.0, 77.0, 76.0, '2024-02-10 19:00:00'),
  (1003, 13, 84.0, 85.0, 86.0, 87.0, 83.0, 84.0, 85.0, 86.0, 87.0, 78.0, 82.0, 80.0, '2024-02-10 20:00:00'),
  (1004, 14, 87.0, 88.0, 89.0, 90.0, 86.0, 87.0, 88.0, 89.0, 90.0, 82.0, 87.0, 84.0, '2024-02-10 21:00:00'),
  (1005, 15, 90.0, 91.0, 92.0, 93.0, 89.0, 90.0, 91.0, 92.0, 93.0, 86.0, 92.0, 88.0, '2024-02-10 22:00:00'),
  (1006, 16, 78.0, 79.0, 80.0, 81.0, 77.0, 78.0, 79.0, 80.0, 81.0, 90.0, 72.0, 92.0, '2024-02-10 23:00:00'),
  (1007, 17, 81.0, 82.0, 83.0, 84.0, 80.0, 81.0, 82.0, 83.0, 84.0, 70.0, 77.0, 96.0, '2024-02-11 18:00:00'),
  (1008, 18, 84.0, 85.0, 86.0, 87.0, 83.0, 84.0, 85.0, 86.0, 87.0, 74.0, 82.0, 72.0, '2024-02-11 19:00:00'),
  (1009, 19, 87.0, 88.0, 89.0, 90.0, 86.0, 87.0, 88.0, 89.0, 90.0, 78.0, 87.0, 76.0, '2024-02-11 20:00:00'),
  (1010, 20, 90.0, 91.0, 92.0, 93.0, 89.0, 90.0, 91.0, 92.0, 93.0, 82.0, 92.0, 80.0, '2024-02-11 21:00:00'),
  (1011, 21, 78.0, 79.0, 80.0, 81.0, 77.0, 78.0, 79.0, 80.0, 81.0, 86.0, 72.0, 84.0, '2024-02-11 22:00:00'),
  (1012, 22, 81.0, 82.0, 83.0, 84.0, 80.0, 81.0, 82.0, 83.0, 84.0, 90.0, 77.0, 88.0, '2024-02-11 23:00:00'),
  (1013, 23, 84.0, 85.0, 86.0, 87.0, 83.0, 84.0, 85.0, 86.0, 87.0, 70.0, 82.0, 92.0, '2024-02-12 18:00:00'),
  (1014, 24, 87.0, 88.0, 89.0, 90.0, 86.0, 87.0, 88.0, 89.0, 90.0, 74.0, 87.0, 96.0, '2024-02-12 19:00:00'),
  (1015, 25, 90.0, 91.0, 92.0, 93.0, 89.0, 90.0, 91.0, 92.0, 93.0, 78.0, 92.0, 72.0, '2024-02-12 20:00:00'),
  (1016, 26, 78.0, 79.0, 80.0, 81.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 72.0, 76.0, '2024-02-12 21:00:00'),
  (1017, 27, 81.0, 82.0, 83.0, 84.0, 80.0, 81.0, 82.0, 83.0, 84.0, 86.0, 77.0, 80.0, '2024-02-12 22:00:00'),
  (1018, 28, 84.0, 85.0, 86.0, 87.0, 83.0, 84.0, 85.0, 86.0, 87.0, 90.0, 82.0, 84.0, '2024-02-12 23:00:00'),
  (1019, 29, 87.0, 88.0, 89.0, 90.0, 86.0, 87.0, 88.0, 89.0, 90.0, 70.0, 87.0, 88.0, '2024-02-13 18:00:00'),
  (1020, 30, 90.0, 91.0, 92.0, 93.0, 89.0, 90.0, 91.0, 92.0, 93.0, 74.0, 92.0, 92.0, '2024-02-13 19:00:00'),
  (1021, 31, 78.0, 79.0, 80.0, 81.0, 77.0, 78.0, 79.0, 80.0, 81.0, 78.0, 72.0, 96.0, '2024-02-13 20:00:00'),
  (1022, 32, 81.0, 82.0, 83.0, 84.0, 80.0, 81.0, 82.0, 83.0, 84.0, 82.0, 77.0, 72.0, '2024-02-13 21:00:00'),
  (1023, 33, 84.0, 85.0, 86.0, 87.0, 83.0, 84.0, 85.0, 86.0, 87.0, 86.0, 82.0, 76.0, '2024-02-13 22:00:00'),
  (1024, 34, 87.0, 88.0, 89.0, 90.0, 86.0, 87.0, 88.0, 89.0, 90.0, 90.0, 87.0, 80.0, '2024-02-13 23:00:00'),
  (1025, 35, 90.0, 91.0, 92.0, 93.0, 89.0, 90.0, 91.0, 92.0, 93.0, 70.0, 92.0, 84.0, '2024-02-14 18:00:00'),
  (1026, 36, 78.0, 79.0, 80.0, 81.0, 77.0, 78.0, 79.0, 80.0, 81.0, 74.0, 72.0, 88.0, '2024-02-14 19:00:00'),
  (1027, 37, 81.0, 82.0, 83.0, 84.0, 80.0, 81.0, 82.0, 83.0, 84.0, 78.0, 77.0, 92.0, '2024-02-14 20:00:00'),
  (1028, 38, 84.0, 85.0, 86.0, 87.0, 83.0, 84.0, 85.0, 86.0, 87.0, 82.0, 82.0, 96.0, '2024-02-14 21:00:00'),
  (1029, 39, 87.0, 88.0, 89.0, 90.0, 86.0, 87.0, 88.0, 89.0, 90.0, 86.0, 87.0, 72.0, '2024-02-14 22:00:00'),
  (1030, 40, 90.0, 91.0, 92.0, 93.0, 89.0, 90.0, 91.0, 92.0, 93.0, 90.0, 92.0, 76.0, '2024-02-14 23:00:00'),
  (1031, 41, 78.0, 79.0, 80.0, 81.0, 77.0, 78.0, 79.0, 80.0, 81.0, 70.0, 72.0, 80.0, '2024-02-15 18:00:00'),
  (1032, 42, 81.0, 82.0, 83.0, 84.0, 80.0, 81.0, 82.0, 83.0, 84.0, 74.0, 77.0, 84.0, '2024-02-15 19:00:00'),
  (1033, 43, 84.0, 85.0, 86.0, 87.0, 83.0, 84.0, 85.0, 86.0, 87.0, 78.0, 82.0, 88.0, '2024-02-15 20:00:00'),
  (1034, 44, 87.0, 88.0, 89.0, 90.0, 86.0, 87.0, 88.0, 89.0, 90.0, 82.0, 87.0, 92.0, '2024-02-15 21:00:00'),
  (1035, 45, 90.0, 91.0, 92.0, 93.0, 89.0, 90.0, 91.0, 92.0, 93.0, 86.0, 92.0, 96.0, '2024-02-15 22:00:00'),
  (1036, 46, 78.0, 79.0, 80.0, 81.0, 77.0, 78.0, 79.0, 80.0, 81.0, 90.0, 72.0, 72.0, '2024-02-15 23:00:00'),
  (1037, 47, 81.0, 82.0, 83.0, 84.0, 80.0, 81.0, 82.0, 83.0, 84.0, 70.0, 77.0, 76.0, '2024-02-16 18:00:00'),
  (1038, 48, 84.0, 85.0, 86.0, 87.0, 83.0, 84.0, 85.0, 86.0, 87.0, 74.0, 82.0, 80.0, '2024-02-16 19:00:00'),
  (1039, 49, 87.0, 88.0, 89.0, 90.0, 86.0, 87.0, 88.0, 89.0, 90.0, 78.0, 87.0, 84.0, '2024-02-16 20:00:00'),
  (1040, 50, 90.0, 91.0, 92.0, 93.0, 89.0, 90.0, 91.0, 92.0, 93.0, 82.0, 92.0, 88.0, '2024-02-16 21:00:00'),
  (1041, 51, 78.0, 79.0, 80.0, 81.0, 77.0, 78.0, 79.0, 80.0, 81.0, 86.0, 72.0, 92.0, '2024-02-16 22:00:00'),
  (1042, 52, 81.0, 82.0, 83.0, 84.0, 80.0, 81.0, 82.0, 83.0, 84.0, 90.0, 77.0, 96.0, '2024-02-16 23:00:00'),
  (1043, 53, 84.0, 85.0, 86.0, 87.0, 83.0, 84.0, 85.0, 86.0, 87.0, 70.0, 82.0, 72.0, '2024-02-17 18:00:00'),
  (1044, 54, 87.0, 88.0, 89.0, 90.0, 86.0, 87.0, 88.0, 89.0, 90.0, 74.0, 87.0, 76.0, '2024-02-17 19:00:00'),
  (1045, 55, 90.0, 91.0, 92.0, 93.0, 89.0, 90.0, 91.0, 92.0, 93.0, 78.0, 92.0, 80.0, '2024-02-17 20:00:00'),
  (1046, 56, 78.0, 79.0, 80.0, 81.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 72.0, 84.0, '2024-02-17 21:00:00'),
  (1047, 57, 81.0, 82.0, 83.0, 84.0, 80.0, 81.0, 82.0, 83.0, 84.0, 86.0, 77.0, 88.0, '2024-02-17 22:00:00'),
  (1048, 58, 84.0, 85.0, 86.0, 87.0, 83.0, 84.0, 85.0, 86.0, 87.0, 90.0, 82.0, 92.0, '2024-02-17 23:00:00'),
  (1049, 59, 87.0, 88.0, 89.0, 90.0, 86.0, 87.0, 88.0, 89.0, 90.0, 70.0, 87.0, 96.0, '2024-02-18 18:00:00'),
  (1050, 60, 90.0, 91.0, 92.0, 93.0, 89.0, 90.0, 91.0, 92.0, 93.0, 74.0, 92.0, 72.0, '2024-02-18 19:00:00');

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
