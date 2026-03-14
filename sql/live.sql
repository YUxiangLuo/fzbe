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

INSERT INTO `users` (`user_id`, `username`, `password_hash`, `full_name`, `email`, `phone_number`, `role`, `created_at`) VALUES
(1, 'admin', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', '系统管理员', 'admin@university.edu', '010-1234-5678', 'Admin', '2024-01-01 09:00:00');
