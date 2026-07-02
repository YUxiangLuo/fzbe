-- Add academic year/semester metadata to classes for teacher/assistant term filtering.
-- Compatible with MariaDB/MySQL. Safe to re-run.

-- Use the current academic term only to backfill legacy rows during migration.
-- Future API writes must provide explicit academic_year_start/semester values.
SET @current_academic_year_start := IF(MONTH(CURDATE()) >= 9, YEAR(CURDATE()), YEAR(CURDATE()) - 1);
SET @current_semester := IF(MONTH(CURDATE()) >= 9 OR MONTH(CURDATE()) <= 1, 1, 2);

SET @column_exists := (
  SELECT COUNT(*)
    FROM information_schema.COLUMNS
   WHERE TABLE_SCHEMA = DATABASE()
     AND TABLE_NAME = 'classes'
     AND COLUMN_NAME = 'academic_year_start'
);
SET @sql := IF(
  @column_exists = 0,
  'ALTER TABLE `classes` ADD COLUMN `academic_year_start` SMALLINT UNSIGNED NULL AFTER `class_code`',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

UPDATE `classes`
   SET `academic_year_start` = @current_academic_year_start
 WHERE `academic_year_start` IS NULL;

ALTER TABLE `classes`
  MODIFY COLUMN `academic_year_start` SMALLINT UNSIGNED NOT NULL;

SET @column_exists := (
  SELECT COUNT(*)
    FROM information_schema.COLUMNS
   WHERE TABLE_SCHEMA = DATABASE()
     AND TABLE_NAME = 'classes'
     AND COLUMN_NAME = 'semester'
);
SET @sql := IF(
  @column_exists = 0,
  'ALTER TABLE `classes` ADD COLUMN `semester` TINYINT UNSIGNED NULL AFTER `academic_year_start`',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

UPDATE `classes`
   SET `semester` = @current_semester
 WHERE `semester` IS NULL;

ALTER TABLE `classes`
  MODIFY COLUMN `semester` TINYINT UNSIGNED NOT NULL;

SET @index_exists := (
  SELECT COUNT(*)
    FROM information_schema.STATISTICS
   WHERE TABLE_SCHEMA = DATABASE()
     AND TABLE_NAME = 'classes'
     AND INDEX_NAME = 'idx_teacher_academic_term'
);
SET @sql := IF(
  @index_exists = 0,
  'ALTER TABLE `classes` ADD INDEX `idx_teacher_academic_term` (`teacher_id`, `academic_year_start`, `semester`)',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
