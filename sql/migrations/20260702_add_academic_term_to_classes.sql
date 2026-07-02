-- Add academic year/semester metadata to classes for teacher/assistant term filtering.
-- Safe to re-run on MariaDB.

ALTER TABLE classes
  ADD COLUMN IF NOT EXISTS academic_year_start SMALLINT UNSIGNED NOT NULL DEFAULT 2025 AFTER class_code;

ALTER TABLE classes
  ADD COLUMN IF NOT EXISTS semester TINYINT UNSIGNED NOT NULL DEFAULT 1 AFTER academic_year_start;

ALTER TABLE classes
  ADD INDEX IF NOT EXISTS idx_teacher_academic_term (teacher_id, academic_year_start, semester);
