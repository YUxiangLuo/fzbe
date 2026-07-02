import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import path from "node:path";

const repoRoot = path.resolve(import.meta.dir, "../../..");
const migrationPath = path.join(repoRoot, "sql", "migrations", "20260702_add_academic_term_to_classes.sql");
const schemaPath = path.join(repoRoot, "sql", "schema.sql");

describe("academic term migration", () => {
  test("uses information_schema guards instead of MariaDB-only ALTER IF NOT EXISTS syntax", async () => {
    const migrationSql = await readFile(migrationPath, "utf8");

    expect(migrationSql).toContain("information_schema.COLUMNS");
    expect(migrationSql).toContain("information_schema.STATISTICS");
    expect(migrationSql).toContain("PREPARE stmt FROM @sql");
    expect(migrationSql).not.toContain("ADD COLUMN IF NOT EXISTS");
    expect(migrationSql).not.toContain("ADD INDEX IF NOT EXISTS");
  });

  test("backfills legacy rows before enforcing non-null academic term columns", async () => {
    const migrationSql = await readFile(migrationPath, "utf8");

    expect(migrationSql).toContain("SET @current_academic_year_start");
    expect(migrationSql).toContain("SET @current_semester");
    expect(migrationSql).toContain("WHERE `academic_year_start` IS NULL");
    expect(migrationSql).toContain("WHERE `semester` IS NULL");
    expect(migrationSql).toContain("MODIFY COLUMN `academic_year_start` SMALLINT UNSIGNED NOT NULL");
    expect(migrationSql).toContain("MODIFY COLUMN `semester` TINYINT UNSIGNED NOT NULL");
  });

  test("fresh schema requires explicit academic term values without hard-coded defaults", async () => {
    const schemaSql = await readFile(schemaPath, "utf8");

    expect(schemaSql).toContain("`academic_year_start` SMALLINT UNSIGNED NOT NULL,");
    expect(schemaSql).toContain("`semester` TINYINT UNSIGNED NOT NULL,");
    expect(schemaSql).not.toContain("`academic_year_start` SMALLINT UNSIGNED NOT NULL DEFAULT");
    expect(schemaSql).not.toContain("`semester` TINYINT UNSIGNED NOT NULL DEFAULT");
  });
});
