import { expect, test } from "bun:test";
import path from "node:path";
import { upgradeTeacherPresentationSql } from "../../../scripts/upgrade-teacher-presentation-plans";

test("teacher presentation SQL contains only canonical completed production plans", async () => {
  const sqlPath = path.resolve(
    import.meta.dir,
    "../../../sql/teacher1_presentation_data.sql",
  );
  const source = await Bun.file(sqlPath).text();
  const result = upgradeTeacherPresentationSql(source);

  expect(result.upgradedCount).toBeGreaterThan(0);
  expect(result.sql).toBe(source);
});
