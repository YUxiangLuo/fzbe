import { describe, expect, test } from "bun:test";
import path from "node:path";
import {
  buildMulticlassReportFixturePath,
  isManagedMulticlassReportFixturePath,
} from "../../../scripts/e2e-seed-teacher-multiclass-fixtures";

describe("teacher multiclass report fixtures", () => {
  test("gives every report its own managed PDF path", () => {
    const templatePath = path.resolve("uploads/reports/e2e-teacher-multiclass-report.pdf");
    const firstPath = buildMulticlassReportFixturePath(
      templatePath,
      "fixture-run-1",
      20247001,
      970001,
    );
    const secondPath = buildMulticlassReportFixturePath(
      templatePath,
      "fixture-run-1",
      20247002,
      970002,
    );

    expect(firstPath).not.toBe(secondPath);
    expect(path.dirname(firstPath)).toBe(path.dirname(templatePath));
    expect(firstPath.endsWith(".pdf")).toBe(true);
    expect(isManagedMulticlassReportFixturePath(firstPath, templatePath)).toBe(true);
    expect(isManagedMulticlassReportFixturePath(templatePath, templatePath)).toBe(false);
    expect(isManagedMulticlassReportFixturePath(
      path.join(path.dirname(templatePath), "student-upload.pdf"),
      templatePath,
    )).toBe(false);
    expect(isManagedMulticlassReportFixturePath(
      path.resolve("somewhere-else/e2e-teacher-multiclass-report-fixture-run-1-s1-e1.pdf"),
      templatePath,
    )).toBe(false);
  });

  test("rejects unsafe path tokens and invalid ids", () => {
    const templatePath = path.resolve("uploads/reports/template.pdf");
    expect(() => buildMulticlassReportFixturePath(templatePath, "../escape", 1, 2))
      .toThrow("unsafe characters");
    expect(() => buildMulticlassReportFixturePath(templatePath, "safe", 0, 2))
      .toThrow("positive integer");
  });
});
