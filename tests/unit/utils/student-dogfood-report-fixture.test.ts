import { describe, expect, test } from "bun:test";
import {
  buildMinimalPdf,
  inspectMinimalPdf,
} from "../../../scripts/pdf-fixture";
import { buildStudentDogfoodReportPdf } from "../../../scripts/student-dogfood-report-fixture";

describe("student dogfood report PDF fixtures", () => {
  test.each([
    {
      username: "20990004",
      experimentId: 14,
      status: "submitted" as const,
      grade: null,
      outcome: "Review state: awaiting teacher grading",
    },
    {
      username: "20990005",
      experimentId: 15,
      status: "rejected" as const,
      grade: null,
      outcome: "Required action: complete experiment redo",
    },
    {
      username: "20990009",
      experimentId: 19,
      status: "graded" as const,
      grade: 92,
      outcome: "Grade: 92",
    },
  ])("builds a visible, identifiable $status report", (fixture) => {
    const pdf = buildStudentDogfoodReportPdf({
      runId: "pr-20260803",
      ...fixture,
    });
    const inspection = inspectMinimalPdf(pdf, {
      expectedTextMarkers: [
        "Fangzhen Student Dogfood Report",
        `Account: ${fixture.username}`,
        `Experiment: ${fixture.experimentId}`,
        `Status: ${fixture.status}`,
        fixture.outcome,
      ],
    });

    expect(inspection.ok).toBe(true);
    if (!inspection.ok) throw new Error(inspection.error);
    expect(inspection.objectCount).toBe(6);
    expect(inspection.visibleText).toContain(`Fixture run: pr-20260803`);
  });

  test("escapes PDF literal delimiters while preserving visible text", () => {
    const pdf = buildMinimalPdf([String.raw`Account (test): C:\fixture`]);
    const inspection = inspectMinimalPdf(pdf, {
      expectedTextMarkers: [String.raw`Account (test): C:\fixture`],
    });

    expect(inspection).toEqual({
      ok: true,
      visibleText: String.raw`Account (test): C:\fixture`,
      objectCount: 6,
    });
  });

  test("refuses to label a graded fixture without a real grade", () => {
    expect(() => buildStudentDogfoodReportPdf({
      runId: "pr-20260803",
      username: "20990009",
      experimentId: 19,
      status: "graded",
      grade: null,
    })).toThrow("requires a finite grade");
  });

  test("rejects empty, fontless, unselected-font, and stale-xref fixtures", () => {
    const empty = inspectMinimalPdf(buildMinimalPdf([]));
    expect(empty.ok).toBe(false);
    if (!empty.ok) expect(empty.error).toContain("no visible ASCII text");

    const valid = buildMinimalPdf(["Visible report fixture"]);
    const fontless = inspectMinimalPdf(
      valid.replace("/Resources << /Font << /F1 4 0 R >> >>", "/Resources << >>"),
    );
    expect(fontless.ok).toBe(false);
    if (!fontless.ok) expect(fontless.error).toContain("font resource");

    const unselectedFont = inspectMinimalPdf(valid.replace("/F1 12 Tf", "XXXXXXXXX"));
    expect(unselectedFont.ok).toBe(false);
    if (!unselectedFont.ok) expect(unselectedFont.error).toContain("no visible ASCII text");

    const staleXref = inspectMinimalPdf(
      valid.replace(/\d{10} 00000 n /, "9999999999 00000 n "),
    );
    expect(staleXref.ok).toBe(false);
    if (!staleXref.ok) expect(staleXref.error).toContain("wrong offset");
  });
});
