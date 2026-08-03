import { buildMinimalPdf } from "./pdf-fixture";

export type StudentDogfoodReportStatus = "submitted" | "rejected" | "graded";

export interface StudentDogfoodReportFixture {
  runId: string;
  username: string;
  experimentId: number;
  status: StudentDogfoodReportStatus;
  grade: number | null;
}

/** Build a small but visible report artifact for status/download dogfood flows. */
export function buildStudentDogfoodReportPdf(
  fixture: StudentDogfoodReportFixture,
): string {
  if (
    fixture.status === "graded"
    && (fixture.grade === null || !Number.isFinite(fixture.grade))
  ) {
    throw new Error("A graded student dogfood report fixture requires a finite grade.");
  }

  const reviewLines = fixture.status === "submitted"
    ? ["Review state: awaiting teacher grading"]
    : fixture.status === "rejected"
      ? [
          "Review state: rejected by teacher",
          "Required action: complete experiment redo",
        ]
      : [
          "Review state: graded",
          `Grade: ${fixture.grade}`,
        ];

  return buildMinimalPdf([
    "Fangzhen Student Dogfood Report",
    `Fixture run: ${fixture.runId}`,
    `Account: ${fixture.username}`,
    `Experiment: ${fixture.experimentId}`,
    `Status: ${fixture.status}`,
    ...reviewLines,
  ]);
}
