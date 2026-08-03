process.env.DB_HOST = "127.0.0.1";
process.env.DB_USER = "test";
process.env.DB_PASSWORD = "test";
process.env.DB_DATABASE = "test";
process.env.JWT_SECRET = "test-jwt-secret";
process.env.PORT = "3001";

import { beforeEach, describe, expect, mock, test } from "bun:test";
import { Hono } from "hono";
import { sign } from "hono/jwt";

const VALID_REVIEW_REVISION = "a".repeat(64);

mock.restore();

const gradeReport = mock(async () => ({
  error: "status_conflict" as const,
  current_status: "graded",
}));

mock.module("../../../src/modules/class-grades/class-grades.service", () => ({
  checkClassAccess: mock(async () => ({ allowed: true })),
  getClassReports: mock(async () => []),
  gradeReport,
  getClassExperimentRuns: mock(async () => []),
  getGradingPolicy: mock(async () => null),
  updateGradingPolicy: mock(async () => ({ error: "not_found" })),
  getGradeSummaries: mock(async () => ({ data: [] })),
  getTeacherClassGradeSummaries: mock(async () => ({ data: [] })),
  getAssistantClassGradeSummaries: mock(async () => ({ data: [] })),
  exportReportArchive: mock(async () => ({ error: "no_reports" })),
  exportGradeCsv: mock(async () => ({ error: "no_students" })),
  exportReportCsv: mock(async () => ({ error: "no_students" })),
  getExperimentEvents: mock(async () => []),
}));

const { default: classGradesController } = await import(
  "../../../src/modules/class-grades/class-grades.controller"
);
const app = new Hono();
app.route("/", classGradesController);

const authHeaders = async () => ({
  Authorization: `Bearer ${await sign({
    sub: 99,
    username: "teacher-test",
    role: "Teacher",
    exp: Math.floor(Date.now() / 1000) + 600,
  }, process.env.JWT_SECRET!)}`,
  "Content-Type": "application/json",
});

describe("class grades controller", () => {
  beforeEach(() => gradeReport.mockClear());

  test("maps a terminal report state conflict to HTTP 409", async () => {
    const response = await app.request("/api/v1/reports/9", {
      method: "PUT",
      headers: await authHeaders(),
      body: JSON.stringify({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        status: "rejected",
        feedback: "请完整重做实验",
      }),
    });

    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({
      error: "报告评阅内容已变化，请刷新后重试",
      current_status: "graded",
    });
    expect(gradeReport).toHaveBeenCalledWith(9, 99, "Teacher", {
      expected_status: "submitted",
      expected_review_revision: VALID_REVIEW_REVISION,
      status: "rejected",
      feedback: "请完整重做实验",
    });
  });

  test("requires the status observed by the reviewer before calling the service", async () => {
    const response = await app.request("/api/v1/reports/9", {
      method: "PUT",
      headers: await authHeaders(),
      body: JSON.stringify({
        expected_review_revision: VALID_REVIEW_REVISION,
        status: "graded",
        grade: 90,
      }),
    });

    expect(response.status).toBe(400);
    expect(gradeReport).not.toHaveBeenCalled();
  });

  test("requires the review revision observed by the reviewer", async () => {
    const response = await app.request("/api/v1/reports/9", {
      method: "PUT",
      headers: await authHeaders(),
      body: JSON.stringify({
        expected_status: "graded",
        feedback: "复核评语",
      }),
    });

    expect(response.status).toBe(400);
    expect(gradeReport).not.toHaveBeenCalled();
  });

  test("normalizes blank feedback to null before calling the service", async () => {
    const response = await app.request("/api/v1/reports/9", {
      method: "PUT",
      headers: await authHeaders(),
      body: JSON.stringify({
        expected_status: "graded",
        expected_review_revision: VALID_REVIEW_REVISION,
        feedback: "   ",
      }),
    });

    expect(response.status).toBe(409);
    expect(gradeReport).toHaveBeenCalledWith(9, 99, "Teacher", {
      expected_status: "graded",
      expected_review_revision: VALID_REVIEW_REVISION,
      feedback: null,
    });
  });

  test("rejects an ambiguous rejection payload before calling the service", async () => {
    const response = await app.request("/api/v1/reports/9", {
      method: "PUT",
      headers: await authHeaders(),
      body: JSON.stringify({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        status: "rejected",
        feedback: "请完整重做实验",
        grade: 0,
        experiment_grade: { model_quality: 0 },
      }),
    });

    expect(response.status).toBe(400);
    expect(gradeReport).not.toHaveBeenCalled();
  });
});
