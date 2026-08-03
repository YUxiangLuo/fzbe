process.env.DB_HOST = "127.0.0.1";
process.env.DB_USER = "test";
process.env.DB_PASSWORD = "test";
process.env.DB_DATABASE = "test";
process.env.JWT_SECRET = "test-jwt-secret";
process.env.PORT = "3001";

import { beforeEach, describe, expect, mock, test } from "bun:test";
import { Hono } from "hono";
import { sign } from "hono/jwt";

mock.restore();

const redoStatus = {
  is_rejected: false,
  has_report: false,
  redo_in_progress: true,
  experiment: {
    experiment_id: 990022,
    student_id: 20240002,
    retry_of_experiment_id: 990021,
    status: "In Progress",
  },
  rejected_report: {
    report_id: 77,
    experiment_id: 990021,
    student_id: 20240002,
    status: "rejected",
    feedback: "请完整重做",
    submitted_at: "2026-08-02T10:00:00.000Z",
    pdf_file_path: "reports/rejected.pdf",
    pdf_file_available: true,
  },
};
const getLatestReportStatus = mock(async () => redoStatus);

mock.module("../../../src/modules/reports/reports.service", () => ({
  submitReport: mock(async () => ({
    data: { message: "报告提交成功", report_id: 1, pdf_path: "reports/1.pdf" },
    isUpdate: false,
  })),
  getLatestReportStatus,
  HTMLSanitizationError: class HTMLSanitizationError extends Error {},
}));

const { default: reportsController } = await import(
  "../../../src/modules/reports/reports.controller"
);
const app = new Hono();
app.route("/", reportsController);

const authHeaders = async (role = "Student") => ({
  Authorization: `Bearer ${await sign({
    sub: role === "Student" ? 20240002 : 101,
    username: role === "Student" ? "20240002" : "teacher-test",
    role,
    exp: Math.floor(Date.now() / 1000) + 600,
  }, process.env.JWT_SECRET!)}`,
});

describe("reports controller", () => {
  beforeEach(() => getLatestReportStatus.mockClear());

  test("returns stable rejection context for a redo in progress", async () => {
    const response = await app.request("/api/v1/my-latest-report-status", {
      headers: await authHeaders(),
    });

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual(redoStatus);
    expect(getLatestReportStatus).toHaveBeenCalledWith(20240002);
  });

  test("does not expose archived rejection paths to non-students", async () => {
    const response = await app.request("/api/v1/my-latest-report-status", {
      headers: await authHeaders("Teacher"),
    });

    expect(response.status).toBe(403);
    expect(getLatestReportStatus).not.toHaveBeenCalled();
  });
});
