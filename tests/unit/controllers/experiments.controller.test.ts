process.env.DB_HOST = "127.0.0.1";
process.env.DB_USER = "test";
process.env.DB_PASSWORD = "test";
process.env.DB_DATABASE = "test";
process.env.JWT_SECRET = "test-jwt-secret";
process.env.PORT = "3001";

import { beforeEach, describe, expect, mock, test } from "bun:test";
import { Hono } from "hono";
import { sign } from "hono/jwt";
import { ModelCommitOutcomeUnknownError } from "../../../src/modules/models/model-errors";

mock.restore();

const updateExperiment = mock(async () => ({
  error: "invalid_production_plan" as const,
  code: "PRODUCTION_FORECAST_LEGACY_REJECTED" as const,
  message: "生产预测结果为旧版格式，请重新生成并保存生产计划",
}));
const createExperiment = mock(async (): Promise<any> => ({ data: {} }));
const retryRejectedExperiment = mock(async (): Promise<any> => ({
  data: { experiment_id: 990003, retry_of_experiment_id: 990002 },
  created: true,
}));

mock.module("../../../src/modules/experiments/experiments.service", () => ({
  getActiveExperiment: mock(async () => null),
  createExperiment,
  retryRejectedExperiment,
  getAllExperiments: mock(async () => []),
  deleteExperiment: mock(async () => ({ success: true })),
  updateExperiment,
  createExperimentEvent: mock(async () => ({ data: { event_id: 1 } })),
}));

const { default: experimentsController } = await import(
  "../../../src/modules/experiments/experiments.controller"
);
const app = new Hono();
app.route("/", experimentsController);

const authHeaders = async () => ({
  Authorization: `Bearer ${await sign({
    sub: 20240002,
    username: "20240002",
    role: "Student",
    exp: Math.floor(Date.now() / 1000) + 600,
  }, process.env.JWT_SECRET!)}`,
  "Content-Type": "application/json",
});

describe("experiments controller", () => {
  beforeEach(() => {
    updateExperiment.mockClear();
    createExperiment.mockClear();
    createExperiment.mockImplementation(async () => ({ data: {} }));
    retryRejectedExperiment.mockClear();
    retryRejectedExperiment.mockImplementation(async () => ({
      data: { experiment_id: 990003, retry_of_experiment_id: 990002 },
      created: true,
    }));
  });

  test.each([
    ["/api/v1/students/me/experiment-runs", {}, createExperiment],
    ["/api/v1/students/me/experiment-runs/retry", { rejected_report_id: 77 }, retryRejectedExperiment],
  ] as const)("distinguishes an unknown creation commit outcome for %s", async (pathname, body, operation) => {
    operation.mockRejectedValueOnce(new ModelCommitOutcomeUnknownError());
    const response = await app.request(pathname, {
      method: "POST", headers: await authHeaders(), body: JSON.stringify(body),
    });
    expect(response.status).toBe(503);
    expect(await response.json()).toMatchObject({ code: "EXPERIMENT_COMMIT_OUTCOME_UNKNOWN" });
    expect(operation).toHaveBeenCalledTimes(1);
  });

  test("creates a retry from a numeric rejected report id", async () => {
    const response = await app.request("/api/v1/students/me/experiment-runs/retry", {
      method: "POST",
      headers: await authHeaders(),
      body: JSON.stringify({ rejected_report_id: 77 }),
    });

    expect(response.status).toBe(201);
    expect(await response.json()).toEqual({
      experiment_id: 990003,
      retry_of_experiment_id: 990002,
    });
    expect(retryRejectedExperiment).toHaveBeenCalledWith(20240002, {
      rejected_report_id: 77,
    });
  });

  test("returns the same retry with 200 on an idempotent repeat", async () => {
    retryRejectedExperiment.mockResolvedValueOnce({
      data: { experiment_id: 990003, retry_of_experiment_id: 990002 },
      created: false,
    });

    const response = await app.request("/api/v1/students/me/experiment-runs/retry", {
      method: "POST",
      headers: await authHeaders(),
      body: JSON.stringify({ rejected_report_id: 77 }),
    });

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      experiment_id: 990003,
      retry_of_experiment_id: 990002,
    });
  });

  test("returns the latest generation when retrying a stale rejected ancestor", async () => {
    retryRejectedExperiment.mockResolvedValueOnce({
      error: "report_not_latest",
      data: {
        experiment_id: 990004,
        retry_of_experiment_id: 990003,
        status: "Completed",
      },
    });

    const response = await app.request("/api/v1/students/me/experiment-runs/retry", {
      method: "POST",
      headers: await authHeaders(),
      body: JSON.stringify({ rejected_report_id: 77 }),
    });

    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({
      error: "该驳回报告不是当前正式实验",
      code: "REJECTED_REPORT_NOT_LATEST",
      current: {
        experiment_id: 990004,
        retry_of_experiment_id: 990003,
        status: "Completed",
      },
    });
  });

  test("returns 409 when the latest retry has already produced a report", async () => {
    retryRejectedExperiment.mockResolvedValueOnce({
      error: "report_not_latest",
      data: {
        experiment_id: 990003,
        retry_of_experiment_id: 990002,
        status: "Completed",
      },
    });

    const response = await app.request("/api/v1/students/me/experiment-runs/retry", {
      method: "POST",
      headers: await authHeaders(),
      body: JSON.stringify({ rejected_report_id: 77 }),
    });

    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({
      error: "该驳回报告不是当前正式实验",
      code: "REJECTED_REPORT_NOT_LATEST",
      current: {
        experiment_id: 990003,
        retry_of_experiment_id: 990002,
        status: "Completed",
      },
    });
  });

  test("rejects ordinary creation after an official attempt exists", async () => {
    createExperiment.mockResolvedValueOnce({
      error: "already_exists",
      data: { experiment_id: 990002, status: "Completed" },
    });

    const response = await app.request("/api/v1/students/me/experiment-runs", {
      method: "POST",
      headers: await authHeaders(),
      body: JSON.stringify({}),
    });

    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({
      error: "已经存在正式实验；被驳回后请使用重做入口",
      code: "EXPERIMENT_ALREADY_EXISTS",
      current: { experiment_id: 990002, status: "Completed" },
    });
  });

  test("preserves the legacy forecast rejection code in the update response", async () => {
    const response = await app.request("/api/v1/experiment-runs/990002", {
      method: "PATCH",
      headers: await authHeaders(),
      body: JSON.stringify({ expected_version: 0, highest_completed_step: 1 }),
    });

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({
      error: "生产计划数据无效：生产预测结果为旧版格式，请重新生成并保存生产计划",
      code: "PRODUCTION_FORECAST_LEGACY_REJECTED",
    });
    expect(updateExperiment).toHaveBeenCalledWith(990002, 20240002, {
      expected_version: 0,
      highest_completed_step: 1,
    });
  });
});
