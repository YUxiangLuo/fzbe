import { beforeEach, describe, expect, mock, test } from "bun:test";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { sign } from "hono/jwt";

type AppLike = {
  fetch: (request: Request) => Promise<Response>;
};

const loginMock = mock(async () => ({
  success: true as const,
  token: "test-token",
  user: { role: "Student" },
}));

let ownedReportFilename: string | null = null;

const poolQueryMock = mock(async (sql?: string, values?: unknown[]) => {
  if (String(sql).includes("FROM experiment_manuals")) {
    const filename = String(values?.[0] ?? "");
    return [[{
      manual_id: 1,
      is_active: 1,
      file_path: resolve(import.meta.dir, "../../../uploads/manuals", filename),
    }]];
  }
  if (String(sql).includes("FROM enrollments e JOIN users u")) {
    return [[]];
  }
  if (String(sql).includes("FROM experiment_status")) {
    return [[{
      experiment_id: 77,
      student_id: 123,
      retry_of_experiment_id: null,
      status: "Completed",
      current_step: 7,
      start_time: "2026-08-03T00:00:00.000Z",
      last_activity_at: "2026-08-03T01:00:00.000Z",
      completion_time: "2026-08-03T01:00:00.000Z",
    }]];
  }
  if (
    ownedReportFilename
    && String(sql).includes("FROM experiment_reports")
    && values?.[0] === ownedReportFilename
  ) {
    return [[{
      student_id: 123,
      pdf_file_path: resolve(import.meta.dir, "../../../uploads/reports", ownedReportFilename),
    }]];
  }
  if (String(sql).includes("FROM experiment_reports")) {
    return [[]];
  }
  if (String(sql).includes("SELECT teacher_id FROM classes")) {
    return [[{ teacher_id: 456 }]];
  }
  if (String(sql).includes("FROM model_training_runs")) {
    return [[{
      model_type: "ma",
      parameters: {
        observability: {
          method_name: "移动平均",
          uncertainty_evidence: { sources: ["empirical"] },
        },
      },
      metrics: {},
    }]];
  }

  return [[]];
});

mock.module("../../../src/config", () => ({
  env: {
    DB_HOST: "127.0.0.1",
    DB_PORT: 3306,
    DB_USER: "test",
    DB_PASSWORD: "test",
    DB_DATABASE: "test",
    E2E_ALLOW_DB_RESET: false,
    JWT_SECRET: "test-jwt-secret",
    PORT: 3001,
  },
  connectionLimit: 64,
  pool: {
    query: poolQueryMock,
  },
}));

mock.module("../../../src/modules/auth/auth.service", () => ({
  login: loginMock,
}));

const ensureTestEnv = () => {
  process.env.DB_HOST ??= "127.0.0.1";
  process.env.DB_USER ??= "test";
  process.env.DB_PASSWORD ??= "test";
  process.env.DB_DATABASE ??= "test";
  process.env.JWT_SECRET = "test-jwt-secret";
  process.env.PORT ??= "3001";
  delete process.env.E2E_SERVER;
};

let appPromise: Promise<AppLike> | null = null;

const getApp = async (): Promise<AppLike> => {
  if (!appPromise) {
    ensureTestEnv();
    appPromise = import("../../../src/index").then((module) => module.app as AppLike);
  }

  return appPromise;
};

describe("app route smoke tests", () => {
  beforeEach(() => {
    ownedReportFilename = null;
    loginMock.mockClear();
    poolQueryMock.mockClear();
  });

  test("wraps successful JSON responses from mounted controllers in a data envelope", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/api/v1/sessions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username: "student001", password: "secret123" }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      data: {
        token: "test-token",
        user: { role: "Student" },
      },
    });
    expect(loginMock).toHaveBeenCalledTimes(1);
  });

  test("wraps completed-experiment model observability in the documented data envelope", async () => {
    const app = await getApp();
    const token = await sign(
      {
        sub: 123,
        role: "Student",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );
    const response = await app.fetch(
      new Request("http://localhost/api/v1/models/experiments/990002/observability", {
        headers: { Authorization: `Bearer ${token}` },
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      data: {
        models: {
          ma: {
            method_name: "移动平均",
            uncertainty_evidence: { sources: ["empirical"] },
          },
        },
      },
    });
  });

  test("wraps latest report status in the documented data envelope", async () => {
    const app = await getApp();
    const token = await sign(
      {
        sub: 123,
        role: "Student",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );
    const response = await app.fetch(
      new Request("http://localhost/api/v1/my-latest-report-status", {
        headers: { Authorization: `Bearer ${token}` },
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      data: {
        is_rejected: false,
        has_report: false,
        redo_in_progress: false,
        experiment: {
          experiment_id: 77,
          student_id: 123,
          retry_of_experiment_id: null,
          status: "Completed",
          current_step: 7,
          start_time: "2026-08-03T00:00:00.000Z",
          last_activity_at: "2026-08-03T01:00:00.000Z",
          completion_time: "2026-08-03T01:00:00.000Z",
        },
      },
    });
  });

  test("wraps class report arrays in the documented data envelope", async () => {
    const app = await getApp();
    const token = await sign(
      {
        sub: 456,
        role: "Teacher",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );
    const response = await app.fetch(
      new Request("http://localhost/api/v1/classes/9/reports", {
        headers: { Authorization: `Bearer ${token}` },
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ data: [] });
  });

  test("preserves non-2xx JSON errors without wrapping them in a data envelope", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/api/v1/sessions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username: "student001" }),
      }),
    );

    expect(response.status).toBe(400);
    const payload = await response.json();
    expect(payload).toHaveProperty("error");
    expect(payload).not.toHaveProperty("data");
  });

  test("keeps null self-registration JSON as a validation error", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/api/v1/users/register/Student", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: "null",
      }),
    );

    expect(response.status).toBe(400);
    const payload = await response.json();
    expect(payload).toHaveProperty("error", "验证失败");
    expect(payload).not.toHaveProperty("data");
  });

  test("returns the application-level 404 payload for unknown routes", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/does-not-exist"),
    );

    expect(response.status).toBe(404);
    await expect(response.json()).resolves.toEqual({
      error: "接口不存在",
    });
  });

  test("keeps auth guard behavior when requests enter through the real app", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/api/v1/manuals/active"),
    );

    expect(response.status).toBe(401);
    const payload = await response.json();
    expect(payload).toHaveProperty("error");
    expect(payload).not.toHaveProperty("data");
  });

  test("does not allow an administrator account to become a class teacher", async () => {
    const app = await getApp();
    const token = await sign(
      {
        sub: 1,
        role: "Admin",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );
    const body = new FormData();
    body.set("class_name", "管理员不得归属的班级");

    const response = await app.fetch(
      new Request("http://localhost/api/v1/classes", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body,
      }),
    );

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({ error: "权限不足" });
  });

  test("does not serve an orphan report file to an administrator", async () => {
    const app = await getApp();
    const token = await sign(
      {
        sub: 1,
        role: "Admin",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );

    const response = await app.fetch(
      new Request("http://localhost/reports/orphan-report.pdf", {
        headers: { Authorization: `Bearer ${token}` },
      }),
    );

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({ error: "权限不足" });
  });

  test("serves a report to an administrator only when the database still owns it", async () => {
    const app = await getApp();
    const token = await sign(
      {
        sub: 1,
        role: "Admin",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );
    ownedReportFilename = `owned_admin=report_${process.pid}_${Date.now()}.pdf`;
    const reportPath = resolve(import.meta.dir, "../../../uploads/reports", ownedReportFilename);
    await mkdir(resolve(reportPath, ".."), { recursive: true });
    await writeFile(reportPath, "%PDF-1.4 owned report");

    try {
      const response = await app.fetch(
        new Request(`http://localhost/reports/${ownedReportFilename}`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
      );

      expect(response.status).toBe(200);
      expect(await response.text()).toBe("%PDF-1.4 owned report");
      const ownershipQuery = poolQueryMock.mock.calls.find(([sql]) => (
        String(sql).includes("FROM experiment_reports")
        && String(sql).includes("pdf_file_path = ?")
      ));
      expect(ownershipQuery?.[1]).toEqual([
        ownedReportFilename,
        `%/${ownedReportFilename.replaceAll("=", "==").replaceAll("_", "=_")}`,
      ]);
    } finally {
      await rm(reportPath, { force: true });
    }
  });

  test("exposes only a minimal public health response", async () => {
    const app = await getApp();
    const response = await app.fetch(
      new Request("http://localhost/api/v1/health", {
        headers: { "X-Request-ID": "0123456789abcdef0123456789abcdef" },
      }),
    );

    expect(response.status).toBe(200);
    expect(response.headers.get("X-Request-ID"))
      .toBe("0123456789abcdef0123456789abcdef");
    await expect(response.json()).resolves.toEqual({
      data: { app: "fangzhen-be", status: "ok" },
    });
  });

  test("gives authenticated clients only the PDF timeout configuration they need", async () => {
    const app = await getApp();
    const unauthenticated = await app.fetch(
      new Request("http://localhost/api/v1/client-config"),
    );
    expect(unauthenticated.status).toBe(401);

    const token = await sign(
      {
        sub: 123,
        role: "Student",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );
    const response = await app.fetch(
      new Request("http://localhost/api/v1/client-config", {
        headers: { Authorization: `Bearer ${token}` },
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      data: {
        reportSubmission: {
          pdfQueueMs: 60_000,
          pdfGenerationMs: 30_000,
        },
      },
    });
  });

  test("restricts runtime concurrency observability to administrators", async () => {
    const app = await getApp();
    const unauthenticated = await app.fetch(
      new Request("http://localhost/api/v1/runtime-info"),
    );
    expect(unauthenticated.status).toBe(401);

    const studentToken = await sign(
      {
        sub: 123,
        role: "Student",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );
    const forbidden = await app.fetch(
      new Request("http://localhost/api/v1/runtime-info", {
        headers: { Authorization: `Bearer ${studentToken}` },
      }),
    );
    expect(forbidden.status).toBe(403);

    const adminToken = await sign(
      {
        sub: 1,
        role: "Admin",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );
    const response = await app.fetch(
      new Request("http://localhost/api/v1/runtime-info", {
        headers: { Authorization: `Bearer ${adminToken}` },
      }),
    );

    expect(response.status).toBe(200);
    const payload = await response.json() as {
      data?: {
        deploymentMode?: unknown;
        coordinationScope?: unknown;
        e2eServer?: unknown;
        concurrency?: { maxConcurrentPdfJobs?: unknown };
        timeouts?: {
          modelExecutionHardCapMs?: unknown;
          modelSingleOperationMs?: unknown;
          modelSingleOperationMaxMs?: unknown;
          modelOperationCleanupBudgetMs?: unknown;
          modelFailureRecordMs?: unknown;
          modelNginxReadMs?: unknown;
          modelBrowserRequestMs?: unknown;
          modelTrainingMs?: unknown;
          modelPredictionMs?: unknown;
          adfExecutionMs?: unknown;
          adfOperationMs?: unknown;
          adfOperationCleanupBudgetMs?: unknown;
          adfNginxReadMs?: unknown;
          adfBrowserRequestMs?: unknown;
          pdfQueueMs?: unknown;
          pdfGenerationMs?: unknown;
        };
        requests?: {
          scope?: unknown;
          activeRequests?: unknown;
          totalRequests?: unknown;
          completedRequests?: unknown;
          methods?: Record<string, unknown>;
          statusClasses?: Record<string, unknown>;
          latencyMs?: Record<string, unknown>;
        };
        modelOperations?: Record<string, unknown>;
        modelExecutionLocks?: Record<string, unknown>;
        pdfQueue?: Record<string, unknown>;
      };
    };

    expect(payload.data).toMatchObject({
      deploymentMode: "single_instance",
      coordinationScope: "process",
    });
    expect(typeof payload.data?.e2eServer).toBe("boolean");

    expect(payload.data?.pdfQueue).toEqual({
      activeJobs: 0,
      pendingJobs: 0,
      maxConcurrentJobs: payload.data?.concurrency?.maxConcurrentPdfJobs,
      queueTimeoutMs: payload.data?.timeouts?.pdfQueueMs,
      generationTimeoutMs: payload.data?.timeouts?.pdfGenerationMs,
    });
    expect(payload.data?.timeouts).toMatchObject({
      modelExecutionHardCapMs: 600_000,
      modelSingleOperationMs: 660_000,
      modelSingleOperationMaxMs: 660_000,
      modelOperationCleanupBudgetMs: 11_000,
      modelFailureRecordMs: 2_000,
      modelNginxReadMs: 675_000,
      modelBrowserRequestMs: 690_000,
      modelTrainingMs: 300_000,
      modelPredictionMs: 60_000,
      adfExecutionMs: 30_000,
      adfOperationMs: 34_000,
      adfOperationCleanupBudgetMs: 11_000,
      adfNginxReadMs: 45_000,
      adfBrowserRequestMs: 120_000,
    });
    expect(payload.data?.requests).toMatchObject({
      scope: "process",
      activeRequests: 1,
      methods: expect.objectContaining({ GET: expect.any(Number) }),
      statusClasses: expect.objectContaining({ "2xx": expect.any(Number) }),
      latencyMs: expect.objectContaining({ count: expect.any(Number) }),
    });
    expect(payload.data?.modelOperations).toMatchObject({
      admission: {
        active: 0,
        pending: 0,
      },
    });
    expect(payload.data?.modelExecutionLocks).toMatchObject({
      scope: "process",
      active: 0,
    });
  });

  test("disables Bun idle timeout only for the exact long-running model and ADF routes", async () => {
    ensureTestEnv();
    const { serverConfig } = await import("../../../src/index");
    const timeout = mock((_request: Request, _seconds: number) => undefined);
    const server = { timeout } as unknown as Bun.Server<undefined>;
    const longRunningPaths = [
      "/api/v1/models/ma/guided-training/sessions",
      "/api/v1/models/arima/guided-training/sessions/session-1/steps/fit/run",
      "/api/v1/models/lstm/production-preparation/sessions/session-1/steps/fit_model/run",
      "/api/v1/models/weighted_avg/predict?trace=1",
      "/api/v1/tools/adf",
    ];
    const ordinaryPaths = [
      "/api/v1/models/ma/guided-training/sessions/session-1",
      "/api/v1/models/ma/guided-training/sessions/session-1/steps/fit",
      "/api/v1/models/ma/prepare-production",
      "/api/v1/models/ma/predict/",
      "/api/v1/tools/adf/",
      "/api/v1/runtime-info",
    ];

    for (const path of [...longRunningPaths, ...ordinaryPaths]) {
      await serverConfig.fetch(
        new Request(`http://localhost${path}`, { method: "POST" }),
        server,
      );
    }

    expect(timeout).toHaveBeenCalledTimes(longRunningPaths.length);
    expect(timeout.mock.calls.map(([request, seconds]) => ({
      pathname: new URL(request.url).pathname,
      seconds,
    }))).toEqual([
      { pathname: "/api/v1/models/ma/guided-training/sessions", seconds: 0 },
      {
        pathname: "/api/v1/models/arima/guided-training/sessions/session-1/steps/fit/run",
        seconds: 0,
      },
      {
        pathname: "/api/v1/models/lstm/production-preparation/sessions/session-1/steps/fit_model/run",
        seconds: 0,
      },
      { pathname: "/api/v1/models/weighted_avg/predict", seconds: 0 },
      { pathname: "/api/v1/tools/adf", seconds: 0 },
    ]);
  });

  test("forbids assistants without assigned classes from reading the question bank", async () => {
    const app = await getApp();
    const token = await sign(
      {
        sub: 201,
        role: "Assistant",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );

    const response = await app.fetch(
      new Request("http://localhost/api/v1/question-bank/questions", {
        headers: { Authorization: `Bearer ${token}` },
      }),
    );

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({ error: "权限不足" });
    expect(poolQueryMock).toHaveBeenCalledWith(
      "SELECT 1 FROM class_assistants WHERE assistant_id = ? LIMIT 1",
      [201],
    );
    expect(poolQueryMock.mock.calls.some(([sql]) => String(sql).includes("FROM question_bank"))).toBe(false);
  });

  test("authenticated file responses include length metadata and support byte ranges", async () => {
    const app = await getApp();
    const uploadsDir = resolve(import.meta.dir, "../../../uploads/manuals");
    const filename = "unit-range-fixture.pdf";
    const filePath = resolve(uploadsDir, filename);
    const token = await sign(
      {
        sub: 123,
        role: "Student",
        exp: Math.floor(Date.now() / 1000) + 3600,
      },
      "test-jwt-secret",
    );

    await mkdir(uploadsDir, { recursive: true });
    await writeFile(filePath, "0123456789");

    try {
      const fullResponse = await app.fetch(
        new Request(`http://localhost/manuals/${filename}`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
      );

      expect(fullResponse.status).toBe(200);
      expect(fullResponse.headers.get("content-length")).toBe("10");
      expect(fullResponse.headers.get("accept-ranges")).toBe("bytes");
      expect(await fullResponse.text()).toBe("0123456789");

      const rangeResponse = await app.fetch(
        new Request(`http://localhost/manuals/${filename}`, {
          headers: {
            Authorization: `Bearer ${token}`,
            Range: "bytes=2-5",
          },
        }),
      );

      expect(rangeResponse.status).toBe(206);
      expect(rangeResponse.headers.get("content-length")).toBe("4");
      expect(rangeResponse.headers.get("content-range")).toBe("bytes 2-5/10");
      expect(await rangeResponse.text()).toBe("2345");

      const multiRangeResponse = await app.fetch(
        new Request(`http://localhost/manuals/${filename}`, {
          headers: {
            Authorization: `Bearer ${token}`,
            Range: "bytes=0-0,2-3",
          },
        }),
      );

      expect(multiRangeResponse.status).toBe(200);
      expect(multiRangeResponse.headers.get("content-length")).toBe("10");
      expect(multiRangeResponse.headers.get("content-range")).toBeNull();
      expect(await multiRangeResponse.text()).toBe("0123456789");
    } finally {
      await rm(filePath, { force: true });
    }
  });
});
