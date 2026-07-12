process.env.DB_HOST = "127.0.0.1";
process.env.DB_USER = "test";
process.env.DB_PASSWORD = "test";
process.env.DB_DATABASE = "test";
process.env.JWT_SECRET = "test-jwt-secret";
process.env.PORT = "3001";

import { beforeEach, describe, expect, mock, test } from "bun:test";
import { Hono } from "hono";
import { sign } from "hono/jwt";
import { AdfValidationError } from "../../../src/utils/adf";

mock.restore();

const performAdfTest = mock(async () => ([
  {
    diff_order: 0,
    statistic: -1.23,
    p_value: 0.45,
    stationary: false,
    critical_values: { "1%": -3.5, "5%": -2.9, "10%": -2.6 },
  },
]));
const acquireModelSlot = mock(async () => ({ connection: {} as never, name: "slot:1" }));
const releaseModelSlot = mock(async () => {});

mock.module("../../../src/modules/tools/tools.service", () => ({
  performAdfTest,
}));
mock.module("../../../src/modules/models/model-locks", () => ({
  acquireModelSlot,
  releaseModelSlot,
}));

const { default: toolsController } = await import("../../../src/modules/tools/tools.controller");

const app = new Hono();
app.route("/", toolsController);

const createAuthHeaders = async () => {
  const token = await sign(
    {
      sub: 20240002,
      username: "20240002",
      role: "Student",
      exp: Math.floor(Date.now() / 1000) + 60 * 10,
    },
    process.env.JWT_SECRET!,
  );

  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
};

const requestAdf = async (series: Array<{ month: string; sales: number }>) => {
  const headers = await createAuthHeaders();
  return app.request("/api/v1/tools/adf", {
    method: "POST",
    headers,
    body: JSON.stringify({ series }),
  });
};

describe("tools controller", () => {
  beforeEach(() => {
    performAdfTest.mockReset();
    acquireModelSlot.mockReset();
    releaseModelSlot.mockReset();
    acquireModelSlot.mockResolvedValue({ connection: {} as never, name: "slot:1" });
    releaseModelSlot.mockResolvedValue(undefined);
    performAdfTest.mockResolvedValue([
      {
        diff_order: 0,
        statistic: -1.23,
        p_value: 0.45,
        stationary: false,
        critical_values: { "1%": -3.5, "5%": -2.9, "10%": -2.6 },
      },
    ]);
  });

  test("maps ADF validation errors to 400 with the original message", async () => {
    performAdfTest.mockRejectedValueOnce(new AdfValidationError("数据量不足，无法进行 ADF 检测"));

    const response = await requestAdf([
      { month: "2024-01", sales: 1 },
      { month: "2024-02", sales: 1 },
      { month: "2024-03", sales: 1 },
      { month: "2024-04", sales: 1 },
    ]);
    const data = await response.json();

    expect(response.status).toBe(400);
    expect(data).toEqual({ error: "数据量不足，无法进行 ADF 检测" });
  });

  test("keeps unexpected ADF failures as 500", async () => {
    performAdfTest.mockRejectedValueOnce(new Error("python crashed"));

    const response = await requestAdf([
      { month: "2024-01", sales: 1 },
      { month: "2024-02", sales: 2 },
      { month: "2024-03", sales: 3 },
      { month: "2024-04", sales: 4 },
    ]);
    const data = await response.json();

    expect(response.status).toBe(500);
    expect(data).toEqual({ error: "ADF 检测失败" });
  });

  test("returns 429 before starting Python when all model slots are busy", async () => {
    acquireModelSlot.mockResolvedValueOnce(null);

    const response = await requestAdf([
      { month: "2024-01", sales: 1 },
      { month: "2024-02", sales: 2 },
      { month: "2024-03", sales: 3 },
      { month: "2024-04", sales: 4 },
    ]);

    expect(response.status).toBe(429);
    expect(await response.json()).toEqual({ error: "模型服务繁忙，请稍后再试" });
    expect(performAdfTest).not.toHaveBeenCalled();
    expect(releaseModelSlot).toHaveBeenCalledWith(null);
  });

  test("rejects oversized series before acquiring a model slot", async () => {
    const series = Array.from({ length: 5001 }, (_, index) => ({
      month: `2024-${index + 1}`,
      sales: index,
    }));

    const response = await requestAdf(series);

    expect(response.status).toBe(400);
    expect(acquireModelSlot).not.toHaveBeenCalled();
    expect(performAdfTest).not.toHaveBeenCalled();
  });
});
