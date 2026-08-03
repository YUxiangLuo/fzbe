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

class DatasetFileValidationError extends Error {}
class ManualFileValidationError extends Error {}

const uploadDataset = mock(async (_file: File, _name: string, _description?: string) => ({
  data: { dataset_id: 1 },
}));
const deleteDataset = mock(async () => ({ error: "not_found" as const }));
const uploadManual = mock(async (_input: Record<string, unknown>) => ({ manual_id: 2 }));
const deleteManual = mock(async () => ({ error: "not_found" as const }));

mock.module("../../../src/modules/datasets/datasets.service", () => ({
  uploadDataset,
  getAllDatasets: mock(async () => []),
  getDatasetById: mock(async () => null),
  updateDataset: mock(async () => ({ error: "not_found" })),
  deleteDataset,
  getIndustries: mock(async () => []),
  getCompaniesByIndustry: mock(async () => []),
  getProductsByCompany: mock(async () => []),
  getProductFields: mock(async () => ({ error: "not_found" })),
  getProductSales: mock(async () => ({ error: "not_found" })),
  FileValidationError: DatasetFileValidationError,
}));

mock.module("../../../src/modules/manuals/manuals.service", () => ({
  uploadManual,
  getAllManuals: mock(async () => []),
  getActiveManual: mock(async () => null),
  updateManual: mock(async () => ({ error: "not_found" })),
  deleteManual,
  FileValidationError: ManualFileValidationError,
}));

const { default: datasetsController } = await import(
  "../../../src/modules/datasets/datasets.controller"
);
const { default: manualsController } = await import(
  "../../../src/modules/manuals/manuals.controller"
);

const app = new Hono();
app.route("/", datasetsController);
app.route("/", manualsController);

const adminAuthorization = async () => `Bearer ${await sign({
  sub: 7001,
  username: "admin-test",
  role: "Admin",
  exp: Math.floor(Date.now() / 1000) + 600,
}, process.env.JWT_SECRET!)}`;

const studentAuthorization = async () => `Bearer ${await sign({
  sub: 7002,
  username: "student-test",
  role: "Student",
  exp: Math.floor(Date.now() / 1000) + 600,
}, process.env.JWT_SECRET!)}`;

const upload = async (
  endpoint: "/api/v1/datasets" | "/api/v1/manuals",
  nameField: "data_name" | "file_name",
  name: string,
  description: string | File | undefined = undefined,
) => {
  const form = new FormData();
  form.set("file", new File(["content"], endpoint.endsWith("datasets") ? "data.csv" : "manual.pdf"));
  form.set(nameField, name);
  if (description !== undefined) form.set("description", description);
  return app.request(endpoint, {
    method: "POST",
    headers: { Authorization: await adminAuthorization() },
    body: form,
  });
};

describe("admin content metadata controllers", () => {
  beforeEach(() => {
    uploadDataset.mockReset();
    uploadDataset.mockResolvedValue({ data: { dataset_id: 1 } });
    uploadManual.mockReset();
    uploadManual.mockResolvedValue({ manual_id: 2 });
    deleteDataset.mockReset();
    deleteDataset.mockResolvedValue({ error: "not_found" });
    deleteManual.mockReset();
    deleteManual.mockResolvedValue({ error: "not_found" });
  });

  test("trims validated dataset metadata before calling the service", async () => {
    const response = await upload("/api/v1/datasets", "data_name", "  测试数据  ", "  备注  ");

    expect(response.status).toBe(201);
    expect(await response.json()).toEqual({ dataset_id: 1 });
    expect(uploadDataset).toHaveBeenCalledTimes(1);
    expect(uploadDataset.mock.calls[0]?.slice(1)).toEqual(["测试数据", "备注"]);
  });

  test("rejects invalid dataset names and non-string descriptions before file processing", async () => {
    for (const [name, description] of [
      ["短", "备注"],
      ["x".repeat(101), "备注"],
      ["有效名称", "x".repeat(201)],
      ["有效名称", new File(["not text"], "description.txt")],
    ] as const) {
      const response = await upload("/api/v1/datasets", "data_name", name, description);
      expect(response.status).toBe(400);
    }
    expect(uploadDataset).not.toHaveBeenCalled();
  });

  test("trims manual metadata and binds the authenticated uploader", async () => {
    const response = await upload("/api/v1/manuals", "file_name", "  学生手册  ", "  最新版本  ");

    expect(response.status).toBe(201);
    expect(uploadManual).toHaveBeenCalledWith(expect.objectContaining({
      fileName: "学生手册",
      description: "最新版本",
      uploaderId: 7001,
    }));
  });

  test("rejects invalid manual metadata and preserves typed validation errors", async () => {
    const tooShort = await upload("/api/v1/manuals", "file_name", "短");
    expect(tooShort.status).toBe(400);

    const binaryDescription = await upload(
      "/api/v1/manuals",
      "file_name",
      "有效手册",
      new File(["binary"], "description.bin"),
    );
    expect(binaryDescription.status).toBe(400);
    expect(uploadManual).not.toHaveBeenCalled();

    uploadManual.mockRejectedValueOnce(new ManualFileValidationError("PDF 文件格式无效"));
    const invalidFile = await upload("/api/v1/manuals", "file_name", "有效手册");
    expect(invalidFile.status).toBe(400);
    expect(await invalidFile.json()).toEqual({ error: "PDF 文件格式无效" });
  });

  test("reports dataset file staging failures without claiming deletion succeeded", async () => {
    deleteDataset.mockResolvedValueOnce({ error: "file_cleanup_failed" } as never);

    const response = await app.request("/api/v1/datasets/7", {
      method: "DELETE",
      headers: { Authorization: await adminAuthorization() },
    });

    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({ error: "数据集文件无法安全清理，数据集未删除" });
  });

  test("uses an honest response when only isolated-file purging remains", async () => {
    deleteDataset.mockResolvedValueOnce({ success: true, cleanupPending: true } as never);

    const response = await app.request("/api/v1/datasets/7", {
      method: "DELETE",
      headers: { Authorization: await adminAuthorization() },
    });

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      message: "数据集已删除；隔离文件将在后续清理",
      cleanup_pending: true,
    });
  });

  test("reports manual staging failures without claiming deletion succeeded", async () => {
    deleteManual.mockResolvedValueOnce({ error: "file_cleanup_failed" } as never);

    const response = await app.request("/api/v1/manuals/7", {
      method: "DELETE",
      headers: { Authorization: await adminAuthorization() },
    });

    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({ error: "手册文件无法安全清理，手册未删除" });
  });

  test("reports successful manual row deletion while durable purge remains pending", async () => {
    deleteManual.mockResolvedValueOnce({ success: true, cleanupPending: true } as never);

    const response = await app.request("/api/v1/manuals/7", {
      method: "DELETE",
      headers: { Authorization: await adminAuthorization() },
    });

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      message: "手册已删除；隔离文件将在后续清理",
      cleanup_pending: true,
    });
  });

  test("allows only administrators to delete experiment manuals", async () => {
    deleteManual.mockResolvedValueOnce({ success: true, cleanupPending: false } as never);

    const forbidden = await app.request("/api/v1/manuals/7", {
      method: "DELETE",
      headers: { Authorization: await studentAuthorization() },
    });

    expect(forbidden.status).toBe(403);
    expect(deleteManual).not.toHaveBeenCalled();

    const allowed = await app.request("/api/v1/manuals/7", {
      method: "DELETE",
      headers: { Authorization: await adminAuthorization() },
    });
    expect(allowed.status).toBe(200);
    expect(deleteManual).toHaveBeenCalledWith(7);
  });
});
