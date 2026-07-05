import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { mkdir, readdir, rm } from "node:fs/promises";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const manualsServiceModulePath = path.join(srcRoot, "modules", "manuals", "manuals.service.ts");
const freshManualsServiceModulePath = `${manualsServiceModulePath}?manuals-service-test`;
const manualsDir = path.resolve(srcRoot, "../uploads/manuals");

const poolQuery = mock(async () => [[]]);
const connectionQuery = mock(async () => [[]]);
const beginTransaction = mock(async () => {});
const commit = mock(async () => {});
const rollback = mock(async () => {});
const release = mock(() => {});
const getConnection = mock(async () => ({
  query: connectionQuery,
  beginTransaction,
  commit,
  rollback,
  release,
}));

mock.module(configModulePath, () => ({
  env: {
    DB_HOST: process.env.DB_HOST ?? "127.0.0.1",
    DB_USER: process.env.DB_USER ?? "test",
    DB_PASSWORD: process.env.DB_PASSWORD ?? "test",
    DB_DATABASE: process.env.DB_DATABASE ?? "test",
    JWT_SECRET: "test-jwt-secret",
    PORT: Number(process.env.PORT ?? "3001"),
  },
  connectionLimit: 64,
  pool: {
    query: poolQuery,
    getConnection,
  },
}));

const { uploadManual } = await import(freshManualsServiceModulePath);

describe("manuals service", () => {
  beforeEach(async () => {
    await mkdir(manualsDir, { recursive: true });
    poolQuery.mockReset();
    connectionQuery.mockReset();
    connectionQuery.mockResolvedValue([[]]);
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    release.mockClear();
    getConnection.mockClear();
  });

  afterEach(async () => {
    const entries = await readdir(manualsDir).catch(() => []);
    await Promise.all(
      entries
        .filter((entry) => entry.includes("manual-rollback-test"))
        .map((entry) => rm(path.join(manualsDir, entry), { force: true })),
    );
  });

  test("removes the uploaded PDF when database insert fails", async () => {
    const originalName = `manual-rollback-test-${Date.now()}.pdf`;
    const file = new File(
      [new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])],
      originalName,
      { type: "application/pdf" },
    );
    connectionQuery.mockRejectedValueOnce(new Error("insert failed"));

    await expect(uploadManual({
      file,
      fileName: "回滚测试",
      uploaderId: 1,
    })).rejects.toThrow("insert failed");

    const entries = await readdir(manualsDir);
    expect(entries.some((entry) => entry.includes(path.basename(originalName, ".pdf")))).toBe(false);
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledTimes(1);
  });

  test("rolls back the manual row and removes the PDF when post-insert loading fails", async () => {
    const originalName = `manual-rollback-test-${Date.now()}-select.pdf`;
    const file = new File(
      [new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])],
      originalName,
      { type: "application/pdf" },
    );
    connectionQuery
      .mockResolvedValueOnce([{ insertId: 123 }])
      .mockRejectedValueOnce(new Error("select failed"));

    await expect(uploadManual({
      file,
      fileName: "回滚测试",
      uploaderId: 1,
    })).rejects.toThrow("select failed");

    const entries = await readdir(manualsDir);
    expect(entries.some((entry) => entry.includes(path.basename(originalName, ".pdf")))).toBe(false);
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
  });
});
