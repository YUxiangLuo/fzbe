import { afterEach, beforeEach, describe, expect, mock, spyOn, test } from "bun:test";
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
const destroy = mock(() => {});
const getConnection = mock(async () => ({
  query: connectionQuery,
  beginTransaction,
  commit,
  rollback,
  release,
  destroy,
}));

const testEnvironment: {
  DB_HOST: string;
  DB_PORT: number;
  DB_USER: string;
  DB_PASSWORD: string;
  DB_DATABASE: string;
  E2E_ALLOW_DB_RESET: boolean | string;
  JWT_SECRET: string;
  PORT: number;
} = {
  DB_HOST: process.env.DB_HOST ?? "127.0.0.1",
  DB_PORT: 3306,
  DB_USER: process.env.DB_USER ?? "test",
  DB_PASSWORD: process.env.DB_PASSWORD ?? "test",
  DB_DATABASE: process.env.DB_DATABASE ?? "test",
  E2E_ALLOW_DB_RESET: false,
  JWT_SECRET: "test-jwt-secret",
  PORT: Number(process.env.PORT ?? "3001"),
};

mock.module(configModulePath, () => ({
  env: testEnvironment,
  connectionLimit: 64,
  pool: {
    query: poolQuery,
    getConnection,
  },
}));

const { uploadManual } = await import(freshManualsServiceModulePath);

describe("manuals service", () => {
  beforeEach(async () => {
    testEnvironment.DB_DATABASE = process.env.DB_DATABASE ?? "test";
    testEnvironment.E2E_ALLOW_DB_RESET = false;
    await mkdir(manualsDir, { recursive: true });
    poolQuery.mockReset();
    connectionQuery.mockReset();
    connectionQuery.mockResolvedValue([[]]);
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    release.mockClear();
    destroy.mockClear();
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

  test("removes a partially written PDF when Bun.write fails before database ownership", async () => {
    const originalName = `manual-rollback-test-${Date.now()}-partial-write.pdf`;
    const file = new File(
      [new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])],
      originalName,
      { type: "application/pdf" },
    );
    const originalWrite = Bun.write.bind(Bun);
    const writeSpy = spyOn(Bun, "write").mockImplementationOnce(async (destination) => {
      await originalWrite(destination, "%PDF-partial");
      throw new Error("simulated partial write failure");
    });

    try {
      await expect(uploadManual({
        file,
        fileName: "部分写入失败",
        uploaderId: 1,
      })).rejects.toThrow("simulated partial write failure");
    } finally {
      writeSpy.mockRestore();
    }

    const entries = await readdir(manualsDir);
    expect(entries.some((entry) => entry.includes(path.basename(originalName, ".pdf")))).toBe(false);
    expect(getConnection).not.toHaveBeenCalled();
  });

  test("removes the PDF when no database connection can be acquired", async () => {
    const originalName = `manual-rollback-test-${Date.now()}-connection.pdf`;
    const file = new File(
      [new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])],
      originalName,
      { type: "application/pdf" },
    );
    getConnection.mockRejectedValueOnce(new Error("pool unavailable"));

    await expect(uploadManual({
      file,
      fileName: "连接失败",
      uploaderId: 1,
    })).rejects.toThrow("pool unavailable");

    const entries = await readdir(manualsDir);
    expect(entries.some((entry) => entry.includes(path.basename(originalName, ".pdf")))).toBe(false);
    expect(beginTransaction).not.toHaveBeenCalled();
  });

  test("destroys a connection whose pre-commit rollback fails", async () => {
    const originalName = `manual-rollback-test-${Date.now()}-rollback-failed.pdf`;
    const file = new File(
      [new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])],
      originalName,
      { type: "application/pdf" },
    );
    connectionQuery.mockRejectedValueOnce(new Error("insert failed"));
    rollback.mockRejectedValueOnce(new Error("rollback failed"));

    await expect(uploadManual({
      file,
      fileName: "回滚失败",
      uploaderId: 1,
    })).rejects.toThrow("insert failed");

    expect(destroy).toHaveBeenCalledTimes(1);
    expect(release).not.toHaveBeenCalled();
    const entries = await readdir(manualsDir);
    expect(entries.some((entry) => entry.includes(path.basename(originalName, ".pdf")))).toBe(false);
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

  test("returns the committed manual and preserves its PDF after a lost COMMIT acknowledgement", async () => {
    const originalName = `manual-rollback-test-${Date.now()}-ack-committed.pdf`;
    const file = new File(
      [new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])],
      originalName,
      { type: "application/pdf" },
    );
    let insertedPath = "";
    connectionQuery.mockImplementation(async (sql: string, params?: unknown[]) => {
      if (sql.startsWith("INSERT INTO experiment_manuals")) {
        insertedPath = String(params?.[1]);
        return [{ insertId: 123 }];
      }
      if (sql.includes("FROM experiment_manuals")) {
        return [[{
          manual_id: 123,
          file_name: "ACK 已提交",
          file_path: insertedPath,
          description: null,
          uploader_id: 1,
          is_active: 0,
        }]];
      }
      return [[]];
    });
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));

    const result = await uploadManual({
      file,
      fileName: "ACK 已提交",
      uploaderId: 1,
    });

    expect(result.file_path).toBe(`manuals/${path.basename(insertedPath)}`);
    expect(await Bun.file(insertedPath).exists()).toBe(true);
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
    expect(commit).toHaveBeenCalledTimes(2);
  });

  test("removes the orphan only after an ACK-loss probe confirms the row is absent", async () => {
    const originalName = `manual-rollback-test-${Date.now()}-ack-absent.pdf`;
    const file = new File(
      [new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])],
      originalName,
      { type: "application/pdf" },
    );
    let insertedPath = "";
    let selectCount = 0;
    connectionQuery.mockImplementation(async (sql: string, params?: unknown[]) => {
      if (sql.startsWith("INSERT INTO experiment_manuals")) {
        insertedPath = String(params?.[1]);
        return [{ insertId: 124 }];
      }
      if (sql.includes("FROM experiment_manuals")) {
        selectCount += 1;
        return selectCount === 1
          ? [[{
            manual_id: 124,
            file_name: "ACK 未提交",
            file_path: insertedPath,
            uploader_id: 1,
          }]]
          : [[]];
      }
      return [[]];
    });
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));

    await expect(uploadManual({
      file,
      fileName: "ACK 未提交",
      uploaderId: 1,
    })).rejects.toThrow("commit acknowledgement lost");

    expect(selectCount).toBe(2);
    expect(await Bun.file(insertedPath).exists()).toBe(false);
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(2);
  });

  test("retains the PDF when the ACK-loss owner probe is unavailable", async () => {
    const originalName = `manual-rollback-test-${Date.now()}-ack-unknown.pdf`;
    const file = new File(
      [new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])],
      originalName,
      { type: "application/pdf" },
    );
    let insertedPath = "";
    let selectCount = 0;
    connectionQuery.mockImplementation(async (sql: string, params?: unknown[]) => {
      if (sql.startsWith("INSERT INTO experiment_manuals")) {
        insertedPath = String(params?.[1]);
        return [{ insertId: 125 }];
      }
      if (sql.includes("FROM experiment_manuals")) {
        selectCount += 1;
        if (selectCount === 2) throw new Error("probe unavailable");
        return [[{
          manual_id: 125,
          file_name: "ACK 未知",
          file_path: insertedPath,
          uploader_id: 1,
        }]];
      }
      return [[]];
    });
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));

    await expect(uploadManual({
      file,
      fileName: "ACK 未知",
      uploaderId: 1,
    })).rejects.toThrow("commit acknowledgement lost");

    expect(await Bun.file(insertedPath).exists()).toBe(true);
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(rollback).toHaveBeenCalledTimes(1);
  });

  test("scopes disposable-database storage without changing the public manual URL", async () => {
    testEnvironment.DB_DATABASE = "fz_e2e_manual_storage";
    testEnvironment.E2E_ALLOW_DB_RESET = "1";
    const originalName = `manual-rollback-test-${Date.now()}-scoped.pdf`;
    const file = new File(
      [new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])],
      originalName,
      { type: "application/pdf" },
    );
    let insertedPath = "";
    connectionQuery.mockImplementation(async (sql: string, params?: unknown[]) => {
      if (sql.startsWith("INSERT INTO experiment_manuals")) {
        insertedPath = String(params?.[1]);
        return [{ insertId: 126 }];
      }
      if (sql.includes("FROM experiment_manuals")) {
        return [[{
          manual_id: 126,
          file_name: "隔离手册",
          file_path: insertedPath,
          uploader_id: 1,
        }]];
      }
      return [[]];
    });

    const result = await uploadManual({
      file,
      fileName: "隔离手册",
      uploaderId: 1,
    });

    expect(insertedPath.startsWith(path.join(manualsDir, ".db") + path.sep)).toBe(true);
    expect(result.file_path).toBe(`manuals/${path.basename(insertedPath)}`);
    expect(result.file_path).not.toContain(".db/");
    await rm(insertedPath, { force: true });
  });
});
