import { createHash } from "node:crypto";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { rm, writeFile } from "node:fs/promises";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const uploadsRoot = path.resolve(srcRoot, "../uploads");
const manualsRoot = path.join(uploadsRoot, "manuals");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const stagedDeletionModulePath = path.join(srcRoot, "utils", "staged-file-deletion.ts");
const serviceModulePath = `${path.join(srcRoot, "modules", "manuals", "manuals.service.ts")}?manual-delete-test`;
const TEST_DATABASE = "manual_delete_test";
const targetPath = path.join(manualsRoot, "manual-delete-target.pdf");
const alternatePath = path.join(manualsRoot, "manual-delete-reused.pdf");
const stagedEntry = {
  originalPath: targetPath,
  stagedPath: path.join(manualsRoot, ".deleting-manual-test-0"),
};

const lifecycle: string[] = [];

function createConnection(label: "primary" | "probe") {
  return {
    query: mock(async (sql: string) => {
      if (sql.includes("FROM experiment_manuals") && sql.includes("FOR UPDATE")) {
        return [[{ manual_id: 7, file_path: targetPath }]];
      }
      if (sql.startsWith("DELETE FROM experiment_manuals")) return [{ affectedRows: 1 }];
      return [[]];
    }),
    beginTransaction: mock(async () => { lifecycle.push(`${label}-begin`); }),
    commit: mock(async () => { lifecycle.push(`${label}-commit`); }),
    rollback: mock(async () => { lifecycle.push(`${label}-rollback`); }),
    release: mock(() => { lifecycle.push(`${label}-release`); }),
    destroy: mock(() => { lifecycle.push(`${label}-destroy`); }),
  };
}

let primary = createConnection("primary");
let probe = createConnection("probe");
const poolQuery = mock(async () => [[]]);
const getConnection = mock(async () => primary);

const stageManagedPathsForDeletion = mock(async () => [stagedEntry]);
const restoreStagedManagedPaths = mock(async () => { lifecycle.push("restore"); });
const purgeStagedManagedPaths = mock(async () => {
  lifecycle.push("purge");
  return [] as string[];
});
const reconcileStagedDeletionJournals = mock(async () => ({
  scannedJournals: 0,
  restoredJournals: 0,
  purgedJournals: 0,
  skippedJournals: 0,
  failedJournals: 0,
  failedJournalNames: [],
}));
mock.module(configModulePath, () => ({
  env: {
    DB_DATABASE: TEST_DATABASE,
    DB_HOST: "127.0.0.1",
    DB_PORT: 3306,
    E2E_ALLOW_DB_RESET: false,
    JWT_SECRET: "test-jwt-secret",
    PORT: 3001,
  },
  connectionLimit: 64,
  pool: {
    query: poolQuery,
    getConnection,
  },
}));

mock.module(stagedDeletionModulePath, () => ({
  stageManagedPathsForDeletion,
  restoreStagedManagedPaths,
  purgeStagedManagedPaths,
  reconcileStagedDeletionJournals,
}));

const { deleteManual, reconcileManualDeletionJournals } = await import(serviceModulePath);

function fingerprint(candidatePath: string) {
  return createHash("sha256").update(path.resolve(candidatePath)).digest("hex");
}

describe("manual deletion durable file transaction", () => {
  beforeEach(() => {
    lifecycle.length = 0;
    primary = createConnection("primary");
    probe = createConnection("probe");
    getConnection.mockReset();
    getConnection.mockImplementation(async () => primary);
    poolQuery.mockReset();
    poolQuery.mockResolvedValue([[]]);
    stageManagedPathsForDeletion.mockReset();
    stageManagedPathsForDeletion.mockResolvedValue([stagedEntry]);
    restoreStagedManagedPaths.mockReset();
    restoreStagedManagedPaths.mockImplementation(async () => { lifecycle.push("restore"); });
    purgeStagedManagedPaths.mockReset();
    purgeStagedManagedPaths.mockImplementation(async () => {
      lifecycle.push("purge");
      return [];
    });
    reconcileStagedDeletionJournals.mockReset();
    reconcileStagedDeletionJournals.mockResolvedValue({
      scannedJournals: 0,
      restoredJournals: 0,
      purgedJournals: 0,
      skippedJournals: 0,
      failedJournals: 0,
      failedJournalNames: [],
    });
  });

  afterEach(async () => {
    await rm(path.resolve(uploadsRoot, "manual-delete-outside.pdf"), { force: true });
  });

  test("does not delete the row when its file cannot be staged", async () => {
    stageManagedPathsForDeletion.mockRejectedValueOnce(new Error("EACCES"));

    const result = await deleteManual(7);

    expect(result).toEqual({ error: "file_cleanup_failed" });
    expect(primary.query.mock.calls.some(([sql]) => (
      String(sql).startsWith("DELETE FROM experiment_manuals")
    ))).toBe(false);
    expect(primary.rollback).toHaveBeenCalledTimes(1);
    expect(primary.commit).not.toHaveBeenCalled();
  });

  test("restores the staged file when the database delete fails", async () => {
    primary.query.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_manuals") && sql.includes("FOR UPDATE")) {
        return [[{ manual_id: 7, file_path: targetPath }]];
      }
      if (sql.startsWith("DELETE FROM experiment_manuals")) {
        throw new Error("database unavailable");
      }
      return [[]];
    });

    await expect(deleteManual(7)).rejects.toThrow("database unavailable");

    expect(primary.rollback).toHaveBeenCalledTimes(1);
    expect(restoreStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
    expect(primary.commit).not.toHaveBeenCalled();
  });

  test("destroys an untrusted transaction connection when rollback itself fails", async () => {
    primary.query.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_manuals") && sql.includes("FOR UPDATE")) {
        return [[{ manual_id: 7, file_path: targetPath }]];
      }
      if (sql.startsWith("DELETE FROM experiment_manuals")) {
        throw new Error("database unavailable");
      }
      return [[]];
    });
    primary.rollback.mockRejectedValueOnce(new Error("rollback unavailable"));

    await expect(deleteManual(7)).rejects.toThrow("database unavailable");

    expect(primary.destroy).toHaveBeenCalledTimes(1);
    expect(primary.release).not.toHaveBeenCalled();
    expect(restoreStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
  });

  test("does not stage a file still referenced by another manual", async () => {
    primary.query.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_manuals") && sql.includes("FOR UPDATE")) {
        return [[
          { manual_id: 7, file_path: targetPath },
          { manual_id: 8, file_path: targetPath },
        ]];
      }
      if (sql.startsWith("DELETE FROM experiment_manuals")) return [{ affectedRows: 1 }];
      return [[]];
    });
    stageManagedPathsForDeletion.mockResolvedValueOnce([]);

    const result = await deleteManual(7);

    expect(result).toEqual({ success: true, cleanupPending: false });
    expect(stageManagedPathsForDeletion.mock.calls[0]?.[0]).toEqual([]);
  });

  test("reports cleanup pending only after the row commit", async () => {
    purgeStagedManagedPaths.mockImplementationOnce(async () => {
      lifecycle.push("purge");
      return [stagedEntry.stagedPath];
    });

    const result = await deleteManual(7);

    expect(result).toEqual({ success: true, cleanupPending: true });
    expect(lifecycle.indexOf("primary-commit")).toBeLessThan(lifecycle.indexOf("purge"));
  });

  test("still reports committed success when post-commit purge raises unexpectedly", async () => {
    purgeStagedManagedPaths.mockRejectedValueOnce(new Error("filesystem unavailable"));

    const result = await deleteManual(7);

    expect(result).toEqual({ success: true, cleanupPending: true });
    expect(primary.commit).toHaveBeenCalledTimes(1);
    expect(primary.rollback).not.toHaveBeenCalled();
  });

  test("treats lost COMMIT acknowledgement as success after a locked absent-owner probe", async () => {
    primary.commit.mockImplementationOnce(async () => {
      lifecycle.push("primary-commit-attempt");
      throw new Error("commit acknowledgement lost");
    });
    probe.query.mockResolvedValue([[]]);
    getConnection
      .mockResolvedValueOnce(primary)
      .mockResolvedValueOnce(probe);

    const result = await deleteManual(7);

    expect(result).toEqual({ success: true, cleanupPending: false });
    expect(primary.destroy).toHaveBeenCalledTimes(1);
    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
    expect(lifecycle.indexOf("probe-begin")).toBeLessThan(lifecycle.indexOf("purge"));
    expect(lifecycle.indexOf("purge")).toBeLessThan(lifecycle.indexOf("probe-commit"));
  });

  test("restores and surfaces the original error when the owner still has the same canonical path", async () => {
    primary.commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    probe.query.mockResolvedValue([[{ manual_id: 7, file_path: targetPath }]]);
    getConnection
      .mockResolvedValueOnce(primary)
      .mockResolvedValueOnce(probe);

    await expect(deleteManual(7)).rejects.toThrow("commit acknowledgement lost");

    expect(primary.destroy).toHaveBeenCalledTimes(1);
    expect(restoreStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
    expect(purgeStagedManagedPaths).not.toHaveBeenCalled();
    expect(lifecycle.indexOf("restore")).toBeLessThan(lifecycle.indexOf("probe-commit"));
  });

  test("purges an old-incarnation file when the manual ID was reused with a different path", async () => {
    primary.commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    probe.query.mockResolvedValue([[{ manual_id: 7, file_path: alternatePath }]]);
    getConnection
      .mockResolvedValueOnce(primary)
      .mockResolvedValueOnce(probe);

    const result = await deleteManual(7);

    expect(result).toEqual({ success: true, cleanupPending: false });
    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
  });

  test("fails closed when a live stored path is outside managed manual storage", async () => {
    const outsidePath = path.resolve(uploadsRoot, "manual-delete-outside.pdf");
    await writeFile(outsidePath, "outside");
    primary.query.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_manuals") && sql.includes("FOR UPDATE")) {
        return [[{ manual_id: 7, file_path: outsidePath }]];
      }
      if (sql.startsWith("DELETE FROM experiment_manuals")) return [{ affectedRows: 1 }];
      return [[]];
    });

    const result = await deleteManual(7);

    expect(result).toEqual({ error: "file_cleanup_failed" });
    expect(stageManagedPathsForDeletion).not.toHaveBeenCalled();
    expect(primary.query.mock.calls.some(([sql]) => (
      String(sql).startsWith("DELETE FROM experiment_manuals")
    ))).toBe(false);
  });

  test.each([
    ["restore", [{ manual_id: 7, file_path: targetPath }]],
    ["purge", []],
    ["purge", [{ manual_id: 7, file_path: alternatePath }]],
  ] as const)("startup reconciliation chooses %s for current owner state", async (
    expectedAction,
    ownerRows,
  ) => {
    probe.query.mockResolvedValue([ownerRows]);
    getConnection.mockResolvedValue(probe);
    let actualAction: string | undefined;
    reconcileStagedDeletionJournals.mockImplementationOnce(async (_root, resolver) => {
      const resolution = await resolver({
        version: 1,
        operationId: "58d350a3-536d-4d69-81df-7ed8426e0946",
        createdAt: new Date().toISOString(),
        metadata: {
          databaseName: TEST_DATABASE,
          kind: "manual",
          entityIds: [7],
          filePathFingerprint: fingerprint(targetPath),
        },
        entries: [stagedEntry],
      });
      actualAction = typeof resolution === "string" ? resolution : resolution.action;
      lifecycle.push(`filesystem-${actualAction}`);
      if (typeof resolution !== "string") await resolution.release?.();
      return {
        scannedJournals: 1,
        restoredJournals: actualAction === "restore" ? 1 : 0,
        purgedJournals: actualAction === "purge" ? 1 : 0,
        skippedJournals: 0,
        failedJournals: 0,
        failedJournalNames: [],
      };
    });

    await reconcileManualDeletionJournals();

    expect(actualAction).toBe(expectedAction);
    expect(lifecycle.indexOf(`filesystem-${expectedAction}`)).toBeLessThan(
      lifecycle.indexOf("probe-commit"),
    );
  });
});
