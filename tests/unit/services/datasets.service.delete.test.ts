import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const stagedDeletionModulePath = path.join(srcRoot, "utils", "staged-file-deletion.ts");
const serviceModulePath = `${path.join(srcRoot, "modules", "datasets", "datasets.service.ts")}?dataset-delete-test`;
const TEST_DATABASE = "dataset_delete_test";

const poolQuery = mock(async () => [[]]);

const connectionQuery = mock(async (sql: string) => {
  if (sql.startsWith("SELECT file_path")) {
    return [[{ file_path: path.resolve(srcRoot, "../uploads/datasets/source.csv") }]];
  }
  if (sql.startsWith("DELETE FROM datasets")) return [{ affectedRows: 1 }];
  return [[]];
});
const lifecycle: string[] = [];
const beginTransaction = mock(async () => {});
const commit = mock(async () => { lifecycle.push("commit"); });
const rollback = mock(async () => {});
const release = mock(() => { lifecycle.push("release"); });
const destroy = mock(() => { lifecycle.push("destroy"); });
const getConnection = mock(async () => ({
  query: connectionQuery,
  beginTransaction,
  commit,
  rollback,
  release,
  destroy,
}));

const stagedEntry = {
  originalPath: path.resolve(srcRoot, "../uploads/datasets/source.csv"),
  stagedPath: path.resolve(srcRoot, "../uploads/datasets/.deleting-test-0"),
};
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
const classifyManagedDeletionOwnership = (
  entries: Array<{ originalPath: string }>,
  references: Iterable<string>,
) => {
  const referenceSet = new Set(references);
  const matches = entries.filter((entry) => referenceSet.has(entry.originalPath)).length;
  return matches === 0 ? "none" : matches === entries.length ? "all" : "partial";
};
mock.module(configModulePath, () => ({
  env: {
    DB_DATABASE: TEST_DATABASE,
    DB_HOST: "127.0.0.1",
    DB_PORT: 3306,
    E2E_ALLOW_DB_RESET: false,
  },
  pool: {
    query: poolQuery,
    getConnection,
  },
}));

mock.module(stagedDeletionModulePath, () => ({
  classifyManagedDeletionOwnership,
  stageManagedPathsForDeletion,
  restoreStagedManagedPaths,
  purgeStagedManagedPaths,
  reconcileStagedDeletionJournals,
}));

const { deleteDataset, reconcileDatasetDeletionJournals } = await import(serviceModulePath);

describe("deleteDataset file transaction", () => {
  beforeEach(() => {
    connectionQuery.mockReset();
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.startsWith("SELECT file_path")) {
        return [[{ file_path: path.resolve(srcRoot, "../uploads/datasets/source.csv") }]];
      }
      if (sql.startsWith("DELETE FROM datasets")) return [{ affectedRows: 1 }];
      return [[]];
    });
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    release.mockClear();
    destroy.mockClear();
    getConnection.mockClear();
    stageManagedPathsForDeletion.mockReset();
    stageManagedPathsForDeletion.mockResolvedValue([stagedEntry]);
    restoreStagedManagedPaths.mockClear();
    purgeStagedManagedPaths.mockReset();
    purgeStagedManagedPaths.mockImplementation(async () => {
      lifecycle.push("purge");
      return [];
    });
    poolQuery.mockReset();
    poolQuery.mockResolvedValue([[]]);
    reconcileStagedDeletionJournals.mockReset();
    reconcileStagedDeletionJournals.mockResolvedValue({
      scannedJournals: 0,
      restoredJournals: 0,
      purgedJournals: 0,
      skippedJournals: 0,
      failedJournals: 0,
      failedJournalNames: [],
    });
    lifecycle.length = 0;
  });

  test("does not delete the database row when a file cannot be staged", async () => {
    stageManagedPathsForDeletion.mockRejectedValueOnce(new Error("EBUSY"));

    const result = await deleteDataset(7);

    expect(result).toEqual({ error: "file_cleanup_failed" });
    expect(connectionQuery.mock.calls.some(([sql]) => String(sql).startsWith("DELETE FROM datasets"))).toBe(false);
    expect(commit).not.toHaveBeenCalled();
    expect(rollback).toHaveBeenCalledTimes(1);
  });

  test("commits only after the original paths have been staged", async () => {
    const result = await deleteDataset(7);

    expect(result).toEqual({ success: true, cleanupPending: false });
    expect(stageManagedPathsForDeletion).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(purgeStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
    expect(lifecycle).toEqual(["commit", "release", "purge"]);
  });

  test("returns cleanupPending while retaining the durable journal for a retry", async () => {
    purgeStagedManagedPaths.mockImplementationOnce(async () => {
      lifecycle.push("purge");
      return [stagedEntry.stagedPath];
    });

    const result = await deleteDataset(7);

    expect(result).toEqual({ success: true, cleanupPending: true });
    expect(lifecycle).toEqual(["commit", "release", "purge"]);
  });

  test("does not turn a committed delete into an API failure when purge raises", async () => {
    purgeStagedManagedPaths.mockImplementationOnce(async () => {
      lifecycle.push("purge");
      throw new Error("filesystem unavailable");
    });

    const result = await deleteDataset(7);

    expect(result).toEqual({ success: true, cleanupPending: true });
    expect(lifecycle).toEqual(["commit", "release", "purge"]);
  });

  test("restores staged paths when the database delete fails", async () => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.startsWith("SELECT file_path")) {
        return [[{ file_path: path.resolve(srcRoot, "../uploads/datasets/source.csv") }]];
      }
      if (sql.startsWith("DELETE FROM datasets")) throw new Error("database unavailable");
      return [[]];
    });

    await expect(deleteDataset(7)).rejects.toThrow("database unavailable");
    expect(commit).not.toHaveBeenCalled();
    expect(rollback).toHaveBeenCalled();
    expect(restoreStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
  });

  test("destroys a pre-commit transaction connection when rollback also fails", async () => {
    rollback.mockRejectedValueOnce(new Error("rollback acknowledgement lost"));
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.startsWith("SELECT file_path")) {
        return [[{ file_path: stagedEntry.originalPath }]];
      }
      if (sql.startsWith("DELETE FROM datasets")) throw new Error("database unavailable");
      return [[]];
    });

    await expect(deleteDataset(7)).rejects.toThrow("database unavailable");

    expect(destroy).toHaveBeenCalledTimes(1);
    expect(release).not.toHaveBeenCalled();
    expect(restoreStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
  });

  test("treats a lost commit acknowledgement as success only after a fresh locking probe proves deletion", async () => {
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.startsWith("SELECT file_path")) {
        return [[{ file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("SELECT dataset_id, file_path FROM datasets")) return [[]];
      if (sql.startsWith("DELETE FROM datasets")) return [{ affectedRows: 1 }];
      return [[]];
    });

    const result = await deleteDataset(7);

    expect(result).toEqual({ success: true, cleanupPending: false });
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();
    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
  });

  test("keeps ACK-loss deletion success when its post-probe purge raises", async () => {
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    purgeStagedManagedPaths.mockImplementationOnce(async () => {
      lifecycle.push("purge");
      throw new Error("filesystem unavailable");
    });
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.startsWith("SELECT file_path")) {
        return [[{ file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("SELECT dataset_id, file_path FROM datasets")) return [[]];
      if (sql.startsWith("DELETE FROM datasets")) return [{ affectedRows: 1 }];
      return [[]];
    });

    const result = await deleteDataset(7);

    expect(result).toEqual({ success: true, cleanupPending: true });
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
  });

  test("restores and reports the original commit error when the fresh probe finds the same owner paths", async () => {
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.startsWith("SELECT file_path")) {
        return [[{ file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("SELECT dataset_id, file_path FROM datasets")) {
        return [[{ dataset_id: 7, file_path: stagedEntry.originalPath }]];
      }
      if (sql.startsWith("DELETE FROM datasets")) return [{ affectedRows: 1 }];
      return [[]];
    });

    await expect(deleteDataset(7)).rejects.toThrow("commit acknowledgement lost");

    expect(destroy).toHaveBeenCalledTimes(1);
    expect(restoreStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
    expect(purgeStagedManagedPaths).not.toHaveBeenCalled();
  });

  test("retains staged evidence when the post-commit owner probe itself fails", async () => {
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    getConnection
      .mockImplementationOnce(async () => ({
        query: connectionQuery,
        beginTransaction,
        commit,
        rollback,
        release,
        destroy,
      }))
      .mockRejectedValueOnce(new Error("probe unavailable"));

    await expect(deleteDataset(7)).rejects.toThrow("commit acknowledgement lost");

    expect(destroy).toHaveBeenCalledTimes(1);
    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).not.toHaveBeenCalled();
  });

  test("destroys both ambiguous and probe sessions when the probe rollback acknowledgement is lost", async () => {
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    rollback.mockRejectedValueOnce(new Error("probe rollback acknowledgement lost"));
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.startsWith("SELECT file_path")) {
        return [[{ file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("SELECT dataset_id, file_path FROM datasets")) {
        throw new Error("probe unavailable");
      }
      if (sql.startsWith("DELETE FROM datasets")) return [{ affectedRows: 1 }];
      return [[]];
    });

    await expect(deleteDataset(7)).rejects.toThrow("commit acknowledgement lost");

    expect(destroy).toHaveBeenCalledTimes(2);
    expect(release).not.toHaveBeenCalled();
    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).not.toHaveBeenCalled();
  });

  test.each([
    ["restore", [[{
      dataset_id: 7,
      file_path: path.resolve(srcRoot, "../uploads/datasets/source.csv"),
    }]]],
    ["purge", [[]]],
  ] as const)("reconciliation chooses %s from the dataset owner row", async (expectedAction, rows) => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT dataset_id, file_path FROM datasets")) return rows as never;
      return [[]];
    });
    let actualAction: string | undefined;
    reconcileStagedDeletionJournals.mockImplementationOnce(async (_root, resolver) => {
      const resolution = await resolver({
        version: 1,
        operationId: "dataset-op",
        createdAt: new Date().toISOString(),
        metadata: {
          databaseName: TEST_DATABASE,
          kind: "dataset",
          entityIds: [7],
        },
        entries: [{
          originalPath: path.resolve(srcRoot, "../uploads/datasets/source.csv"),
          stagedPath: stagedEntry.stagedPath,
        }],
      });
      actualAction = typeof resolution === "string" ? resolution : resolution.action;
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

    await reconcileDatasetDeletionJournals();

    expect(actualAction).toBe(expectedAction);
  });

  test("purges an old-incarnation journal when a reused dataset ID references different paths", async () => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT dataset_id, file_path FROM datasets")) {
        return [[{
          dataset_id: 7,
          file_path: path.resolve(srcRoot, "../uploads/datasets/reused-id.csv"),
        }]];
      }
      return [[]];
    });
    let actualAction: string | undefined;
    reconcileStagedDeletionJournals.mockImplementationOnce(async (_root, resolver) => {
      const resolution = await resolver({
        version: 1,
        operationId: "dataset-aba-op",
        createdAt: new Date().toISOString(),
        metadata: { databaseName: TEST_DATABASE, kind: "dataset", entityIds: [7] },
        entries: [{
          originalPath: stagedEntry.originalPath,
          stagedPath: stagedEntry.stagedPath,
        }],
      });
      actualAction = typeof resolution === "string" ? resolution : resolution.action;
      if (typeof resolution !== "string") await resolution.release?.();
      return {
        scannedJournals: 1,
        restoredJournals: 0,
        purgedJournals: 1,
        skippedJournals: 0,
        failedJournals: 0,
        failedJournalNames: [],
      };
    });

    await reconcileDatasetDeletionJournals();

    expect(actualAction).toBe("purge");
  });

  test("reconciliation skips a journal owned by another database without taking a lock", async () => {
    let actualAction: string | undefined;
    reconcileStagedDeletionJournals.mockImplementationOnce(async (_root, resolver) => {
      const resolution = await resolver({
        version: 1,
        operationId: "foreign-dataset-op",
        createdAt: new Date().toISOString(),
        metadata: { databaseName: "other_database", kind: "dataset", entityIds: [7] },
        entries: [],
      });
      actualAction = typeof resolution === "string" ? resolution : resolution.action;
      return {
        scannedJournals: 1,
        restoredJournals: 0,
        purgedJournals: 0,
        skippedJournals: 1,
        failedJournals: 0,
        failedJournalNames: [],
      };
    });

    await reconcileDatasetDeletionJournals();

    expect(actualAction).toBe("skip");
    expect(poolQuery).not.toHaveBeenCalled();
  });
});
