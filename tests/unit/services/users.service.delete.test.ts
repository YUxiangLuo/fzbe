import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";
import { getDatabaseManagedStorageRoot } from "../../../src/utils/managed-storage";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const passwordModulePath = path.join(srcRoot, "utils", "password.ts");
const stagedDeletionModulePath = path.join(srcRoot, "utils", "staged-file-deletion.ts");
const serviceModulePath = `${path.join(srcRoot, "modules", "users", "users.service.ts")}?user-delete-test`;
const TEST_DATABASE = "fz_test_user_delete";
const reportsRoot = path.resolve(srcRoot, "../uploads/reports");
const TEST_STORAGE_ENV = {
  DB_DATABASE: TEST_DATABASE,
  DB_HOST: "127.0.0.1",
  DB_PORT: 3306,
  E2E_ALLOW_DB_RESET: true,
};
const reportsStorageRoot = getDatabaseManagedStorageRoot(reportsRoot, TEST_STORAGE_ENV);

const lifecycle: string[] = [];
const poolQuery = mock(async () => [[]]);
const connectionQuery = mock(async (sql: string) => {
  if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
  if (sql.includes("FROM experiment_reports")) {
    return [[{
      report_id: 90,
      student_id: 77,
      pdf_file_path: path.join(reportsStorageRoot, "student-77.pdf"),
    }]];
  }
  if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
    return [[{ user_id: 77, role: "Student" }]];
  }
  if (sql.includes("FROM enrollments")) return [[]];
  return [{ affectedRows: 1 }];
});
const beginTransaction = mock(async () => {});
const commit = mock(async () => { lifecycle.push("commit"); });
const rollback = mock(async () => { lifecycle.push("rollback"); });
const transactionRelease = mock(() => { lifecycle.push("transaction-release"); });
const transactionDestroy = mock(() => { lifecycle.push("transaction-destroy"); });
const getConnection = mock(async () => ({
  query: connectionQuery,
  beginTransaction,
  commit,
  rollback,
  release: transactionRelease,
  destroy: transactionDestroy,
}));

const stagedEntry = {
  originalPath: path.join(reportsStorageRoot, "student-77.pdf"),
  stagedPath: path.join(reportsStorageRoot, ".deleting-unit-0"),
  journalPath: path.join(reportsStorageRoot, ".deletion-journal", "unit.json"),
};
const secondStagedEntry = {
  originalPath: path.join(reportsStorageRoot, "student-77-second.pdf"),
  stagedPath: path.join(reportsStorageRoot, ".deleting-unit-1"),
  journalPath: stagedEntry.journalPath,
};
const stageManagedPathsForDeletion = mock(async () => {
  lifecycle.push("stage");
  return [stagedEntry];
});
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
  env: TEST_STORAGE_ENV,
  pool: { query: poolQuery, getConnection },
}));

mock.module(passwordModulePath, () => ({
  hashPassword: mock(async () => "hash"),
  shouldChangePassword: mock(async () => false),
  verifyPassword: mock(async () => true),
}));

mock.module(stagedDeletionModulePath, () => ({
  classifyManagedDeletionOwnership,
  isPathStrictlyInside: () => true,
  stageManagedPathsForDeletion,
  restoreStagedManagedPaths,
  purgeStagedManagedPaths,
  reconcileStagedDeletionJournals,
}));

const { deleteUser, batchDeleteUsers, reconcileReportDeletionJournals } = await import(serviceModulePath);

const successfulReconcileResult = (action: "restore" | "purge" | "skip") => ({
  scannedJournals: 1,
  restoredJournals: action === "restore" ? 1 : 0,
  purgedJournals: action === "purge" ? 1 : 0,
  skippedJournals: action === "skip" ? 1 : 0,
  failedJournals: 0,
  failedJournalNames: [],
});

describe("user deletion report file transaction", () => {
  beforeEach(() => {
    lifecycle.length = 0;
    poolQuery.mockReset();
    poolQuery.mockResolvedValue([[]]);
    connectionQuery.mockReset();
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("FROM experiment_reports")) {
        return [[{
          report_id: 90,
          student_id: 77,
          pdf_file_path: path.join(reportsStorageRoot, "student-77.pdf"),
        }]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    transactionRelease.mockClear();
    transactionDestroy.mockClear();
    getConnection.mockClear();
    stageManagedPathsForDeletion.mockReset();
    stageManagedPathsForDeletion.mockImplementation(async () => {
      lifecycle.push("stage");
      return [stagedEntry];
    });
    restoreStagedManagedPaths.mockReset();
    restoreStagedManagedPaths.mockImplementation(async () => { lifecycle.push("restore"); });
    purgeStagedManagedPaths.mockReset();
    purgeStagedManagedPaths.mockImplementation(async () => {
      lifecycle.push("purge");
      return [];
    });
    reconcileStagedDeletionJournals.mockReset();
    reconcileStagedDeletionJournals.mockResolvedValue(successfulReconcileResult("skip"));
  });

  test("locks user then experiment and report, commits, then purges staged files", async () => {
    const result = await deleteUser(77, 1);

    expect(result).toEqual({
      success: true,
      cleanup: { removedReportFiles: 1 },
    });
    const lockingQueries = connectionQuery.mock.calls
      .map(([sql]) => String(sql))
      .filter((sql) => sql.includes("FOR UPDATE"));
    expect(lockingQueries.map((sql) => (
      sql.includes("experiment_status")
        ? "experiment"
        : sql.includes("experiment_reports") ? "report" : "user"
    ))).toEqual(["user", "experiment", "report"]);
    expect(lifecycle).toEqual([
      "stage",
      "commit",
      "transaction-release",
      "purge",
    ]);
  });

  test("resolves a legacy relative report path inside the disposable database namespace", async () => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("FROM experiment_reports")) {
        return [[{ report_id: 90, student_id: 77, pdf_file_path: "reports/legacy.pdf" }]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });

    const result = await deleteUser(77, 1);

    expect(result).toEqual({ success: true, cleanup: { removedReportFiles: 1 } });
    expect(stageManagedPathsForDeletion).toHaveBeenCalledWith(
      [path.join(reportsStorageRoot, "legacy.pdf")],
      reportsStorageRoot,
      { databaseName: TEST_DATABASE, kind: "student-reports", entityIds: [77] },
    );
  });

  test("keeps committed user-deletion success when report purge raises", async () => {
    purgeStagedManagedPaths.mockImplementationOnce(async () => {
      lifecycle.push("purge");
      throw new Error("filesystem unavailable");
    });

    const result = await deleteUser(77, 1);

    expect(result).toEqual({
      success: true,
      cleanup: { removedReportFiles: 1, reportFileCleanupPending: 1 },
    });
    expect(lifecycle).toEqual([
      "stage",
      "commit",
      "transaction-release",
      "purge",
    ]);
  });

  test("fails closed when report staging fails", async () => {
    stageManagedPathsForDeletion.mockRejectedValueOnce(new Error("EBUSY"));

    const result = await deleteUser(77, 1);

    expect(result).toEqual({ error: "file_cleanup_failed" });
    expect(connectionQuery.mock.calls.some(([sql]) => String(sql).startsWith("DELETE FROM users"))).toBe(false);
    expect(commit).not.toHaveBeenCalled();
    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).not.toHaveBeenCalled();
    expect(lifecycle).toEqual(["rollback", "transaction-release"]);
  });

  test("restores staged reports when the database delete rolls back", async () => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("FROM experiment_reports")) {
        return [[{ report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM enrollments")) return [[]];
      if (sql.startsWith("DELETE FROM users")) throw new Error("database unavailable");
      return [{ affectedRows: 1 }];
    });

    await expect(deleteUser(77, 1)).rejects.toThrow("database unavailable");

    expect(commit).not.toHaveBeenCalled();
    expect(restoreStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
    expect(purgeStagedManagedPaths).not.toHaveBeenCalled();
    expect(lifecycle).toEqual([
      "stage",
      "rollback",
      "restore",
      "transaction-release",
    ]);
  });

  test("destroys a pre-commit transaction connection when rollback also fails", async () => {
    rollback.mockImplementationOnce(async () => {
      lifecycle.push("rollback");
      throw new Error("rollback acknowledgement lost");
    });
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("FROM experiment_reports")) {
        return [[{ report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM enrollments")) return [[]];
      if (sql.startsWith("DELETE FROM users")) throw new Error("database unavailable");
      return [{ affectedRows: 1 }];
    });

    await expect(deleteUser(77, 1)).rejects.toThrow("database unavailable");

    expect(transactionDestroy).toHaveBeenCalledTimes(1);
    expect(transactionRelease).not.toHaveBeenCalled();
    expect(restoreStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
  });

  test("turns a lost commit acknowledgement into success only after a fresh locking probe finds no owner", async () => {
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("user-deletion-owner-probe")) return [[]];
      if (sql.includes("FROM experiment_reports")) {
        return [[{ report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM users") && sql.includes("user_id IN")) return [[]];
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });

    const result = await deleteUser(77, 1);

    expect(result).toEqual({
      success: true,
      cleanup: { removedReportFiles: 1 },
    });
    expect(transactionDestroy).toHaveBeenCalledTimes(1);
    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
    expect(connectionQuery.mock.calls
      .map(([sql]) => String(sql))
      .filter((sql) => sql.includes("FOR UPDATE"))
      .map((sql) => sql.includes("experiment_status")
        ? "experiment"
        : sql.includes("experiment_reports") ? "report" : "user"))
      .toEqual(["user", "experiment", "report", "user", "experiment", "report"]);
  });

  test("restores a committed deletion tombstone when a surviving report now shares its path", async () => {
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("user-deletion-owner-probe")) {
        return [[{ report_id: 91, student_id: 88, pdf_file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("FROM experiment_reports")) {
        return [[{ report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM users") && sql.includes("user_id IN")) return [[]];
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });

    const result = await deleteUser(77, 1);

    expect(result).toEqual({ success: true, cleanup: {} });
    expect(restoreStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
    expect(purgeStagedManagedPaths).not.toHaveBeenCalled();
  });

  test("keeps ACK-loss user-deletion success when its post-probe purge raises", async () => {
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    purgeStagedManagedPaths.mockImplementationOnce(async () => {
      lifecycle.push("purge");
      throw new Error("filesystem unavailable");
    });
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("user-deletion-owner-probe")) return [[]];
      if (sql.includes("FROM experiment_reports")) {
        return [[{ report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM users") && sql.includes("user_id IN")) return [[]];
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });

    const result = await deleteUser(77, 1);

    expect(result).toEqual({
      success: true,
      cleanup: { removedReportFiles: 1, reportFileCleanupPending: 1 },
    });
    expect(transactionDestroy).toHaveBeenCalledTimes(1);
  });

  test("destroys both ambiguous and probe sessions when the probe rollback acknowledgement is lost", async () => {
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    rollback.mockImplementationOnce(async () => {
      lifecycle.push("rollback");
      throw new Error("probe rollback acknowledgement lost");
    });
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("user-deletion-owner-probe")) {
        throw new Error("probe unavailable");
      }
      if (sql.includes("FROM experiment_reports")) {
        return [[{ report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });

    await expect(deleteUser(77, 1)).rejects.toThrow("commit acknowledgement lost");

    expect(transactionDestroy).toHaveBeenCalledTimes(2);
    expect(transactionRelease).not.toHaveBeenCalled();
    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).not.toHaveBeenCalled();
  });

  test("restores and surfaces the commit error when the fresh probe finds the same report owner", async () => {
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("FROM experiment_reports")) {
        return [[{ report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM users") && sql.includes("user_id IN")) {
        return [[{ user_id: 77 }]];
      }
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });

    await expect(deleteUser(77, 1)).rejects.toThrow("commit acknowledgement lost");

    expect(transactionDestroy).toHaveBeenCalledTimes(1);
    expect(restoreStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
    expect(purgeStagedManagedPaths).not.toHaveBeenCalled();
  });

  test("purges an old report generation but conservatively surfaces the commit error when the owner ID was reused", async () => {
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    const newIncarnationPath = path.join(reportsStorageRoot, "student-77-new-incarnation.pdf");
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("user-deletion-owner-probe")) {
        return [[{ report_id: 91, student_id: 77, pdf_file_path: newIncarnationPath }]];
      }
      if (sql.includes("FROM experiment_reports")) {
        return [[{ report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM users") && sql.includes("user_id IN")) {
        return [[{ user_id: 77 }]];
      }
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });

    // Owner-ID reuse and an uncommitted delete followed by a concurrent report
    // generation are indistinguishable without an immutable user generation.
    // The old unreferenced PDF is safe to purge, but the API must not claim the
    // user deletion committed.
    await expect(deleteUser(77, 1)).rejects.toThrow("commit acknowledgement lost");

    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
  });

  test("retains every tombstone and surfaces the commit error when owner paths match only partially", async () => {
    stageManagedPathsForDeletion.mockImplementationOnce(async () => {
      lifecycle.push("stage");
      return [stagedEntry, secondStagedEntry];
    });
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("user-deletion-owner-probe")) {
        return [[{
          report_id: 90,
          student_id: 77,
          pdf_file_path: stagedEntry.originalPath,
        }]];
      }
      if (sql.includes("FROM experiment_reports")) {
        return [[
          { report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath },
          { report_id: 91, student_id: 77, pdf_file_path: secondStagedEntry.originalPath },
        ]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM users") && sql.includes("user_id IN")) {
        return [[{ user_id: 77 }]];
      }
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });

    await expect(deleteUser(77, 1)).rejects.toThrow("commit acknowledgement lost");

    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).not.toHaveBeenCalled();
  });

  test("uses surviving owner rows as the not-committed outcome when no report file was staged", async () => {
    stageManagedPathsForDeletion.mockResolvedValueOnce([]);
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("FROM experiment_reports")) {
        return [[{ report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM users") && sql.includes("user_id IN")) {
        return [[{ user_id: 77 }]];
      }
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });

    await expect(deleteUser(77, 1)).rejects.toThrow("commit acknowledgement lost");

    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).not.toHaveBeenCalled();
  });

  test("uses absent owner rows as the committed outcome when no report file was staged", async () => {
    stageManagedPathsForDeletion.mockResolvedValueOnce([]);
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("user-deletion-owner-probe")) return [[]];
      if (sql.includes("FROM experiment_reports")) {
        return [[{ report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("FROM users WHERE user_id = ? FOR UPDATE")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("FROM users") && sql.includes("user_id IN")) return [[]];
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });

    const result = await deleteUser(77, 1);

    expect(result).toEqual({ success: true, cleanup: {} });
    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).toHaveBeenCalledWith([]);
  });

  test("retains staged files and surfaces the commit error when a batch probe finds a mixed owner set", async () => {
    const user78Entry = {
      originalPath: path.join(reportsStorageRoot, "student-78.pdf"),
      stagedPath: path.join(reportsStorageRoot, ".deleting-unit-1"),
      journalPath: stagedEntry.journalPath,
    };
    stageManagedPathsForDeletion.mockImplementationOnce(async () => {
      lifecycle.push("stage");
      return [stagedEntry, user78Entry];
    });
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }, { experiment_id: 80 }]];
      if (sql.includes("user-deletion-owner-probe")) return [[]];
      if (sql.includes("FROM experiment_reports")) {
        return [[
          { report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath },
          { report_id: 91, student_id: 78, pdf_file_path: user78Entry.originalPath },
        ]];
      }
      if (sql.includes("SELECT user_id, role FROM users") && sql.includes("user_id IN")) {
        return [[
          { user_id: 77, role: "Student" },
          { user_id: 78, role: "Student" },
        ]];
      }
      if (sql.includes("SELECT user_id") && sql.includes("user_id IN")) {
        return [[{ user_id: 78 }]];
      }
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });

    await expect(batchDeleteUsers([77, 78], 1)).rejects.toThrow("commit acknowledgement lost");

    expect(restoreStagedManagedPaths).not.toHaveBeenCalled();
    expect(purgeStagedManagedPaths).not.toHaveBeenCalled();
  });

  test("returns batch-delete success after an ambiguous commit probe proves every owner absent", async () => {
    commit.mockRejectedValueOnce(new Error("commit acknowledgement lost"));
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_status")) return [[{ experiment_id: 70 }]];
      if (sql.includes("user-deletion-owner-probe")) return [[]];
      if (sql.includes("FROM experiment_reports")) {
        return [[{ report_id: 90, student_id: 77, pdf_file_path: stagedEntry.originalPath }]];
      }
      if (sql.includes("SELECT user_id, role FROM users") && sql.includes("user_id IN")) {
        return [[{ user_id: 77, role: "Student" }]];
      }
      if (sql.includes("SELECT user_id") && sql.includes("user_id IN")) return [[]];
      if (sql.includes("FROM enrollments")) return [[]];
      return [{ affectedRows: 1 }];
    });

    const result = await batchDeleteUsers([77], 1);

    expect(result).toEqual({
      success: true,
      deletedCount: 1,
      cleanup: { removedReportFiles: 1 },
    });
    expect(transactionDestroy).toHaveBeenCalledTimes(1);
    expect(purgeStagedManagedPaths).toHaveBeenCalledWith([stagedEntry]);
  });

  test.each([
    ["restore", [[{ user_id: 77 }]]],
    ["purge", [[]]],
  ] as const)("report journal reconciliation chooses %s from owner existence", async (expectedAction, rows) => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_reports")) {
        return expectedAction === "restore"
          ? [[{
            report_id: 90,
            student_id: 77,
            pdf_file_path: stagedEntry.originalPath,
          }]]
          : [[]];
      }
      if (sql.includes("FROM users") && sql.includes("FOR UPDATE")) return rows as never;
      return [[]];
    });
    let actualAction: string | undefined;
    reconcileStagedDeletionJournals.mockImplementationOnce(async (_root, resolver) => {
      const resolution = await resolver({
        version: 1,
        operationId: "report-op",
        createdAt: new Date().toISOString(),
        metadata: {
          databaseName: TEST_DATABASE,
          kind: "student-reports",
          entityIds: [77],
        },
        entries: [{
          originalPath: stagedEntry.originalPath,
          stagedPath: stagedEntry.stagedPath,
        }],
      });
      actualAction = typeof resolution === "string" ? resolution : resolution.action;
      lifecycle.push("filesystem-action");
      if (typeof resolution !== "string") await resolution.release?.();
      return successfulReconcileResult(actualAction as "restore" | "purge");
    });

    await reconcileReportDeletionJournals();

    expect(actualAction).toBe(expectedAction);
    expect(lifecycle).toEqual([
      "filesystem-action",
      "commit",
      "transaction-release",
    ]);
  });

  test("purges an old-incarnation report journal when a reused user ID references another PDF", async () => {
    connectionQuery.mockImplementation(async (sql: string) => {
      if (sql.includes("FROM experiment_reports")) {
        return [[{
          report_id: 91,
          student_id: 77,
          pdf_file_path: path.join(reportsStorageRoot, "new-incarnation.pdf"),
        }]];
      }
      if (sql.includes("FROM users") && sql.includes("FOR UPDATE")) {
        return [[{ user_id: 77 }]];
      }
      return [[]];
    });
    let actualAction: string | undefined;
    reconcileStagedDeletionJournals.mockImplementationOnce(async (_root, resolver) => {
      const resolution = await resolver({
        version: 1,
        operationId: "report-aba-op",
        createdAt: new Date().toISOString(),
        metadata: {
          databaseName: TEST_DATABASE,
          kind: "student-reports",
          entityIds: [77],
        },
        entries: [{
          originalPath: stagedEntry.originalPath,
          stagedPath: stagedEntry.stagedPath,
        }],
      });
      actualAction = typeof resolution === "string" ? resolution : resolution.action;
      if (typeof resolution !== "string") await resolution.release?.();
      return successfulReconcileResult("purge");
    });

    await reconcileReportDeletionJournals();

    expect(actualAction).toBe("purge");
  });

  test("skips foreign-database journals without touching their owner rows", async () => {
    let action: string | undefined;
    reconcileStagedDeletionJournals.mockImplementationOnce(async (_root, resolver) => {
      const foreign = await resolver({
        version: 1,
        operationId: "foreign-op",
        createdAt: new Date().toISOString(),
        metadata: { databaseName: "other_database", kind: "student-reports", entityIds: [77] },
        entries: [],
      });
      action = typeof foreign === "string" ? foreign : foreign.action;
      return successfulReconcileResult("skip");
    });

    await reconcileReportDeletionJournals();

    expect(action).toBe("skip");
    expect(poolQuery).not.toHaveBeenCalled();
  });
});
