import { afterEach, describe, expect, test } from "bun:test";
import path from "node:path";
import {
  access,
  link,
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rename,
  rm,
  symlink,
  unlink,
  utimes,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import {
  fsyncDirectory,
  isPathStrictlyInside,
  purgeStagedManagedPaths,
  reconcileStagedDeletionJournals,
  restoreStagedManagedPaths,
  stageManagedPathsForDeletion,
} from "../../../src/utils/staged-file-deletion";

const tempRoots: string[] = [];
const TEST_METADATA = {
  databaseName: "unit_test",
  kind: "test-artifact",
  entityIds: [1],
};

async function makeTempRoot() {
  const root = await mkdtemp(path.join(tmpdir(), "fz-staged-delete-"));
  tempRoots.push(root);
  return root;
}

async function exists(candidatePath: string) {
  try {
    await access(candidatePath);
    return true;
  } catch {
    return false;
  }
}

afterEach(async () => {
  await Promise.all(tempRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("staged managed file deletion", () => {
  test("only accepts paths strictly inside the managed root", async () => {
    const root = await makeTempRoot();
    expect(isPathStrictlyInside(path.join(root, "report.pdf"), root)).toBe(true);
    expect(isPathStrictlyInside(root, root)).toBe(false);
    expect(isPathStrictlyInside(path.resolve(root, "../outside.pdf"), root)).toBe(false);

    await expect(stageManagedPathsForDeletion([root], root, TEST_METADATA)).rejects.toThrow(
      "拒绝删除受管目录之外的路径",
    );
  });

  test("can restore multiple regular files after staging", async () => {
    const root = await makeTempRoot();
    const reportPath = path.join(root, "report.pdf");
    const segmentPath = path.join(root, "part.csv");
    await Bun.write(reportPath, "pdf");
    await Bun.write(segmentPath, "csv");

    const staged = await stageManagedPathsForDeletion(
      [reportPath, segmentPath],
      root,
      TEST_METADATA,
    );
    expect(staged).toHaveLength(2);
    expect(await exists(reportPath)).toBe(false);
    expect(await exists(segmentPath)).toBe(false);

    await restoreStagedManagedPaths(staged);
    expect(await Bun.file(reportPath).text()).toBe("pdf");
    expect(await Bun.file(segmentPath).text()).toBe("csv");
  });

  test("rejects directories and symbolic links before creating a journal", async () => {
    const root = await makeTempRoot();
    const directoryPath = path.join(root, "segments");
    const targetPath = path.join(root, "target.csv");
    const symlinkPath = path.join(root, "alias.csv");
    await mkdir(directoryPath);
    await Bun.write(targetPath, "csv");
    await symlink(targetPath, symlinkPath);

    await expect(
      stageManagedPathsForDeletion([directoryPath], root, TEST_METADATA),
    ).rejects.toThrow("只允许普通文件");
    await expect(
      stageManagedPathsForDeletion([symlinkPath], root, TEST_METADATA),
    ).rejects.toThrow("只允许普通文件");
    expect(await exists(path.join(root, ".deletion-journal"))).toBe(false);
  });

  test("purges staged paths and ignores already-missing managed paths", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "report.pdf");
    await Bun.write(filePath, "pdf");

    const staged = await stageManagedPathsForDeletion(
      [filePath, path.join(root, "missing.pdf")],
      root,
      TEST_METADATA,
    );
    expect(staged).toHaveLength(1);
    expect(await purgeStagedManagedPaths(staged)).toEqual([]);
    // A request-path purge and a just-unblocked reconciler may race after the
    // database commit. Both choose purge, and ENOENT must remain successful.
    expect(await purgeStagedManagedPaths(staged)).toEqual([]);
    expect(await exists(staged[0]!.stagedPath)).toBe(false);
    expect(await exists(filePath)).toBe(false);
  });

  test("uses short tombstones for valid near-NAME_MAX filenames", async () => {
    const root = await makeTempRoot();
    const longName = `${"a".repeat(220)}.csv`;
    const filePath = path.join(root, longName);
    await Bun.write(filePath, "csv");

    const staged = await stageManagedPathsForDeletion([filePath], root, {
      databaseName: "unit_test",
      kind: "dataset",
      entityIds: [7],
    });

    expect(staged).toHaveLength(1);
    expect(Buffer.byteLength(path.basename(staged[0]!.stagedPath))).toBeLessThanOrEqual(255);
    expect(await purgeStagedManagedPaths(staged)).toEqual([]);
  });

  test("rejects a managed path whose ancestor symlink escapes the real root", async () => {
    const root = await makeTempRoot();
    const outside = await makeTempRoot();
    const victimPath = path.join(outside, "victim.csv");
    await Bun.write(victimPath, "outside");
    await symlink(outside, path.join(root, "link"));

    await expect(
      stageManagedPathsForDeletion(
        [path.join(root, "link", "victim.csv")],
        root,
        TEST_METADATA,
      ),
    ).rejects.toThrow("拒绝通过符号链接删除受管目录之外的路径");
    expect(await Bun.file(victimPath).text()).toBe("outside");
  });

  test("rejects a journal directory symlink without writing or cleaning outside the root", async () => {
    const root = await makeTempRoot();
    const outside = await makeTempRoot();
    const filePath = path.join(root, "managed.csv");
    const outsideTempPath = path.join(outside, "keep.tmp");
    await Bun.write(filePath, "csv");
    await Bun.write(outsideTempPath, "keep");
    await symlink(outside, path.join(root, ".deletion-journal"));

    await expect(
      stageManagedPathsForDeletion([filePath], root, TEST_METADATA),
    ).rejects.toThrow("文件隔离日志路径必须是受管目录内的真实目录");
    await expect(
      reconcileStagedDeletionJournals(root, async () => "purge"),
    ).rejects.toThrow("文件隔离日志路径必须是受管目录内的真实目录");
    expect(await Bun.file(filePath).text()).toBe("csv");
    expect(await Bun.file(outsideTempPath).text()).toBe("keep");
  });

  test("does not remove a fresh journal temp owned by a concurrent writer", async () => {
    const root = await makeTempRoot();
    const directory = path.join(root, ".deletion-journal");
    await mkdir(directory);
    const freshPath = path.join(directory, ".00000000-0000-4000-8000-000000000001.tmp");
    const stalePath = path.join(directory, ".00000000-0000-4000-8000-000000000002.tmp");
    const unrelatedPath = path.join(directory, "writer.tmp");
    await Bun.write(freshPath, "in progress");
    await Bun.write(stalePath, "abandoned");
    await Bun.write(unrelatedPath, "not ours");
    const staleTime = new Date(Date.now() - 48 * 60 * 60 * 1000);
    await utimes(stalePath, staleTime, staleTime);

    await reconcileStagedDeletionJournals(root, async () => "skip");

    expect(await Bun.file(freshPath).text()).toBe("in progress");
    expect(await exists(stalePath)).toBe(false);
    expect(await Bun.file(unrelatedPath).text()).toBe("not ours");
  });

  test("startup reconciliation restores a staged path while its DB owner exists", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "restore-after-crash.pdf");
    await Bun.write(filePath, "pdf");
    const staged = await stageManagedPathsForDeletion([filePath], root, {
      databaseName: "unit_test",
      kind: "student-reports",
      entityIds: [42],
    });
    expect(await exists(filePath)).toBe(false);

    const result = await reconcileStagedDeletionJournals(root, async (journal) => {
      expect(journal.metadata).toEqual({
        databaseName: "unit_test",
        kind: "student-reports",
        entityIds: [42],
      });
      return "restore";
    });

    expect(result).toEqual({
      scannedJournals: 1,
      restoredJournals: 1,
      purgedJournals: 0,
      skippedJournals: 0,
      failedJournals: 0,
      failedJournalNames: [],
    });
    expect(await Bun.file(filePath).text()).toBe("pdf");
    expect(await readdir(path.join(root, ".deletion-journal"))).toEqual([]);
    expect(staged[0]!.journalPath).toBeTruthy();
  });

  test("purge fails closed when an original path proves staging never completed", async () => {
    const root = await makeTempRoot();
    const firstPath = path.join(root, "first.csv");
    const secondPath = path.join(root, "second.csv");
    await Bun.write(firstPath, "first");
    await Bun.write(secondPath, "second");
    const staged = await stageManagedPathsForDeletion([firstPath, secondPath], root, {
      databaseName: "unit_test",
      kind: "dataset",
      entityIds: [9],
    });
    // Model a crash after the journal was durable but before the second rename:
    // one path is staged while another is still at its original location.
    await rename(staged[1]!.stagedPath, staged[1]!.originalPath);

    const result = await reconcileStagedDeletionJournals(root, async () => "purge");

    expect(result.purgedJournals).toBe(0);
    expect(result.failedJournals).toBe(1);
    expect(await exists(firstPath)).toBe(false);
    expect(await exists(staged[0]!.stagedPath)).toBe(true);
    expect(await Bun.file(secondPath).text()).toBe("second");
    expect(await readdir(path.join(root, ".deletion-journal"))).toHaveLength(1);
  });

  test("purges a fully staged committed journal without touching any original path", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "committed.csv");
    await Bun.write(filePath, "committed");
    const staged = await stageManagedPathsForDeletion([filePath], root, TEST_METADATA);

    const result = await reconcileStagedDeletionJournals(root, async () => "purge");

    expect(result.purgedJournals).toBe(1);
    expect(result.failedJournals).toBe(0);
    expect(await exists(filePath)).toBe(false);
    expect(await exists(staged[0]!.stagedPath)).toBe(false);
    expect(await readdir(path.join(root, ".deletion-journal"))).toEqual([]);
  });

  test("rejects a populated managed directory instead of recursively staging it", async () => {
    const root = await makeTempRoot();
    const directoryPath = path.join(root, "dataset-7");
    const nestedPath = path.join(directoryPath, "segments", "part.csv");
    await mkdir(path.dirname(nestedPath), { recursive: true });
    await Bun.write(nestedPath, "csv");

    await expect(stageManagedPathsForDeletion(
      [directoryPath],
      root,
      TEST_METADATA,
    )).rejects.toThrow("只允许普通文件");
    expect(await Bun.file(nestedPath).text()).toBe("csv");
  });

  test("reconciliation rejects a directory occupying a journal original path", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "directory-placeholder.pdf");
    await Bun.write(filePath, "pdf");
    const staged = await stageManagedPathsForDeletion([filePath], root, TEST_METADATA);
    await mkdir(filePath);
    let resolverCalled = false;

    const result = await reconcileStagedDeletionJournals(root, async () => {
      resolverCalled = true;
      return "restore";
    });

    expect(result.failedJournals).toBe(1);
    expect(resolverCalled).toBe(false);
    expect((await lstat(filePath)).isDirectory()).toBe(true);
    expect(await Bun.file(staged[0]!.stagedPath).text()).toBe("pdf");
    expect(await exists(staged[0]!.journalPath!)).toBe(true);
  });

  test("reconciliation rejects a symbolic link occupying a journal tombstone", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "symlink-tombstone.pdf");
    const symlinkTarget = path.join(root, "do-not-delete.pdf");
    await Bun.write(filePath, "pdf");
    await Bun.write(symlinkTarget, "protected");
    const staged = await stageManagedPathsForDeletion([filePath], root, TEST_METADATA);
    await unlink(staged[0]!.stagedPath);
    await symlink(symlinkTarget, staged[0]!.stagedPath);
    let resolverCalled = false;

    const result = await reconcileStagedDeletionJournals(root, async () => {
      resolverCalled = true;
      return "purge";
    });

    expect(result.failedJournals).toBe(1);
    expect(resolverCalled).toBe(false);
    expect((await lstat(staged[0]!.stagedPath)).isSymbolicLink()).toBe(true);
    expect(await Bun.file(symlinkTarget).text()).toBe("protected");
    expect(await exists(staged[0]!.journalPath!)).toBe(true);
  });

  test("purge never recursively removes a directory substituted for a tombstone", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "non-recursive-purge.pdf");
    await Bun.write(filePath, "pdf");
    const staged = await stageManagedPathsForDeletion([filePath], root, TEST_METADATA);
    await unlink(staged[0]!.stagedPath);
    await mkdir(staged[0]!.stagedPath);
    const childPath = path.join(staged[0]!.stagedPath, "keep.txt");
    await Bun.write(childPath, "keep");

    expect(await purgeStagedManagedPaths(staged)).toEqual([staged[0]!.stagedPath]);
    expect(await Bun.file(childPath).text()).toBe("keep");
    expect(await exists(staged[0]!.journalPath!)).toBe(true);
  });

  test("restore never overwrites a different placeholder at the original path", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "occupied-no-replace.pdf");
    await Bun.write(filePath, "old-generation");
    const staged = await stageManagedPathsForDeletion([filePath], root, TEST_METADATA);
    await Bun.write(filePath, "new-generation");

    await expect(restoreStagedManagedPaths(staged)).rejects.toThrow("恢复已隔离文件失败");

    expect(await Bun.file(filePath).text()).toBe("new-generation");
    expect(await Bun.file(staged[0]!.stagedPath).text()).toBe("old-generation");
    expect(await exists(staged[0]!.journalPath!)).toBe(true);
  });

  test("restore persists the new hard link before unlinking the tombstone", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "durable-link-order.pdf");
    await Bun.write(filePath, "pdf");
    const staged = await stageManagedPathsForDeletion([filePath], root, TEST_METADATA);
    const operations: string[] = [];

    await restoreStagedManagedPaths(staged, {
      createHardLink: async (existingPath, newPath) => {
        operations.push("link");
        await link(existingPath, newPath);
      },
      fsyncDirectory: async (directoryPath) => {
        operations.push("fsync-parent");
        await fsyncDirectory(directoryPath);
      },
      unlink: async (candidatePath) => {
        operations.push("unlink-tombstone");
        await unlink(candidatePath);
      },
    });

    expect(operations).toEqual(["link", "fsync-parent", "unlink-tombstone"]);
    expect(await Bun.file(filePath).text()).toBe("pdf");
    expect(await exists(staged[0]!.stagedPath)).toBe(false);
  });

  test("an intermediate fsync failure retains both hard links and the journal", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "durable-link-fsync-failure.pdf");
    await Bun.write(filePath, "pdf");
    const staged = await stageManagedPathsForDeletion([filePath], root, TEST_METADATA);
    const operations: string[] = [];

    await expect(restoreStagedManagedPaths(staged, {
      createHardLink: async (existingPath, newPath) => {
        operations.push("link");
        await link(existingPath, newPath);
      },
      fsyncDirectory: async () => {
        operations.push("fsync-parent");
        throw new Error("injected fsync failure");
      },
      unlink: async (candidatePath) => {
        operations.push("unlink-tombstone");
        await unlink(candidatePath);
      },
    })).rejects.toThrow("恢复已隔离文件失败");

    expect(operations).toEqual(["link", "fsync-parent"]);
    const [originalStats, stagedStats] = await Promise.all([
      lstat(staged[0]!.originalPath),
      lstat(staged[0]!.stagedPath),
    ]);
    expect(originalStats.dev).toBe(stagedStats.dev);
    expect(originalStats.ino).toBe(stagedStats.ino);
    expect(await exists(staged[0]!.journalPath!)).toBe(true);
  });

  test("restore completes a crash state containing two hard links to the staged inode", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "hard-link-crash.pdf");
    await Bun.write(filePath, "pdf");
    const staged = await stageManagedPathsForDeletion([filePath], root, TEST_METADATA);

    // Model a crash after link(tombstone, original) and before unlink(tombstone).
    await link(staged[0]!.stagedPath, staged[0]!.originalPath);
    const [originalStats, stagedStats] = await Promise.all([
      lstat(staged[0]!.originalPath),
      lstat(staged[0]!.stagedPath),
    ]);
    expect(originalStats.dev).toBe(stagedStats.dev);
    expect(originalStats.ino).toBe(stagedStats.ino);

    const result = await reconcileStagedDeletionJournals(root, async () => "restore");

    expect(result.restoredJournals).toBe(1);
    expect(result.failedJournals).toBe(0);
    expect(await Bun.file(filePath).text()).toBe("pdf");
    expect(await exists(staged[0]!.stagedPath)).toBe(false);
    expect(await readdir(path.join(root, ".deletion-journal"))).toEqual([]);
  });

  test("skip leaves a foreign-database journal and its tombstone untouched", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "foreign.pdf");
    await Bun.write(filePath, "pdf");
    const staged = await stageManagedPathsForDeletion([filePath], root, {
      databaseName: "other_database",
      kind: "student-reports",
      entityIds: [42],
    });

    const skipped = await reconcileStagedDeletionJournals(root, async () => "skip");

    expect(skipped.skippedJournals).toBe(1);
    expect(await exists(filePath)).toBe(false);
    expect(await exists(staged[0]!.stagedPath)).toBe(true);
    expect(await readdir(path.join(root, ".deletion-journal"))).toHaveLength(1);

    const restored = await reconcileStagedDeletionJournals(root, async () => "restore");
    expect(restored.restoredJournals).toBe(1);
    expect(await Bun.file(filePath).text()).toBe("pdf");
  });

  test("rejects corrupt metadata as failed instead of silently skipping it", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "corrupt.pdf");
    await Bun.write(filePath, "pdf");
    const staged = await stageManagedPathsForDeletion([filePath], root, TEST_METADATA);
    const journalPath = staged[0]!.journalPath!;
    const journal = JSON.parse(await readFile(journalPath, "utf8"));
    delete journal.metadata.databaseName;
    await Bun.write(journalPath, `${JSON.stringify(journal)}\n`);
    let resolverCalled = false;

    const result = await reconcileStagedDeletionJournals(root, async () => {
      resolverCalled = true;
      return "skip";
    });

    expect(resolverCalled).toBe(false);
    expect(result.failedJournals).toBe(1);
    expect(result.failedJournalNames).toEqual([path.basename(journalPath)]);
    expect(await exists(staged[0]!.stagedPath)).toBe(true);
  });

  test("rejects an empty-entry journal instead of reporting a cleanup that cannot remove it", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "empty-entries.pdf");
    await Bun.write(filePath, "pdf");
    const staged = await stageManagedPathsForDeletion([filePath], root, TEST_METADATA);
    const journalPath = staged[0]!.journalPath!;
    const journal = JSON.parse(await readFile(journalPath, "utf8"));
    journal.entries = [];
    await Bun.write(journalPath, `${JSON.stringify(journal)}\n`);

    const result = await reconcileStagedDeletionJournals(root, async () => "purge");

    expect(result.failedJournals).toBe(1);
    expect(result.failedJournalNames).toEqual([path.basename(journalPath)]);
    expect(await exists(journalPath)).toBe(true);
    expect(await exists(staged[0]!.stagedPath)).toBe(true);
  });

  test("rejects non-regular and oversized journal entries before reading them", async () => {
    const root = await makeTempRoot();
    const directory = path.join(root, ".deletion-journal");
    await mkdir(directory);
    const directoryJournal = path.join(
      directory,
      "00000000-0000-4000-8000-000000000001.json",
    );
    const oversizedJournal = path.join(
      directory,
      "00000000-0000-4000-8000-000000000002.json",
    );
    await mkdir(directoryJournal);
    await Bun.write(oversizedJournal, "x".repeat(1024 * 1024 + 1));

    const result = await reconcileStagedDeletionJournals(root, async () => "skip");

    expect(result.scannedJournals).toBe(2);
    expect(result.failedJournals).toBe(2);
    expect(result.failedJournalNames).toEqual([
      path.basename(directoryJournal),
      path.basename(oversizedJournal),
    ]);
  });

  test("releases a reconciliation lease even when the filesystem action fails", async () => {
    const root = await makeTempRoot();
    const filePath = path.join(root, "occupied.pdf");
    await Bun.write(filePath, "original");
    const staged = await stageManagedPathsForDeletion([filePath], root, TEST_METADATA);
    await Bun.write(filePath, "replacement");
    let releaseCalls = 0;

    const result = await reconcileStagedDeletionJournals(root, async () => ({
      action: "restore",
      release: async () => { releaseCalls += 1; },
    }));

    expect(result.failedJournals).toBe(1);
    expect(releaseCalls).toBe(1);
    expect(await exists(staged[0]!.journalPath!)).toBe(true);
  });
});
