import { describe, expect, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  MODEL_EDGE_ROOTS,
  acquireOwnedFileLock,
  cleanupModelEdgeOwnedArtifacts,
  withOwnedFileLock,
  type CleanupOperations,
  type ModelEdgeOwnedArtifacts,
} from "../../system/model-edge-runtime";

const fakeStats = (directory: boolean) => ({
  isDirectory: () => directory,
  isSymbolicLink: () => false,
}) as fs.Stats;

const fixture = () => {
  const prefix = "exp900-student42-";
  const entries = {
    guided: new Set([`${prefix}baseline`, `${prefix}owned-guided`, "unrelated-guided"]),
    preparation: new Set([`${prefix}owned-preparation`, "unrelated-preparation"]),
    trained: new Set([
      `${prefix}ma.pkl`,
      `${prefix}ma_metadata.json`,
      `${prefix}baseline.pkl`,
      "exp901-student42-unrelated.pkl",
    ]),
  };
  const removed: string[] = [];
  const failed = new Set<string>();
  const operations: CleanupOperations = {
    list: (root) => {
      if (root === MODEL_EDGE_ROOTS.guided) return [...entries.guided];
      if (root === MODEL_EDGE_ROOTS.preparation) return [...entries.preparation];
      if (root === MODEL_EDGE_ROOTS.trained) return [...entries.trained];
      throw new Error(`unexpected root ${root}`);
    },
    lstat: (entryPath) => {
      if (failed.has(entryPath)) throw new Error(`injected failure ${entryPath}`);
      if (removed.includes(entryPath)) {
        throw Object.assign(new Error(`missing ${entryPath}`), { code: "ENOENT" });
      }
      return fakeStats(path.dirname(entryPath) !== MODEL_EDGE_ROOTS.trained);
    },
    unlink: (entryPath) => removed.push(entryPath),
    removeGuidedDir: (entryPath) => {
      removed.push(entryPath);
      return true;
    },
    removePreparationDir: (entryPath) => {
      removed.push(entryPath);
      return true;
    },
    removeModelGroup: (entryPath) => removed.push(entryPath),
  };
  const owned: ModelEdgeOwnedArtifacts = {
    baseline: {
      guided: new Set([`${prefix}baseline`]),
      preparation: new Set(),
      trained: new Set([`${prefix}baseline.pkl`]),
    },
    experimentIds: [900],
    studentId: 42,
    guidedDirs: [path.join(MODEL_EDGE_ROOTS.guided, `${prefix}owned-guided`)],
    preparationDirs: [path.join(MODEL_EDGE_ROOTS.preparation, `${prefix}owned-preparation`)],
    modelPaths: [path.join(MODEL_EDGE_ROOTS.trained, `${prefix}ma.pkl`)],
  };
  return { prefix, removed, failed, operations, owned };
};

describe("model-edge exact-owned teardown", () => {
  test("successful cleanup targets only new exact-prefix entries", async () => {
    const { removed, operations, owned, prefix } = fixture();
    await cleanupModelEdgeOwnedArtifacts(owned, operations);

    expect(removed).toContain(path.join(MODEL_EDGE_ROOTS.guided, `${prefix}owned-guided`));
    expect(removed).toContain(path.join(
      MODEL_EDGE_ROOTS.preparation,
      `${prefix}owned-preparation`,
    ));
    expect(removed).toContain(path.join(MODEL_EDGE_ROOTS.trained, `${prefix}ma.pkl`));
    expect(removed).toContain(path.join(
      MODEL_EDGE_ROOTS.trained,
      `${prefix}ma_metadata.json`,
    ));
    expect(removed.some((entry) => entry.includes("baseline"))).toBe(false);
    expect(removed.some((entry) => entry.includes("unrelated"))).toBe(false);
  });

  test("one owned-path failure does not skip remaining cleanup tasks", async () => {
    const { removed, failed, operations, owned, prefix } = fixture();
    failed.add(path.join(MODEL_EDGE_ROOTS.guided, `${prefix}owned-guided`));

    await expect(cleanupModelEdgeOwnedArtifacts(owned, operations)).rejects.toThrow(
      "cleanup task failed",
    );
    expect(removed).toContain(path.join(
      MODEL_EDGE_ROOTS.preparation,
      `${prefix}owned-preparation`,
    ));
    expect(removed).toContain(path.join(MODEL_EDGE_ROOTS.trained, `${prefix}ma.pkl`));
    expect(removed).toContain(path.join(
      MODEL_EDGE_ROOTS.trained,
      `${prefix}ma_metadata.json`,
    ));
  });

  test("rejects an out-of-root DB path without skipping valid owned cleanup", async () => {
    const { removed, operations, owned, prefix } = fixture();
    owned.guidedDirs.push(path.join(os.tmpdir(), "exp900-student42-escape"));
    await expect(cleanupModelEdgeOwnedArtifacts(owned, operations)).rejects.toThrow(
      "cleanup task failed",
    );
    expect(removed).toContain(path.join(
      MODEL_EDGE_ROOTS.preparation,
      `${prefix}owned-preparation`,
    ));
    expect(removed.some((entry) => entry.includes("unrelated"))).toBe(false);
  });
});

describe("model-edge runner lock ownership", () => {
  test("is exclusive and reusable after its owner releases it", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "fz-model-edge-lock-"));
    const lockPath = path.join(directory, "runner.lock");
    try {
      const release = acquireOwnedFileLock(lockPath);
      expect(() => acquireOwnedFileLock(lockPath)).toThrow("manually remove this stale lock");
      release();
      const releaseAgain = acquireOwnedFileLock(lockPath);
      releaseAgain();
      expect(fs.existsSync(lockPath)).toBe(false);
    } finally {
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });

  test("does not remove a replacement lock it no longer owns", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "fz-model-edge-replace-"));
    const lockPath = path.join(directory, "runner.lock");
    try {
      const release = acquireOwnedFileLock(lockPath);
      fs.unlinkSync(lockPath);
      fs.writeFileSync(lockPath, "replacement-owner", "utf8");
      expect(release).toThrow("ownership changed");
      expect(fs.readFileSync(lockPath, "utf8")).toBe("replacement-owner");
    } finally {
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });

  test("aggregates the original work error with a lock-release error", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "fz-model-edge-aggregate-"));
    const lockPath = path.join(directory, "runner.lock");
    try {
      let observed: unknown;
      try {
        await withOwnedFileLock(lockPath, async () => {
          fs.unlinkSync(lockPath);
          fs.writeFileSync(lockPath, "replacement-owner", "utf8");
          throw new Error("injected work failure");
        });
      } catch (error) {
        observed = error;
      }
      expect(observed).toBeInstanceOf(AggregateError);
      const messages = (observed as AggregateError).errors.map(String).join("\n");
      expect(messages).toContain("injected work failure");
      expect(messages).toContain("ownership changed");
      expect(fs.readFileSync(lockPath, "utf8")).toBe("replacement-owner");
    } finally {
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });
});
