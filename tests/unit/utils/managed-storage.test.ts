import { afterEach, describe, expect, test } from "bun:test";
import { lstat, mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  databaseStorageNamespace,
  ensureDatabaseManagedStorageRoot,
  getDatabaseManagedStorageRoot,
  resolveStoredManagedPath,
} from "../../../src/utils/managed-storage";

const roots: string[] = [];
const disposableEnv = {
  DB_DATABASE: "fz_test_upload_cleanup",
  DB_HOST: "127.0.0.1",
  DB_PORT: 3306,
  E2E_ALLOW_DB_RESET: true,
};

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("database managed storage", () => {
  test("uses a database identity namespace only for opted-in disposable databases", () => {
    const root = "/srv/fangzhen/uploads/reports";
    expect(getDatabaseManagedStorageRoot(root, disposableEnv)).toBe(
      path.join(root, ".db", "4d1616ef0ca689df"),
    );
    expect(getDatabaseManagedStorageRoot(root, {
      ...disposableEnv,
      E2E_ALLOW_DB_RESET: false,
    })).toBe(root);
    expect(getDatabaseManagedStorageRoot(root, {
      ...disposableEnv,
      DB_DATABASE: "fangzhen",
    })).toBe(root);
  });

  test("hashes the effective default MySQL port and every database identity field", () => {
    const withoutPort = { ...disposableEnv, DB_PORT: undefined };
    expect(databaseStorageNamespace(withoutPort)).toBe(
      databaseStorageNamespace(disposableEnv),
    );
    expect(databaseStorageNamespace({ ...disposableEnv, DB_HOST: "db.internal" }))
      .not.toBe(databaseStorageNamespace(disposableEnv));
    expect(databaseStorageNamespace({ ...disposableEnv, DB_PORT: 3307 }))
      .not.toBe(databaseStorageNamespace(disposableEnv));
    expect(databaseStorageNamespace({ ...disposableEnv, DB_DATABASE: "fz_test_other" }))
      .not.toBe(databaseStorageNamespace(disposableEnv));
  });

  test("rejects ambiguous whitespace in the storage identity", () => {
    expect(() => databaseStorageNamespace({
      ...disposableEnv,
      DB_HOST: " 127.0.0.1",
    })).toThrow("数据库存储身份格式无效");
  });

  test("never resolves a live absolute path outside the managed root", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fz-storage-root-"));
    const outside = await mkdtemp(path.join(os.tmpdir(), "fz-storage-outside-"));
    roots.push(root, outside);
    const outsideFile = path.join(outside, "secret.pdf");
    await writeFile(outsideFile, "secret");

    await expect(resolveStoredManagedPath({
      storedPath: outsideFile,
      managedRoot: root,
      environment: disposableEnv,
      legacyFallbackDirectory: getDatabaseManagedStorageRoot(root, disposableEnv),
    })).rejects.toMatchObject({ code: "outside_existing" });
  });

  test("never aliases a missing outside absolute path to a same-basename current file", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fz-storage-fallback-"));
    roots.push(root);
    const databaseRoot = getDatabaseManagedStorageRoot(root, disposableEnv);
    await mkdir(databaseRoot, { recursive: true });
    const currentFile = path.join(databaseRoot, "report.pdf");
    await writeFile(currentFile, "current-owner");

    await expect(resolveStoredManagedPath({
      storedPath: path.join(root, "..", "missing-old-deployment", "report.pdf"),
      managedRoot: root,
      environment: disposableEnv,
      legacyFallbackDirectory: databaseRoot,
    })).rejects.toMatchObject({ code: "outside_existing" });

    expect(await Bun.file(currentFile).text()).toBe("current-owner");
  });

  test("fails closed when the namespace parent is a symlink", async () => {
    const workspace = await mkdtemp(path.join(os.tmpdir(), "fz-storage-symlink-"));
    const outside = await mkdtemp(path.join(os.tmpdir(), "fz-storage-symlink-outside-"));
    roots.push(workspace, outside);
    const uploads = path.join(workspace, "uploads");
    const categoryRoot = path.join(uploads, "datasets");
    await mkdir(categoryRoot, { recursive: true });
    await writeFile(path.join(outside, "sentinel"), "untouched");
    await symlink(outside, path.join(categoryRoot, ".db"), "dir");

    await expect(ensureDatabaseManagedStorageRoot(categoryRoot, disposableEnv))
      .rejects.toMatchObject({ code: "invalid_path" });
    expect(await Bun.file(path.join(outside, "sentinel")).text()).toBe("untouched");
    await expect(lstat(path.join(outside, databaseStorageNamespace(disposableEnv))))
      .rejects.toMatchObject({ code: "ENOENT" });
  });

  test("rejects a legacy category root symlink before recursive seed cleanup", async () => {
    const workspace = await mkdtemp(path.join(os.tmpdir(), "fz-storage-category-link-"));
    const outside = await mkdtemp(path.join(os.tmpdir(), "fz-storage-category-outside-"));
    roots.push(workspace, outside);
    const uploads = path.join(workspace, "uploads");
    await mkdir(uploads);
    const categoryRoot = path.join(uploads, "dogfood");
    await symlink(outside, categoryRoot, "dir");
    const sentinel = path.join(outside, "must-survive");
    await writeFile(sentinel, "safe");

    await expect(ensureDatabaseManagedStorageRoot(categoryRoot, {
      ...disposableEnv,
      E2E_ALLOW_DB_RESET: false,
    })).rejects.toMatchObject({ code: "invalid_path" });
    expect(await Bun.file(sentinel).text()).toBe("safe");
  });

  test("destructive operations reject an existing unscoped disposable artifact", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fz-storage-unscoped-"));
    roots.push(root);
    const legacy = path.join(root, "legacy.pdf");
    await writeFile(legacy, "legacy");

    await expect(resolveStoredManagedPath({
      storedPath: legacy,
      managedRoot: root,
      environment: disposableEnv,
      enforceDisposableNamespace: true,
    })).rejects.toMatchObject({ code: "legacy_unscoped" });
  });
});
