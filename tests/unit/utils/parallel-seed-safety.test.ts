import path from "node:path";
import { tmpdir } from "node:os";
import {
  access,
  mkdir,
  mkdtemp,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { describe, expect, test } from "bun:test";
import {
  assertParallelManagedPath,
  assertParallelManagedPaths,
  ensureDatasetsRoot,
  getDatasetsRoot,
  getGuidedTrainingRoot,
  getManualsRoot,
  getProductionPreparationRoot,
  getReportsRoot,
  getTrainedModelsRoot,
  getUploadsRoot,
  rmMany,
  validateParallelManagedPaths,
} from "../../../scripts/parallel-seed.shared";
import { databaseStorageNamespace } from "../../../src/utils/managed-storage";

const STORAGE_ENV_KEYS = [
  "DB_DATABASE",
  "DB_HOST",
  "DB_PORT",
  "E2E_ALLOW_DB_RESET",
] as const;

async function withStorageEnvironment(
  values: Partial<Record<(typeof STORAGE_ENV_KEYS)[number], string>>,
  run: () => void | Promise<void>,
) {
  const original = Object.fromEntries(
    STORAGE_ENV_KEYS.map((key) => [key, process.env[key]]),
  );
  try {
    for (const key of STORAGE_ENV_KEYS) {
      const value = values[key];
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    await run();
  } finally {
    for (const key of STORAGE_ENV_KEYS) {
      const value = original[key];
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

describe("parallel seed managed-path safety", () => {
  test("scopes upload fixtures only for an opted-in disposable database", async () => {
    const databaseEnvironment = {
      DB_DATABASE: "fz_test_parallel_storage",
      DB_HOST: "LOCALHOST",
      DB_PORT: "3306",
      E2E_ALLOW_DB_RESET: "1",
    } as const;
    await withStorageEnvironment(databaseEnvironment, () => {
      const namespace = databaseStorageNamespace({
        ...databaseEnvironment,
        DB_PORT: 3306,
      });
      expect(getDatasetsRoot()).toBe(
        path.resolve(getUploadsRoot(), "datasets", ".db", namespace),
      );
      expect(getManualsRoot()).toBe(
        path.resolve(getUploadsRoot(), "manuals", ".db", namespace),
      );
      expect(getReportsRoot()).toBe(
        path.resolve(getUploadsRoot(), "reports", ".db", namespace),
      );
    });

    await withStorageEnvironment({
      ...databaseEnvironment,
      E2E_ALLOW_DB_RESET: "0",
    }, () => {
      expect(getDatasetsRoot()).toBe(path.resolve(getUploadsRoot(), "datasets"));
      expect(getManualsRoot()).toBe(path.resolve(getUploadsRoot(), "manuals"));
      expect(getReportsRoot()).toBe(path.resolve(getUploadsRoot(), "reports"));
    });
  });

  test("allows child paths inside every managed fixture root", () => {
    const uploadsChild = path.join(
      getReportsRoot(),
      "parallel-report.pdf",
    );
    const modelChild = path.join(
      getTrainedModelsRoot(),
      "exp1-student2-ma-run-backtest.pkl",
    );
    const guidedTrainingChild = path.join(
      getGuidedTrainingRoot(),
      "exp1-student2-ma-session",
      "state.pkl",
    );
    const productionPreparationChild = path.join(
      getProductionPreparationRoot(),
      "exp1-student2-ma-session",
      "dataset.json",
    );

    expect(assertParallelManagedPath(uploadsChild)).toBe(
      path.resolve(uploadsChild),
    );
    expect(assertParallelManagedPath(modelChild)).toBe(
      path.resolve(modelChild),
    );
    expect(assertParallelManagedPath(guidedTrainingChild)).toBe(
      path.resolve(guidedTrainingChild),
    );
    expect(assertParallelManagedPath(productionPreparationChild)).toBe(
      path.resolve(productionPreparationChild),
    );
  });

  test("rejects managed roots themselves", () => {
    expect(() => assertParallelManagedPath(getUploadsRoot())).toThrow(
      "outside parallel fixture roots",
    );
    expect(() => assertParallelManagedPath(getManualsRoot())).toThrow(
      "outside parallel fixture roots",
    );
    expect(() => assertParallelManagedPath(getDatasetsRoot())).toThrow(
      "outside parallel fixture roots",
    );
    expect(() => assertParallelManagedPath(getReportsRoot())).toThrow(
      "outside parallel fixture roots",
    );
    expect(() => assertParallelManagedPath(getTrainedModelsRoot())).toThrow(
      "outside parallel fixture roots",
    );
    expect(() => assertParallelManagedPath(getGuidedTrainingRoot())).toThrow(
      "outside parallel fixture roots",
    );
    expect(() => assertParallelManagedPath(getProductionPreparationRoot())).toThrow(
      "outside parallel fixture roots",
    );
  });

  test("rejects empty and whitespace-only paths", () => {
    expect(() => assertParallelManagedPath("")).toThrow(
      "empty parallel fixture path",
    );
    expect(() => assertParallelManagedPath("   ")).toThrow(
      "empty parallel fixture path",
    );
  });

  test("rejects repository paths outside the managed roots", () => {
    const repositoryFile = path.resolve(process.cwd(), "package.json");
    const repositorySibling = path.resolve(
      process.cwd(),
      "..",
      "outside-parallel-fixtures.txt",
    );
    const unrelatedUpload = path.join(getUploadsRoot(), "private", "file.txt");

    expect(() => assertParallelManagedPath(repositoryFile)).toThrow(
      "outside parallel fixture roots",
    );
    expect(() => assertParallelManagedPath(repositorySibling)).toThrow(
      "outside parallel fixture roots",
    );
    expect(() => assertParallelManagedPath(unrelatedUpload)).toThrow(
      "outside parallel fixture roots",
    );
  });

  test("rejects traversal that resolves outside a managed root", () => {
    const uploadsTraversal = path.join(
      getUploadsRoot(),
      "reports",
      "..",
      "..",
      "outside.txt",
    );
    const modelsTraversal = path.join(
      getTrainedModelsRoot(),
      "nested",
      "..",
      "..",
      "outside.pkl",
    );

    expect(() => assertParallelManagedPath(uploadsTraversal)).toThrow(
      "outside parallel fixture roots",
    );
    expect(() => assertParallelManagedPath(modelsTraversal)).toThrow(
      "outside parallel fixture roots",
    );
  });

  test("normalizes and deduplicates batches while validating every path", () => {
    const modelPath = path.join(
      getTrainedModelsRoot(),
      "exp3-student4-ma-run-backtest.pkl",
    );
    const equivalentModelPath = path.join(
      getTrainedModelsRoot(),
      "nested",
      "..",
      path.basename(modelPath),
    );

    expect(assertParallelManagedPaths([
      modelPath,
      equivalentModelPath,
      null,
      undefined,
    ])).toEqual([path.resolve(modelPath)]);

    expect(() => assertParallelManagedPaths([
      modelPath,
      path.resolve(process.cwd(), "outside.pkl"),
    ])).toThrow("outside parallel fixture roots");
    expect(() => assertParallelManagedPaths([modelPath, ""])).toThrow(
      "empty parallel fixture path",
    );
  });

  test("prevalidates the entire batch before deleting anything", async () => {
    await mkdir(getTrainedModelsRoot(), { recursive: true });
    const temporaryDirectory = await mkdtemp(
      path.join(getTrainedModelsRoot(), "parallel-seed-safety-"),
    );
    const safePath = path.join(temporaryDirectory, "safe.pkl");

    try {
      await writeFile(safePath, "fixture");
      await expect(rmMany([
        safePath,
        path.resolve(process.cwd(), "package.json"),
      ])).rejects.toThrow("outside parallel fixture roots");
      await expect(access(safePath)).resolves.toBeNull();
    } finally {
      await rm(temporaryDirectory, { recursive: true, force: true });
    }
  });

  test("removes model sidecars without touching unrelated siblings", async () => {
    await mkdir(getTrainedModelsRoot(), { recursive: true });
    const temporaryDirectory = await mkdtemp(
      path.join(getTrainedModelsRoot(), "parallel-seed-sidecars-"),
    );
    const modelPath = path.join(temporaryDirectory, "model.pkl");
    const sidecarPath = path.join(temporaryDirectory, "model_keras_model.h5");
    const unrelatedPath = path.join(temporaryDirectory, "other.pkl");

    try {
      await Promise.all([
        writeFile(modelPath, "model"),
        writeFile(sidecarPath, "sidecar"),
        writeFile(unrelatedPath, "unrelated"),
      ]);
      await rmMany([modelPath]);

      await expect(access(modelPath)).rejects.toThrow();
      await expect(access(sidecarPath)).rejects.toThrow();
      await expect(access(unrelatedPath)).resolves.toBeNull();
    } finally {
      await rm(temporaryDirectory, { recursive: true, force: true });
    }
  });

  test("rejects an intermediate symbolic link that escapes a managed root", async () => {
    await mkdir(getReportsRoot(), { recursive: true });
    const managedDirectory = await mkdtemp(
      path.join(getReportsRoot(), "parallel-seed-symlink-"),
    );
    const outsideDirectory = await mkdtemp(
      path.join(tmpdir(), "fz-parallel-seed-outside-"),
    );
    const outsideFile = path.join(outsideDirectory, "must-survive.txt");
    const escapeLink = path.join(managedDirectory, "escape");

    try {
      await writeFile(outsideFile, "do not delete");
      await symlink(outsideDirectory, escapeLink, "dir");

      await expect(
        rmMany([path.join(escapeLink, path.basename(outsideFile))]),
      ).rejects.toThrow("symbolic link");
      await expect(access(outsideFile)).resolves.toBeNull();
    } finally {
      await rm(managedDirectory, { recursive: true, force: true });
      await rm(outsideDirectory, { recursive: true, force: true });
    }
  });

  test("rejects a missing database namespace leaf beneath an escaping .db symlink", async () => {
    const originalWorkingDirectory = process.cwd();
    const workspace = await mkdtemp(path.join(tmpdir(), "fz-parallel-seed-workspace-"));
    const outsideDirectory = await mkdtemp(path.join(tmpdir(), "fz-parallel-seed-db-outside-"));
    const databaseEnvironment = {
      DB_DATABASE: "fz_test_parallel_symlink_escape",
      DB_HOST: "127.0.0.1",
      DB_PORT: "3306",
      E2E_ALLOW_DB_RESET: "1",
    } as const;

    try {
      await mkdir(path.join(workspace, "uploads", "datasets"), { recursive: true });
      await symlink(outsideDirectory, path.join(workspace, "uploads", "datasets", ".db"), "dir");
      process.chdir(workspace);

      await withStorageEnvironment(databaseEnvironment, async () => {
        const targetPath = path.join(getDatasetsRoot(), "fixture.csv");
        await expect(validateParallelManagedPaths([targetPath]))
          .rejects.toThrow("symbolic link outside the workspace");
        await expect(ensureDatasetsRoot()).rejects.toThrow(
          "符号链接或不是目录",
        );
        await expect(access(path.join(outsideDirectory, databaseStorageNamespace({
          ...databaseEnvironment,
          DB_PORT: 3306,
        })))).rejects.toThrow();
      });
    } finally {
      process.chdir(originalWorkingDirectory);
      await rm(workspace, { recursive: true, force: true });
      await rm(outsideDirectory, { recursive: true, force: true });
    }
  });
});
