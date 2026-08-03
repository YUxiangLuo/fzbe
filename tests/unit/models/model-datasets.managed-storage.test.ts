import { afterEach, describe, expect, test } from "bun:test";
import {
  mkdir,
  mkdtemp,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { resolveModelDatasetFilePath } from "../../../src/modules/models/model-datasets";
import { getDatabaseManagedStorageRoot } from "../../../src/utils/managed-storage";

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryRoots.splice(0).map((root) =>
      rm(root, { recursive: true, force: true }),
    ),
  );
});

const productionEnvironment = {
  DB_DATABASE: "fangzhen",
  DB_HOST: "localhost",
} as const;

describe("model dataset managed-storage resolution", () => {
  test("accepts a regular file inside the managed dataset root", async () => {
    const managedRoot = await mkdtemp(path.join(os.tmpdir(), "fz-model-dataset-root-"));
    temporaryRoots.push(managedRoot);
    const datasetPath = path.join(managedRoot, "segment.csv");
    await writeFile(datasetPath, "年份,月份,销售数量\n2024,1,100\n");

    await expect(resolveModelDatasetFilePath(datasetPath, {
      managedRoot,
      environment: productionEnvironment,
    })).resolves.toBe(datasetPath);
  });

  test("rejects file and root symbolic links", async () => {
    const managedRoot = await mkdtemp(path.join(os.tmpdir(), "fz-model-dataset-root-"));
    const outsideRoot = await mkdtemp(path.join(os.tmpdir(), "fz-model-dataset-outside-"));
    temporaryRoots.push(managedRoot, outsideRoot);
    const outsideFile = path.join(outsideRoot, "outside.csv");
    await writeFile(outsideFile, "年份,月份,销售数量\n2024,1,100\n");
    const fileLink = path.join(managedRoot, "linked.csv");
    await symlink(outsideFile, fileLink);

    await expect(resolveModelDatasetFilePath(fileLink, {
      managedRoot,
      environment: productionEnvironment,
    })).rejects.toThrow("不是常规文件");

    await writeFile(
      path.join(managedRoot, "segment.csv"),
      "年份,月份,销售数量\n2024,1,100\n",
    );
    const linkedRoot = path.join(outsideRoot, "linked-root");
    await symlink(managedRoot, linkedRoot, "dir");
    await expect(resolveModelDatasetFilePath(path.join(linkedRoot, "segment.csv"), {
      managedRoot: linkedRoot,
      environment: productionEnvironment,
    })).rejects.toThrow("符号链接");
  });

  test("reads only the current disposable database namespace", async () => {
    const managedRoot = await mkdtemp(path.join(os.tmpdir(), "fz-model-dataset-root-"));
    temporaryRoots.push(managedRoot);
    const environment = {
      DB_DATABASE: "fz_e2e_model_dataset",
      DB_HOST: "LOCALHOST",
      E2E_ALLOW_DB_RESET: "1",
    } as const;
    const databaseRoot = getDatabaseManagedStorageRoot(managedRoot, environment);
    await mkdir(databaseRoot, { recursive: true });
    const scopedFile = path.join(databaseRoot, "segment.csv");
    const unscopedFile = path.join(managedRoot, "legacy.csv");
    await Promise.all([
      writeFile(scopedFile, "年份,月份,销售数量\n2024,1,100\n"),
      writeFile(unscopedFile, "年份,月份,销售数量\n2024,1,999\n"),
    ]);

    await expect(resolveModelDatasetFilePath(scopedFile, {
      managedRoot,
      environment,
    })).resolves.toBe(scopedFile);
    await expect(resolveModelDatasetFilePath(unscopedFile, {
      managedRoot,
      environment,
    })).rejects.toThrow("共享目录");
  });
});
