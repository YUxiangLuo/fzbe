import { afterEach, expect, mock, test } from "bun:test";
import { lstat, mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { getDatabaseManagedStorageRoot } from "../../../src/utils/managed-storage";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const loggerModulePath = path.join(srcRoot, "utils", "logger.ts");
const modelsServiceModulePath = path.join(srcRoot, "modules", "models", "models.service.ts");
const reportsServiceModulePath = `${path.join(
  srcRoot,
  "modules",
  "reports",
  "reports.service.ts",
)}?managed-cleanup-test`;

const environment = {
  DB_HOST: "127.0.0.1",
  DB_PORT: 3306,
  DB_USER: "test",
  DB_PASSWORD: "test",
  DB_DATABASE: "fz_test_report_cleanup",
  E2E_ALLOW_DB_RESET: true,
  JWT_SECRET: "test-jwt-secret",
  ALLOW_LEGACY_PRODUCTION_PLAN_WRITES: true,
  PORT: 3001,
};
const reportsRoot = path.resolve(srcRoot, "../uploads/reports");
const databaseReportsRoot = getDatabaseManagedStorageRoot(reportsRoot, environment);
const temporaryRoots: string[] = [];

mock.module(configModulePath, () => ({
  env: environment,
  connectionLimit: 64,
  pool: {
    query: mock(async () => [[]]),
    getConnection: mock(async () => {
      throw new Error("database access is not expected");
    }),
  },
}));
mock.module(loggerModulePath, () => ({
  default: { info: mock(() => {}), warn: mock(() => {}), error: mock(() => {}) },
}));
mock.module(modelsServiceModulePath, () => ({
  cleanupExperimentModelArtifacts: mock(() => {}),
}));
mock.module("puppeteer-core", () => ({ default: {} }));

const { cleanupGeneratedReportPdf } = await import(reportsServiceModulePath);

afterEach(async () => {
  await rm(databaseReportsRoot, { recursive: true, force: true });
  await Promise.all(temporaryRoots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ));
});

test("cleans a legacy relative report path from the disposable database namespace", async () => {
  await mkdir(databaseReportsRoot, { recursive: true });
  const reportPath = path.join(databaseReportsRoot, "legacy-report.pdf");
  await writeFile(reportPath, "pdf-fixture");

  await cleanupGeneratedReportPdf("reports/legacy-report.pdf", databaseReportsRoot);

  await expect(lstat(reportPath)).rejects.toMatchObject({ code: "ENOENT" });
});

test("never follows an ancestor symlink while cleaning a report", async () => {
  const managedRoot = await mkdtemp(path.join(os.tmpdir(), "fz-report-cleanup-root-"));
  const outsideRoot = await mkdtemp(path.join(os.tmpdir(), "fz-report-cleanup-outside-"));
  temporaryRoots.push(managedRoot, outsideRoot);
  const outsideFile = path.join(outsideRoot, "must-survive.pdf");
  await writeFile(outsideFile, "outside-owner");
  await symlink(outsideRoot, path.join(managedRoot, "linked"), "dir");

  await cleanupGeneratedReportPdf(
    path.join(managedRoot, "linked", path.basename(outsideFile)),
    managedRoot,
  );

  expect(await Bun.file(outsideFile).text()).toBe("outside-owner");
});
