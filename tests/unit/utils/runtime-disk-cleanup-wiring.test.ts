import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import path from "node:path";

describe("runtime disk cleanup wiring", () => {
  test("runs every durable deletion reconciler at startup and on the interval schedule", async () => {
    const source = await readFile(
      path.resolve(import.meta.dir, "../../../src/index.ts"),
      "utf8",
    );

    expect(source).toContain('"dataset-deletion-journals"');
    expect(source).toContain("reconcileDatasetDeletionJournals");
    expect(source).toContain('"report-deletion-journals"');
    expect(source).toContain("reconcileReportDeletionJournals");
    expect(source).toContain('"manual-deletion-journals"');
    expect(source).toContain("reconcileManualDeletionJournals");
    expect(source).toContain("await runStartupDeletionJournalReconciliation()");
    expect(source).toContain('await runRuntimeDiskCleanup("startup", { skipDeletionJournals: true })');
    expect(source).toContain('void runRuntimeDiskCleanup("interval")');
    expect(source.indexOf('"dataset-deletion-journals"')).toBeLessThan(
      source.indexOf('"production-preparation"'),
    );
    expect(source).toContain("STARTUP_DELETION_RECONCILIATION_ATTEMPTS = 3");
    expect(source).toContain("result.failedJournals > 0 || result.skippedJournals > 0");
    expect(source).toContain("if (!deletionRecoveryComplete)");
    expect(source).toContain("本轮跳过后续保留期清理");
  });
});
