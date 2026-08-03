import { describe, expect, mock, test } from "bun:test";
import type { Connection } from "mysql2/promise";
import { auditExperimentRetryLineage } from "../../../scripts/experiment-retry-lineage-audit";

const buildConnection = (rowSets: Array<Array<Record<string, unknown>>>) => {
  const pending = [...rowSets];
  const query = mock(async () => [pending.shift() ?? []]);
  return {
    connection: { query } as unknown as Connection,
    query,
  };
};

describe("experiment retry lineage audit", () => {
  test("accepts explicit lineage with no ambiguous legacy candidate", async () => {
    const { connection, query } = buildConnection([[], []]);

    await expect(auditExperimentRetryLineage(connection)).resolves.toBeUndefined();
    expect(query).toHaveBeenCalledTimes(2);

    const explicitLineageSql = String(query.mock.calls[0]?.[0]);
    expect(explicitLineageSql).toContain("parent.experiment_id >= child.experiment_id");
    expect(explicitLineageSql).toContain("FROM experiment_status intervening");
    expect(explicitLineageSql).toContain(
      "intervening.experiment_id > parent.experiment_id",
    );
    expect(explicitLineageSql).toContain(
      "intervening.experiment_id < child.experiment_id",
    );
  });

  test("fails before the legacy scan when explicit lineage is invalid", async () => {
    const { connection, query } = buildConnection([[
      {
        child_experiment_id: 12,
        child_student_id: 2,
        parent_experiment_id: 11,
        parent_student_id: 3,
        parent_status: "Completed",
        parent_report_status: "rejected",
      },
    ]]);

    await expect(auditExperimentRetryLineage(connection))
      .rejects.toThrow("Existing retry lineage violates the rejected-parent contract");
    expect(query).toHaveBeenCalledTimes(1);
  });

  test("fails closed instead of guessing an unlinked legacy retry", async () => {
    const { connection, query } = buildConnection([[], [
      {
        parent_experiment_id: 21,
        student_id: 7,
        first_unlinked_experiment_id: 22,
        unlinked_later_experiment_count: 1,
      },
    ]]);

    await expect(auditExperimentRetryLineage(connection))
      .rejects.toThrow("Ambiguous legacy retry lineage detected");
    expect(query).toHaveBeenCalledTimes(2);

    const ambiguousLineageSql = String(query.mock.calls[1]?.[0]);
    expect(ambiguousLineageSql).toContain("later.retry_of_experiment_id IS NULL");
    expect(ambiguousLineageSql).not.toContain("explicit_retry");
  });

  test("does not let a valid explicit child mask a later unlinked attempt", async () => {
    const { connection, query } = buildConnection([[], [
      {
        parent_experiment_id: 31,
        student_id: 8,
        first_unlinked_experiment_id: 33,
        unlinked_later_experiment_count: 1,
      },
    ]]);

    await expect(auditExperimentRetryLineage(connection))
      .rejects.toThrow("Ambiguous legacy retry lineage detected");

    const ambiguousLineageSql = String(query.mock.calls[1]?.[0]);
    expect(ambiguousLineageSql).toContain("later.retry_of_experiment_id IS NULL");
    expect(ambiguousLineageSql).not.toContain("explicit_retry.experiment_id IS NULL");
  });

  test("rejects an explicit child linked to a stale rejected ancestor", async () => {
    const { connection, query } = buildConnection([[
      {
        child_experiment_id: 43,
        child_student_id: 9,
        parent_experiment_id: 41,
        parent_student_id: 9,
        parent_status: "Completed",
        parent_report_status: "rejected",
      },
    ]]);

    await expect(auditExperimentRetryLineage(connection))
      .rejects.toThrow("Existing retry lineage violates the rejected-parent contract");
    expect(query).toHaveBeenCalledTimes(1);

    const explicitLineageSql = String(query.mock.calls[0]?.[0]);
    expect(explicitLineageSql).toContain("FROM experiment_status intervening");
  });
});
