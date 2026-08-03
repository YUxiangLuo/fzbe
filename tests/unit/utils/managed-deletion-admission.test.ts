import { describe, expect, test } from "bun:test";
import {
  acquireManagedDeletionPermit,
  getManagedDeletionAdmissionStats,
  tryAcquireManagedDeletionPermit,
} from "../../../src/utils/managed-deletion-admission";

describe("managed deletion admission", () => {
  test("shares one process-wide slot and releases idempotently", () => {
    const permit = tryAcquireManagedDeletionPermit();
    expect(permit).not.toBeNull();
    expect(tryAcquireManagedDeletionPermit()).toBeNull();
    expect(getManagedDeletionAdmissionStats()).toEqual({ capacity: 1, active: 1, pending: 0 });

    permit!.release();
    permit!.release();
    expect(getManagedDeletionAdmissionStats()).toEqual({ capacity: 1, active: 0, pending: 0 });
  });

  test("times out instead of consuming another pair of database connections", async () => {
    const permit = await acquireManagedDeletionPermit();
    expect(permit).not.toBeNull();
    const blocked = await acquireManagedDeletionPermit(5);
    expect(blocked).toBeNull();
    expect(getManagedDeletionAdmissionStats()).toEqual({ capacity: 1, active: 1, pending: 0 });
    permit!.release();
  });
});
