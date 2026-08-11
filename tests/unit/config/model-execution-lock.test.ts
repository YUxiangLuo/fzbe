import { describe, expect, mock, test } from "bun:test";
import { createModelExecutionLockManager } from "../../../src/modules/models/model-execution-lock";

const silentLogger = { error: mock(() => {}) };

describe("process-local model execution lock", () => {
  test("fails fast for the same experiment/model and permits independent keys", () => {
    const manager = createModelExecutionLockManager({ logger: silentLogger });
    const first = manager.tryAcquire(42, "ma");
    const conflict = manager.tryAcquire(42, "ma");
    const otherModel = manager.tryAcquire(42, "es");
    const otherExperiment = manager.tryAcquire(43, "ma");

    expect(first).not.toBeNull();
    expect(conflict).toBeNull();
    expect(otherModel).not.toBeNull();
    expect(otherExperiment).not.toBeNull();
    expect(manager.getStats()).toMatchObject({
      scope: "process",
      active: 3,
      acquired: 3,
      conflicts: 1,
    });

    manager.release(first);
    manager.release(otherModel);
    manager.release(otherExperiment);
    expect(manager.getStats().active).toBe(0);
  });

  test("verifies ownership and makes release idempotent", () => {
    const manager = createModelExecutionLockManager({ logger: silentLogger });
    const handle = manager.tryAcquire(7, "weighted_avg");

    expect(() => manager.verifyOwned(handle)).not.toThrow();
    manager.release(handle);
    manager.release(handle);
    expect(() => manager.verifyOwned(handle)).toThrow(
      "进程内模型执行锁已失效",
    );
    expect(manager.getStats()).toMatchObject({
      active: 0,
      ownershipLosses: 1,
      releaseFailures: 0,
    });
  });

  test("never lets a stale token release the current owner", () => {
    silentLogger.error.mockClear();
    const manager = createModelExecutionLockManager({ logger: silentLogger });
    const owner = manager.tryAcquire(9, "lstm");
    expect(owner).not.toBeNull();

    manager.release({ key: owner!.key, token: Symbol("stale") });

    expect(manager.getStats()).toMatchObject({ active: 1, releaseFailures: 1 });
    expect(manager.tryAcquire(9, "lstm")).toBeNull();
    expect(silentLogger.error).toHaveBeenCalledTimes(1);
    manager.release(owner);
  });
});
