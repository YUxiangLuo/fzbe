import { beforeEach, describe, expect, mock, test } from "bun:test";
import {
  createModelOperationLeaseManager,
  type ModelExecutionLockAdapter,
} from "../../../src/modules/models/model-operation-lease";
import {
  ModelCapacityBusyError,
  ModelConflictError,
  ModelOperationDeadlineError,
} from "../../../src/modules/models/model-errors";
import type { ModelExecutionLockHandle } from "../../../src/modules/models/model-execution-lock";
import { getCurrentModelOperationSignal } from "../../../src/modules/models/model-operation-scope";
import type { ModelType } from "../../../src/modules/models/models.schema";

const events: string[] = [];
let executionAvailable = true;
let handleSequence = 0;

const tryAcquire = mock((experimentId: number, modelType: ModelType) => {
  events.push(`acquire:execution:${experimentId}:${modelType}`);
  if (!executionAvailable) return null;
  handleSequence += 1;
  return {
    key: `${experimentId}:${modelType}`,
    token: Symbol(`execution-${handleSequence}`),
  } satisfies ModelExecutionLockHandle;
});
const release = mock((handle: ModelExecutionLockHandle | null | undefined) => {
  if (handle) events.push("release:execution");
});
const verifyOwned = mock((handle: ModelExecutionLockHandle | null | undefined) => {
  if (handle) events.push("verify:execution");
});
const executionLocks: ModelExecutionLockAdapter = {
  tryAcquire,
  release,
  verifyOwned,
};

const silentLogger = { info: () => {}, warn: () => {} };
let manager = createModelOperationLeaseManager({
  capacity: 2,
  deadlineMs: 25,
  executionLocks,
  logger: silentLogger,
});

const execution = {
  execution: { experimentId: 1, modelType: "ma" as const },
};

describe("model operation lease", () => {
  beforeEach(() => {
    events.length = 0;
    executionAvailable = true;
    manager = createModelOperationLeaseManager({
      capacity: 2,
      deadlineMs: 25,
      executionLocks,
      logger: silentLogger,
    });
    tryAcquire.mockClear();
    release.mockClear();
    verifyOwned.mockClear();
  });

  test("acquires the process-local execution mutex inside admission and releases it first", async () => {
    const result = await manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        execution,
        async (lease) => {
          events.push("callback");
          expect(getCurrentModelOperationSignal()).toBe(lease.terminationSignal);
          await lease.verifyOwned();
          return "ok";
        },
      ));

    expect(result).toBe("ok");
    expect(getCurrentModelOperationSignal()).toBeUndefined();
    expect(events).toEqual([
      "acquire:execution:1:ma",
      "callback",
      "verify:execution",
      "release:execution",
    ]);
    expect(manager.getModelOperationStats().admission).toMatchObject({
      scope: "process",
      active: 0,
      pending: 0,
    });
  });

  test("supports capacity-only operations without an execution mutex", async () => {
    await manager.withModelOperationContext("adf", async (context) =>
      await manager.withModelOperationLease(context, {}, async () => undefined));

    expect(tryAcquire).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledWith(null);
    expect(manager.getModelOperationStats().admission.active).toBe(0);
  });

  test("releases every acquired resource when the callback fails", async () => {
    await expect(manager.withModelOperationContext("production_preparation", async (context) =>
      await manager.withModelOperationLease(
        context,
        execution,
        async () => { throw new Error("boom"); },
      ))).rejects.toThrow("boom");

    expect(events.at(-1)).toBe("release:execution");
    expect(manager.getModelOperationStats().admission.active).toBe(0);
  });

  test("deadline is internal and releases the lease", async () => {
    const error = await manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        execution,
        async (lease) => await new Promise<never>((_, reject) => {
          lease.terminationSignal.addEventListener(
            "abort",
            () => reject(lease.terminationSignal.reason),
            { once: true },
          );
        }),
      )).catch((caught) => caught);

    expect(error).toBeInstanceOf(ModelOperationDeadlineError);
    expect(error).toMatchObject({
      code: "MODEL_OPERATION_DEADLINE_EXCEEDED",
      operation: "prediction",
      timeoutMs: 25,
    });
    expect(events.at(-1)).toBe("release:execution");
    expect(manager.getModelOperationStats().admission.active).toBe(0);
  });

  test("supports a shorter endpoint-specific operation deadline", async () => {
    const error = await manager.withModelOperationContext("adf", async (context) =>
      await new Promise<never>((_, reject) => {
        context.deadlineSignal.addEventListener(
          "abort",
          () => reject(context.deadlineSignal.reason),
          { once: true },
        );
      }), 5).catch((caught) => caught);

    expect(error).toBeInstanceOf(ModelOperationDeadlineError);
    expect(error).toMatchObject({ operation: "adf", timeoutMs: 5 });
  });

  test("rejects above-capacity work immediately without touching execution locks", async () => {
    const releases: Array<() => void> = [];
    const hold = (experimentId: number) => manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        { execution: { experimentId, modelType: "ma" } },
        async () => await new Promise<void>((resolve) => releases.push(resolve)),
      ));

    const first = hold(1);
    const second = hold(2);
    while (releases.length < 2) await Bun.sleep(1);
    const lockCallsBeforeRejection = tryAcquire.mock.calls.length;

    const rejected = await manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        { execution: { experimentId: 3, modelType: "ma" } },
        async () => undefined,
      )).catch((caught) => caught);

    expect(rejected).toBeInstanceOf(ModelCapacityBusyError);
    expect(tryAcquire).toHaveBeenCalledTimes(lockCallsBeforeRejection);
    expect(manager.getModelOperationStats().admission).toMatchObject({
      capacity: 2,
      active: 2,
      pending: 0,
    });

    for (const resolve of releases) resolve();
    await Promise.all([first, second]);
    expect(manager.getModelOperationStats().admission.active).toBe(0);
  });

  test("releases admission when the execution mutex is busy", async () => {
    executionAvailable = false;
    const error = await manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        execution,
        async () => undefined,
      )).catch((caught) => caught);

    expect(error).toBeInstanceOf(ModelConflictError);
    expect(error).toMatchObject({ code: "MODEL_BUSY" });
    expect(events).toEqual(["acquire:execution:1:ma"]);
    expect(manager.getModelOperationStats().admission.active).toBe(0);
  });
});
