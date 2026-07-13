import { beforeEach, describe, expect, mock, test } from "bun:test";
import {
  createModelOperationLeaseManager,
  type ModelOperationLockAdapter,
} from "../../../src/modules/models/model-operation-lease";
import {
  ModelCapacityBusyError,
  ModelConflictError,
  ModelOperationDeadlineError,
} from "../../../src/modules/models/model-errors";
import type { AdvisoryLockHandle } from "../../../src/modules/models/model-locks";
import type { ModelType } from "../../../src/modules/models/models.schema";

const events: string[] = [];
let slotAvailable = true;
let executionAvailable = true;
let handleSequence = 0;

const makeHandle = (prefix: string): AdvisoryLockHandle => ({
  connection: {} as never,
  name: `${prefix}-${++handleSequence}`,
  ownerConnectionId: handleSequence,
  signal: new AbortController().signal,
  stopHeartbeat: () => {},
});

const acquireModelSlot = mock(async (_signal?: AbortSignal) => {
  events.push("acquire:slot");
  return slotAvailable ? makeHandle("slot") : null;
});
const acquireModelExecutionLock = mock(async (
  _experimentId: number,
  _modelType: ModelType,
  _signal?: AbortSignal,
) => {
  events.push("acquire:execution");
  return executionAvailable ? makeHandle("execution") : null;
});
const releaseModelExecutionLock = mock(async (handle: AdvisoryLockHandle | null | undefined) => {
  if (handle) events.push("release:execution");
});
const releaseModelSlot = mock(async (handle: AdvisoryLockHandle | null | undefined) => {
  if (handle) events.push("release:slot");
});
const verifyModelLockOwned = mock(async () => undefined);
const locks: ModelOperationLockAdapter = {
  acquireModelSlot,
  acquireModelExecutionLock,
  releaseModelExecutionLock,
  releaseModelSlot,
  verifyModelLockOwned,
  combineModelLockSignals: (...handles: Array<{ signal?: AbortSignal } | null>) => {
    const signals = handles.map((handle) => handle?.signal).filter(Boolean) as AbortSignal[];
    if (signals.length === 0) return undefined;
    return signals.length === 1 ? signals[0] : AbortSignal.any(signals);
  },
};

const silentLogger = { info: () => {}, warn: () => {} };
let manager = createModelOperationLeaseManager({
  capacity: 2,
  deadlineMs: 25,
  locks,
  logger: silentLogger,
});

describe("model operation lease", () => {
  beforeEach(() => {
    events.length = 0;
    slotAvailable = true;
    executionAvailable = true;
    manager = createModelOperationLeaseManager({
      capacity: 2,
      deadlineMs: 25,
      locks,
      logger: silentLogger,
    });
    acquireModelSlot.mockClear();
    acquireModelExecutionLock.mockClear();
    releaseModelExecutionLock.mockClear();
    releaseModelSlot.mockClear();
    verifyModelLockOwned.mockClear();
  });

  test("uses slot-before-execution order and releases in reverse order", async () => {
    const result = await manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        { mode: "slot_and_execution", experimentId: 1, modelType: "ma" },
        async (lease) => {
          events.push("callback");
          await lease.verifyOwned();
          return "ok";
        },
      ));

    expect(result).toBe("ok");
    expect(events).toEqual([
      "acquire:slot",
      "acquire:execution",
      "callback",
      "release:execution",
      "release:slot",
    ]);
    expect(manager.getModelOperationStats().admission.active).toBe(0);
  });

  test("passes the operation deadline signal through both lock acquisitions", async () => {
    await manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        { mode: "slot_and_execution", experimentId: 1, modelType: "ma" },
        async () => undefined,
      ));

    const slotSignal = acquireModelSlot.mock.calls[0]?.[0];
    const executionSignal = acquireModelExecutionLock.mock.calls[0]?.[2];
    expect(slotSignal).toBeInstanceOf(AbortSignal);
    expect(executionSignal).toBe(slotSignal);
  });

  test("releases every acquired resource when the callback fails", async () => {
    await expect(manager.withModelOperationContext("production_preparation", async (context) =>
      await manager.withModelOperationLease(
        context,
        { mode: "slot_and_execution", experimentId: 1, modelType: "ma" },
        async () => { throw new Error("boom"); },
      ))).rejects.toThrow("boom");

    expect(events.slice(-2)).toEqual(["release:execution", "release:slot"]);
    expect(manager.getModelOperationStats().admission.active).toBe(0);
  });

  test("deadline is internal and releases the lease", async () => {
    const error = await manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        { mode: "slot_and_execution", experimentId: 1, modelType: "ma" },
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
    expect(events.slice(-2)).toEqual(["release:execution", "release:slot"]);
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

  test("deadline aborts lock acquisition and releases admission immediately", async () => {
    acquireModelSlot.mockImplementationOnce(async (signal?: AbortSignal) =>
      await new Promise<never>((_, reject) => {
        signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
      }));

    const error = await manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        { mode: "slot_and_execution", experimentId: 1, modelType: "ma" },
        async () => undefined,
      )).catch((caught) => caught);

    expect(error).toBeInstanceOf(ModelOperationDeadlineError);
    expect(events).toEqual([]);
    expect(releaseModelSlot).toHaveBeenCalledWith(null);
    expect(manager.getModelOperationStats().admission.active).toBe(0);
  });

  test("rejects above-capacity work immediately without acquiring DB locks", async () => {
    const releases: Array<() => void> = [];
    const hold = () => manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        { mode: "slot_and_execution", experimentId: 1, modelType: "ma" },
        async () => await new Promise<void>((resolve) => releases.push(resolve)),
      ));

    const first = hold();
    const second = hold();
    while (releases.length < 2) await Bun.sleep(1);
    const lockCallsBeforeRejection = acquireModelSlot.mock.calls.length;

    const rejected = await manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        { mode: "slot_and_execution", experimentId: 2, modelType: "ma" },
        async () => undefined,
      )).catch((caught) => caught);

    expect(rejected).toBeInstanceOf(ModelCapacityBusyError);
    expect(acquireModelSlot).toHaveBeenCalledTimes(lockCallsBeforeRejection);
    expect(manager.getModelOperationStats().admission).toMatchObject({ capacity: 2, active: 2, pending: 0 });

    for (const release of releases) release();
    await Promise.all([first, second]);
    expect(manager.getModelOperationStats().admission.active).toBe(0);
  });

  test("releases slot and admission when the model execution lock is busy", async () => {
    executionAvailable = false;
    const error = await manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        { mode: "slot_and_execution", experimentId: 1, modelType: "ma" },
        async () => undefined,
      )).catch((caught) => caught);

    expect(error).toBeInstanceOf(ModelConflictError);
    expect(error).toMatchObject({ code: "MODEL_BUSY" });
    expect(events).toEqual(["acquire:slot", "acquire:execution", "release:slot"]);
    expect(manager.getModelOperationStats().admission.active).toBe(0);
  });
});
