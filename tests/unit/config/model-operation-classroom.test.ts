import { expect, mock, test } from "bun:test";
import {
  createModelOperationLeaseManager,
  type ModelOperationLockAdapter,
} from "../../../src/modules/models/model-operation-lease";
import { ModelCapacityBusyError } from "../../../src/modules/models/model-errors";
import type { AdvisoryLockHandle } from "../../../src/modules/models/model-locks";

let sequence = 0;
const makeHandle = (prefix: string): AdvisoryLockHandle => ({
  connection: {} as never,
  name: `${prefix}-${++sequence}`,
  ownerConnectionId: sequence,
  signal: new AbortController().signal,
  stopHeartbeat: () => {},
});
const acquireModelSlot = mock(async () => makeHandle("slot"));
const acquireModelExecutionLock = mock(async () => makeHandle("execution"));
const releaseModelExecutionLock = mock(async () => undefined);
const releaseModelSlot = mock(async () => undefined);

const locks: ModelOperationLockAdapter = {
  acquireModelSlot,
  acquireModelExecutionLock,
  releaseModelExecutionLock,
  releaseModelSlot,
  verifyModelLockOwned: async () => undefined,
  combineModelLockSignals: (...handles: Array<{ signal?: AbortSignal } | null>) => {
    const signals = handles.map((handle) => handle?.signal).filter(Boolean) as AbortSignal[];
    return signals.length === 1 ? signals[0] : AbortSignal.any(signals);
  },
};
const manager = createModelOperationLeaseManager({
  capacity: 50,
  deadlineMs: 5_000,
  locks,
  logger: { info: () => {}, warn: () => {} },
});

test("64-core classroom: 100 simultaneous students admit 50 and fail-fast 50", async () => {
  const releaseCallbacks: Array<() => void> = [];
  const startHeldOperation = (experimentId: number) =>
    manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        { mode: "slot_and_execution", experimentId, modelType: "ma" },
        async () => await new Promise<void>((resolve) => releaseCallbacks.push(resolve)),
      ));

  const admitted = Array.from({ length: 50 }, (_, index) => startHeldOperation(index + 1));
  while (releaseCallbacks.length < 50) await Bun.sleep(1);

  const rejected = await Promise.all(
    Array.from({ length: 50 }, (_, index) =>
      startHeldOperation(index + 51).catch((error) => error)),
  );

  expect(rejected.every((error) => error instanceof ModelCapacityBusyError)).toBe(true);
  expect(acquireModelSlot).toHaveBeenCalledTimes(50);
  expect(acquireModelExecutionLock).toHaveBeenCalledTimes(50);
  expect(manager.getModelOperationStats().admission).toEqual({
    capacity: 50,
    active: 50,
    pending: 0,
  });

  for (const release of releaseCallbacks) release();
  await Promise.all(admitted);

  expect(releaseModelExecutionLock).toHaveBeenCalledTimes(50);
  expect(releaseModelSlot).toHaveBeenCalledTimes(50);
  expect(manager.getModelOperationStats().admission.active).toBe(0);
});
