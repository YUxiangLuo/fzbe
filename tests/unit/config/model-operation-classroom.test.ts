import { expect, mock, test } from "bun:test";
import {
  createModelOperationLeaseManager,
  type ModelExecutionLockAdapter,
} from "../../../src/modules/models/model-operation-lease";
import { ModelCapacityBusyError } from "../../../src/modules/models/model-errors";
import type { ModelExecutionLockHandle } from "../../../src/modules/models/model-execution-lock";

const tryAcquire = mock((experimentId: number) => ({
  key: `${experimentId}:ma`,
  token: Symbol(String(experimentId)),
}) satisfies ModelExecutionLockHandle);
const release = mock(() => undefined);
const executionLocks: ModelExecutionLockAdapter = {
  tryAcquire,
  release,
  verifyOwned: () => undefined,
};
const manager = createModelOperationLeaseManager({
  capacity: 50,
  deadlineMs: 5_000,
  executionLocks,
  logger: { info: () => {}, warn: () => {} },
});

test("single 64-core instance: 100 simultaneous students admit 50 and fail-fast 50", async () => {
  const releaseCallbacks: Array<() => void> = [];
  const startHeldOperation = (experimentId: number) =>
    manager.withModelOperationContext("prediction", async (context) =>
      await manager.withModelOperationLease(
        context,
        { execution: { experimentId, modelType: "ma" } },
        async () => await new Promise<void>((resolve) => releaseCallbacks.push(resolve)),
      ));

  const admitted = Array.from({ length: 50 }, (_, index) => startHeldOperation(index + 1));
  while (releaseCallbacks.length < 50) await Bun.sleep(1);

  const rejected = await Promise.all(
    Array.from({ length: 50 }, (_, index) =>
      startHeldOperation(index + 51).catch((error) => error)),
  );

  expect(rejected.every((error) => error instanceof ModelCapacityBusyError)).toBe(true);
  expect(tryAcquire).toHaveBeenCalledTimes(50);
  expect(manager.getModelOperationStats().admission).toEqual({
    scope: "process",
    capacity: 50,
    active: 50,
    pending: 0,
  });

  for (const resolve of releaseCallbacks) resolve();
  await Promise.all(admitted);

  expect(release).toHaveBeenCalledTimes(50);
  expect(manager.getModelOperationStats().admission.active).toBe(0);
});
