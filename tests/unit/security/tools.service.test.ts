import { expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const adfModulePath = path.join(srcRoot, "utils/adf.ts");
const leaseModulePath = path.join(srcRoot, "modules/models/model-operation-lease.ts");
const toolsServiceModulePath = path.join(srcRoot, "modules/tools/tools.service.ts");

const terminationController = new AbortController();
const runAdf = mock(async () => [{
  diff_order: 0,
  statistic: -3,
  p_value: 0.01,
  stationary: true,
  critical_values: { "1%": -3, "5%": -2, "10%": -1 },
}]);
const withModelOperationContext = mock(async (
  _operation: string,
  callback: (context: Record<string, unknown>) => Promise<unknown>,
  _deadlineMs?: number,
) => await callback({}));
const withModelOperationLease = mock(async (
  _context: Record<string, unknown>,
  _options: Record<string, unknown>,
  callback: (lease: { terminationSignal: AbortSignal; throwIfTerminated(): void }) => Promise<unknown>,
) => await callback({
  terminationSignal: terminationController.signal,
  throwIfTerminated: () => {},
}));

mock.module(adfModulePath, () => ({ runAdf }));
mock.module(leaseModulePath, () => ({
  withModelOperationContext,
  withModelOperationLease,
}));

const { performAdfTest } = await import(`${toolsServiceModulePath}?tools-service-signal-test`);

test("passes the internal lease termination signal to ADF", async () => {
  const series = [{ month: "2025-01", sales: 100 }];
  await performAdfTest(series);

  expect(withModelOperationContext).toHaveBeenCalledWith("adf", expect.any(Function), 32_000);
  expect(withModelOperationLease).toHaveBeenCalledWith(
    expect.anything(),
    { mode: "slot_only" },
    expect.any(Function),
  );
  expect(runAdf).toHaveBeenCalledWith(series, terminationController.signal);
});
