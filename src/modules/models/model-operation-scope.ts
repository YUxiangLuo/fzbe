import { AsyncLocalStorage } from "node:async_hooks";

const operationSignalStorage = new AsyncLocalStorage<AbortSignal>();

export const runWithModelOperationSignal = <T>(
  signal: AbortSignal,
  callback: () => T,
): T => operationSignalStorage.run(signal, callback);

export const getCurrentModelOperationSignal = (): AbortSignal | undefined =>
  operationSignalStorage.getStore();
