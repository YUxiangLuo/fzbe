import { afterEach, expect, jest, mock, test } from "bun:test";
import path from "node:path";
import { ModelTimeoutError } from "../../../src/modules/models/model-errors";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const pythonModulePath = path.join(srcRoot, "utils/python.ts");
const adfModulePath = path.join(srcRoot, "utils/adf.ts");

mock.module(pythonModulePath, () => ({
  PY_DIR: "/tmp/fangzhen-adf-test",
  resolvePythonExecutable: () => "/usr/bin/python",
}));

const { runAdf } = await import(`${adfModulePath}?adf-cancellation-test`);
const originalSpawn = Bun.spawn;

afterEach(() => {
  jest.useRealTimers();
  (Bun as typeof Bun & { spawn: typeof Bun.spawn }).spawn = originalSpawn;
});

test("terminates and awaits the ADF subprocess even when killed pipes error", async () => {
  let stdoutController!: ReadableStreamDefaultController<Uint8Array>;
  let stderrController!: ReadableStreamDefaultController<Uint8Array>;
  let resolveExit!: (exitCode: number) => void;
  const exited = new Promise<number>((resolve) => {
    resolveExit = resolve;
  });
  const proc = {
    stdout: new ReadableStream<Uint8Array>({
      start(controller) { stdoutController = controller; },
    }),
    stderr: new ReadableStream<Uint8Array>({
      start(controller) { stderrController = controller; },
    }),
    exited,
    exitCode: null as number | null,
    pid: 999_999_997,
    kill: mock((signal: number | NodeJS.Signals) => {
      proc.exitCode = 137;
      stdoutController.error(new Error("stdout pipe closed"));
      stderrController.error(new Error("stderr pipe closed"));
      return true;
    }),
  };
  const spawn = mock(() => proc as never);
  (Bun as typeof Bun & { spawn: typeof Bun.spawn }).spawn = spawn as typeof Bun.spawn;

  const controller = new AbortController();
  const lockLost = Object.assign(new Error("数据库模型锁已失效"), {
    code: "MODEL_LOCK_LOST",
  });
  const result = runAdf([{ month: "2025-01", sales: 100 }], controller.signal);
  while (spawn.mock.calls.length === 0) await Bun.sleep(1);
  controller.abort(lockLost);

  let settled = false;
  void result.finally(() => { settled = true; }).catch(() => {});
  await Bun.sleep(1);
  expect(settled).toBe(false);

  resolveExit(137);
  await expect(result).rejects.toBe(lockLost);
  expect(proc.kill).toHaveBeenCalledWith("SIGTERM");
  expect(proc.exitCode).toBe(137);
});

test("returns a typed timeout only after the ADF subprocess exits", async () => {
  jest.useFakeTimers();
  let stdoutController!: ReadableStreamDefaultController<Uint8Array>;
  let stderrController!: ReadableStreamDefaultController<Uint8Array>;
  let resolveExit!: (exitCode: number) => void;
  const exited = new Promise<number>((resolve) => {
    resolveExit = resolve;
  });
  const proc = {
    stdout: new ReadableStream<Uint8Array>({
      start(controller) { stdoutController = controller; },
    }),
    stderr: new ReadableStream<Uint8Array>({
      start(controller) { stderrController = controller; },
    }),
    exited,
    exitCode: null as number | null,
    pid: 999_999_996,
    kill: mock((_: number | NodeJS.Signals) => {
      proc.exitCode = 143;
      stdoutController.error(new Error("stdout pipe closed"));
      stderrController.error(new Error("stderr pipe closed"));
      resolveExit(143);
      return true;
    }),
  };
  const spawn = mock(() => proc as never);
  (Bun as typeof Bun & { spawn: typeof Bun.spawn }).spawn = spawn as typeof Bun.spawn;

  const result = runAdf([{ month: "2025-01", sales: 100 }]);
  for (let attempt = 0; attempt < 10 && spawn.mock.calls.length === 0; attempt += 1) {
    await Promise.resolve();
  }
  expect(spawn).toHaveBeenCalledTimes(1);
  jest.advanceTimersByTime(30_000);

  const error = await result.catch((caughtError) => caughtError);
  expect(error).toBeInstanceOf(ModelTimeoutError);
  expect(error).toMatchObject({
    code: "MODEL_TIMEOUT",
    timeoutMs: 30_000,
    source: "python_subprocess",
  });
  expect(proc.kill).toHaveBeenCalledWith("SIGTERM");
});
