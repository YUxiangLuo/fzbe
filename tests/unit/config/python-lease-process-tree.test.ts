import { afterAll, expect, mock, test } from "bun:test";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const python = Bun.which("python3");
const setsid = Bun.which("setsid");
const fixture = await mkdtemp(path.join(os.tmpdir(), "fz-python-lease-"));
const originalGrace = process.env.MODEL_PROCESS_TERMINATION_GRACE_MS;
process.env.MODEL_PROCESS_TERMINATION_GRACE_MS = "100";
await mkdir(path.join(fixture, "src"));
const script = [
  "import json, os, signal, subprocess, sys",
  "child_code = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(\"READY\",flush=True); time.sleep(30)'",
  "child = subprocess.Popen([sys.executable,'-c',child_code],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)",
  "child.stdout.readline()",
  "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
  `with open(${JSON.stringify(path.join(fixture, "pids.json"))},'w') as f: json.dump({'parent':os.getpid(),'child':child.pid},f)`,
  "child.wait()",
].join("\n");
for (const entry of ["run_model.py", "guided_model.py", "src/adf.py"]) {
  await Bun.write(path.join(fixture, entry), script);
}
await Bun.write(path.join(fixture, "src/__init__.py"), "");

mock.module(path.resolve(import.meta.dir, "../../../src/utils/python.ts"), () => ({
  PY_DIR: fixture,
  resolvePythonExecutable: () => python!,
}));
// Auxiliary process waiting does not use the store. Keep the process/lease
// integration independent of local DB credentials and fail if it touches SQL.
const unexpectedDatabaseWrite = () => { throw new Error("Process-tree regression must not access the database"); };
mock.module(path.resolve(import.meta.dir, "../../../src/modules/models/model-run-store.ts"), () => ({
  backfillArtifactSha256ForRun: unexpectedDatabaseWrite,
  backfillProductionArtifactSha256ForRun: unexpectedDatabaseWrite,
  clearProductionArtifactForRun: unexpectedDatabaseWrite,
}));
const { runPythonModel, runPythonGuidedTrainingStep } = await import("../../../src/modules/models/model-python-runner");
const { waitForAuxiliaryPythonProcess } = await import("../../../src/modules/models/model-artifacts");
const { runAdf } = await import("../../../src/utils/adf");
const { createModelOperationLeaseManager } = await import("../../../src/modules/models/model-operation-lease");
const { createModelExecutionLockManager } = await import("../../../src/modules/models/model-execution-lock");
const { ModelTimeoutError } = await import("../../../src/modules/models/model-errors");

afterAll(async () => {
  if (originalGrace === undefined) delete process.env.MODEL_PROCESS_TERMINATION_GRACE_MS;
  else process.env.MODEL_PROCESS_TERMINATION_GRACE_MS = originalGrace;
  await rm(fixture, { recursive: true, force: true });
});

const groupExists = (pid: number) => {
  try { process.kill(-pid, 0); return true; } catch { return false; }
};

for (const kind of ["ordinary", "guided", "auxiliary", "adf"] as const) {
  test.skipIf(!python || !setsid)(`${kind} keeps its lease until a TERM-resistant descendant exits`, async () => {
    const pidFile = path.join(fixture, "pids.json");
    await rm(pidFile, { force: true });
    const locks = createModelExecutionLockManager();
    const manager = createModelOperationLeaseManager({
      capacity: 1, deadlineMs: 10_000, executionLocks: locks,
      logger: { info() {}, warn() {} },
    });
    const controller = new AbortController();
    const execution = { execution: { experimentId: 1, modelType: "ma" as const } };
    const operation = manager.withModelOperationContext("prediction", (context) =>
      manager.withModelOperationLease(context, execution, async () => {
        if (kind === "ordinary") return runPythonModel("ma", {}, 500, undefined, controller.signal);
        if (kind === "guided") {
          return runPythonGuidedTrainingStep("ma", "save_artifact", {}, "attempt", null,
            path.join(fixture, "candidate"), path.join(fixture, "receipt"), fixture, 500, controller.signal);
        }
        if (kind === "adf") return runAdf([{ month: "1", sales: 1 }], controller.signal);
        const proc = Bun.spawn([setsid!, python!, path.join(fixture, "run_model.py")], {
          stdout: "pipe", stderr: "pipe", env: { ...process.env },
        });
        return waitForAuxiliaryPythonProcess(proc, "test auxiliary", {
          timeoutMs: 500, terminationGraceMs: 100, processGroup: true,
          terminationSignal: controller.signal,
        });
      }),
    );
    // The ADF path uses the same injected deadline reason; its normal hard cap
    // is deliberately longer than this process-tree regression needs.
    const timer = kind === "adf"
      ? setTimeout(() => controller.abort(new ModelTimeoutError("test deadline", 500)), 500)
      : undefined;
    let pids: { parent: number; child: number } | undefined;
    try {
      const error = await operation.catch((caught) => caught);
      expect(error).toBeInstanceOf(ModelTimeoutError);
      pids = await Bun.file(pidFile).json();
      expect(groupExists(pids!.parent)).toBe(false);
      expect(manager.getModelOperationStats().admission.active).toBe(0);
      expect(locks.getStats().active).toBe(0);
      await manager.withModelOperationContext("prediction", (context) =>
        manager.withModelOperationLease(context, execution, async () => {
          expect(groupExists(pids!.parent)).toBe(false);
        }),
      );
    } finally {
      if (timer) clearTimeout(timer);
      if (!pids && await Bun.file(pidFile).exists()) pids = await Bun.file(pidFile).json();
      if (pids) {
        try { process.kill(-pids.parent, "SIGKILL"); } catch { /* already gone */ }
      }
      controller.abort(new Error("test cleanup"));
      await operation.catch(() => {});
    }
  }, 15_000);
}
