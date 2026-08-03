import { expect, test } from "bun:test";
import { terminateProcessTree } from "../../../src/utils/process-tree";

const processExists = (pid: number) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return !(
      error instanceof Error
      && "code" in error
      && (error as NodeJS.ErrnoException).code === "ESRCH"
    );
  }
};

const processGroupExists = (pid: number) => {
  try {
    process.kill(-pid, 0);
    return true;
  } catch (error) {
    return !(
      error instanceof Error
      && "code" in error
      && (error as NodeJS.ErrnoException).code === "ESRCH"
    );
  }
};

const readFirstLine = async (stream: ReadableStream<Uint8Array>) => {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let text = "";
  try {
    while (!text.includes("\n")) {
      const { done, value } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
    }
  } finally {
    reader.releaseLock();
  }
  return text.split("\n", 1)[0]?.trim() ?? "";
};

test("kills surviving process-group members after the setsid leader exits", async () => {
  const setsid = process.platform === "win32" ? null : Bun.which("setsid");
  const python = Bun.which("python3");
  if (!setsid || !python) return;

  const leaderScript = [
    "import signal, subprocess, sys",
    "child_code = 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGHUP, signal.SIG_IGN); print(\"READY\", flush=True); time.sleep(1000)'",
    "child = subprocess.Popen([sys.executable, '-c', child_code], stdout=subprocess.PIPE, text=True)",
    "child.stdout.readline()",
    "print(child.pid, flush=True)",
    "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
    "child.wait()",
  ].join("\n");

  const proc = Bun.spawn([
    setsid,
    python,
    "-c",
    leaderScript,
  ], {
    stdout: "pipe",
    stderr: "pipe",
  });
  const childPid = Number(await readFirstLine(proc.stdout));
  expect(Number.isInteger(childPid)).toBe(true);
  expect(processExists(childPid)).toBe(true);

  try {
    const startedAt = Date.now();
    await terminateProcessTree(proc, {
      graceMs: 50,
      processGroup: true,
      label: "process-tree-test",
    });

    // Bun may publish the convenience exitCode property one scheduler turn
    // after the authoritative exited promise resolves under heavy load.
    expect(Number.isInteger(await proc.exited)).toBe(true);
    expect(Date.now() - startedAt).toBeGreaterThanOrEqual(40);
    expect(processGroupExists(proc.pid)).toBe(false);
  } finally {
    try {
      process.kill(-proc.pid, "SIGKILL");
    } catch {
      // The expected path has already removed the complete process group.
    }
    await proc.exited;
  }
});
