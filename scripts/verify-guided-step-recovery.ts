import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { PY_DIR, resolvePythonExecutable } from "../src/utils/python";

const assert: (condition: unknown, message: string) => asserts condition = (condition, message) => {
  if (!condition) throw new Error(message);
};

const python = resolvePythonExecutable();
const script = path.join(PY_DIR, "guided_model.py");
const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "fz-guided-recovery-"));
const artifactDir = path.join(workDir, "artifact");
const stateFile = path.join(artifactDir, "state.pkl");
const modelFile = path.join(artifactDir, "ma.pkl");

const runPrepare = async (data: unknown[]) => {
  const proc = Bun.spawn([
    python,
    script,
    "ma",
    "prepare_data",
    "--state-file",
    stateFile,
    "--artifact-dir",
    artifactDir,
    "--data-stdin",
    "--train-start",
    "0",
    "--train-end",
    "4",
    "--eval-start",
    "5",
    "--eval-end",
    "7",
    "--window",
    "2",
    "--save-model",
    modelFile,
  ], {
    cwd: PY_DIR,
    stdin: new Blob([JSON.stringify(data)]),
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return { stdout, stderr, exitCode };
};

const validData = Array.from({ length: 8 }, (_, index) => ({
  月份: `2026-${String(index + 1).padStart(2, "0")}`,
  销售数量: 100 + index,
}));

const verifyAtomicMemberPublication = async () => {
  const memberDir = path.join(workDir, "member-publication");
  const checkCode = `
import json, os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "src"))
import ensemble_utils

member_dir = sys.argv[2]
os.makedirs(member_dir, exist_ok=True)
source = os.path.join(member_dir, "source.pkl")
destination = os.path.join(member_dir, "destination.pkl")
with open(source, "wb") as handle:
    handle.write(b"complete-new-artifact")
with open(destination, "wb") as handle:
    handle.write(b"previous-valid-artifact")

real_copy = ensemble_utils.shutil.copy2
def interrupted_copy(source_path, temp_path):
    with open(temp_path, "wb") as handle:
        handle.write(b"partial")
    raise OSError("simulated interrupted copy")

ensemble_utils.shutil.copy2 = interrupted_copy
try:
    ensemble_utils.persist_member_model_artifact(
        "ma", source, destination, move_source=False
    )
except OSError:
    pass
else:
    raise AssertionError("interrupted member copy unexpectedly succeeded")
finally:
    ensemble_utils.shutil.copy2 = real_copy

with open(destination, "rb") as handle:
    assert handle.read() == b"previous-valid-artifact"
with open(source, "rb") as handle:
    assert handle.read() == b"complete-new-artifact"

ensemble_utils.persist_member_model_artifact(
    "ma", source, destination, move_source=False
)
with open(destination, "rb") as handle:
    assert handle.read() == b"complete-new-artifact"
print(json.dumps({"partial_copy_not_published": True, "retry_published_complete_copy": True}))
`;
  const proc = Bun.spawn([python, "-c", checkCode, PY_DIR, memberDir], {
    cwd: PY_DIR,
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  assert(exitCode === 0, `atomic ensemble member verification failed: ${stderr}`);
  return JSON.parse(stdout) as {
    partial_copy_not_published: boolean;
    retry_published_complete_copy: boolean;
  };
};

try {
  const firstRun = await runPrepare(validData);
  assert(firstRun.exitCode === 0, `first guided step failed: ${firstRun.stderr}`);
  const firstPayload = JSON.parse(firstRun.stdout) as Record<string, unknown>;
  const firstState = fs.readFileSync(stateFile);
  const firstStateHash = crypto.createHash("sha256").update(firstState).digest("hex");

  // This input cannot satisfy the requested ranges. Success therefore proves
  // that the durable cached payload was replayed instead of recomputing the step.
  const replayRun = await runPrepare([{ 月份: "invalid", 销售数量: 1 }]);
  assert(replayRun.exitCode === 0, `cached guided step replay failed: ${replayRun.stderr}`);
  const replayPayload = JSON.parse(replayRun.stdout) as Record<string, unknown>;
  const replayStateHash = crypto.createHash("sha256").update(fs.readFileSync(stateFile)).digest("hex");
  assert(
    JSON.stringify(replayPayload) === JSON.stringify(firstPayload),
    "replayed guided payload differs from the durable first result",
  );
  assert(replayStateHash === firstStateHash, "cached replay unexpectedly rewrote guided state");

  const readyFile = path.join(workDir, "holder-ready");
  const holderCode = [
    "import fcntl, os, sys, time",
    "artifact_dir, ready_file = sys.argv[1], sys.argv[2]",
    "os.makedirs(artifact_dir, exist_ok=True)",
    "handle = open(os.path.join(artifact_dir, '.step.lock'), 'a+b')",
    "fcntl.flock(handle.fileno(), fcntl.LOCK_EX)",
    "open(ready_file, 'w').close()",
    "time.sleep(30)",
  ].join("; ");
  const holder = Bun.spawn([python, "-c", holderCode, artifactDir, readyFile], {
    cwd: PY_DIR,
    stdout: "ignore",
    stderr: "pipe",
  });

  for (let attempt = 0; attempt < 100 && !fs.existsSync(readyFile); attempt += 1) {
    await Bun.sleep(20);
  }
  assert(fs.existsSync(readyFile), "workspace lock holder did not become ready");

  const blockedRun = await runPrepare(validData);
  assert(blockedRun.exitCode !== 0, "a second process unexpectedly entered the locked workspace");
  assert(blockedRun.stderr.includes("[CONFLICT]"), "workspace conflict was not returned as a typed error");

  holder.kill("SIGTERM");
  await holder.exited;
  const recoveredRun = await runPrepare(validData);
  assert(recoveredRun.exitCode === 0, `guided step did not recover after lock release: ${recoveredRun.stderr}`);
  const memberPublication = await verifyAtomicMemberPublication();

  console.log(JSON.stringify({
    first_step_id: firstPayload.step_id,
    cached_replay_identical: true,
    state_hash_unchanged_on_replay: true,
    concurrent_workspace_exit_code: blockedRun.exitCode,
    concurrent_workspace_error: blockedRun.stderr.trim(),
    recovered_after_lock_release: true,
    ensemble_member_partial_copy_not_published: memberPublication.partial_copy_not_published,
    ensemble_member_retry_published_complete_copy: memberPublication.retry_published_complete_copy,
  }, null, 2));
} finally {
  fs.rmSync(workDir, { recursive: true, force: true });
}
