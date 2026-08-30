import { afterEach, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { calculateModelRuntimeRevision } from "../../../src/modules/models/model-runtime-identity";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) => rm(directory, {
    recursive: true,
    force: true,
  })));
});

describe("model runtime revision inputs", () => {
  test("ignores development dependency metadata but tracks production runtime changes", async () => {
    const pythonRoot = await mkdtemp(path.join(tmpdir(), "fzbe-runtime-revision-"));
    temporaryDirectories.push(pythonRoot);
    await mkdir(path.join(pythonRoot, "src"));
    await Promise.all([
      writeFile(path.join(pythonRoot, ".python-version"), "3.13\n"),
      writeFile(path.join(pythonRoot, "production-dependencies.lock"), "numpy==2.4.4\n"),
      writeFile(path.join(pythonRoot, "guided_model.py"), "print('guided')\n"),
      writeFile(path.join(pythonRoot, "src", "runtime.py"), "VALUE = 1\n"),
      writeFile(path.join(pythonRoot, "pyproject.toml"), "[dependency-groups]\ndev=['ruff==1']\n"),
      writeFile(path.join(pythonRoot, "uv.lock"), "ruff==1\n"),
    ]);

    const baseline = calculateModelRuntimeRevision(pythonRoot);
    await writeFile(path.join(pythonRoot, "pyproject.toml"), "[dependency-groups]\ndev=['ruff==2']\n");
    await writeFile(path.join(pythonRoot, "uv.lock"), "ruff==2\n");
    expect(calculateModelRuntimeRevision(pythonRoot)).toBe(baseline);

    await writeFile(path.join(pythonRoot, "production-dependencies.lock"), "numpy==2.5.0\n");
    expect(calculateModelRuntimeRevision(pythonRoot)).not.toBe(baseline);

    await writeFile(path.join(pythonRoot, "production-dependencies.lock"), "numpy==2.4.4\n");
    await writeFile(path.join(pythonRoot, "src", "runtime.py"), "VALUE = 2\n");
    expect(calculateModelRuntimeRevision(pythonRoot)).not.toBe(baseline);
  });
});
