import { afterEach, describe, expect, test } from "bun:test";
import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

const projectRoot = path.resolve(import.meta.dir, "../../..");
const bunRunner = path.join(projectRoot, "scripts/run-pinned-bun.sh");
const uvRunner = path.join(projectRoot, "py/run-pinned-uv.sh");
const pythonVersionFile = path.join(projectRoot, "py/.python-version");
const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) => rm(directory, {
    recursive: true,
    force: true,
  })));
});

const createExecutable = async (filePath: string, contents: string) => {
  await writeFile(filePath, contents);
  await chmod(filePath, 0o755);
};

describe("commit-pinned tool runners", () => {
  test("keeps the Python runtime version in the checkout", async () => {
    expect(await Bun.file(pythonVersionFile).text()).toBe("3.13\n");
  });

  test("uses a versioned Bun package when the bootstrap version differs", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "fzbe-pinned-bun-"));
    temporaryDirectories.push(directory);
    const fakeBin = path.join(directory, "bin");
    const invocationLog = path.join(directory, "bun-x-args");
    await mkdir(fakeBin);
    await createExecutable(path.join(fakeBin, "bun"), `#!/usr/bin/env bash
if [[ "\${1-}" == "--version" ]]; then
  printf '1.3.14\\n'
  exit 0
fi
printf '%s\\n' "$*" > "$PINNED_TOOL_TEST_LOG"
printf '1.4.2\\n'
`);

    const result = Bun.spawnSync([bunRunner, "--version"], {
      cwd: projectRoot,
      env: {
        ...process.env,
        PATH: `${fakeBin}:/usr/bin:/bin`,
        PINNED_TOOL_TEST_LOG: invocationLog,
      },
      stdout: "pipe",
      stderr: "pipe",
    });

    expect([result.exitCode, result.stdout.toString().trim()]).toEqual([0, "1.4.2"]);
    expect((await Bun.file(invocationLog).text()).trim()).toBe("x bun@1.4.2 --version");
  });

  test("uses a versioned uv tool environment instead of replacing the bootstrap", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "fzbe-pinned-uv-"));
    temporaryDirectories.push(directory);
    const fakeBin = path.join(directory, "bin");
    const invocationLog = path.join(directory, "uv-args");
    await mkdir(fakeBin);
    await createExecutable(path.join(fakeBin, "uv"), `#!/usr/bin/env bash
if [[ "\${1-}" == "--version" ]]; then
  printf 'uv 0.11.0 (fake)\\n'
  exit 0
fi
printf '%s\\n' "$*" > "$PINNED_TOOL_TEST_LOG"
printf 'uv 0.12.7 (versioned fake)\\n'
`);

    const result = Bun.spawnSync([uvRunner, "--version"], {
      cwd: projectRoot,
      env: {
        ...process.env,
        PATH: `${fakeBin}:/usr/bin:/bin`,
        PINNED_TOOL_TEST_LOG: invocationLog,
      },
      stdout: "pipe",
      stderr: "pipe",
    });

    expect([result.exitCode, result.stdout.toString().trim()]).toEqual([
      0,
      "uv 0.12.7 (versioned fake)",
    ]);
    expect((await Bun.file(invocationLog).text()).trim()).toBe(
      "tool run --from uv==0.12.7 uv --version",
    );
  });
});
