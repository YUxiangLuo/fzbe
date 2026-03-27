import { describe, expect, mock, test } from "bun:test";
import { runStartupEnvironmentChecks, type StartupCheckDependencies } from "../../../src/utils/startup-checks";

const createDeps = (): StartupCheckDependencies & {
  warnings: string[];
  infos: string[];
  commands: Array<{ command: string[]; cwd?: string; timeoutMs?: number }>;
} => {
  const warnings: string[] = [];
  const infos: string[] = [];
  const commands: Array<{ command: string[]; cwd?: string; timeoutMs?: number }> = [];

  return {
    warnings,
    infos,
    commands,
    ensureExecutableFile: mock(() => {}),
    ensureWritableDirectory: mock(() => {}),
    fileExists: mock(() => true),
    getChromiumPath: () => "/usr/bin/chromium",
    getPyDir: () => "/tmp/py",
    logInfo: (message) => {
      infos.push(message);
    },
    logWarn: (message) => {
      warnings.push(message);
    },
    queryDatabase: mock(async () => {}),
    resolvePythonExecutable: mock(() => "/tmp/py/venv/bin/python"),
    runCommand: mock(async (input) => {
      commands.push(input);
    }),
  };
};

describe("startup environment checks", () => {
  test("runs lightweight runtime probes for Python and Chromium", async () => {
    const deps = createDeps();

    await runStartupEnvironmentChecks(deps);

    expect(deps.commands).toEqual([
      {
        label: "Python ADF 自检",
        command: ["/tmp/py/venv/bin/python", "-m", "src.adf", "--help"],
        cwd: "/tmp/py",
      },
      {
        label: "Python 模型运行时自检",
        command: ["/tmp/py/venv/bin/python", "run_model.py", "--help"],
        cwd: "/tmp/py",
      },
      {
        command: ["/usr/bin/chromium", "--version"],
        timeoutMs: 10_000,
      },
    ]);
    expect(deps.warnings).toHaveLength(0);
    expect(deps.infos).toContain("Startup environment checks passed.");
  });

  test("fails startup when a runtime probe cannot execute", async () => {
    const deps = createDeps();
    deps.runCommand = mock(async (input) => {
      deps.commands.push(input);
      if (input.command[1] === "run_model.py") {
        throw new Error("No module named tensorflow");
      }
    });

    await expect(runStartupEnvironmentChecks(deps)).rejects.toThrow("启动环境检查失败");
    expect(deps.warnings).toContain("Python 模型运行时自检失败: No module named tensorflow");
  });
});
