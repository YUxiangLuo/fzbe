import { describe, expect, mock, test } from "bun:test";
import { runStartupEnvironmentChecks, type StartupCheckDependencies } from "../../../src/utils/startup-checks";

const createDeps = (): StartupCheckDependencies & {
  warnings: string[];
  infos: string[];
  chromiumProbes: Array<{ label?: string; chromiumPath: string; timeoutMs?: number }>;
  commands: Array<{ label?: string; command: string[]; cwd?: string; timeoutMs?: number }>;
} => {
  const warnings: string[] = [];
  const infos: string[] = [];
  const chromiumProbes: Array<{ label?: string; chromiumPath: string; timeoutMs?: number }> = [];
  const commands: Array<{ label?: string; command: string[]; cwd?: string; timeoutMs?: number }> = [];

  return {
    warnings,
    infos,
    chromiumProbes,
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
    probeChromiumRuntime: mock(async (input) => {
      chromiumProbes.push(input);
    }),
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
    ]);
    expect(deps.chromiumProbes).toEqual([
      {
        label: "Chromium/Puppeteer 自检",
        chromiumPath: "/usr/bin/chromium",
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

  test("fails startup when the database runtime probe fails", async () => {
    const deps = createDeps();
    deps.queryDatabase = mock(async () => {
      throw new Error("GET_LOCK permission denied");
    });

    await expect(runStartupEnvironmentChecks(deps)).rejects.toThrow("启动环境检查失败");
    expect(deps.warnings).toContain("数据库运行时检查失败: GET_LOCK permission denied");
  });
});
