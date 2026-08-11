import { describe, expect, mock, test } from "bun:test";
import {
  probeStartupDatabaseRuntime,
  runStartupEnvironmentChecks,
  type StartupCheckDependencies,
} from "../../../src/utils/startup-checks";

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
    ensureManagedStorageRoot: mock(async (managedRoot: string) => managedRoot),
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
  test("returns a healthy startup probe connection after validating its session", async () => {
    const release = mock(() => {});
    const destroy = mock(() => {});
    const connection = {
      query: mock(async () => [[{
        ok: 1,
        session_time_zone: "+00:00",
        utc_offset_seconds: 0,
      }]]),
      release,
      destroy,
    };

    await probeStartupDatabaseRuntime({
      getConnection: mock(async () => connection),
    } as never);

    expect(release).toHaveBeenCalledTimes(1);
    expect(destroy).not.toHaveBeenCalled();
  });

  test("fails closed when the database wall clock is not aligned with DB_TIMEZONE", async () => {
    const destroy = mock(() => {});
    const release = mock(() => {});
    const connection = {
      query: mock(async () => [[{
        ok: 1,
        session_time_zone: "SYSTEM",
        utc_offset_seconds: 32_400,
      }]]),
      release,
      destroy,
    };

    await expect(probeStartupDatabaseRuntime({
      getConnection: mock(async () => connection),
    } as never, "Z")).rejects.toThrow("数据库会话时区不一致");
    expect(release).not.toHaveBeenCalled();
    expect(destroy).toHaveBeenCalledTimes(1);
  });

  test("destroys a startup probe connection after the health query fails", async () => {
    const destroy = mock(() => {});
    const release = mock(() => {});
    const connection = {
      query: mock(async () => { throw new Error("database unavailable"); }),
      release,
      destroy,
    };

    await expect(probeStartupDatabaseRuntime({
      getConnection: mock(async () => connection),
    } as never)).rejects.toThrow("database unavailable");
    expect(release).not.toHaveBeenCalled();
    expect(destroy).toHaveBeenCalledTimes(1);
  });

  test("destroys a healthy probe connection when returning it to the pool fails", async () => {
    const destroy = mock(() => {});
    const connection = {
      query: mock(async () => [[{
        ok: 1,
        session_time_zone: "+00:00",
        utc_offset_seconds: 0,
      }]]),
      release: mock(() => { throw new Error("pool release failed"); }),
      destroy,
    };

    await expect(probeStartupDatabaseRuntime({
      getConnection: mock(async () => connection),
    } as never)).rejects.toThrow("pool release failed");
    expect(destroy).toHaveBeenCalledTimes(1);
  });

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
      throw new Error("database unavailable");
    });

    await expect(runStartupEnvironmentChecks(deps)).rejects.toThrow("启动环境检查失败");
    expect(deps.warnings).toContain("数据库运行时检查失败: database unavailable");
  });
});
