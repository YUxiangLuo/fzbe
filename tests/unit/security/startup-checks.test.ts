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
  test("returns a healthy startup probe session only after lock release and rollback are acknowledged", async () => {
    const release = mock(() => {});
    const destroy = mock(() => {});
    const rollback = mock(async () => {});
    const query = mock(async (sql: string) => {
      if (sql === "SELECT 1 AS ok") return [[{ ok: 1 }]];
      if (sql.includes("GET_LOCK")) return [[{ acquired: 1 }]];
      if (sql.includes("RELEASE_LOCK")) return [[{ released: 1 }]];
      return [[]];
    });
    const connection = {
      beginTransaction: mock(async () => {}),
      query,
      rollback,
      release,
      destroy,
    };

    await probeStartupDatabaseRuntime({
      getConnection: mock(async () => connection),
    } as never);

    expect(rollback).toHaveBeenCalledTimes(1);
    expect(release).toHaveBeenCalledTimes(1);
    expect(destroy).not.toHaveBeenCalled();
  });

  test.each([
    ["GET_LOCK acknowledgement loss", "get-lock-throws"],
    ["release query failure", "release-throws"],
    ["unconfirmed release", "release-zero"],
    ["rollback acknowledgement loss", "rollback-throws"],
    ["pool release failure", "pool-release-throws"],
  ] as const)("destroys a startup probe session after %s", async (_label, failure) => {
    const release = mock(() => {
      if (failure === "pool-release-throws") throw new Error("pool release failed");
    });
    const destroy = mock(() => {});
    const rollback = mock(async () => {
      if (failure === "rollback-throws") throw new Error("rollback acknowledgement lost");
    });
    const query = mock(async (sql: string) => {
      if (sql === "SELECT 1 AS ok") return [[{ ok: 1 }]];
      if (sql.includes("GET_LOCK")) {
        if (failure === "get-lock-throws") throw new Error("GET_LOCK acknowledgement lost");
        return [[{ acquired: 1 }]];
      }
      if (sql.includes("RELEASE_LOCK")) {
        if (failure === "release-throws") throw new Error("release acknowledgement lost");
        return [[{ released: failure === "release-zero" ? 0 : 1 }]];
      }
      return [[]];
    });
    const connection = {
      beginTransaction: mock(async () => {}),
      query,
      rollback,
      release,
      destroy,
    };

    await expect(probeStartupDatabaseRuntime({
      getConnection: mock(async () => connection),
    } as never)).rejects.toThrow();

    if (failure === "pool-release-throws") {
      expect(release).toHaveBeenCalledTimes(1);
    } else {
      expect(release).not.toHaveBeenCalled();
    }
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
      throw new Error("GET_LOCK permission denied");
    });

    await expect(runStartupEnvironmentChecks(deps)).rejects.toThrow("启动环境检查失败");
    expect(deps.warnings).toContain("数据库运行时检查失败: GET_LOCK permission denied");
  });
});
