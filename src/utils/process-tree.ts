import logger from "./logger";

export interface TerminableProcess {
  readonly pid: number;
  readonly exited: Promise<number>;
  readonly exitCode: number | null;
  kill(signal?: number | NodeJS.Signals): unknown;
}

interface ProcessTreeTerminationOptions {
  graceMs: number;
  processGroup: boolean;
  label: string;
  logContext?: Record<string, unknown>;
}

const PROCESS_GROUP_POLL_INTERVAL_MS = 20;

const isNoSuchProcessError = (error: unknown) =>
  error instanceof Error
  && "code" in error
  && (error as NodeJS.ErrnoException).code === "ESRCH";

/**
 * A setsid leader can exit before its descendants. A negative PID probes the
 * whole POSIX process group, so proc.exited alone is not a tree-exit signal.
 */
const isProcessGroupAlive = (pid: number): boolean => {
  try {
    process.kill(-pid, 0);
    return true;
  } catch (error) {
    return !isNoSuchProcessError(error);
  }
};

const waitForProcessGroupExit = async (pid: number) => {
  while (isProcessGroupAlive(pid)) {
    await Bun.sleep(PROCESS_GROUP_POLL_INTERVAL_MS);
  }
};

/**
 * Terminates a subprocess and, when it was launched through setsid, does not
 * return until every member of that process group has disappeared.
 */
export async function terminateProcessTree(
  proc: TerminableProcess,
  options: ProcessTreeTerminationOptions,
): Promise<void> {
  const useProcessGroup = options.processGroup && process.platform !== "win32";
  const context = {
    ...options.logContext,
    processPid: proc.pid,
    label: options.label,
  };

  const signalTree = (signal: "SIGTERM" | "SIGKILL") => {
    if (useProcessGroup) {
      try {
        // Do not gate this on proc.exitCode: the setsid leader may already be
        // gone while descendants with the same PGID are still alive.
        process.kill(-proc.pid, signal);
        return;
      } catch (error) {
        if (!isNoSuchProcessError(error)) {
          logger.warn(
            { ...context, err: error, signal },
            "向进程组发送终止信号失败，回退为主进程",
          );
        }
      }
    }

    if (proc.exitCode !== null) return;
    try {
      proc.kill(signal);
    } catch (error) {
      logger.warn(
        { ...context, err: error, signal },
        "向主进程发送终止信号失败",
      );
    }
  };

  const waitForTreeExit = async () => {
    await proc.exited;
    if (useProcessGroup) {
      await waitForProcessGroupExit(proc.pid);
    }
  };

  signalTree("SIGTERM");
  const exitedGracefully = await Promise.race([
    waitForTreeExit().then(() => true),
    Bun.sleep(options.graceMs).then(() => false),
  ]);

  if (exitedGracefully) return;

  logger.warn(
    { ...context, waitMs: options.graceMs },
    "进程树未在宽限期内退出，发送 SIGKILL",
  );
  signalTree("SIGKILL");
  await waitForTreeExit();
}
