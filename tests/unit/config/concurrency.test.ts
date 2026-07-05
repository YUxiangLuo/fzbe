import { describe, expect, test } from "bun:test";
import path from "node:path";

const projectRoot = path.resolve(import.meta.dir, "../../..");

const CONFIG_ENV_NAMES = [
  "MAX_CONCURRENT_MODEL_JOBS",
  "MODEL_CONCURRENCY",
  "MAX_CONCURRENT_PDF_JOBS",
  "PDF_CONCURRENCY",
  "PDF_QUEUE_TIMEOUT_MS",
  "PDF_GENERATION_TIMEOUT_MS",
  "DB_QUERY_HEADROOM",
  "DB_CONNECTION_LIMIT",
  "PYTHON_MODEL_THREADS",
  "TF_NUM_INTEROP_THREADS",
  "AUTH_LOGIN_RATE_LIMIT_WINDOW_MS",
  "AUTH_LOGIN_RATE_LIMIT_MAX_PER_IP",
  "AUTH_LOGIN_RATE_LIMIT_MAX_PER_ACCOUNT",
  "AUTH_REGISTER_RATE_LIMIT_WINDOW_MS",
  "AUTH_REGISTER_RATE_LIMIT_MAX_PER_IP",
  "AUTH_REGISTER_RATE_LIMIT_MAX_PER_ACCOUNT",
] as const;

const SNAPSHOT_SCRIPT = `
import { CONCURRENCY_LIMITS, DB_POOL_CONFIG, PYTHON_RUNTIME_CONFIG, RATE_LIMITS, TIMEOUTS } from "./src/utils/constants.ts";

console.log(JSON.stringify({
  CONCURRENCY_LIMITS,
  DB_POOL_CONFIG,
  PYTHON_RUNTIME_CONFIG,
  RATE_LIMITS,
  TIMEOUTS,
}));
`;

interface ConstantsSnapshot {
  CONCURRENCY_LIMITS: {
    SYSTEM_PARALLELISM: number;
    DEFAULT_MODEL_JOB_CAP: number;
    MAX_CONCURRENT_MODEL_JOBS: number;
    MAX_CONCURRENT_PDF_JOBS: number;
  };
  DB_POOL_CONFIG: {
    LOCK_CONNECTIONS: number;
    QUERY_HEADROOM: number;
    CONNECTION_LIMIT: number;
  };
  PYTHON_RUNTIME_CONFIG: {
    MODEL_PROCESS_THREADS: number;
    TF_INTER_OP_THREADS: number;
  };
  RATE_LIMITS: {
    AUTH_LOGIN_WINDOW_MS: number;
    AUTH_LOGIN_MAX_PER_IP: number;
    AUTH_LOGIN_MAX_PER_ACCOUNT: number;
    AUTH_REGISTER_WINDOW_MS: number;
    AUTH_REGISTER_MAX_PER_IP: number;
    AUTH_REGISTER_MAX_PER_ACCOUNT: number;
  };
  TIMEOUTS: {
    PDF_QUEUE: number;
    PDF_GENERATION: number;
  };
}

async function loadConstantsSnapshot(
  configEnv: Partial<Record<(typeof CONFIG_ENV_NAMES)[number], string>> = {},
): Promise<ConstantsSnapshot> {
  const childEnv: Record<string, string> = {};
  for (const [name, value] of Object.entries(process.env)) {
    if (value !== undefined) {
      childEnv[name] = value;
    }
  }
  for (const name of CONFIG_ENV_NAMES) {
    delete childEnv[name];
  }
  Object.assign(childEnv, configEnv);

  const proc = Bun.spawn(["bun", "-e", SNAPSHOT_SCRIPT], {
    cwd: projectRoot,
    env: childEnv,
    stdout: "pipe",
    stderr: "pipe",
  });

  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);

  if (proc.exitCode !== 0) {
    throw new Error(stderr.trim() || stdout.trim() || `exit code ${proc.exitCode}`);
  }

  return JSON.parse(stdout) as ConstantsSnapshot;
}

describe("concurrency safeguards", () => {
  test("defaults are sized for a 64-core / 50-student classroom deployment", async () => {
    const { CONCURRENCY_LIMITS, PYTHON_RUNTIME_CONFIG, RATE_LIMITS, TIMEOUTS } =
      await loadConstantsSnapshot();

    expect(CONCURRENCY_LIMITS.DEFAULT_MODEL_JOB_CAP).toBe(50);
    expect(CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS).toBe(
      Math.min(CONCURRENCY_LIMITS.SYSTEM_PARALLELISM, 50),
    );
    expect(CONCURRENCY_LIMITS.MAX_CONCURRENT_PDF_JOBS).toBe(12);
    expect(TIMEOUTS.PDF_QUEUE).toBe(60_000);
    expect(TIMEOUTS.PDF_GENERATION).toBe(30_000);
    expect(PYTHON_RUNTIME_CONFIG.MODEL_PROCESS_THREADS).toBe(1);
    expect(PYTHON_RUNTIME_CONFIG.TF_INTER_OP_THREADS).toBe(1);
    expect(RATE_LIMITS.AUTH_LOGIN_WINDOW_MS).toBe(15 * 60 * 1000);
    expect(RATE_LIMITS.AUTH_LOGIN_MAX_PER_IP).toBe(100);
    expect(RATE_LIMITS.AUTH_LOGIN_MAX_PER_ACCOUNT).toBe(10);
    expect(RATE_LIMITS.AUTH_REGISTER_WINDOW_MS).toBe(60 * 60 * 1000);
    expect(RATE_LIMITS.AUTH_REGISTER_MAX_PER_IP).toBe(50);
    expect(RATE_LIMITS.AUTH_REGISTER_MAX_PER_ACCOUNT).toBe(3);
  });

  test("CONNECTION_LIMIT covers model advisory locks plus query headroom", async () => {
    const { CONCURRENCY_LIMITS, DB_POOL_CONFIG } = await loadConstantsSnapshot();
    const expectedQueryHeadroom = Math.max(
      40,
      Math.ceil(CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS / 2),
    );

    expect(DB_POOL_CONFIG.LOCK_CONNECTIONS).toBe(
      2 * CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS,
    );
    expect(DB_POOL_CONFIG.QUERY_HEADROOM).toBe(expectedQueryHeadroom);
    expect(DB_POOL_CONFIG.CONNECTION_LIMIT).toBe(
      DB_POOL_CONFIG.LOCK_CONNECTIONS + DB_POOL_CONFIG.QUERY_HEADROOM,
    );
  });

  test("explicit environment values override concurrency, timeout, database, and Python defaults", async () => {
    const {
      CONCURRENCY_LIMITS,
      DB_POOL_CONFIG,
      PYTHON_RUNTIME_CONFIG,
      RATE_LIMITS,
      TIMEOUTS,
    } =
      await loadConstantsSnapshot({
        MAX_CONCURRENT_MODEL_JOBS: "37",
        MAX_CONCURRENT_PDF_JOBS: "9",
        PDF_QUEUE_TIMEOUT_MS: "70000",
        PDF_GENERATION_TIMEOUT_MS: "45000",
        DB_QUERY_HEADROOM: "44",
        DB_CONNECTION_LIMIT: "150",
        PYTHON_MODEL_THREADS: "2",
        TF_NUM_INTEROP_THREADS: "3",
        AUTH_LOGIN_RATE_LIMIT_WINDOW_MS: "120000",
        AUTH_LOGIN_RATE_LIMIT_MAX_PER_IP: "20",
        AUTH_LOGIN_RATE_LIMIT_MAX_PER_ACCOUNT: "5",
        AUTH_REGISTER_RATE_LIMIT_WINDOW_MS: "600000",
        AUTH_REGISTER_RATE_LIMIT_MAX_PER_IP: "10",
        AUTH_REGISTER_RATE_LIMIT_MAX_PER_ACCOUNT: "2",
      });

    expect(CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS).toBe(37);
    expect(CONCURRENCY_LIMITS.MAX_CONCURRENT_PDF_JOBS).toBe(9);
    expect(TIMEOUTS.PDF_QUEUE).toBe(70_000);
    expect(TIMEOUTS.PDF_GENERATION).toBe(45_000);
    expect(DB_POOL_CONFIG.LOCK_CONNECTIONS).toBe(74);
    expect(DB_POOL_CONFIG.QUERY_HEADROOM).toBe(44);
    expect(DB_POOL_CONFIG.CONNECTION_LIMIT).toBe(150);
    expect(PYTHON_RUNTIME_CONFIG.MODEL_PROCESS_THREADS).toBe(2);
    expect(PYTHON_RUNTIME_CONFIG.TF_INTER_OP_THREADS).toBe(3);
    expect(RATE_LIMITS.AUTH_LOGIN_WINDOW_MS).toBe(120_000);
    expect(RATE_LIMITS.AUTH_LOGIN_MAX_PER_IP).toBe(20);
    expect(RATE_LIMITS.AUTH_LOGIN_MAX_PER_ACCOUNT).toBe(5);
    expect(RATE_LIMITS.AUTH_REGISTER_WINDOW_MS).toBe(600_000);
    expect(RATE_LIMITS.AUTH_REGISTER_MAX_PER_IP).toBe(10);
    expect(RATE_LIMITS.AUTH_REGISTER_MAX_PER_ACCOUNT).toBe(2);
  });

  test("legacy concurrency environment names remain supported", async () => {
    const { CONCURRENCY_LIMITS } = await loadConstantsSnapshot({
      MODEL_CONCURRENCY: "33",
      PDF_CONCURRENCY: "7",
    });

    expect(CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS).toBe(33);
    expect(CONCURRENCY_LIMITS.MAX_CONCURRENT_PDF_JOBS).toBe(7);
  });

  test("invalid positive-integer environment values fail fast", async () => {
    await expect(loadConstantsSnapshot({ MAX_CONCURRENT_MODEL_JOBS: "0" })).rejects.toThrow(
      "MAX_CONCURRENT_MODEL_JOBS must be a positive integer",
    );

    await expect(loadConstantsSnapshot({ PDF_QUEUE_TIMEOUT_MS: "30s" })).rejects.toThrow(
      "PDF_QUEUE_TIMEOUT_MS must be a positive integer",
    );

    await expect(
      loadConstantsSnapshot({ AUTH_LOGIN_RATE_LIMIT_MAX_PER_ACCOUNT: "0" }),
    ).rejects.toThrow("AUTH_LOGIN_RATE_LIMIT_MAX_PER_ACCOUNT must be a positive integer");
  });
});
