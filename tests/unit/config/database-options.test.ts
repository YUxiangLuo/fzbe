import { describe, expect, test } from "bun:test";
import path from "node:path";

const projectRoot = path.resolve(import.meta.dir, "../../..");
const CONFIG_ENV_NAMES = [
  "DB_PORT",
  "DB_CONNECT_TIMEOUT_MS",
  "DB_QUEUE_LIMIT",
  "DB_TIMEZONE",
  "DB_DATE_STRINGS",
  "DB_SSL_MODE",
  "DB_SSL_CA_PATH",
  "DB_SSL_REJECT_UNAUTHORIZED",
] as const;

const SNAPSHOT_SCRIPT = `
import { buildDatabasePoolOptions } from "./src/config/database-options.ts";

const options = buildDatabasePoolOptions();
const ssl = options.ssl && typeof options.ssl === "object"
  ? {
      rejectUnauthorized: options.ssl.rejectUnauthorized,
      ca: Boolean(options.ssl.ca),
    }
  : options.ssl;

console.log(JSON.stringify({
  host: options.host,
  port: options.port,
  user: options.user,
  database: options.database,
  queueLimit: options.queueLimit,
  connectTimeout: options.connectTimeout,
  timezone: options.timezone,
  dateStrings: options.dateStrings,
  ssl,
  enableKeepAlive: options.enableKeepAlive,
  keepAliveInitialDelay: options.keepAliveInitialDelay,
}));
`;

interface DatabaseOptionsSnapshot {
  host: string;
  port?: number;
  user: string;
  database: string;
  queueLimit: number;
  connectTimeout?: number;
  timezone?: string;
  dateStrings?: boolean;
  ssl?: {
    rejectUnauthorized?: boolean;
    ca: boolean;
  };
  enableKeepAlive?: boolean;
  keepAliveInitialDelay?: number;
}

async function loadDatabaseOptionsSnapshot(
  configEnv: Record<string, string> = {},
): Promise<DatabaseOptionsSnapshot> {
  const childEnv: Record<string, string> = {};
  for (const [name, value] of Object.entries(process.env)) {
    if (value !== undefined) {
      childEnv[name] = value;
    }
  }
  for (const name of CONFIG_ENV_NAMES) {
    delete childEnv[name];
  }

  Object.assign(childEnv, {
    DB_HOST: "127.0.0.1",
    DB_USER: "test",
    DB_PASSWORD: "test-password",
    DB_DATABASE: "test-db",
    JWT_SECRET: "test-jwt-secret",
    PORT: "3001",
  }, configEnv);

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

  return JSON.parse(stdout) as DatabaseOptionsSnapshot;
}

describe("database pool options", () => {
  test("defaults to bounded queueing without TLS when no production overrides are set", async () => {
    const options = await loadDatabaseOptionsSnapshot();

    expect(options.host).toBe("127.0.0.1");
    expect(options.port).toBeUndefined();
    expect(options.queueLimit).toBe(100);
    expect(options.connectTimeout).toBeUndefined();
    expect(options.timezone).toBeUndefined();
    expect(options.dateStrings).toBeUndefined();
    expect(options.ssl).toBeUndefined();
    expect(options.enableKeepAlive).toBeTrue();
    expect(options.keepAliveInitialDelay).toBe(0);
  });

  test("treats empty optional database environment values as unset", async () => {
    const options = await loadDatabaseOptionsSnapshot({
      DB_PORT: "",
      DB_CONNECT_TIMEOUT_MS: "",
      DB_QUEUE_LIMIT: "",
      DB_TIMEZONE: "",
      DB_DATE_STRINGS: "",
      DB_SSL_MODE: "",
      DB_SSL_CA_PATH: "",
      DB_SSL_REJECT_UNAUTHORIZED: "",
    });

    expect(options.port).toBeUndefined();
    expect(options.queueLimit).toBe(100);
    expect(options.connectTimeout).toBeUndefined();
    expect(options.timezone).toBeUndefined();
    expect(options.dateStrings).toBeUndefined();
    expect(options.ssl).toBeUndefined();
  });

  test("applies explicit production hardening options", async () => {
    const options = await loadDatabaseOptionsSnapshot({
      DB_PORT: "3307",
      DB_CONNECT_TIMEOUT_MS: "12345",
      DB_QUEUE_LIMIT: "12",
      DB_TIMEZONE: "Z",
      DB_DATE_STRINGS: "true",
      DB_SSL_MODE: "required",
      DB_SSL_REJECT_UNAUTHORIZED: "false",
    });

    expect(options.port).toBe(3307);
    expect(options.queueLimit).toBe(12);
    expect(options.connectTimeout).toBe(12_345);
    expect(options.timezone).toBe("Z");
    expect(options.dateStrings).toBeTrue();
    expect(options.ssl).toEqual({
      rejectUnauthorized: false,
      ca: false,
    });
  });

  test("rejects invalid database hardening options before creating the pool", async () => {
    await expect(loadDatabaseOptionsSnapshot({ DB_SSL_MODE: "maybe" })).rejects.toThrow(
      "Missing or invalid environment variables: DB_SSL_MODE",
    );

    await expect(loadDatabaseOptionsSnapshot({ DB_QUEUE_LIMIT: "0" })).rejects.toThrow(
      "Missing or invalid environment variables: DB_QUEUE_LIMIT",
    );
  });
});
