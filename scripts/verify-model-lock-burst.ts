import { pool } from "../src/config";
import { CONCURRENCY_LIMITS } from "../src/utils/constants";
import {
  acquireModelJobLocks,
  releaseModelJobLocks,
  type ModelJobLockHandle,
} from "../src/modules/models/model-locks";

function parsePositiveIntegerEnv(name: string, fallback: number): number {
  const rawValue = process.env[name]?.trim();
  if (!rawValue) return fallback;
  if (!/^[1-9]\d*$/.test(rawValue)) {
    throw new Error(`${name} must be a positive integer, got: ${rawValue}`);
  }
  return Number(rawValue);
}

const modelJobs = CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS;
const requestCount = parsePositiveIntegerEnv(
  "MODEL_LOCK_BURST_REQUESTS",
  Math.max(100, modelJobs * 2),
);
const timeoutMs = parsePositiveIntegerEnv("MODEL_LOCK_BURST_TIMEOUT_MS", 30_000);

if (requestCount <= modelJobs) {
  throw new Error(
    `MODEL_LOCK_BURST_REQUESTS (${requestCount}) must exceed model capacity (${modelJobs})`,
  );
}

const acquiredLocks: ModelJobLockHandle[] = [];
const startedAt = Date.now();
const watchdog = setTimeout(() => {
  console.error(JSON.stringify({
    error: "model lock burst timed out",
    timeoutMs,
    modelJobs,
    requestCount,
    acquired: acquiredLocks.length,
  }));
  process.exit(1);
}, timeoutMs);

try {
  const experimentBase = 800_000_000 + Math.floor(Math.random() * 100_000_000);
  const acquisitions = await Promise.all(
    Array.from({ length: requestCount }, async (_, index) =>
      acquireModelJobLocks(experimentBase + index, "ma")),
  );

  const capacityCount = acquisitions.filter((result) => result.status === "capacity").length;
  const conflictCount = acquisitions.filter((result) => result.status === "conflict").length;
  for (const result of acquisitions) {
    if (result.status === "acquired") {
      acquiredLocks.push(result.locks);
    }
  }

  if (acquiredLocks.length !== modelJobs) {
    throw new Error(`Expected ${modelJobs} acquired jobs, got ${acquiredLocks.length}`);
  }
  if (capacityCount !== requestCount - modelJobs) {
    throw new Error(
      `Expected ${requestCount - modelJobs} capacity results, got ${capacityCount}`,
    );
  }
  if (conflictCount !== 0) {
    throw new Error(`Expected no experiment conflicts, got ${conflictCount}`);
  }

  // This query is the key regression assertion: active jobs must retain enough
  // pool headroom for guided claims, heartbeats, commits, and ordinary APIs.
  const queryStartedAt = Date.now();
  await pool.query("SELECT 1 AS healthy");
  const queryElapsedMs = Date.now() - queryStartedAt;

  console.log(JSON.stringify({
    modelJobs,
    requestCount,
    acquired: acquiredLocks.length,
    capacity: capacityCount,
    conflicts: conflictCount,
    queryElapsedMs,
    totalElapsedMs: Date.now() - startedAt,
  }));
} finally {
  clearTimeout(watchdog);
  await Promise.all(acquiredLocks.map((locks) => releaseModelJobLocks(locks)));
  await pool.end();
}
