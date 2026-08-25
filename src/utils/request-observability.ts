import { randomUUID } from "node:crypto";

const FORWARDED_REQUEST_ID_PATTERN = /^(?:[a-f0-9]{32}|[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12})$/i;
const TRACKED_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"] as const;
type TrackedMethod = (typeof TRACKED_METHODS)[number];
type StatusClass = "1xx" | "2xx" | "3xx" | "4xx" | "5xx";

export interface RequestObservation {
  requestId: string;
  finish(status: number): {
    durationMs: number;
    statusClass: StatusClass;
  } | null;
}

interface RequestObservabilityClock {
  now(): number;
}

const defaultClock: RequestObservabilityClock = {
  now: () => performance.now(),
};

const statusClassFor = (status: number): StatusClass => {
  if (status >= 500) return "5xx";
  if (status >= 400) return "4xx";
  if (status >= 300) return "3xx";
  if (status >= 200) return "2xx";
  return "1xx";
};

const normalizeMethod = (method: string): TrackedMethod | "OTHER" => {
  const normalized = method.toUpperCase();
  return (TRACKED_METHODS as readonly string[]).includes(normalized)
    ? normalized as TrackedMethod
    : "OTHER";
};

export function resolveRequestId(forwardedRequestId?: string | null): string {
  const candidate = forwardedRequestId?.trim();
  if (
    candidate
    && candidate === forwardedRequestId
    && FORWARDED_REQUEST_ID_PATTERN.test(candidate)
  ) {
    return candidate.toLowerCase();
  }
  return randomUUID();
}

export function createRequestObservability(clock: RequestObservabilityClock = defaultClock) {
  let activeRequests = 0;
  let totalRequests = 0;
  let completedRequests = 0;
  let latencyTotalMs = 0;
  let latencyMaxMs = 0;
  const methods: Record<TrackedMethod | "OTHER", number> = {
    GET: 0,
    POST: 0,
    PUT: 0,
    PATCH: 0,
    DELETE: 0,
    HEAD: 0,
    OPTIONS: 0,
    OTHER: 0,
  };
  const statusClasses: Record<StatusClass, number> = {
    "1xx": 0,
    "2xx": 0,
    "3xx": 0,
    "4xx": 0,
    "5xx": 0,
  };

  const begin = (method: string, forwardedRequestId?: string | null): RequestObservation => {
    const requestId = resolveRequestId(forwardedRequestId);
    const startedAt = clock.now();
    const methodKey = normalizeMethod(method);
    let finished = false;

    totalRequests += 1;
    activeRequests += 1;
    methods[methodKey] += 1;

    return {
      requestId,
      finish(status) {
        if (finished) return null;
        finished = true;

        const durationMs = Math.max(0, clock.now() - startedAt);
        const statusClass = statusClassFor(status);
        activeRequests = Math.max(0, activeRequests - 1);
        completedRequests += 1;
        statusClasses[statusClass] += 1;
        latencyTotalMs += durationMs;
        latencyMaxMs = Math.max(latencyMaxMs, durationMs);
        return { durationMs, statusClass };
      },
    };
  };

  const snapshot = () => ({
    scope: "process" as const,
    activeRequests,
    totalRequests,
    completedRequests,
    methods: { ...methods },
    statusClasses: { ...statusClasses },
    latencyMs: {
      count: completedRequests,
      total: latencyTotalMs,
      average: completedRequests > 0 ? latencyTotalMs / completedRequests : 0,
      max: latencyMaxMs,
    },
  });

  return { begin, snapshot };
}

const requestObservability = createRequestObservability();

export const beginRequestObservation = requestObservability.begin;
export const getRequestObservabilityStats = requestObservability.snapshot;
