import { describe, expect, test } from "bun:test";
import {
  createRequestObservability,
  resolveRequestId,
} from "../../../src/utils/request-observability";

describe("request observability", () => {
  test("preserves only nginx or UUID-shaped correlation ids", () => {
    expect(resolveRequestId("0123456789abcdef0123456789ABCDEF"))
      .toBe("0123456789abcdef0123456789abcdef");
    expect(resolveRequestId("550e8400-e29b-41d4-a716-446655440000"))
      .toBe("550e8400-e29b-41d4-a716-446655440000");

    const generated = resolveRequestId("student-controlled-value\nforged");
    expect(generated).toMatch(
      /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/,
    );
  });

  test("tracks bounded method, status, active, and latency aggregates", () => {
    let now = 100;
    const observability = createRequestObservability({ now: () => now });
    const first = observability.begin("GET", "0123456789abcdef0123456789abcdef");
    const second = observability.begin("BREW");

    expect(observability.snapshot()).toMatchObject({
      activeRequests: 2,
      totalRequests: 2,
      completedRequests: 0,
      methods: { GET: 1, OTHER: 1 },
    });

    now = 112.5;
    expect(first.finish(200)).toEqual({ durationMs: 12.5, statusClass: "2xx" });
    expect(first.finish(500)).toBeNull();
    now = 140;
    expect(second.finish(503)).toEqual({ durationMs: 40, statusClass: "5xx" });

    expect(observability.snapshot()).toEqual({
      scope: "process",
      activeRequests: 0,
      totalRequests: 2,
      completedRequests: 2,
      methods: {
        GET: 1,
        POST: 0,
        PUT: 0,
        PATCH: 0,
        DELETE: 0,
        HEAD: 0,
        OPTIONS: 0,
        OTHER: 1,
      },
      statusClasses: {
        "1xx": 0,
        "2xx": 1,
        "3xx": 0,
        "4xx": 0,
        "5xx": 1,
      },
      latencyMs: {
        count: 2,
        total: 52.5,
        average: 26.25,
        max: 40,
      },
    });
  });
});
