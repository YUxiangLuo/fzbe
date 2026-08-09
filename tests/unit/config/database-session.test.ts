import { describe, expect, mock, test } from "bun:test";
import {
  configureDatabasePoolSessions,
  configureDatabaseSession,
  resolveDatabaseSessionTimezone,
} from "../../../src/config/database-session";

describe("database session timezone", () => {
  test("maps mysql2 UTC and fixed offsets to an equivalent server session", () => {
    expect(resolveDatabaseSessionTimezone("Z")).toEqual({
      mysqlValue: "+00:00",
      offsetSeconds: 0,
    });
    expect(resolveDatabaseSessionTimezone("+08:00")).toEqual({
      mysqlValue: "+08:00",
      offsetSeconds: 28_800,
    });
    expect(resolveDatabaseSessionTimezone("-04:30")).toEqual({
      mysqlValue: "-04:30",
      offsetSeconds: -16_200,
    });
  });

  test.each(["local", "Asia/Shanghai", "+14:01", "-14:00", "+08:60"])(
    "rejects unsupported DB_TIMEZONE %s",
    (timezone) => {
      expect(() => resolveDatabaseSessionTimezone(timezone)).toThrow("DB_TIMEZONE");
    },
  );

  test("configures standalone migration connections before use", async () => {
    const query = mock(async () => [[], []]);
    await configureDatabaseSession({ query } as never, "Z");
    expect(query).toHaveBeenCalledWith("SET SESSION time_zone = ?", ["+00:00"]);
  });

  test("enqueues timezone initialization on every new pooled connection", () => {
    let listener: ((connection: unknown) => void) | null = null;
    const on = mock((_event: string, nextListener: (connection: unknown) => void) => {
      listener = nextListener;
      return {};
    });
    const onError = mock(() => {});
    const query = mock((_sql: string, _values: unknown[], callback: (error: Error | null) => void) => {
      callback(null);
    });
    const destroy = mock(() => {});

    configureDatabasePoolSessions({ on } as never, "Z", onError);
    listener?.({ query, destroy });

    expect(on).toHaveBeenCalledWith("connection", expect.any(Function));
    expect(query).toHaveBeenCalledWith(
      "SET SESSION time_zone = ?",
      ["+00:00"],
      expect.any(Function),
    );
    expect(onError).not.toHaveBeenCalled();
    expect(destroy).not.toHaveBeenCalled();
  });

  test("destroys a pooled connection when session initialization fails", () => {
    let listener: ((connection: unknown) => void) | null = null;
    const onError = mock(() => {});
    const destroy = mock(() => {});
    const failure = new Error("unknown time zone");

    configureDatabasePoolSessions({
      on: mock((_event: string, nextListener: (connection: unknown) => void) => {
        listener = nextListener;
        return {};
      }),
    } as never, "Z", onError);
    listener?.({
      query: mock((_sql: string, _values: unknown[], callback: (error: Error | null) => void) => {
        callback(failure);
      }),
      destroy,
    });

    expect(onError).toHaveBeenCalledWith(failure);
    expect(destroy).toHaveBeenCalledTimes(1);
  });
});
