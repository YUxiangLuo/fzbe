import { describe, expect, test } from "bun:test";
import {
  databaseDatetimeToIsoString,
  normalizeMysqlConnectionTimezone,
  parseDatabaseDatetime,
} from "../../../src/utils/database-datetime";

describe("database datetime parsing", () => {
  test("interprets timezone-free mysql strings using UTC when DB_TIMEZONE is Z", () => {
    expect(databaseDatetimeToIsoString("2026-08-04 12:34:56.123456", "Z"))
      .toBe("2026-08-04T12:34:56.123Z");
  });

  test("interprets timezone-free mysql strings using a fixed connection offset", () => {
    expect(databaseDatetimeToIsoString("2026-08-04 12:34:56", "+09:00"))
      .toBe("2026-08-04T03:34:56.000Z");
    expect(databaseDatetimeToIsoString("2026-08-04 12:34:56", "-04:30"))
      .toBe("2026-08-04T17:04:56.000Z");
  });

  test("preserves explicit offsets and valid mysql2 Date objects", () => {
    expect(databaseDatetimeToIsoString("2026-08-04T12:34:56+02:00", "Z"))
      .toBe("2026-08-04T10:34:56.000Z");
    const date = new Date("2026-08-04T12:34:56.000Z");
    expect(parseDatabaseDatetime(date, "+09:00")).toBe(date);
  });

  test("uses local time only when mysql2 does and rejects invalid values", () => {
    const parsed = parseDatabaseDatetime("2026-08-04 12:34:56", "local");
    expect(parsed?.getTime()).toBe(new Date(2026, 7, 4, 12, 34, 56).getTime());
    expect(databaseDatetimeToIsoString("0000-00-00 00:00:00", "Z")).toBeNull();
    expect(databaseDatetimeToIsoString("not-a-date", "Z")).toBeNull();
  });

  test("mirrors mysql2's UTC fallback for invalid timezone options", () => {
    expect(normalizeMysqlConnectionTimezone("Asia/Tokyo")).toBe("Z");
    expect(databaseDatetimeToIsoString("2026-08-04 12:34:56", "Asia/Tokyo"))
      .toBe("2026-08-04T12:34:56.000Z");
  });
});
