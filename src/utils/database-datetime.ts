const MYSQL_DATETIME_WITHOUT_TIMEZONE =
  /^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?)?$/;
const MYSQL_FIXED_OFFSET = /^[+-]\d{2}:\d{2}$/;

/**
 * Mirror mysql2's connection-timezone interpretation for values returned as
 * strings. mysql2 falls back to UTC for an invalid timezone option, while an
 * unset option means application-local time.
 */
export function normalizeMysqlConnectionTimezone(
  databaseTimezone: string | null | undefined,
): "local" | "Z" | string {
  const normalized = databaseTimezone?.trim();
  if (!normalized || normalized === "local") return "local";
  if (normalized === "Z" || MYSQL_FIXED_OFFSET.test(normalized)) return normalized;
  return "Z";
}

/**
 * Parse either mysql2 Date objects or DATE/DATETIME/TIMESTAMP strings without
 * silently interpreting a database wall-clock value in the process timezone.
 * Explicit ISO timestamps keep their own offset.
 */
export function parseDatabaseDatetime(
  value: unknown,
  databaseTimezone?: string | null,
): Date | null {
  if (value === null || value === undefined || value === "") return null;
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value;
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed || trimmed.startsWith("0000-00-00")) return null;
    const mysqlMatch = MYSQL_DATETIME_WITHOUT_TIMEZONE.exec(trimmed);
    if (mysqlMatch) {
      const [, year, month, day, hour = "00", minute = "00", second = "00", fraction] =
        mysqlMatch;
      const milliseconds = fraction ? `.${fraction}` : "";
      const canonical = `${year}-${month}-${day}T${hour}:${minute}:${second}${milliseconds}`;
      const timezone = normalizeMysqlConnectionTimezone(databaseTimezone);
      const parsed = new Date(timezone === "local" ? canonical : `${canonical}${timezone}`);
      return Number.isNaN(parsed.getTime()) ? null : parsed;
    }

    const parsed = new Date(trimmed);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  const parsed = new Date(value as string | number);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

export function databaseDatetimeToIsoString(
  value: unknown,
  databaseTimezone?: string | null,
): string | null {
  return parseDatabaseDatetime(value, databaseTimezone)?.toISOString() ?? null;
}
