import type { Connection, Pool } from "mysql2/promise";

const MYSQL_FIXED_OFFSET = /^[+-]\d{2}:\d{2}$/;

function parseFixedOffsetSeconds(value: string): number | null {
  if (!MYSQL_FIXED_OFFSET.test(value)) return null;

  const sign = value[0] === "-" ? -1 : 1;
  const hours = Number(value.slice(1, 3));
  const minutes = Number(value.slice(4, 6));
  if (minutes > 59 || hours > 14 || (hours === 14 && minutes !== 0)) {
    return null;
  }
  if (sign < 0 && hours === 14) {
    return null;
  }

  return sign * (hours * 60 + minutes) * 60;
}

/**
 * Convert mysql2's Date parsing timezone to the equivalent MySQL/MariaDB
 * session timezone. Keeping both sides aligned prevents NOW()/CURRENT_TIMESTAMP
 * wall-clock values from being interpreted a second time in another timezone.
 */
export function resolveDatabaseSessionTimezone(databaseTimezone: string): {
  mysqlValue: string;
  offsetSeconds: number;
} {
  const normalized = databaseTimezone.trim();
  if (normalized === "Z") {
    return { mysqlValue: "+00:00", offsetSeconds: 0 };
  }

  const offsetSeconds = parseFixedOffsetSeconds(normalized);
  if (offsetSeconds === null) {
    throw new Error("DB_TIMEZONE must be Z or a valid fixed offset such as +08:00");
  }

  return { mysqlValue: normalized, offsetSeconds };
}

export async function configureDatabaseSession(
  connection: Pick<Connection, "query">,
  databaseTimezone: string,
): Promise<void> {
  const { mysqlValue } = resolveDatabaseSessionTimezone(databaseTimezone);
  await connection.query("SET SESSION time_zone = ?", [mysqlValue]);
}

interface CallbackPoolConnection {
  query(
    sql: string,
    values: unknown[],
    callback: (error: Error | null) => void,
  ): unknown;
  destroy(): void;
}

/**
 * mysql2 emits `connection` synchronously before handing a new physical
 * connection to its first borrower. Enqueueing SET here therefore makes the
 * session timezone the first command on every pooled connection.
 */
export function configureDatabasePoolSessions(
  dbPool: Pick<Pool, "on">,
  databaseTimezone: string,
  onError: (error: Error) => void,
): void {
  const { mysqlValue } = resolveDatabaseSessionTimezone(databaseTimezone);

  dbPool.on("connection", (connection) => {
    const callbackConnection = connection as unknown as CallbackPoolConnection;
    callbackConnection.query(
      "SET SESSION time_zone = ?",
      [mysqlValue],
      (error) => {
        if (!error) return;
        onError(error);
        callbackConnection.destroy();
      },
    );
  });
}
