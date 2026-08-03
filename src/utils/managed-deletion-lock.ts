import { createHash } from "node:crypto";
import type { Pool, PoolConnection, RowDataPacket } from "mysql2/promise";

interface LockStatusRow extends RowDataPacket {
  acquired?: number | string | null;
  released?: number | string | null;
}

export interface ManagedDeletionLockIdentity {
  databaseName: string;
  kind: string;
  entityIds: number[];
  /** Acquire a stable kind-wide lock before the sorted entity locks. */
  includeGlobalLock?: boolean;
}

export interface ManagedDeletionLockLease {
  lockNames: string[];
  release: () => Promise<void>;
}

function normalizedEntityIds(entityIds: number[]): number[] {
  const normalized = [...new Set(entityIds)].sort((left, right) => left - right);
  if (
    normalized.length === 0
    || normalized.some((entityId) => !Number.isInteger(entityId) || entityId <= 0)
  ) {
    throw new Error("Managed deletion locks require positive integer entity IDs");
  }
  return normalized;
}

export function buildManagedDeletionLockNames(
  identity: ManagedDeletionLockIdentity,
): string[] {
  const databaseName = identity.databaseName.trim();
  const kind = identity.kind.trim();
  if (!databaseName || !kind) {
    throw new Error("Managed deletion locks require a database name and kind");
  }

  const lockKeys: Array<number | "global"> = [
    ...(identity.includeGlobalLock ? ["global" as const] : []),
    ...normalizedEntityIds(identity.entityIds),
  ];
  return lockKeys.map((entityId) => {
    const digest = createHash("sha256")
      .update(JSON.stringify([databaseName, kind, entityId]))
      .digest("hex")
      .slice(0, 48);
    // MySQL/MariaDB named locks have a short implementation-defined name
    // limit. Hashing also avoids leaking deployment database names into
    // performance_schema while keeping the stable name well below 64 bytes.
    return `fz-del:${digest}`;
  });
}

async function releaseNamedLocks(
  connection: PoolConnection,
  lockNames: string[],
): Promise<void> {
  for (const lockName of [...lockNames].reverse()) {
    const [rows] = await connection.query<LockStatusRow[]>(
      "SELECT RELEASE_LOCK(?) AS released",
      [lockName],
    );
    if (Number(rows[0]?.released) !== 1) {
      throw new Error(`Failed to release managed deletion lock: ${lockName}`);
    }
  }
}

/**
 * Acquires an optional kind-wide lock, then one server-wide named lock per
 * entity in numeric order. Per-entity
 * locks make overlapping batches serialize (for example [7] versus [7, 8]),
 * while the database identity prevents unrelated deployments from contending.
 * The dedicated connection is returned to the pool only after every lock has
 * been released; on any protocol error it is destroyed so no lock can leak
 * back into the pool.
 */
export async function tryAcquireManagedDeletionLocks(
  dbPool: Pick<Pool, "getConnection">,
  identity: ManagedDeletionLockIdentity,
  timeoutSeconds: number,
): Promise<ManagedDeletionLockLease | null> {
  if (!Number.isInteger(timeoutSeconds) || timeoutSeconds < 0) {
    throw new Error("Managed deletion lock timeout must be a non-negative integer");
  }

  const lockNames = buildManagedDeletionLockNames(identity);
  const connection = await dbPool.getConnection();
  const acquiredLockNames: string[] = [];

  try {
    for (const lockName of lockNames) {
      const [rows] = await connection.query<LockStatusRow[]>(
        "SELECT GET_LOCK(?, ?) AS acquired",
        [lockName, timeoutSeconds],
      );
      if (Number(rows[0]?.acquired) !== 1) {
        await releaseNamedLocks(connection, acquiredLockNames);
        connection.release();
        return null;
      }
      acquiredLockNames.push(lockName);
    }
  } catch (error) {
    // Closing the underlying connection is MariaDB's fail-safe release for all
    // named locks owned by that session.
    connection.destroy();
    throw error;
  }

  let released = false;
  return {
    lockNames,
    release: async () => {
      if (released) return;
      released = true;
      try {
        await releaseNamedLocks(connection, acquiredLockNames);
        connection.release();
      } catch (error) {
        connection.destroy();
        throw error;
      }
    },
  };
}
