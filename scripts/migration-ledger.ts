import { createHash } from "node:crypto";
import type { Connection, RowDataPacket } from "mysql2/promise";
import { loadDatabaseEnvironment, type RuntimeDatabaseEnvSource } from "../src/config/database-connection-options";
import { createMigrationConnection } from "./migration-database";
import type { DatabaseMigration } from "./migrate";

export const DATABASE_MIGRATION_LEDGER_TABLE = "_fzbe_schema_migrations";
export const DATABASE_MIGRATION_LOCK_TIMEOUT_SECONDS = 600;

interface AdvisoryLockRow extends RowDataPacket {
  acquired: number | null;
}

interface AppliedMigrationRow extends RowDataPacket {
  migration_id: string;
  script_path: string;
}

export interface DatabaseMigrationLedger {
  listApplied(): Promise<Map<string, string>>;
  recordApplied(migration: DatabaseMigration): Promise<void>;
}

export interface DatabaseMigrationCoordinator {
  runExclusive<T>(operation: (ledger: DatabaseMigrationLedger) => Promise<T>): Promise<T>;
}

export const getDatabaseMigrationLockName = (databaseName: string): string => {
  const databaseHash = createHash("sha256").update(databaseName).digest("hex").slice(0, 32);
  return `fzbe:migrations:${databaseHash}`;
};

export function createDatabaseMigrationCoordinator(
  source: RuntimeDatabaseEnvSource = process.env,
  lockTimeoutSeconds = DATABASE_MIGRATION_LOCK_TIMEOUT_SECONDS,
  connectionFactory: () => Promise<Connection> = async () =>
    await createMigrationConnection({}, source),
): DatabaseMigrationCoordinator {
  if (!Number.isInteger(lockTimeoutSeconds) || lockTimeoutSeconds < 0) {
    throw new Error("Database migration lock timeout must be a nonnegative integer");
  }
  const databaseEnv = loadDatabaseEnvironment(source);
  const lockName = getDatabaseMigrationLockName(databaseEnv.DB_DATABASE);

  return {
    async runExclusive<T>(
      operation: (ledger: DatabaseMigrationLedger) => Promise<T>,
    ): Promise<T> {
      const connection = await connectionFactory();
      let acquired = false;
      let result: T | undefined;
      let operationFailed = false;
      let operationError: unknown;
      let cleanupError: unknown;

      try {
        const [lockRows] = await connection.query<AdvisoryLockRow[]>(
          "SELECT GET_LOCK(?, ?) AS acquired",
          [lockName, lockTimeoutSeconds],
        );
        if (Number(lockRows[0]?.acquired) !== 1) {
          throw new Error(
            `Timed out waiting ${lockTimeoutSeconds}s for database migration lock`,
          );
        }
        acquired = true;

        await connection.query(`
          CREATE TABLE IF NOT EXISTS \`${DATABASE_MIGRATION_LEDGER_TABLE}\` (
            migration_id VARCHAR(191) NOT NULL,
            script_path VARCHAR(512) NOT NULL,
            applied_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            PRIMARY KEY (migration_id)
          ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        `);

        const ledger: DatabaseMigrationLedger = {
          async listApplied() {
            const [rows] = await connection.query<AppliedMigrationRow[]>(
              `SELECT migration_id, script_path
                 FROM \`${DATABASE_MIGRATION_LEDGER_TABLE}\`
                ORDER BY migration_id ASC`,
            );
            return new Map(rows.map((row) => [row.migration_id, row.script_path]));
          },
          async recordApplied(migration) {
            await connection.execute(
              `INSERT INTO \`${DATABASE_MIGRATION_LEDGER_TABLE}\`
                 (migration_id, script_path)
               VALUES (?, ?)`,
              [migration.id, migration.scriptPath],
            );
          },
        };

        result = await operation(ledger);
      } catch (error) {
        operationFailed = true;
        operationError = error;
      } finally {
        if (acquired) {
          try {
            const [releaseRows] = await connection.query<AdvisoryLockRow[]>(
              "SELECT RELEASE_LOCK(?) AS acquired",
              [lockName],
            );
            if (Number(releaseRows[0]?.acquired) !== 1) {
              throw new Error("Database migration advisory lock was not released cleanly");
            }
          } catch (error) {
            cleanupError = error;
          }
        }
        try {
          await connection.end();
        } catch (error) {
          cleanupError ??= error;
        }
      }

      if (operationFailed) throw operationError;
      if (cleanupError !== undefined) throw cleanupError;
      return result as T;
    },
  };
}
