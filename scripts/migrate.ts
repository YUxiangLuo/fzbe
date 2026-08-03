import path from "node:path";
import {
  createDatabaseMigrationCoordinator,
  type DatabaseMigrationCoordinator,
} from "./migration-ledger";

export interface DatabaseMigration {
  id: string;
  scriptPath: string;
}

export const DATABASE_MIGRATIONS = [
  { id: "001-academic-terms", scriptPath: "scripts/migrate-academic-terms.ts" },
  {
    id: "002-question-answer-explanations",
    scriptPath: "scripts/migrate-question-answer-explanations.ts",
  },
  { id: "003-quiz-answer-snapshots", scriptPath: "scripts/migrate-quiz-answer-snapshots.ts" },
  {
    id: "004-guided-training-sessions",
    scriptPath: "scripts/migrate-guided-training-sessions.ts",
  },
  {
    id: "005-model-training-integrity",
    scriptPath: "scripts/migrate-model-training-integrity.ts",
  },
  {
    id: "006-production-preparation-sessions",
    scriptPath: "scripts/migrate-production-preparation-sessions.ts",
  },
  { id: "007-user-phone-unique", scriptPath: "scripts/migrate-user-phone-unique.ts" },
  {
    id: "008-production-forecast-evidence",
    scriptPath: "scripts/migrate-production-forecast-evidence.ts",
  },
  {
    id: "009-experiment-retry-lineage",
    scriptPath: "scripts/migrate-experiment-retry-lineage.ts",
  },
  {
    id: "010-experiment-report-status-index",
    scriptPath: "scripts/migrate-experiment-report-status-index.ts",
  },
  {
    id: "011-experiment-retry-delete-rule",
    scriptPath: "scripts/migrate-experiment-retry-delete-rule.ts",
  },
  {
    id: "012-user-phone-canonical-unique",
    scriptPath: "scripts/migrate-user-phone-canonical-unique.ts",
  },
  {
    id: "013-production-inventory-precision",
    scriptPath: "scripts/migrate-production-inventory-precision.ts",
  },
] as const satisfies readonly DatabaseMigration[];

export type DatabaseMigrationExecutor = (
  migration: DatabaseMigration,
) => Promise<void>;

const projectRoot = path.resolve(import.meta.dir, "..");

export const executeDatabaseMigration: DatabaseMigrationExecutor = async (migration) => {
  const processHandle = Bun.spawn(
    [process.execPath, "run", migration.scriptPath],
    {
      cwd: projectRoot,
      env: process.env,
      stdout: "inherit",
      stderr: "inherit",
    },
  );
  const exitCode = await processHandle.exited;
  if (exitCode !== 0) {
    throw new Error(
      `Database migration ${migration.id} failed with exit code ${exitCode}`,
    );
  }
};

export interface RunDatabaseMigrationsOptions {
  coordinator: DatabaseMigrationCoordinator;
  execute?: DatabaseMigrationExecutor;
  log?: (message: string) => void;
}

export async function runDatabaseMigrations({
  coordinator,
  execute = executeDatabaseMigration,
  log = console.log,
}: RunDatabaseMigrationsOptions) {
  await coordinator.runExclusive(async (ledger) => {
    const applied = await ledger.listApplied();

    for (const migration of DATABASE_MIGRATIONS) {
      const recordedScriptPath = applied.get(migration.id);
      if (recordedScriptPath !== undefined) {
        if (recordedScriptPath !== migration.scriptPath) {
          throw new Error(
            `Database migration ledger mismatch for ${migration.id}: `
            + `recorded ${recordedScriptPath}, configured ${migration.scriptPath}`,
          );
        }
        log(`Skipping already applied database migration ${migration.id}.`);
        continue;
      }

      log(`Running database migration ${migration.id}...`);
      await execute(migration);
      await ledger.recordApplied(migration);
      applied.set(migration.id, migration.scriptPath);
    }
    log("All database migrations complete.");
  });
}

if (import.meta.main) {
  try {
    await runDatabaseMigrations({
      coordinator: createDatabaseMigrationCoordinator(),
    });
  } catch (error) {
    console.error("Database migration sequence failed; service restart is blocked.", error);
    process.exitCode = 1;
  }
}
