import { describe, expect, test } from "bun:test";
import path from "node:path";
import {
  DATABASE_MIGRATIONS,
  runDatabaseMigrations,
} from "../../../scripts/migrate";
import type {
  DatabaseMigrationCoordinator,
  DatabaseMigrationLedger,
} from "../../../scripts/migration-ledger";

const projectRoot = path.resolve(import.meta.dir, "../../..");

const createMemoryCoordinator = (
  initialApplied: Iterable<readonly [string, string]> = [],
) => {
  const applied = new Map(initialApplied);
  const recorded: string[] = [];
  const ledger: DatabaseMigrationLedger = {
    async listApplied() {
      return new Map(applied);
    },
    async recordApplied(migration) {
      applied.set(migration.id, migration.scriptPath);
      recorded.push(migration.id);
    },
  };
  const coordinator: DatabaseMigrationCoordinator = {
    async runExclusive(operation) {
      return await operation(ledger);
    },
  };
  return { applied, coordinator, recorded };
};

describe("database migration deployment path", () => {
  test("runs every idempotent migration in a stable dependency order", async () => {
    const { coordinator, recorded } = createMemoryCoordinator();
    const executed: string[] = [];
    await runDatabaseMigrations({
      coordinator,
      execute: async (migration) => {
        executed.push(migration.id);
      },
      log: () => {},
    });

    expect(executed).toEqual([
      "001-academic-terms",
      "002-question-answer-explanations",
      "003-quiz-answer-snapshots",
      "004-guided-training-sessions",
      "005-model-training-integrity",
      "006-production-preparation-sessions",
      "007-user-phone-unique",
      "008-production-forecast-evidence",
      "009-experiment-retry-lineage",
      "010-experiment-report-status-index",
      "011-experiment-retry-delete-rule",
      "012-user-phone-canonical-unique",
      "013-production-inventory-precision",
    ]);
    expect(recorded).toEqual(executed);
    expect(DATABASE_MIGRATIONS.map((migration) => migration.scriptPath))
      .toContain("scripts/migrate-user-phone-unique.ts");
    expect(DATABASE_MIGRATIONS.map((migration) => migration.scriptPath))
      .toContain("scripts/migrate-production-forecast-evidence.ts");
    expect(DATABASE_MIGRATIONS.map((migration) => migration.scriptPath))
      .toContain("scripts/migrate-experiment-retry-lineage.ts");
    expect(DATABASE_MIGRATIONS.map((migration) => migration.scriptPath))
      .toContain("scripts/migrate-experiment-report-status-index.ts");
    expect(DATABASE_MIGRATIONS.map((migration) => migration.scriptPath))
      .toContain("scripts/migrate-experiment-retry-delete-rule.ts");
    expect(DATABASE_MIGRATIONS.map((migration) => migration.scriptPath))
      .not.toContain("scripts/migrate-rejected-report-grades.ts");
  });

  test("records each success, stops on failure, and resumes with only pending migrations", async () => {
    const { applied, coordinator } = createMemoryCoordinator();
    const executed: string[] = [];

    await expect(runDatabaseMigrations({
      coordinator,
      execute: async (migration) => {
        executed.push(migration.id);
        if (migration.id === "003-quiz-answer-snapshots") {
          throw new Error("migration failed");
        }
      },
      log: () => {},
    })).rejects.toThrow("migration failed");

    expect(executed).toEqual([
      "001-academic-terms",
      "002-question-answer-explanations",
      "003-quiz-answer-snapshots",
    ]);
    expect([...applied.keys()]).toEqual([
      "001-academic-terms",
      "002-question-answer-explanations",
    ]);

    executed.length = 0;
    await runDatabaseMigrations({
      coordinator,
      execute: async (migration) => {
        executed.push(migration.id);
      },
      log: () => {},
    });
    expect(executed).toEqual([
      "003-quiz-answer-snapshots",
      "004-guided-training-sessions",
      "005-model-training-integrity",
      "006-production-preparation-sessions",
      "007-user-phone-unique",
      "008-production-forecast-evidence",
      "009-experiment-retry-lineage",
      "010-experiment-report-status-index",
      "011-experiment-retry-delete-rule",
      "012-user-phone-canonical-unique",
      "013-production-inventory-precision",
    ]);
  });

  test("does not ledger inventory precision when bounded CAS repair is exhausted", async () => {
    const previouslyApplied = DATABASE_MIGRATIONS
      .filter((migration) => migration.id !== "013-production-inventory-precision")
      .map((migration) => [migration.id, migration.scriptPath] as const);
    const { applied, coordinator, recorded } = createMemoryCoordinator(previouslyApplied);

    await expect(runDatabaseMigrations({
      coordinator,
      execute: async (migration) => {
        expect(migration.id).toBe("013-production-inventory-precision");
        throw new Error("inventory repair CAS attempts exhausted");
      },
      log: () => {},
    })).rejects.toThrow("inventory repair CAS attempts exhausted");

    expect(recorded).toEqual([]);
    expect(applied.has("013-production-inventory-precision")).toBe(false);
  });

  test("runs the delete-rule convergence after the original 009/010 ledger rows", async () => {
    const previouslyApplied = DATABASE_MIGRATIONS
      .filter((migration) => migration.id !== "011-experiment-retry-delete-rule")
      .map((migration) => [migration.id, migration.scriptPath] as const);
    const { coordinator, recorded } = createMemoryCoordinator(previouslyApplied);
    const executed: string[] = [];

    await runDatabaseMigrations({
      coordinator,
      execute: async (migration) => {
        executed.push(migration.id);
      },
      log: () => {},
    });

    expect(executed).toEqual(["011-experiment-retry-delete-rule"]);
    expect(recorded).toEqual(executed);
  });

  test("fails closed when a recorded migration id points at a different script", async () => {
    const { coordinator } = createMemoryCoordinator([
      ["001-academic-terms", "scripts/renamed-or-replaced.ts"],
    ]);

    await expect(runDatabaseMigrations({
      coordinator,
      execute: async () => {
        throw new Error("must not execute");
      },
      log: () => {},
    })).rejects.toThrow("ledger mismatch");
  });

  test("re-audits lineage while quiesced and reconciles rejection grades only after healthy activation", async () => {
    const actions = await Bun.file(path.join(projectRoot, "actions.sh")).text();
    const update = await Bun.file(path.join(projectRoot, "update.sh")).text();

    expect(actions).toContain("set -euo pipefail");
    expect(actions).toContain("run_bun_with_service_environment");
    expect(actions).toContain("run scripts/migrate.ts");
    expect(actions).toContain("run scripts/migrate-experiment-retry-lineage.ts");
    expect(actions).not.toContain("run scripts/migrate-rejected-report-grades.ts");
    expect(update).toContain("set -euo pipefail");
    expect(update).toContain("flock -n 9");
    expect(update).toContain("--property=KillMode --value");
    expect(update).toContain('service_kill_mode" != "control-group"');
    expect(update).toContain("--property=WorkingDirectory");
    expect(update).toContain("--property=EnvironmentFiles");
    expect(update).toContain("--property=Environment --value");
    expect(update).toContain("deployment_environment_has_protected_override");
    expect(update).toContain("assert_single_instance_production_environment");
    expect(update).toContain("NODE_ENV=production");
    expect(update).toContain("production binding must be exactly 127.0.0.1:3001");
    expect(update).toContain("remove obsolete advisory-lock settings");
    expect(update).toContain("deployment_environment_files_match");
    expect(update).toContain("deployment_environment_snapshot");
    expect(update).toContain("run_bun_with_service_environment");
    expect(update).toContain('await import(\"./src/config/env.ts\")');
    expect(update).toContain('await import(\"./src/utils/constants.ts\")');
    expect(update.match(/run scripts\/migrate-experiment-retry-lineage\.ts/g))
      .toHaveLength(2);
    expect(update).not.toContain(
      'run_bun_with_service_environment "$project_root/.env"',
    );
    expect(update).toContain('git worktree add --detach "$release_dir" "$target_commit"');
    expect(update).toContain("assert_live_checkout_unchanged");
    expect(update).toContain("wait_for_service_health");
    expect(update).toContain("/api/v1/health");
    expect(update).toContain('[[ "$(git rev-parse HEAD)" == "$current_commit" ]]');
    expect(update).toContain('git reset --hard "$current_commit"');
    expect(update).not.toContain("git pull");
    expect(update).not.toContain("systemctl restart");

    const targetEnvironmentValidation = update.indexOf(
      'await import(\"./src/config/env.ts\")',
    );
    const stagingActions = update.indexOf("./actions.sh");
    const markServiceStopped = update.indexOf("service_stopped=true", stagingActions);
    const stopService = update.indexOf('sudo systemctl stop "$service_name"', stagingActions);
    const quiescedLineageAudit = update.indexOf(
      "run scripts/migrate-experiment-retry-lineage.ts",
      stopService,
    );
    const quiescedPhoneNormalization = update.indexOf(
      "run scripts/migrate-user-phone-canonical-unique.ts",
      quiescedLineageAudit,
    );
    const activateRelease = update.indexOf('git merge --ff-only "$target_commit"');
    const startService = update.indexOf('sudo systemctl start "$service_name"', activateRelease);
    const finalCleanCheck = update.lastIndexOf("assert_live_checkout_unchanged", stopService);
    const environmentDriftCheck = update.indexOf(
      "assert_deployment_environment_unchanged",
      stopService,
    );
    const healthCheck = update.indexOf("wait_for_service_health", startService);
    const activationComplete = update.indexOf("deployment_complete=true", healthCheck);
    const postHealthLineageAudit = update.indexOf(
      "run scripts/migrate-experiment-retry-lineage.ts",
      activationComplete,
    );
    const postHealthGradeNormalization = update.indexOf(
      "run scripts/migrate-rejected-report-grades.ts",
      activationComplete,
    );
    expect(update.indexOf('cd "$release_dir"')).toBeLessThan(targetEnvironmentValidation);
    expect(targetEnvironmentValidation).toBeLessThan(stagingActions);
    expect(stagingActions).toBeLessThan(stopService);
    expect(stagingActions).toBeLessThan(finalCleanCheck);
    expect(finalCleanCheck).toBeLessThan(stopService);
    expect(markServiceStopped).toBeLessThan(stopService);
    expect(stopService).toBeLessThan(environmentDriftCheck);
    expect(environmentDriftCheck).toBeLessThan(quiescedLineageAudit);
    expect(quiescedLineageAudit).toBeLessThan(quiescedPhoneNormalization);
    expect(quiescedLineageAudit).toBeLessThan(activateRelease);
    expect(environmentDriftCheck).toBeLessThan(quiescedPhoneNormalization);
    expect(stopService).toBeLessThan(quiescedPhoneNormalization);
    expect(quiescedPhoneNormalization).toBeLessThan(activateRelease);
    expect(stopService).toBeLessThan(activateRelease);
    expect(activateRelease).toBeLessThan(startService);
    expect(startService).toBeLessThan(healthCheck);
    expect(healthCheck).toBeLessThan(activationComplete);
    expect(activationComplete).toBeLessThan(postHealthLineageAudit);
    expect(activationComplete).toBeLessThan(postHealthGradeNormalization);
    expect(update.indexOf("post_activation_status=1", activationComplete)).toBeGreaterThan(
      activationComplete,
    );
    expect(update).toContain(
      'if [[ "$service_stopped" == true && "$deployment_complete" != true ]]',
    );
    const quiescedBlock = update.slice(stopService, activateRelease);
    expect(quiescedBlock).toContain('"$deployment_environment_snapshot"');
    expect(quiescedBlock).toContain('"$release_dir"');
    expect(quiescedBlock).not.toMatch(
      /if\s+!\s+run_bun_with_service_environment[\s\S]*?migrate-experiment-retry-lineage\.ts/,
    );
  });

  test("converges already-applied 007 databases to database-enforced canonical phone uniqueness", async () => {
    const migration = await Bun.file(path.join(
      projectRoot,
      "scripts/migrate-user-phone-canonical-unique.ts",
    )).text();
    const schema = await Bun.file(path.join(projectRoot, "sql/schema.sql")).text();

    expect(DATABASE_MIGRATIONS.find((migration) =>
      migration.id === "012-user-phone-canonical-unique"
    )).toEqual({
      id: "012-user-phone-canonical-unique",
      scriptPath: "scripts/migrate-user-phone-canonical-unique.ts",
    });
    expect(migration).toContain("GENERATED ALWAYS AS (NULLIF(TRIM(phone_number), '')) STORED");
    expect(migration).toContain("users_phone_number_canonical_unique");
    expect(migration).toContain("GROUP BY TRIM(phone_number)");
    expect(migration).not.toContain("IS_GENERATED");
    expect(migration).not.toContain("normalized_phone_number");
    expect(schema).toContain("phone_number_canonical");
    expect(schema).toContain("users_phone_number_canonical_unique");
  });

  test("widens production initial inventory so persisted MPS replay keeps JavaScript precision", async () => {
    const migration = await Bun.file(path.join(
      projectRoot,
      "scripts/migrate-production-inventory-precision.ts",
    )).text();
    const schema = await Bun.file(path.join(projectRoot, "sql/schema.sql")).text();

    expect(DATABASE_MIGRATIONS.at(-1)).toEqual({
      id: "013-production-inventory-precision",
      scriptPath: "scripts/migrate-production-inventory-precision.ts",
    });
    expect(schema).toContain("`production_initial_inventory` DOUBLE DEFAULT NULL");
    expect(migration).toContain("MODIFY COLUMN production_initial_inventory DOUBLE");
    expect(migration).toContain("state_version = ?");
    expect(migration).toContain("beginning_inventory");
  });

  test("ships the retry lineage column, uniqueness, and self-reference in the baseline schema", async () => {
    const schema = await Bun.file(path.join(projectRoot, "sql/schema.sql")).text();
    const migration = await Bun.file(path.join(
      projectRoot,
      "scripts/migrate-experiment-retry-lineage.ts",
    )).text();

    expect(schema).toContain("retry_of_experiment_id");
    expect(schema).toContain("experiment_status_retry_parent_unique");
    expect(schema).toMatch(
      /CONSTRAINT `experiment_status_retry_parent_fk`[\s\S]*?ON DELETE SET NULL/,
    );

    expect(migration).toContain("retry_of_experiment_id");
    expect(migration).toContain("experiment_status_retry_parent_unique");
    expect(migration).toContain("experiment_status_retry_parent_fk");
    expect(migration).toContain("ON DELETE SET NULL");
    expect(migration).toContain('foreignKey.delete_rule === "CASCADE"');
    expect(migration).toContain("DROP FOREIGN KEY experiment_status_retry_parent_fk");
    expect(migration).toContain('foreignKey.delete_rule !== "SET NULL"');
  });

  test("fails closed instead of guessing legacy retry parents without a rejection timestamp", async () => {
    const lineageMigration = await Bun.file(path.join(
      projectRoot,
      "scripts/migrate-experiment-retry-lineage.ts",
    )).text();
    const deleteRuleMigration = await Bun.file(path.join(
      projectRoot,
      "scripts/migrate-experiment-retry-delete-rule.ts",
    )).text();
    const audit = await Bun.file(path.join(
      projectRoot,
      "scripts/experiment-retry-lineage-audit.ts",
    )).text();

    expect(lineageMigration).toContain("await auditExperimentRetryLineage(connection)");
    expect(deleteRuleMigration).toContain("await auditExperimentRetryLineage(connection)");
    expect(audit).toContain("Ambiguous legacy retry lineage detected");
    expect(audit).toContain("later.experiment_id > parent.experiment_id");
    expect(audit).toContain("later.retry_of_experiment_id IS NULL");
    expect(audit).not.toContain("explicit_retry.experiment_id IS NULL");
    expect(audit).toContain("parent.experiment_id >= child.experiment_id");
    expect(audit).toContain("FROM experiment_status intervening");
    expect(audit).toContain("intervening.experiment_id < child.experiment_id");
    expect(audit).not.toContain("SET child.retry_of_experiment_id");
  });

  test("ships a new ledger migration that converges already-applied 009 databases", async () => {
    const migration = await Bun.file(path.join(
      projectRoot,
      "scripts/migrate-experiment-retry-delete-rule.ts",
    )).text();
    const packageJson = JSON.parse(
      await Bun.file(path.join(projectRoot, "package.json")).text(),
    ) as { scripts: Record<string, string> };
    const documentation = await Bun.file(path.join(
      projectRoot,
      "docs/database-migrations.md",
    )).text();

    expect(migration).toContain('foreignKey.delete_rule === "CASCADE"');
    expect(migration).toContain("DROP FOREIGN KEY experiment_status_retry_parent_fk");
    expect(migration).toContain("ON DELETE SET NULL");
    expect(migration).toContain("loadForeignKeys()");
    expect(migration).toContain("delete-rule audit did not converge to SET NULL");
    expect(migration).toContain("await auditExperimentRetryLineage(connection)");
    expect(packageJson.scripts["db:migrate:experiment-retry-delete-rule"])
      .toBe("bun run scripts/migrate-experiment-retry-delete-rule.ts");
    expect(documentation).toContain("011-experiment-retry-delete-rule");
    expect(documentation).toContain("already recorded migration 009");
  });

  test("normalizes only official report grades for historical rejected attempts", async () => {
    const migration = await Bun.file(path.join(
      projectRoot,
      "scripts/migrate-rejected-report-grades.ts",
    )).text();

    expect(migration).toContain("WHERE er.status = 'rejected'");
    expect(migration).toContain("SET grade = NULL");
    expect(migration).toContain("SET eg.report_quality = NULL");
    expect(migration).toContain("LIMIT ?");
    expect(migration).toContain("FOR UPDATE");
    expect(migration).toContain("await connection.commit()");
    expect(migration).not.toContain("model_quality = NULL");
    expect(migration).not.toContain("knowledge_test = NULL");
    expect(migration).not.toContain("exp_flow_");
  });

  test("ships and upgrades the rejected-report reconciliation index", async () => {
    const schema = await Bun.file(path.join(projectRoot, "sql/schema.sql")).text();
    const migration = await Bun.file(path.join(
      projectRoot,
      "scripts/migrate-experiment-report-status-index.ts",
    )).text();

    expect(schema).toContain(
      "INDEX `idx_experiment_reports_status_report` (`status`, `report_id`)",
    );
    expect(migration).toContain("ADD INDEX idx_experiment_reports_status_report (status, report_id)");
    expect(migration).toContain("SEQ_IN_INDEX");
    expect(migration).toContain("Number(row.non_unique) === 1");
    expect(migration).toContain("incompatible definition");
  });
});
