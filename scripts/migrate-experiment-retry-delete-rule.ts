import type { RowDataPacket } from "mysql2/promise";
import { createMigrationConnection } from "./migration-database";
import { auditExperimentRetryLineage } from "./experiment-retry-lineage-audit";

interface ColumnRow extends RowDataPacket {
  column_name: string;
  is_nullable: string;
}

interface ForeignKeyRow extends RowDataPacket {
  constraint_name: string;
  column_name: string;
  referenced_table_name: string;
  referenced_column_name: string;
  delete_rule: string;
}

const columnName = "retry_of_experiment_id";
const foreignKeyName = "experiment_status_retry_parent_fk";

const connection = await createMigrationConnection();

const loadForeignKeys = async () => {
  const [rows] = await connection.query<ForeignKeyRow[]>(
    `SELECT kcu.CONSTRAINT_NAME AS constraint_name,
            kcu.COLUMN_NAME AS column_name,
            kcu.REFERENCED_TABLE_NAME AS referenced_table_name,
            kcu.REFERENCED_COLUMN_NAME AS referenced_column_name,
            rc.DELETE_RULE AS delete_rule
       FROM information_schema.KEY_COLUMN_USAGE kcu
       JOIN information_schema.REFERENTIAL_CONSTRAINTS rc
         ON rc.CONSTRAINT_SCHEMA = kcu.CONSTRAINT_SCHEMA
        AND rc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
        AND rc.TABLE_NAME = kcu.TABLE_NAME
      WHERE kcu.TABLE_SCHEMA = DATABASE()
        AND kcu.TABLE_NAME = 'experiment_status'
        AND (kcu.CONSTRAINT_NAME = ? OR kcu.COLUMN_NAME = ?)
        AND kcu.REFERENCED_TABLE_NAME IS NOT NULL`,
    [foreignKeyName, columnName],
  );
  return rows;
};

const assertExpectedForeignKeyIdentity = (foreignKeys: ForeignKeyRow[]) => {
  const foreignKey = foreignKeys[0];
  if (
    foreignKeys.length !== 1
    || !foreignKey
    || foreignKey.constraint_name !== foreignKeyName
    || foreignKey.column_name !== columnName
    || foreignKey.referenced_table_name !== "experiment_status"
    || foreignKey.referenced_column_name !== "experiment_id"
  ) {
    throw new Error(`${foreignKeyName} exists with an incompatible definition`);
  }
  return foreignKey;
};

try {
  const [columns] = await connection.query<ColumnRow[]>(
    `SELECT COLUMN_NAME AS column_name,
            IS_NULLABLE AS is_nullable
       FROM information_schema.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'experiment_status'
        AND COLUMN_NAME = ?
      LIMIT 1`,
    [columnName],
  );
  if (columns.length !== 1 || columns[0]?.is_nullable !== "YES") {
    throw new Error(
      "retry_of_experiment_id is missing or non-nullable; run migration 009 before 011",
    );
  }

  let foreignKeys = await loadForeignKeys();
  if (foreignKeys.length === 0) {
    await connection.query(
      `ALTER TABLE experiment_status
         ADD CONSTRAINT experiment_status_retry_parent_fk
         FOREIGN KEY (retry_of_experiment_id)
         REFERENCES experiment_status(experiment_id)
         ON DELETE SET NULL`,
    );
    console.log("Added experiment retry parent foreign key with SET NULL.");
  } else {
    const foreignKey = assertExpectedForeignKeyIdentity(foreignKeys);
    if (foreignKey.delete_rule === "CASCADE") {
      // 009 was already ledgered by an early rollout that used CASCADE. A new
      // migration id is required because the coordinator correctly skips that
      // historical ledger row.
      // Split DROP and ADD into separate ALTER statements: MySQL 8 rejects
      // dropping and re-adding the same foreign key constraint name in a single
      // ALTER TABLE with ER_FK_DUP_NAME (1826).
      await connection.query(
        `ALTER TABLE experiment_status
           DROP FOREIGN KEY experiment_status_retry_parent_fk`,
      );
      await connection.query(
        `ALTER TABLE experiment_status
           ADD CONSTRAINT experiment_status_retry_parent_fk
           FOREIGN KEY (retry_of_experiment_id)
           REFERENCES experiment_status(experiment_id)
           ON DELETE SET NULL`,
      );
      console.log("Upgraded experiment retry parent delete rule from CASCADE to SET NULL.");
    } else if (foreignKey.delete_rule !== "SET NULL") {
      throw new Error(`${foreignKeyName} must use ON DELETE SET NULL`);
    }
  }

  // Audit the postcondition after DDL instead of trusting an ALTER that may
  // have been interrupted between its implicit commits.
  foreignKeys = await loadForeignKeys();
  const auditedForeignKey = assertExpectedForeignKeyIdentity(foreignKeys);
  if (auditedForeignKey.delete_rule !== "SET NULL") {
    throw new Error(`${foreignKeyName} delete-rule audit did not converge to SET NULL`);
  }

  // A database may have ledgered an older 009 before the fail-closed lineage
  // audit existed. Running the same shared audit here makes bare `db:migrate`
  // sufficient; deployment-specific direct reruns of 009 are defense in depth.
  await auditExperimentRetryLineage(connection);

  console.log("Experiment retry parent delete-rule migration complete.");
} finally {
  await connection.end();
}
