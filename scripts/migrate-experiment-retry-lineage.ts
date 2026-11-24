import { type RowDataPacket } from "mysql2/promise";
import { createMigrationConnection } from "./migration-database";
import { auditExperimentRetryLineage } from "./experiment-retry-lineage-audit";

interface ColumnRow extends RowDataPacket {
  column_name: string;
  column_type: string;
  is_nullable: string;
}

interface IndexRow extends RowDataPacket {
  index_name: string;
  column_name: string;
  non_unique: number;
}

interface ForeignKeyRow extends RowDataPacket {
  constraint_name: string;
  column_name: string;
  referenced_table_name: string;
  referenced_column_name: string;
  delete_rule: string;
}

const columnName = "retry_of_experiment_id";
const indexName = "experiment_status_retry_parent_unique";
const foreignKeyName = "experiment_status_retry_parent_fk";

const connection = await createMigrationConnection();

try {
  const [columns] = await connection.query<ColumnRow[]>(
    `SELECT COLUMN_NAME AS column_name,
            COLUMN_TYPE AS column_type,
            IS_NULLABLE AS is_nullable
       FROM information_schema.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'experiment_status'
        AND COLUMN_NAME = ?
      LIMIT 1`,
    [columnName],
  );
  if (columns.length === 0) {
    await connection.query(
      `ALTER TABLE experiment_status
         ADD COLUMN retry_of_experiment_id INT DEFAULT NULL
         AFTER student_id`,
    );
    console.log("Added experiment_status.retry_of_experiment_id.");
  } else if (
    columns[0]?.column_type.toLowerCase() !== "int(11)"
    && columns[0]?.column_type.toLowerCase() !== "int"
  ) {
    throw new Error("retry_of_experiment_id exists with an incompatible integer type");
  } else if (columns[0]?.is_nullable !== "YES") {
    throw new Error("retry_of_experiment_id must remain nullable for first attempts");
  }

  const [indexes] = await connection.query<IndexRow[]>(
    `SELECT INDEX_NAME AS index_name,
            COLUMN_NAME AS column_name,
            NON_UNIQUE AS non_unique
       FROM information_schema.STATISTICS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'experiment_status'
        AND (INDEX_NAME = ? OR (COLUMN_NAME = ? AND NON_UNIQUE = 0))
      ORDER BY INDEX_NAME, SEQ_IN_INDEX`,
    [indexName, columnName],
  );
  if (indexes.length === 0) {
    await connection.query(
      `ALTER TABLE experiment_status
         ADD UNIQUE KEY experiment_status_retry_parent_unique (retry_of_experiment_id)`,
    );
    console.log("Added experiment retry parent uniqueness constraint.");
  } else if (
    indexes.length !== 1
    || indexes[0]?.index_name !== indexName
    || indexes[0]?.column_name !== columnName
    || Number(indexes[0]?.non_unique) !== 0
  ) {
    throw new Error(`${indexName} exists with an incompatible definition`);
  }

  const [foreignKeys] = await connection.query<ForeignKeyRow[]>(
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
  if (foreignKeys.length === 0) {
    await connection.query(
      `ALTER TABLE experiment_status
         ADD CONSTRAINT experiment_status_retry_parent_fk
         FOREIGN KEY (retry_of_experiment_id)
         REFERENCES experiment_status(experiment_id)
         ON DELETE SET NULL`,
    );
    console.log("Added experiment retry parent foreign key.");
  } else {
    const foreignKey = foreignKeys[0]!;
    const hasExpectedIdentity = (
      foreignKeys.length !== 1
      ? false
      : foreignKey.constraint_name === foreignKeyName
        && foreignKey.column_name === columnName
        && foreignKey.referenced_table_name === "experiment_status"
        && foreignKey.referenced_column_name === "experiment_id"
    );
    if (!hasExpectedIdentity) {
      throw new Error(`${foreignKeyName} exists with an incompatible definition`);
    }
    if (foreignKey.delete_rule === "CASCADE") {
      // An early rollout used CASCADE.  Recursive retry chains can exceed
      // InnoDB's finite cascade depth when an administrator deletes a user;
      // student_id already owns deletion of every attempt, so severing the
      // self-reference is both sufficient and unbounded.
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

  await auditExperimentRetryLineage(connection);

  console.log("Experiment retry lineage migration complete.");
} catch (error) {
  console.error("Experiment retry lineage migration failed:", error);
  process.exitCode = 1;
} finally {
  await connection.end();
}
