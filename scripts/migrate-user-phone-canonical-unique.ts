import {
  type Connection,
  type ResultSetHeader,
  type RowDataPacket,
} from "mysql2/promise";
import { createMigrationConnection } from "./migration-database";

const CANONICAL_COLUMN = "phone_number_canonical";
const CANONICAL_INDEX = "users_phone_number_canonical_unique";

interface DuplicateSummaryRow extends RowDataPacket {
  duplicate_group_count: number;
  affected_account_count: number;
}

interface ColumnRow extends RowDataPacket {
  data_type: string;
  character_maximum_length: number | null;
  generation_expression: string | null;
  extra: string;
}

interface IndexRow extends RowDataPacket {
  column_name: string;
  non_unique: number;
  seq_in_index: number;
}

interface MismatchCountRow extends RowDataPacket {
  mismatch_count: number;
}

export interface CanonicalPhoneMigrationOptions {
  log?: (message: string) => void;
}

function normalizeGenerationExpression(expression: string | null): string {
  return (expression ?? "")
    .toLowerCase()
    .replaceAll("`", "")
    .replace(/\s+/g, "")
    // MySQL exposes the empty literal as e.g. _latin1\'\' while MariaDB
    // returns ''. Both describe the same generated expression.
    .replaceAll("\\'", "'")
    .replace(/_[a-z0-9]+(?='')/g, "");
}

function assertCanonicalColumnDefinition(column: ColumnRow): void {
  const expression = normalizeGenerationExpression(column.generation_expression);
  if (
    column.data_type.toLowerCase() !== "varchar"
    || Number(column.character_maximum_length) !== 50
    || !column.extra.toUpperCase().includes("STORED GENERATED")
    || expression !== "nullif(trim(phone_number),'')"
  ) {
    throw new Error(
      `${CANONICAL_COLUMN} exists with an incompatible definition; manual repair is required.`,
    );
  }
}

function assertCanonicalIndexDefinition(indexRows: IndexRow[]): void {
  if (
    indexRows.length !== 1
    || Number(indexRows[0]?.non_unique) !== 0
    || Number(indexRows[0]?.seq_in_index) !== 1
    || indexRows[0]?.column_name !== CANONICAL_COLUMN
  ) {
    throw new Error(
      `${CANONICAL_INDEX} exists with an incompatible definition; manual repair is required.`,
    );
  }
}

async function loadCanonicalColumn(connection: Connection): Promise<ColumnRow[]> {
  const [rows] = await connection.query<ColumnRow[]>(
    `SELECT DATA_TYPE AS data_type,
            CHARACTER_MAXIMUM_LENGTH AS character_maximum_length,
            GENERATION_EXPRESSION AS generation_expression,
            EXTRA AS extra
       FROM information_schema.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'users'
        AND COLUMN_NAME = ?`,
    [CANONICAL_COLUMN],
  );
  return rows;
}

async function loadCanonicalIndex(connection: Connection): Promise<IndexRow[]> {
  const [rows] = await connection.query<IndexRow[]>(
    `SELECT COLUMN_NAME AS column_name,
            NON_UNIQUE AS non_unique,
            SEQ_IN_INDEX AS seq_in_index
       FROM information_schema.STATISTICS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'users'
        AND INDEX_NAME = ?
      ORDER BY SEQ_IN_INDEX`,
    [CANONICAL_INDEX],
  );
  return rows;
}

async function assertNoCanonicalDuplicates(connection: Connection): Promise<void> {
  const [[summary]] = await connection.query<DuplicateSummaryRow[]>(
    `SELECT COUNT(*) AS duplicate_group_count,
            COALESCE(SUM(duplicate_count), 0) AS affected_account_count
       FROM (
         SELECT COUNT(*) AS duplicate_count
           FROM users
          WHERE phone_number IS NOT NULL
            AND TRIM(phone_number) <> ''
          GROUP BY TRIM(phone_number)
         HAVING COUNT(*) > 1
       ) canonical_duplicates`,
  );

  if (Number(summary?.duplicate_group_count ?? 0) > 0) {
    throw new Error(
      "Cannot enforce canonical unique phone numbers: "
      + `${Number(summary?.duplicate_group_count)} normalized value(s) conflict across `
      + `${Number(summary?.affected_account_count)} account(s). `
      + "Resolve every conflict before rerunning this migration.",
    );
  }
}

export async function migrateUserPhoneCanonicalUnique(
  connection: Connection,
  { log = console.log }: CanonicalPhoneMigrationOptions = {},
): Promise<void> {
  // The old release can still write while staging migrations run. This check
  // gives a non-PII error for existing conflicts; the unique generated-column
  // index below is the authoritative barrier and also catches a concurrent
  // legacy write that lands after this query.
  await assertNoCanonicalDuplicates(connection);

  const [normalizedEmptyPhones] = await connection.query<ResultSetHeader>(
    "UPDATE users SET phone_number = NULL WHERE phone_number IS NOT NULL AND TRIM(phone_number) = ''",
  );
  if (normalizedEmptyPhones.affectedRows > 0) {
    log(`Normalized ${normalizedEmptyPhones.affectedRows} empty phone number(s) to NULL.`);
  }

  const [trimmedPhones] = await connection.query<ResultSetHeader>(
    "UPDATE users SET phone_number = TRIM(phone_number) WHERE phone_number IS NOT NULL AND BINARY phone_number <> BINARY TRIM(phone_number)",
  );
  if (trimmedPhones.affectedRows > 0) {
    log(`Trimmed ${trimmedPhones.affectedRows} phone number(s).`);
  }

  let columns = await loadCanonicalColumn(connection);
  if (columns.length === 0) {
    await connection.query(
      `ALTER TABLE users
         ADD COLUMN phone_number_canonical VARCHAR(50)
         GENERATED ALWAYS AS (NULLIF(TRIM(phone_number), '')) STORED`,
    );
    log(`Added generated column ${CANONICAL_COLUMN}.`);
    columns = await loadCanonicalColumn(connection);
  }
  if (columns.length !== 1) {
    throw new Error(`Expected exactly one ${CANONICAL_COLUMN} column.`);
  }
  assertCanonicalColumnDefinition(columns[0]!);

  let indexes = await loadCanonicalIndex(connection);
  if (indexes.length === 0) {
    // ALTER TABLE either builds the complete unique index or fails. Once it
    // succeeds, old writers cannot insert a whitespace variant of a phone
    // number even though they do not normalize values themselves.
    await connection.query(
      `ALTER TABLE users
         ADD UNIQUE KEY users_phone_number_canonical_unique (phone_number_canonical)`,
    );
    log(`Added ${CANONICAL_INDEX}.`);
    indexes = await loadCanonicalIndex(connection);
  }
  assertCanonicalIndexDefinition(indexes);

  const [[mismatches]] = await connection.query<MismatchCountRow[]>(
    `SELECT COUNT(*) AS mismatch_count
       FROM users
      WHERE NOT (
        phone_number_canonical <=> NULLIF(TRIM(phone_number), '')
      )`,
  );
  if (Number(mismatches?.mismatch_count ?? 0) !== 0) {
    throw new Error(`${CANONICAL_COLUMN} does not match the canonical phone expression.`);
  }

  // Keep this final audit even on an idempotent rerun. It turns unexpected
  // manual index/column drift into a deployment failure rather than silently
  // relying on application-level trimming.
  await assertNoCanonicalDuplicates(connection);
  log("Canonical phone uniqueness is enforced by the database.");
}

if (import.meta.main) {
  const connection = await createMigrationConnection();
  try {
    await migrateUserPhoneCanonicalUnique(connection);
  } finally {
    await connection.end();
  }
}
