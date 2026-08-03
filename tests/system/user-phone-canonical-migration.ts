import mysql, { type Connection, type RowDataPacket } from "mysql2/promise";
import { buildMigrationDatabaseOptions } from "../../scripts/migration-database";
import { migrateUserPhoneCanonicalUnique } from "../../scripts/migrate-user-phone-canonical-unique";
import { assertE2eDatabaseResetAllowed } from "../../scripts/test-database-reset-guard";

const LOCAL_DATABASE_HOSTS = new Set(["127.0.0.1", "localhost", "::1"]);

function assertDisposableTarget(): void {
  assertE2eDatabaseResetAllowed();
  const host = process.env.DB_HOST?.trim().toLowerCase() ?? "";
  if (!LOCAL_DATABASE_HOSTS.has(host)) {
    throw new Error(
      "Canonical-phone SQL test additionally requires a local database host.",
    );
  }
}

async function resetUsersTable(connection: Connection): Promise<void> {
  await connection.query("DROP TABLE IF EXISTS users");
  await connection.query(`
    CREATE TABLE users (
      user_id INT AUTO_INCREMENT PRIMARY KEY,
      username VARCHAR(255) NOT NULL UNIQUE,
      phone_number VARCHAR(50) NULL,
      UNIQUE KEY users_phone_number_unique (phone_number)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  `);
}

assertDisposableTarget();
const connection = await mysql.createConnection(buildMigrationDatabaseOptions());

try {
  await resetUsersTable(connection);
  await connection.query(
    "INSERT INTO users (username, phone_number) VALUES ('legacy-a', '13800138000'), ('legacy-b', ' 13800138000 ')",
  );

  let duplicateError: unknown;
  try {
    await migrateUserPhoneCanonicalUnique(connection, { log: () => {} });
  } catch (error) {
    duplicateError = error;
  }
  if (!(duplicateError instanceof Error)) {
    throw new Error("Migration should fail closed when already-ledgered 007 data has canonical duplicates.");
  }
  if (duplicateError.message.includes("13800138000")) {
    throw new Error("Canonical duplicate diagnostics must not disclose phone numbers.");
  }

  await connection.query("DELETE FROM users WHERE username = 'legacy-b'");
  await migrateUserPhoneCanonicalUnique(connection, { log: () => {} });

  const [indexRows] = await connection.query<RowDataPacket[]>(
    `SELECT COLUMN_NAME, NON_UNIQUE
       FROM information_schema.STATISTICS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'users'
        AND INDEX_NAME = 'users_phone_number_canonical_unique'`,
  );
  if (
    indexRows.length !== 1
    || indexRows[0]?.COLUMN_NAME !== "phone_number_canonical"
    || Number(indexRows[0]?.NON_UNIQUE) !== 0
  ) {
    throw new Error("Canonical phone unique index was not installed correctly.");
  }

  let legacyWriterError: unknown;
  try {
    await connection.query(
      "INSERT INTO users (username, phone_number) VALUES ('legacy-c', ' 13800138000 ')",
    );
  } catch (error) {
    legacyWriterError = error;
  }
  if (
    !legacyWriterError
    || typeof legacyWriterError !== "object"
    || !("code" in legacyWriterError)
    || legacyWriterError.code !== "ER_DUP_ENTRY"
  ) {
    throw new Error("Database did not reject a legacy writer's whitespace phone duplicate.");
  }

  // A legacy writer can still store a brand-new unique value with whitespace.
  // The quiesced deployment rerun must converge that raw representation after
  // the old service is stopped, even though the ledger would skip 007 and 012.
  await connection.query(
    "INSERT INTO users (username, phone_number) VALUES ('legacy-d', ' 13900139000 ')",
  );
  await migrateUserPhoneCanonicalUnique(connection, { log: () => {} });
  const [[normalizedPhone]] = await connection.query<RowDataPacket[]>(
    "SELECT phone_number FROM users WHERE username = 'legacy-d'",
  );
  if (normalizedPhone?.phone_number !== "13900139000") {
    throw new Error("Quiesced migration rerun did not normalize the legacy writer's raw value.");
  }
  console.log("Canonical-phone migration SQL behavior verified.");
} finally {
  await connection.query("DROP TABLE IF EXISTS users");
  await connection.end();
}
