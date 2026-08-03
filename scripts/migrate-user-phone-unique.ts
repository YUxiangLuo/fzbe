import { type ResultSetHeader, type RowDataPacket } from "mysql2/promise";
import { createMigrationConnection } from "./migration-database";

interface IndexRow extends RowDataPacket {
  index_name: string;
}

interface DuplicatePhoneRow extends RowDataPacket {
  normalized_phone_number: string;
  duplicate_count: number;
}

const connection = await createMigrationConnection();

try {
  const [duplicates] = await connection.query<DuplicatePhoneRow[]>(
    `SELECT TRIM(phone_number) AS normalized_phone_number, COUNT(*) AS duplicate_count
       FROM users
      WHERE phone_number IS NOT NULL
        AND TRIM(phone_number) <> ''
      GROUP BY TRIM(phone_number)
     HAVING COUNT(*) > 1
      ORDER BY duplicate_count DESC, normalized_phone_number ASC
      LIMIT 20`,
  );

  if (duplicates.length > 0) {
    const summary = duplicates
      .map((row) => `${row.normalized_phone_number} (${Number(row.duplicate_count)} accounts)`)
      .join(", ");
    throw new Error(
      "Cannot enforce canonical unique phone numbers because duplicates exist after trimming: "
      + `${summary}. Resolve every duplicate before rerunning this migration.`,
    );
  }

  const [normalizedEmptyPhones] = await connection.query<ResultSetHeader>(
    "UPDATE users SET phone_number = NULL WHERE phone_number IS NOT NULL AND TRIM(phone_number) = ''",
  );
  if (normalizedEmptyPhones.affectedRows > 0) {
    console.log(`Normalized ${normalizedEmptyPhones.affectedRows} empty phone number(s) to NULL.`);
  }

  const [trimmedPhones] = await connection.query<ResultSetHeader>(
    "UPDATE users SET phone_number = TRIM(phone_number) WHERE phone_number IS NOT NULL AND BINARY phone_number <> BINARY TRIM(phone_number)",
  );
  if (trimmedPhones.affectedRows > 0) {
    console.log(`Trimmed ${trimmedPhones.affectedRows} phone number(s).`);
  }

  const [existingIndexes] = await connection.query<IndexRow[]>(
    `SELECT INDEX_NAME AS index_name
       FROM information_schema.STATISTICS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'users'
        AND INDEX_NAME = 'users_phone_number_unique'
      LIMIT 1`,
  );

  if (existingIndexes.length > 0) {
    console.log("users_phone_number_unique already exists; phone values are canonical.");
  } else {
    await connection.query(
      "ALTER TABLE users ADD UNIQUE KEY users_phone_number_unique (phone_number)",
    );
    console.log("Added users_phone_number_unique.");
  }
} finally {
  await connection.end();
}
