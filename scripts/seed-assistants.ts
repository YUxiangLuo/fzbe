// scripts/seed-assistants.ts
import mysql from "mysql2/promise";
import type { Pool } from "mysql2/promise";

// --- Configuration ---
const ASSISTANTS_TO_CREATE = 10;
const STARTING_ASSISTANT_NUMBER = 2; // Starts from assistant02
const DEFAULT_PASSWORD = "password123";

// --- Environment Configuration ---
const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE } = process.env;

if (!DB_HOST || !DB_USER || !DB_PASSWORD || !DB_DATABASE) {
  console.error("Database environment variables are not set!");
  process.exit(1);
}

// --- Data Generation ---
function generateAssistants() {
  const users = [];
  for (let i = 0; i < ASSISTANTS_TO_CREATE; i++) {
    const assistantNum = (STARTING_ASSISTANT_NUMBER + i).toString().padStart(2, "0");
    users.push({
      username: `assistant${assistantNum}`,
      full_name: `助教${assistantNum}`,
      email: `assistant${assistantNum}@example.com`,
      role: "Assistant" as const,
    });
  }
  return users;
}

// --- Main Seeding Function ---
async function seedAssistants(pool: Pool) {
  const assistantsToInsert = generateAssistants();
  console.log(`Generated ${assistantsToInsert.length} assistants to insert.`);

  const password_hash = await Bun.password.hash(DEFAULT_PASSWORD, {
    algorithm: "bcrypt",
    cost: 10,
  });
  console.log(`All new assistants will have the default password: "${DEFAULT_PASSWORD}"`);

  let insertedCount = 0;
  for (const user of assistantsToInsert) {
    try {
      await pool.query(
        "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?, ?, ?, ?, ?)",
        [user.username, password_hash, user.full_name, user.email, user.role],
      );
      insertedCount++;
      console.log(`  -> Inserted Assistant: ${user.username}`);
    } catch (error: any) {
      if (error.code === "ER_DUP_ENTRY") {
        console.warn(
          `  -> Skipping duplicate user: ${user.username} or ${user.email}`,
        );
      } else {
        console.error(
          `  -> Failed to insert user "${user.username}":`,
          error,
        );
      }
    }
  }
  console.log(
    `\nSeeding complete. Inserted ${insertedCount} new assistants.`, 
  );
}

// --- Script Execution ---
(async () => {
  let pool;
  try {
    pool = mysql.createPool({
      host: DB_HOST,
      user: DB_USER,
      password: DB_PASSWORD,
      database: DB_DATABASE,
      waitForConnections: true,
      connectionLimit: 1,
      queueLimit: 0,
    });

    await seedAssistants(pool);
  } catch (error) {
    console.error("An unexpected error occurred:", error);
  } finally {
    if (pool) {
      await pool.end();
      console.log("Database connection closed.");
    }
    process.exit(0);
  }
})();
