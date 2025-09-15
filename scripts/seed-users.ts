// scripts/seed-users.ts
import mysql from "mysql2/promise";
import type { Pool } from "mysql2/promise";

// --- Environment Configuration ---
const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE } = process.env;

if (!DB_HOST || !DB_USER || !DB_PASSWORD || !DB_DATABASE) {
  console.error("Database environment variables are not set!");
  process.exit(1);
}

const DEFAULT_PASSWORD = "password123";

// --- Data Generation ---
function generateUsers() {
  const users = [];
  const TEACHER_COUNT = 3;
  const STUDENT_COUNT = 100;

  // Generate Teachers
  for (let i = 2; i < 2 + TEACHER_COUNT; i++) {
    const teacherNum = i.toString().padStart(2, "0");
    users.push({
      username: `teacher${teacherNum}`,
      full_name: `教师${teacherNum}`,
      email: `teacher${teacherNum}@example.com`,
      role: "Teacher" as const,
    });
  }

  // Generate Students
  for (let i = 2; i < 2 + STUDENT_COUNT; i++) {
    const studentNum = i.toString().padStart(3, "0");
    users.push({
      username: `student${studentNum}`,
      full_name: `学生${studentNum}`,
      email: `student${studentNum}@example.com`,
      role: "Student" as const,
    });
  }

  return users;
}

// --- Main Seeding Function ---
async function seedUsers(pool: Pool) {
  const usersToInsert = generateUsers();
  console.log(`Generated ${usersToInsert.length} users to insert.`);

  const password_hash = await Bun.password.hash(DEFAULT_PASSWORD, {
    algorithm: "bcrypt",
    cost: 10,
  });
  console.log(`All new users will have the default password: "${DEFAULT_PASSWORD}"`);

  let insertedCount = 0;
  for (const user of usersToInsert) {
    try {
      await pool.query(
        "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?, ?, ?, ?, ?)",
        [user.username, password_hash, user.full_name, user.email, user.role],
      );
      insertedCount++;
      console.log(`  -> Inserted ${user.role}: ${user.username}`);
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
    `\nSeeding complete. Inserted ${insertedCount} new users.`, 
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

    await seedUsers(pool);
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
