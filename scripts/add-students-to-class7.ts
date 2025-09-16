// scripts/add-students-to-class7.ts
// Utility script to create 50 students and enroll them into class with ID 7.

import mysql from "mysql2/promise";
import type { Pool, PoolConnection } from "mysql2/promise";

const TARGET_CLASS_ID = 7;
const STUDENT_COUNT = 50;
const USERNAME_PREFIX = "class7_student";
const DEFAULT_PASSWORD = "password123";

const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE } = process.env;

if (!DB_HOST || !DB_USER || !DB_PASSWORD || !DB_DATABASE) {
  console.error("Database environment variables (DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE) must be set.");
  process.exit(1);
}

type GeneratedStudent = {
  username: string;
  fullName: string;
  email: string;
};

function buildStudent(index: number): GeneratedStudent {
  const padded = index.toString().padStart(3, "0");
  return {
    username: `${USERNAME_PREFIX}${padded}`,
    fullName: `班级7学生${padded}`,
    email: `${USERNAME_PREFIX}${padded}@example.com`,
  };
}

async function ensureClassExists(pool: Pool) {
  const [rows]: any[] = await pool.query(
    "SELECT class_id FROM classes WHERE class_id = ?",
    [TARGET_CLASS_ID],
  );
  if (rows.length === 0) {
    throw new Error(`Class with id ${TARGET_CLASS_ID} does not exist.`);
  }
}

async function seedStudents(pool: Pool) {
  await ensureClassExists(pool);

  const passwordHash = await Bun.password.hash(DEFAULT_PASSWORD, {
    algorithm: "bcrypt",
    cost: 10,
  });

  console.log(`Creating ${STUDENT_COUNT} students for class ${TARGET_CLASS_ID}...`);

  let successCount = 0;
  for (let i = 1; i <= STUDENT_COUNT; i++) {
    const student = buildStudent(i);
    let connection: PoolConnection | null = null;

    try {
      connection = await pool.getConnection();
      await connection.beginTransaction();

      const [insertResult]: any = await connection.query(
        "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?, ?, ?, ?, 'Student')",
        [student.username, passwordHash, student.fullName, student.email],
      );
      const newUserId = insertResult.insertId;

      await connection.query(
        "INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)",
        [newUserId, TARGET_CLASS_ID],
      );

      await connection.commit();
      successCount++;
      console.log(`  Added ${student.username}`);
    } catch (error: any) {
      if (connection) {
        await connection.rollback();
      }

      if (error?.code === "ER_DUP_ENTRY") {
        console.warn(
          `  Duplicate detected for ${student.username} / ${student.email}. Skipping enrollment but continuing.`,
        );
      } else {
        console.error(`  Failed to create ${student.username}:`, error);
      }
    } finally {
      if (connection) {
        connection.release();
      }
    }
  }

  console.log(
    `Finished. Successfully created and enrolled ${successCount} students into class ${TARGET_CLASS_ID}.`,
  );
}

(async () => {
  let pool: Pool | null = null;
  try {
    pool = mysql.createPool({
      host: DB_HOST,
      user: DB_USER,
      password: DB_PASSWORD,
      database: DB_DATABASE,
      waitForConnections: true,
      connectionLimit: 10,
      queueLimit: 0,
    });

    await seedStudents(pool);
  } catch (error) {
    console.error("Script terminated with error:", error);
    process.exitCode = 1;
  } finally {
    if (pool) {
      await pool.end();
    }
    console.log("Database connection closed.");
  }
})();
