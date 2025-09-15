// scripts/seed-students-for-class.ts
import mysql from "mysql2/promise";
import type { Pool, PoolConnection } from "mysql2/promise";

// --- Configuration ---
const CLASS_ID_TO_ENROLL = 2;
const STUDENTS_TO_CREATE = 200;
const STARTING_STUDENT_NUMBER = 152; // Start after previously created students
const DEFAULT_PASSWORD = "password123";

// --- Environment Configuration ---
const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE } = process.env;

if (!DB_HOST || !DB_USER || !DB_PASSWORD || !DB_DATABASE) {
  console.error("Database environment variables are not set!");
  process.exit(1);
}

// --- Data Generation ---
function generateStudent(studentNumber: number) {
  const studentNumStr = studentNumber.toString().padStart(3, "0");
  return {
    username: `student${studentNumStr}`,
    full_name: `学生${studentNumStr}`,
    email: `student${studentNumStr}@example.com`,
    role: "Student" as const,
  };
}

// --- Main Seeding Function ---
async function seedAndEnrollStudents(pool: Pool) {
  // 1. Verify the class exists
  try {
    const [rows]: any[] = await pool.query(
      "SELECT class_id FROM classes WHERE class_id = ?",
      [CLASS_ID_TO_ENROLL],
    );
    if (rows.length === 0) {
      console.error(
        `Error: Class with ID ${CLASS_ID_TO_ENROLL} does not exist.`,
      );
      return;
    }
    console.log(
      `Verified class ${CLASS_ID_TO_ENROLL} exists. Starting student creation...`,
    );
  } catch (error) {
    console.error("Error verifying class existence:", error);
    return;
  }

  // 2. Hash the default password
  const password_hash = await Bun.password.hash(DEFAULT_PASSWORD, {
    algorithm: "bcrypt",
    cost: 10,
  });
  console.log(
    `All new students will have the default password: "${DEFAULT_PASSWORD}"`,
  );

  // 3. Create and enroll students
  let insertedCount = 0;
  for (let i = 0; i < STUDENTS_TO_CREATE; i++) {
    const studentNumber = STARTING_STUDENT_NUMBER + i;
    const student = generateStudent(studentNumber);
    let connection: PoolConnection | null = null;

    try {
      connection = await pool.getConnection();
      await connection.beginTransaction();

      // Insert into users table
      const [userResult]: any = await connection.query(
        "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?, ?, ?, ?, ?)",
        [
          student.username,
          password_hash,
          student.full_name,
          student.email,
          student.role,
        ],
      );
      const newUserId = userResult.insertId;

      // Insert into enrollments table
      await connection.query(
        "INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)",
        [newUserId, CLASS_ID_TO_ENROLL],
      );

      await connection.commit();
      insertedCount++;
      console.log(
        `  -> Successfully created and enrolled student: ${student.username}`,
      );
    } catch (error: any) {
      if (connection) await connection.rollback();

      if (error.code === "ER_DUP_ENTRY") {
        console.warn(
          `  -> Skipping duplicate user: ${student.username} or ${student.email}`,
        );
      } else {
        console.error(
          `  -> Failed to process student "${student.username}":`,
          error,
        );
      }
    } finally {
      if (connection) connection.release();
    }
  }
  console.log(
    `
Seeding complete. Inserted and enrolled ${insertedCount} new students into class ${CLASS_ID_TO_ENROLL}.`,
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
      connectionLimit: 10, // Allow more connections for parallel potential
      queueLimit: 0,
    });

    await seedAndEnrollStudents(pool);
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
