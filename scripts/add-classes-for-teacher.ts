// scripts/add-classes-for-teacher.ts
import mysql from "mysql2/promise";
import type { Pool } from "mysql2/promise";

// --- Configuration ---
const TEACHER_ID = 14;
const CLASSES_TO_CREATE = [
  "高级生产仿真 A班",
  "高级生产仿真 B班",
  "物流战略模拟 2025",
  "运营管理实践",
  "数字孪生与制造",
  "工业大数据分析",
  "智能决策支持系统",
  "虚拟现实技术应用",
  "人因工程学仿真",
  "质量管理与六西格玛",
];

// --- Environment Configuration ---
const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE } = process.env;

if (!DB_HOST || !DB_USER || !DB_PASSWORD || !DB_DATABASE) {
  console.error("Database environment variables are not set!");
  process.exit(1);
}

// --- Main Seeding Function ---
async function addClasses(pool: Pool) {
  // 1. Verify the teacher exists and is a teacher
  try {
    const [rows]: any[] = await pool.query(
      "SELECT user_id, role FROM users WHERE user_id = ?",
      [TEACHER_ID],
    );
    if (rows.length === 0) {
      console.error(`Error: User with ID ${TEACHER_ID} does not exist.`);
      return;
    }
    if (rows[0].role !== "Teacher") {
      console.error(
        `Error: User with ID ${TEACHER_ID} is a '${rows[0].role}', not a 'Teacher'.`,
      );
      return;
    }
    console.log(
      `Verified teacher with ID ${TEACHER_ID} exists. Starting class creation...`,
    );
  } catch (error) {
    console.error("Error verifying teacher existence:", error);
    return;
  }

  // 2. Insert the new classes
  let insertedCount = 0;
  for (const className of CLASSES_TO_CREATE) {
    try {
      const classCode =
        "CLS" +
        Math.random().toString(36).substring(2, 8).toUpperCase();

      await pool.query(
        "INSERT INTO classes (class_name, class_code, teacher_id) VALUES (?, ?, ?)",
        [className, classCode, TEACHER_ID],
      );
      insertedCount++;
      console.log(`  -> Successfully inserted class: ${className}`);
    } catch (error: any) {
      if (error.code === "ER_DUP_ENTRY") {
        console.warn(`  -> Skipping duplicate class code for: ${className}`);
      } else {
        console.error(`  -> Failed to insert class "${className}":`, error);
      }
    }
  }
  console.log(
    `\nSeeding complete. Inserted ${insertedCount} new classes for teacher ID ${TEACHER_ID}.`,
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

    await addClasses(pool);
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
