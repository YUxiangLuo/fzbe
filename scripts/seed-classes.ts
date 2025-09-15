// scripts/seed-classes.ts
import mysql from "mysql2/promise";
import type { Pool } from "mysql2/promise";

// --- Environment Configuration ---
const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE } = process.env;

if (!DB_HOST || !DB_USER || !DB_PASSWORD || !DB_DATABASE) {
  console.error("Database environment variables are not set!");
  process.exit(1);
}

// --- Fake Data ---
const FAKE_CLASSES = [
  "先进制造仿真实验班",
  "供应链管理虚拟仿真",
  "工业工程综合实验2025",
  "数据驱动决策模拟",
  "智能生产系统规划",
  "物流与仓储管理仿真",
  "企业资源计划(ERP)模拟",
  "市场营销决策仿真",
  "项目管理虚拟实践",
  "金融投资与风险模拟",
];

// --- Main Seeding Function ---
async function seedClasses(pool: Pool) {
  let teacherId: number | null = null;

  // 1. Find a teacher to assign to the classes
  try {
    const [rows]: any[] = await pool.query(
      "SELECT user_id FROM users WHERE role = 'Teacher' LIMIT 1",
    );
    if (rows.length > 0) {
      teacherId = rows[0].user_id;
      console.log(`Found teacher with ID: ${teacherId}`);
    } else {
      console.error(
        "Error: No users with the role 'Teacher' found in the database.",
      );
      console.error(
        "Please add a teacher first before running this script.",
      );
      return; // Exit the function if no teacher is found
    }
  } catch (error) {
    console.error("Error fetching teacher:", error);
    return;
  }

  // 2. Insert the new classes
  console.log("Inserting new classes...");
  let insertedCount = 0;
  for (const className of FAKE_CLASSES) {
    try {
      // Generate a unique class code
      const classCode = 
        className.substring(0, 4).toUpperCase() +
        Math.random().toString(36).substring(2, 6).toUpperCase();

      await pool.query(
        "INSERT INTO classes (class_name, class_code, teacher_id) VALUES (?, ?, ?)",
        [className, classCode, teacherId],
      );
      console.log(`  -> Successfully inserted class: ${className}`);
      insertedCount++;
    } catch (error: any) {
      if (error.code === "ER_DUP_ENTRY") {
        console.warn(`  -> Skipping duplicate class code for: ${className}`);
      } else {
        console.error(`  -> Failed to insert class "${className}":`, error);
      }
    }
  }
  console.log(`\nSeeding complete. Inserted ${insertedCount} new classes.`);
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

    await seedClasses(pool);
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
