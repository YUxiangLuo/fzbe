
import mysql from 'mysql2/promise';
import fs from 'fs/promises';
import path from 'path';

// --- 配置 ---
// 从 .env 文件加载环境变量
// Bun 会自动加载 .env, 所以我们直接使用 process.env
const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE } = process.env;

const SQL_DIR = path.join(process.cwd(), 'sql');
const SCHEMA_FILE = path.join(SQL_DIR, 'schema.sql');
const SEED_FILE = path.join(SQL_DIR, 'seed.sql');

/**
 * 执行一个 .sql 文件中的多个 SQL 语句
 * @param connection 数据库连接实例
 * @param filePath .sql 文件的路径
 */
async function executeSqlFile(connection: mysql.Connection, filePath: string) {
  try {
    const sql = await fs.readFile(filePath, 'utf-8');
    // 移除注释并按分号分割语句
    const statements = sql
      .split(/;(?=\s*CREATE|INSERT|ALTER|SET|TRUNCATE)/)
      .map(s => s.trim())
      .filter(s => s.length > 0);

    for (const statement of statements) {
      if (statement) {
        await connection.query(statement);
        console.log(`✅ Executed statement from ${path.basename(filePath)}`);
      }
    }
    console.log(`✨ Successfully executed all statements from ${path.basename(filePath)}.`);
  } catch (error) {
    console.error(`❌ Error executing SQL file ${filePath}:`, error);
    throw error;
  }
}

/**
 * 主函数：设置数据库
 */
async function setupDatabase() {
  let connection: mysql.Connection | null = null;
  try {
    // 1. 连接到 MySQL/MariaDB 服务器 (不指定具体数据库)
    console.log(`Connecting to MySQL server at ${DB_HOST}...`);
    connection = await mysql.createConnection({
      host: DB_HOST,
      user: DB_USER,
      password: DB_PASSWORD,
      multipleStatements: true, // 允许执行多个语句
    });
    console.log('✅ Connection successful.');

    // 2. 创建数据库
    console.log(`Creating database '${DB_DATABASE}' if it doesn't exist...`);
    await connection.query(`CREATE DATABASE IF NOT EXISTS \`${DB_DATABASE}\`;`);
    console.log(`✅ Database '${DB_DATABASE}' is ready.`);

    // 3. 切换到指定数据库
    await connection.changeUser({ database: DB_DATABASE });
    console.log(`Switched to database '${DB_DATABASE}'.`);

    // 4. 执行 schema.sql 来创建表结构
    console.log('Executing schema.sql...');
    await executeSqlFile(connection, SCHEMA_FILE);

    // 5. 执行 seed.sql 来填充数据
    console.log('Executing seed.sql...');
    await executeSqlFile(connection, SEED_FILE);

    console.log('\n🎉 Database setup complete! All tables created and seeded.');

  } catch (error) {
    console.error('\n❌ An error occurred during database setup:', error);
    process.exit(1); // 出错时退出
  } finally {
    if (connection) {
      await connection.end();
      console.log('\n🔌 Database connection closed.');
    }
  }
}

// --- 运行脚本 ---
setupDatabase();
