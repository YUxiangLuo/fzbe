import mysql from 'mysql2/promise';
import path from 'path';
import {
  buildDatabaseConnectionOptions,
  loadDatabaseEnvironment,
} from '../src/config/database-connection-options';

// --- 配置 ---
// 从 .env 文件加载环境变量
// Bun 会自动加载 .env, 所以我们直接使用 process.env
const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE } = process.env;

const SQL_DIR = path.join(process.cwd(), 'sql');
const SCHEMA_FILE = path.join(SQL_DIR, 'schema.sql');
const SEED_FILE = path.join(SQL_DIR, 'seed.sql');
const SYSTEM_DATABASES = new Set(['mysql', 'information_schema', 'performance_schema', 'sys']);
const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '::1']);

function exitWithSafetyError(message: string): never {
  console.error(`\n🚫 Safety check failed: ${message}`);
  process.exit(1);
}

function requireNonEmptyEnv(name: string, value: string | undefined): string {
  const normalized = value?.trim();
  if (!normalized) {
    exitWithSafetyError(`Missing required env var: ${name}`);
  }
  return normalized;
}

function hasForceFlag() {
  return process.argv.includes('--force');
}

function isEnabled(...values: Array<string | undefined>) {
  return values.some(value => value === '1');
}

function warnDeprecatedEnv(oldName: string, newName: string) {
  if (process.env[oldName]) {
    console.warn(`⚠️  ${oldName} is deprecated, use ${newName}`);
  }
}

function assertDbResetSafety(force: boolean) {
  if (!force) {
    exitWithSafetyError('Destructive reset requires --force. Re-run with `bun run db:reset --force`.');
  }

  const dbHost = requireNonEmptyEnv('DB_HOST', DB_HOST);
  const dbDatabase = requireNonEmptyEnv('DB_DATABASE', DB_DATABASE);
  const dbUser = requireNonEmptyEnv('DB_USER', DB_USER);

  if (SYSTEM_DATABASES.has(dbDatabase.toLowerCase())) {
    exitWithSafetyError(`Refusing to run against system database "${dbDatabase}".`);
  }

  warnDeprecatedEnv('DB_SETUP_ALLOW_PROD', 'DB_RESET_ALLOW_PROD');
  if (
    process.env.NODE_ENV === 'production' &&
    !isEnabled(process.env.DB_RESET_ALLOW_PROD, process.env.DB_SETUP_ALLOW_PROD)
  ) {
    exitWithSafetyError('Refusing to run in production. Set DB_RESET_ALLOW_PROD=1 to override.');
  }

  warnDeprecatedEnv('DB_SETUP_ALLOW_REMOTE', 'DB_RESET_ALLOW_REMOTE');
  if (
    !LOCAL_HOSTS.has(dbHost.toLowerCase()) &&
    !isEnabled(process.env.DB_RESET_ALLOW_REMOTE, process.env.DB_SETUP_ALLOW_REMOTE)
  ) {
    exitWithSafetyError(
      `Refusing to run against remote host "${dbHost}". Set DB_RESET_ALLOW_REMOTE=1 to override.`,
    );
  }

  return { dbHost, dbUser, dbDatabase };
}

/**
 * 执行一个 .sql 文件中的多个 SQL 语句
 * @param connection 数据库连接实例
 * @param filePath .sql 文件的路径
 */
async function executeSqlFile(connection: mysql.Connection, filePath: string) {
  try {
    const sql = await Bun.file(filePath).text();
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
 * 删除指定数据库中的所有表
 * @param connection 数据库连接实例
 * @param database 数据库名称
 */
async function dropAllTables(connection: mysql.Connection, database: string) {
  const [rows] = await connection.query<mysql.RowDataPacket[]>(
    "SELECT TABLE_NAME AS table_name FROM information_schema.tables WHERE table_schema = ? AND table_type = 'BASE TABLE'",
    [database]
  );
  if (rows.length === 0) {
    console.log('No existing tables to drop.');
    return;
  }

  const tableNames = rows.map(row => row.table_name as string);
  console.log(`Dropping ${tableNames.length} tables...`);
  await connection.query('SET FOREIGN_KEY_CHECKS = 0;');
  try {
    await connection.query(
      `DROP TABLE IF EXISTS ${tableNames.map(name => `\`${name.replaceAll('`', '``')}\``).join(', ')};`
    );
  } finally {
    await connection.query('SET FOREIGN_KEY_CHECKS = 1;');
  }
  console.log('✅ All existing tables dropped.');
}

/**
 * 主函数：重置数据库
 */
async function resetDatabase() {
  let connection: mysql.Connection | null = null;
  try {
    const { dbHost, dbUser, dbDatabase } = assertDbResetSafety(hasForceFlag());
    const escapedDatabase = dbDatabase.replaceAll('`', '``');
    const { database: _ignoredDatabase, ...serverOptions } = buildDatabaseConnectionOptions(
      loadDatabaseEnvironment({
        ...process.env,
        DB_HOST: dbHost,
        DB_USER: dbUser,
        DB_PASSWORD: DB_PASSWORD ?? '',
        DB_DATABASE: dbDatabase,
      }),
    );

    // 1. 连接到 MySQL/MariaDB 服务器 (不指定具体数据库)
    console.log(`Connecting to MySQL server at ${dbHost}...`);
    connection = await mysql.createConnection({
      ...serverOptions,
      multipleStatements: true, // 允许执行多个语句
    });
    console.log('✅ Connection successful.');

    // 2. 创建数据库
    console.log(`Creating database '${dbDatabase}' if it doesn't exist...`);
    await connection.query(
      `CREATE DATABASE IF NOT EXISTS \`${escapedDatabase}\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;`,
    );
    console.log(`✅ Database '${dbDatabase}' is ready.`);

    // 3. 切换到指定数据库
    await connection.query(`USE \`${escapedDatabase}\`;`);
    console.log(`Switched to database '${dbDatabase}'.`);

    // 4. 删除已有表
    await dropAllTables(connection, dbDatabase);

    // 5. 执行 schema.sql 来创建表结构
    console.log('Executing schema.sql...');
    await executeSqlFile(connection, SCHEMA_FILE);

    // 6. 执行 seed.sql 来填充数据
    console.log('Executing seed.sql...');
    await executeSqlFile(connection, SEED_FILE);

    console.log('\n🎉 Database reset complete! All tables recreated and seeded.');

  } catch (error) {
    console.error('\n❌ An error occurred during database reset:', error);
    process.exit(1); // 出错时退出
  } finally {
    if (connection) {
      await connection.end();
      console.log('\n🔌 Database connection closed.');
    }
  }
}

// --- 运行脚本 ---
resetDatabase();
