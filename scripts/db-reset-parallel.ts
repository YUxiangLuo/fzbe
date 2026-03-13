import mysql from "mysql2/promise";
import path from "path";

const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE } = process.env;

const SQL_DIR = path.join(process.cwd(), "sql");
const SCHEMA_FILE = path.join(SQL_DIR, "schema.sql");
const SEED_FILE = path.join(SQL_DIR, "seed.parallel-base.sql");
const SYSTEM_DATABASES = new Set(["mysql", "information_schema", "performance_schema", "sys"]);
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);

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
  return process.argv.includes("--force");
}

function isEnabled(...values: Array<string | undefined>) {
  return values.some((value) => value === "1");
}

function warnDeprecatedEnv(oldName: string, newName: string) {
  if (process.env[oldName]) {
    console.warn(`⚠️  ${oldName} is deprecated, use ${newName}`);
  }
}

function assertDbResetSafety(force: boolean) {
  if (!force) {
    exitWithSafetyError("Destructive reset requires --force. Re-run with `bun run db:reset:parallel --force`.");
  }

  const dbHost = requireNonEmptyEnv("DB_HOST", DB_HOST);
  const dbDatabase = requireNonEmptyEnv("DB_DATABASE", DB_DATABASE);
  requireNonEmptyEnv("DB_USER", DB_USER);

  if (SYSTEM_DATABASES.has(dbDatabase.toLowerCase())) {
    exitWithSafetyError(`Refusing to run against system database "${dbDatabase}".`);
  }

  warnDeprecatedEnv("DB_SETUP_ALLOW_PROD", "DB_RESET_ALLOW_PROD");
  if (
    process.env.NODE_ENV === "production"
    && !isEnabled(process.env.DB_RESET_ALLOW_PROD, process.env.DB_SETUP_ALLOW_PROD)
  ) {
    exitWithSafetyError("Refusing to run in production. Set DB_RESET_ALLOW_PROD=1 to override.");
  }

  warnDeprecatedEnv("DB_SETUP_ALLOW_REMOTE", "DB_RESET_ALLOW_REMOTE");
  if (
    !LOCAL_HOSTS.has(dbHost.toLowerCase())
    && !isEnabled(process.env.DB_RESET_ALLOW_REMOTE, process.env.DB_SETUP_ALLOW_REMOTE)
  ) {
    exitWithSafetyError(
      `Refusing to run against remote host "${dbHost}". Set DB_RESET_ALLOW_REMOTE=1 to override.`,
    );
  }

  return { dbHost, dbDatabase };
}

async function executeSqlFile(connection: mysql.Connection, filePath: string) {
  try {
    const sql = await Bun.file(filePath).text();
    const statements = sql
      .split(/;(?=\s*CREATE|\s*INSERT|\s*ALTER|\s*SET|\s*TRUNCATE)/)
      .map((statement) => statement.trim())
      .filter((statement) => statement.length > 0);

    for (const statement of statements) {
      await connection.query(statement);
      console.log(`✅ Executed statement from ${path.basename(filePath)}`);
    }
    console.log(`✨ Successfully executed all statements from ${path.basename(filePath)}.`);
  } catch (error) {
    console.error(`❌ Error executing SQL file ${filePath}:`, error);
    throw error;
  }
}

async function dropAllTables(connection: mysql.Connection, database: string) {
  const [rows] = await connection.query<mysql.RowDataPacket[]>(
    'SELECT TABLE_NAME FROM information_schema.tables WHERE table_schema = ? AND table_type = "BASE TABLE"',
    [database],
  );

  if (rows.length === 0) {
    console.log("No existing tables to drop.");
    return;
  }

  const tableNames = rows.map((row) => row.TABLE_NAME as string);
  console.log(`Dropping ${tableNames.length} tables...`);
  await connection.query("SET FOREIGN_KEY_CHECKS = 0;");
  await connection.query(
    `DROP TABLE IF EXISTS ${tableNames.map((name) => `\`${name}\``).join(", ")};`,
  );
  await connection.query("SET FOREIGN_KEY_CHECKS = 1;");
  console.log("✅ All existing tables dropped.");
}

async function resetDatabase() {
  let connection: mysql.Connection | null = null;

  try {
    const { dbHost, dbDatabase } = assertDbResetSafety(hasForceFlag());

    console.log(`Connecting to MySQL server at ${dbHost}...`);
    connection = await mysql.createConnection({
      host: dbHost,
      user: DB_USER,
      password: DB_PASSWORD,
      multipleStatements: true,
      charset: "utf8mb4",
    });
    console.log("✅ Connection successful.");

    console.log(`Creating database '${dbDatabase}' if it doesn't exist...`);
    await connection.query(`CREATE DATABASE IF NOT EXISTS \`${dbDatabase}\`;`);
    console.log(`✅ Database '${dbDatabase}' is ready.`);

    await connection.changeUser({ database: dbDatabase });
    console.log(`Switched to database '${dbDatabase}'.`);

    await dropAllTables(connection, dbDatabase);

    console.log("Executing schema.sql...");
    await executeSqlFile(connection, SCHEMA_FILE);

    console.log("Executing seed.parallel-base.sql...");
    await executeSqlFile(connection, SEED_FILE);

    console.log("\n🎉 Parallel database reset complete.");
    console.log("Next steps:");
    console.log("  1. bun run seed:parallel:assets --force");
    console.log("  2. bun run seed:parallel:scenario training-burst --force");
  } catch (error) {
    console.error("\n❌ An error occurred during parallel database reset:", error);
    process.exit(1);
  } finally {
    if (connection) {
      await connection.end();
      console.log("\n🔌 Database connection closed.");
    }
  }
}

await resetDatabase();
