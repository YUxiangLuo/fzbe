import mysql from "mysql2/promise";

const requireEnv = (name: string) => {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing required env var: ${name}`);
  return value;
};

const database = requireEnv("DB_DATABASE");
if (!/(?:^|_)(?:test|ci)(?:_|$)/i.test(database)) {
  throw new Error("Test database preparation requires a dedicated test/CI database");
}

const connection = await mysql.createConnection({
  host: requireEnv("DB_HOST"),
  user: requireEnv("DB_USER"),
  password: process.env.DB_PASSWORD ?? "",
  database,
});

try {
  await connection.query(`
    CREATE TABLE IF NOT EXISTS users (
      user_id INT PRIMARY KEY
    ) ENGINE=InnoDB
  `);
  await connection.query(`
    CREATE TABLE IF NOT EXISTS experiment_status (
      experiment_id INT PRIMARY KEY,
      student_id INT NOT NULL,
      FOREIGN KEY (student_id) REFERENCES users(user_id) ON DELETE CASCADE
    ) ENGINE=InnoDB
  `);
  console.log("Guided concurrency test database base schema prepared.");
} finally {
  await connection.end();
}
