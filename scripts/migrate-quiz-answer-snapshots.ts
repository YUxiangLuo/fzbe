import mysql from "mysql2/promise";

const { DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE } = process.env;

function requireEnv(name: string, value: string | undefined): string {
  const normalized = value?.trim();
  if (!normalized) throw new Error(`Missing required env var: ${name}`);
  return normalized;
}

const connection = await mysql.createConnection({
  host: requireEnv("DB_HOST", DB_HOST),
  user: requireEnv("DB_USER", DB_USER),
  password: DB_PASSWORD ?? "",
  database: requireEnv("DB_DATABASE", DB_DATABASE),
  multipleStatements: true,
});

interface AnswerBackfillRow extends mysql.RowDataPacket {
  answer_id: number;
  question_id: number;
  knowledge_point: string | null;
  question_type: string | null;
  question_text: string | null;
  options: unknown;
  correct_answers: unknown;
  answer_explanation?: string | null;
}

interface ForeignKeyRow extends mysql.RowDataPacket {
  CONSTRAINT_NAME: string;
  DELETE_RULE: string;
}

type QuestionOptions = Record<string, string> | string[] | null;

function quoteIdentifier(identifier: string): string {
  return `\`${identifier.replace(/`/g, "``")}\``;
}

function parseJsonField(value: unknown): unknown {
  if (value === null || value === undefined) return value;
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function parseStringArrayField(value: unknown): string[] {
  const parsed = parseJsonField(value);
  if (!Array.isArray(parsed)) return [];
  return parsed.map((item) => String(item));
}

function parseOptionsField(value: unknown): QuestionOptions {
  const parsed = parseJsonField(value);
  if (Array.isArray(parsed)) return parsed.map((item) => String(item));
  if (parsed && typeof parsed === "object") {
    return Object.fromEntries(
      Object.entries(parsed as Record<string, unknown>).map(([key, optionValue]) => [
        key,
        String(optionValue),
      ]),
    );
  }
  return null;
}

function normalizeAnswerExplanationField(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const normalized = value.trim();
  return normalized ? normalized : null;
}

async function hasColumn(tableName: string, columnName: string) {
  const [rows] = await connection.query<mysql.RowDataPacket[]>(
    `SELECT 1
       FROM INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = ?
        AND COLUMN_NAME = ?
      LIMIT 1`,
    [tableName, columnName],
  );
  return rows.length > 0;
}

async function hasIndexOnColumn(tableName: string, columnName: string) {
  const [rows] = await connection.query<mysql.RowDataPacket[]>(
    `SELECT 1
       FROM INFORMATION_SCHEMA.STATISTICS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = ?
        AND COLUMN_NAME = ?
      LIMIT 1`,
    [tableName, columnName],
  );
  return rows.length > 0;
}

async function getQuestionBankForeignKeys() {
  const [rows] = await connection.query<ForeignKeyRow[]>(
    `SELECT kcu.CONSTRAINT_NAME,
            rc.DELETE_RULE
       FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
       JOIN INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
         ON rc.CONSTRAINT_SCHEMA = kcu.CONSTRAINT_SCHEMA
        AND rc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
      WHERE kcu.TABLE_SCHEMA = DATABASE()
        AND kcu.TABLE_NAME = 'quiz_answers'
        AND kcu.COLUMN_NAME = 'question_id'
        AND kcu.REFERENCED_TABLE_NAME = 'question_bank'
        AND kcu.REFERENCED_COLUMN_NAME = 'question_id'`,
  );
  return rows;
}

function buildQuestionSnapshot(row: AnswerBackfillRow) {
  return {
    question_id: Number(row.question_id),
    knowledge_point: row.knowledge_point,
    question_type: row.question_type || "Single Choice",
    question_text: row.question_text || `Question #${row.question_id}`,
    options: parseOptionsField(row.options),
    correct_answers: parseStringArrayField(row.correct_answers),
    answer_explanation: normalizeAnswerExplanationField(row.answer_explanation),
  };
}

async function backfillSnapshots() {
  const hasAnswerExplanationColumn = await hasColumn("question_bank", "answer_explanation");
  const [rows] = await connection.query<AnswerBackfillRow[]>(
    `SELECT qa.answer_id,
            qa.question_id,
            qb.knowledge_point,
            qb.question_type,
            qb.question_text,
            qb.options,
            qb.correct_answers,
            ${hasAnswerExplanationColumn ? "qb.answer_explanation" : "NULL AS answer_explanation"}
       FROM quiz_answers qa
       JOIN question_bank qb ON qb.question_id = qa.question_id
      WHERE qa.question_snapshot IS NULL`,
  );

  for (const row of rows) {
    await connection.query(
      "UPDATE quiz_answers SET question_snapshot = ? WHERE answer_id = ?",
      [JSON.stringify(buildQuestionSnapshot(row)), row.answer_id],
    );
  }

  return rows.length;
}

try {
  if (!(await hasColumn("quiz_answers", "question_snapshot"))) {
    await connection.query(
      "ALTER TABLE quiz_answers ADD COLUMN question_snapshot JSON NULL AFTER submitted_answer",
    );
  }

  const backfilledCount = await backfillSnapshots();

  const foreignKeys = await getQuestionBankForeignKeys();
  for (const foreignKey of foreignKeys) {
    if (
      foreignKey.CONSTRAINT_NAME !== "quiz_answers_question_id_fk" ||
      foreignKey.DELETE_RULE !== "SET NULL"
    ) {
      await connection.query(
        `ALTER TABLE quiz_answers DROP FOREIGN KEY ${quoteIdentifier(foreignKey.CONSTRAINT_NAME)}`,
      );
    }
  }

  await connection.query("ALTER TABLE quiz_answers MODIFY question_id INT NULL");

  if (!(await hasIndexOnColumn("quiz_answers", "question_id"))) {
    await connection.query("ALTER TABLE quiz_answers ADD INDEX idx_quiz_answers_question_id (question_id)");
  }

  const currentForeignKeys = await getQuestionBankForeignKeys();
  if (currentForeignKeys.length === 0) {
    await connection.query(
      `ALTER TABLE quiz_answers
         ADD CONSTRAINT quiz_answers_question_id_fk
         FOREIGN KEY (question_id)
         REFERENCES question_bank(question_id)
         ON DELETE SET NULL`,
    );
  }

  console.log(`Quiz answer snapshot migration complete. Backfilled ${backfilledCount} answer snapshots.`);
} catch (error) {
  console.error("Quiz answer snapshot migration failed:", error);
  process.exitCode = 1;
} finally {
  await connection.end();
}
