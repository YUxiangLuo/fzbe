import type { RowDataPacket, ResultSetHeader } from "mysql2";
import { pool } from "../../config";
import type { CreateQuestionInput, UpdateQuestionInput } from "./question-bank.schema";

interface QuestionRow extends RowDataPacket {
  question_id: number;
  knowledge_point: string | null;
  question_type: string;
  question_text: string;
  options: string | null;
  correct_answers: string;
  creator_id: number | null;
  creator_name?: string;
}

function parseJsonField(value: unknown) {
  if (value === null || value === undefined) return value;
  if (typeof value === "string") {
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  }
  return value;
}

export function normalizeQuestionRow(row: QuestionRow, includeAnswers = true) {
  const parsedOptions = parseJsonField(row.options);
  const normalized: Record<string, unknown> = {
    ...row,
    options:
      Array.isArray(parsedOptions) || (parsedOptions && typeof parsedOptions === "object")
        ? parsedOptions
        : null,
  };

  if (includeAnswers) {
    const parsedAnswers = Array.isArray(row.correct_answers)
      ? row.correct_answers
      : parseJsonField(row.correct_answers);
    normalized.correct_answers = Array.isArray(parsedAnswers) ? parsedAnswers : [];
  } else {
    delete normalized.correct_answers;
  }

  return normalized;
}

export async function getAllQuestions() {
  const [questions] = await pool.query<QuestionRow[]>(
    `SELECT qb.question_id,
            qb.knowledge_point,
            qb.question_type,
            qb.question_text,
            qb.options,
            qb.correct_answers,
            qb.creator_id,
            creator.full_name AS creator_name
       FROM question_bank qb
  LEFT JOIN users creator ON creator.user_id = qb.creator_id
      ORDER BY qb.question_id DESC`,
  );
  return questions.map((row) => normalizeQuestionRow(row, true));
}

export async function createQuestion(input: CreateQuestionInput, creatorId: number) {
  const { knowledge_point, question_type, question_text, options, correct_answers } = input;

  let optionsToStore = options;
  if (question_type === "True/False" && !optionsToStore) {
    optionsToStore = ["正确", "错误"];
  }

  const [result] = await pool.query<ResultSetHeader>(
    `INSERT INTO question_bank
       (knowledge_point, question_type, question_text, options, correct_answers, creator_id)
     VALUES (?, ?, ?, ?, ?, ?)`,
    [
      knowledge_point || null,
      question_type,
      question_text,
      optionsToStore ? JSON.stringify(optionsToStore) : null,
      JSON.stringify(correct_answers),
      creatorId,
    ],
  );

  const [rows] = await pool.query<QuestionRow[]>(
    `SELECT qb.question_id,
            qb.knowledge_point,
            qb.question_type,
            qb.question_text,
            qb.options,
            qb.correct_answers,
            qb.creator_id,
            creator.full_name AS creator_name
       FROM question_bank qb
  LEFT JOIN users creator ON creator.user_id = qb.creator_id
      WHERE qb.question_id = ?
      LIMIT 1`,
    [result.insertId],
  );

  return normalizeQuestionRow(rows[0]!, true);
}

export async function updateQuestion(questionId: number, input: UpdateQuestionInput) {
  const fields: string[] = [];
  const values: unknown[] = [];

  if (input.knowledge_point !== undefined) {
    fields.push("knowledge_point = ?");
    values.push(input.knowledge_point);
  }
  if (input.question_type !== undefined) {
    fields.push("question_type = ?");
    values.push(input.question_type);
  }
  if (input.question_text !== undefined) {
    fields.push("question_text = ?");
    values.push(input.question_text);
  }
  if (input.options !== undefined) {
    fields.push("options = ?");
    values.push(JSON.stringify(input.options));
  } else if (input.question_type === "True/False") {
    fields.push("options = ?");
    values.push(JSON.stringify(["正确", "错误"]));
  }
  if (input.correct_answers !== undefined) {
    fields.push("correct_answers = ?");
    values.push(JSON.stringify(input.correct_answers));
  }

  if (fields.length === 0) {
    return { error: "no_fields" as const };
  }

  values.push(questionId);
  const [result] = await pool.query<ResultSetHeader>(
    `UPDATE question_bank SET ${fields.join(", ")} WHERE question_id = ?`,
    values,
  );

  if (result.affectedRows === 0) {
    return { error: "not_found" as const };
  }

  const [rows] = await pool.query<QuestionRow[]>(
    `SELECT qb.question_id,
            qb.knowledge_point,
            qb.question_type,
            qb.question_text,
            qb.options,
            qb.correct_answers,
            qb.creator_id,
            creator.full_name AS creator_name
       FROM question_bank qb
  LEFT JOIN users creator ON creator.user_id = qb.creator_id
      WHERE qb.question_id = ?
      LIMIT 1`,
    [questionId],
  );

  return { data: normalizeQuestionRow(rows[0]!, true) };
}

export async function deleteQuestion(questionId: number) {
  const [result] = await pool.query<ResultSetHeader>(
    "DELETE FROM question_bank WHERE question_id = ?",
    [questionId],
  );

  if (result.affectedRows === 0) {
    return { error: "not_found" as const };
  }

  return { success: true };
}
