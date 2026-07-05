import type { RowDataPacket, ResultSetHeader } from "mysql2";
import { pool } from "../../config";
import type { CreateQuestionInput, UpdateQuestionInput } from "./question-bank.schema";
import { hasValidQuestionAnswerMapping } from "./question-bank.schema";

interface QuestionRow extends RowDataPacket {
  question_id: number;
  knowledge_point: string | null;
  question_type: string;
  question_text: string;
  options: string | null;
  correct_answers: string;
  answer_explanation?: string | null;
  creator_id: number | null;
  creator_name?: string;
}

interface AssignmentRow extends RowDataPacket { }

type QuestionType = CreateQuestionInput["question_type"];
type QuestionOptions = CreateQuestionInput["options"];

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

function normalizeCorrectAnswersForStorage(
  questionType: QuestionType,
  options: QuestionOptions,
  correctAnswers: string[],
) {
  const normalizedAnswers = Array.from(
    new Set(correctAnswers.map((answer) => String(answer).trim()).filter(Boolean)),
  );

  if (questionType === "True/False") {
    return normalizedAnswers;
  }

  if (Array.isArray(options)) {
    const order = new Map(options.map((option, index) => [String(option).trim(), index]));
    return [...normalizedAnswers].sort((left, right) => {
      const leftIndex = order.get(left) ?? Number.MAX_SAFE_INTEGER;
      const rightIndex = order.get(right) ?? Number.MAX_SAFE_INTEGER;
      return leftIndex - rightIndex || left.localeCompare(right);
    });
  }

  if (options && typeof options === "object") {
    const order = new Map(Object.keys(options).map((key, index) => [String(key).trim(), index]));
    return [...normalizedAnswers].sort((left, right) => {
      const leftIndex = order.get(left) ?? Number.MAX_SAFE_INTEGER;
      const rightIndex = order.get(right) ?? Number.MAX_SAFE_INTEGER;
      return leftIndex - rightIndex || left.localeCompare(right);
    });
  }

  return normalizedAnswers;
}

function normalizeAnswerExplanationForStorage(value: string | null | undefined): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

export function normalizeQuestionRow(row: QuestionRow, includeAnswers = true) {
  const parsedOptions = parseJsonField(row.options);
  const normalized: Record<string, unknown> = {
    ...row,
    options:
      Array.isArray(parsedOptions) || (parsedOptions && typeof parsedOptions === "object")
        ? parsedOptions
        : null,
    answer_explanation: normalizeAnswerExplanationForStorage(row.answer_explanation),
  };

  if (includeAnswers) {
    const parsedAnswers = Array.isArray(row.correct_answers)
      ? row.correct_answers
      : parseJsonField(row.correct_answers);
    normalized.correct_answers = Array.isArray(parsedAnswers) ? parsedAnswers : [];
  } else {
    delete normalized.correct_answers;
    delete normalized.answer_explanation;
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
            qb.answer_explanation,
            qb.creator_id,
            creator.full_name AS creator_name
       FROM question_bank qb
  LEFT JOIN users creator ON creator.user_id = qb.creator_id
      ORDER BY qb.question_id DESC`,
  );
  return questions.map((row) => normalizeQuestionRow(row, true));
}

export async function isAssistantAssignedToAnyClass(assistantId: number): Promise<boolean> {
  const [[assignment]] = await pool.query<AssignmentRow[]>(
    "SELECT 1 FROM class_assistants WHERE assistant_id = ? LIMIT 1",
    [assistantId],
  );
  return Boolean(assignment);
}

export async function createQuestion(input: CreateQuestionInput, creatorId: number) {
  const {
    knowledge_point,
    question_type,
    question_text,
    options,
    correct_answers,
    answer_explanation,
  } = input;

  let optionsToStore = options;
  if (question_type === "True/False" && !optionsToStore) {
    optionsToStore = ["正确", "错误"];
  }
  const normalizedCorrectAnswers = normalizeCorrectAnswersForStorage(
    question_type,
    optionsToStore,
    correct_answers,
  );

  const [result] = await pool.query<ResultSetHeader>(
    `INSERT INTO question_bank
       (knowledge_point, question_type, question_text, options, correct_answers, answer_explanation, creator_id)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
    [
      knowledge_point || null,
      question_type,
      question_text,
      optionsToStore ? JSON.stringify(optionsToStore) : null,
      JSON.stringify(normalizedCorrectAnswers),
      normalizeAnswerExplanationForStorage(answer_explanation),
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
            qb.answer_explanation,
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

export async function updateQuestion(
  questionId: number,
  input: UpdateQuestionInput,
  requesterId: number,
  requesterRole: string,
) {
  const [ownerRows] = await pool.query<QuestionRow[]>(
    "SELECT creator_id FROM question_bank WHERE question_id = ? LIMIT 1",
    [questionId],
  );
  const existing = ownerRows[0];
  if (!existing) return { error: "not_found" as const };
  if (requesterRole !== "Admin" && requesterRole !== "Teacher" && existing.creator_id !== requesterId) {
    return { error: "forbidden" as const };
  }

  const [questionRows] = await pool.query<QuestionRow[]>(
    "SELECT question_type, options, correct_answers FROM question_bank WHERE question_id = ? LIMIT 1",
    [questionId],
  );
  const currentQuestion = questionRows[0];
  if (!currentQuestion) return { error: "not_found" as const };

  const nextQuestionType = (input.question_type ?? currentQuestion.question_type) as CreateQuestionInput["question_type"];
  const nextOptions = input.options ?? (parseJsonField(currentQuestion.options) as CreateQuestionInput["options"]);
  const nextCorrectAnswers = normalizeCorrectAnswersForStorage(
    nextQuestionType,
    nextOptions,
    input.correct_answers ?? (parseJsonField(currentQuestion.correct_answers) as string[]),
  );

  if (!hasValidQuestionAnswerMapping({
    question_type: nextQuestionType,
    options: nextOptions,
    correct_answers: nextCorrectAnswers,
  })) {
    return { error: "invalid_answers" as const };
  }

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
    values.push(JSON.stringify(nextCorrectAnswers));
  }
  if (input.answer_explanation !== undefined) {
    fields.push("answer_explanation = ?");
    values.push(normalizeAnswerExplanationForStorage(input.answer_explanation));
  }

  if (fields.length === 0) {
    return { error: "no_fields" as const };
  }

  values.push(questionId);
  const [updateResult] = await pool.query<ResultSetHeader>(
    `UPDATE question_bank SET ${fields.join(", ")} WHERE question_id = ?`,
    values,
  );
  if (updateResult.affectedRows === 0) {
    return { error: "not_found" as const };
  }

  const [rows] = await pool.query<QuestionRow[]>(
    `SELECT qb.question_id,
            qb.knowledge_point,
            qb.question_type,
            qb.question_text,
            qb.options,
            qb.correct_answers,
            qb.answer_explanation,
            qb.creator_id,
            creator.full_name AS creator_name
       FROM question_bank qb
  LEFT JOIN users creator ON creator.user_id = qb.creator_id
      WHERE qb.question_id = ?
      LIMIT 1`,
    [questionId],
  );

  if (!rows[0]) return { error: "not_found" as const };
  return { data: normalizeQuestionRow(rows[0], true) };
}

export async function deleteQuestion(
  questionId: number,
  requesterId: number,
  requesterRole: string,
) {
  const [ownerRows] = await pool.query<QuestionRow[]>(
    "SELECT creator_id FROM question_bank WHERE question_id = ? LIMIT 1",
    [questionId],
  );
  const existing = ownerRows[0];
  if (!existing) return { error: "not_found" as const };
  if (requesterRole !== "Admin" && requesterRole !== "Teacher" && existing.creator_id !== requesterId) {
    return { error: "forbidden" as const };
  }

  await pool.query<ResultSetHeader>(
    "DELETE FROM question_bank WHERE question_id = ?",
    [questionId],
  );

  return { success: true };
}
