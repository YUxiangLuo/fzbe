import type { RowDataPacket, ResultSetHeader } from "mysql2";
import { pool } from "../../config";
import logger from "../../utils/logger";
import { normalizeQuestionRow } from "../question-bank/question-bank.service";

interface QuestionIdRow extends RowDataPacket {
  question_id: number;
}

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

interface QuestionAnswerRow extends RowDataPacket {
  question_id: number;
  quiz_type?: QuizType;
  knowledge_point?: string | null;
  question_type?: string | null;
  question_text?: string | null;
  options: unknown;
  correct_answers: unknown;
  submitted_answer?: unknown;
  is_correct?: number | boolean;
}

interface QuizAnswerResultRow extends RowDataPacket {
  answer_id: number;
  question_id: number | null;
  quiz_type: QuizType;
  knowledge_point?: string | null;
  question_type?: string | null;
  question_text?: string | null;
  options?: unknown;
  correct_answers?: unknown;
  submitted_answer: unknown;
  is_correct: number | boolean;
  question_snapshot?: unknown;
}

interface ExperimentAccessRow extends RowDataPacket {
  student_id: number;
}

type QuestionOptions = Record<string, string> | string[] | null;

export interface QuizAnswerDetail {
  question_id: number;
  quiz_type: QuizType;
  knowledge_point: string | null;
  question_type: string;
  question_text: string;
  options: QuestionOptions;
  submitted_answer: string[];
  correct_answers: string[];
  is_correct: boolean;
}

interface ExperimentOwnerRow extends RowDataPacket {
  student_id: number;
  status: string;
}

interface StatsRow extends RowDataPacket {
  correct_count: number;
  total_count: number;
}

interface AvailableQuestionCountRow extends RowDataPacket {
  available_count: number;
}

type QuizType = "quiz_about_model" | "quiz_about_plan";
type QuizKind = "model" | "plan";
interface QuestionSnapshot {
  question_id: number;
  knowledge_point: string | null;
  question_type: string;
  question_text: string;
  options: QuestionOptions;
  correct_answers: string[];
}

const QUIZ_QUESTION_TARGET_COUNT = 4;
const QUIZ_SUBMIT_MAX_ATTEMPTS = 3;
const QUIZ_SUBMIT_RETRY_DELAYS_MS = [50, 150] as const;
const DEFAULT_QUESTION_TYPE = "Single Choice";

function getQuizTypeFromKnowledgePoint(knowledgePoint: string): QuizType {
  return knowledgePoint.includes("生产计划") ? "quiz_about_plan" : "quiz_about_model";
}

function getQuizTypeFromKind(kind: QuizKind): QuizType {
  return kind === "plan" ? "quiz_about_plan" : "quiz_about_model";
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
  if (Array.isArray(parsed)) {
    return parsed.map((item) => String(item));
  }
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

function normalizeBooleanField(value: unknown): boolean {
  return value === true || value === 1 || value === "1";
}

function normalizeQuestionSnapshot(question: {
  question_id?: unknown;
  knowledge_point?: unknown;
  question_type?: unknown;
  question_text?: unknown;
  options?: unknown;
  correct_answers?: unknown;
}): QuestionSnapshot {
  const questionId = Number(question.question_id ?? 0);
  return {
    question_id: Number.isFinite(questionId) ? questionId : 0,
    knowledge_point: question.knowledge_point == null ? null : String(question.knowledge_point),
    question_type:
      typeof question.question_type === "string" && question.question_type.trim()
        ? question.question_type
        : DEFAULT_QUESTION_TYPE,
    question_text:
      typeof question.question_text === "string" && question.question_text.trim()
        ? question.question_text
        : `题目 #${Number.isFinite(questionId) ? questionId : 0}`,
    options: parseOptionsField(question.options),
    correct_answers: parseStringArrayField(question.correct_answers),
  };
}

function parseQuestionSnapshotField(value: unknown): QuestionSnapshot | null {
  const parsed = parseJsonField(value);
  if (!parsed || typeof parsed !== "object") return null;

  const snapshot = normalizeQuestionSnapshot(parsed as Record<string, unknown> & { question_id?: number | null });
  return snapshot.question_id > 0 ? snapshot : null;
}

function buildAnswerDetailFromSnapshot(
  snapshot: QuestionSnapshot,
  quizType: QuizType,
  submittedAnswer: string[],
  isCorrect: boolean,
): QuizAnswerDetail {
  return {
    question_id: snapshot.question_id,
    quiz_type: quizType,
    knowledge_point: snapshot.knowledge_point,
    question_type: snapshot.question_type,
    question_text: snapshot.question_text,
    options: snapshot.options,
    submitted_answer: submittedAnswer,
    correct_answers: snapshot.correct_answers,
    is_correct: isCorrect,
  };
}

function normalizeStoredAnswerRow(row: QuizAnswerResultRow): QuizAnswerDetail {
  const snapshot = parseQuestionSnapshotField(row.question_snapshot) ?? normalizeQuestionSnapshot(row);
  return buildAnswerDetailFromSnapshot(
    snapshot,
    row.quiz_type,
    parseStringArrayField(row.submitted_answer),
    normalizeBooleanField(row.is_correct),
  );
}

async function getRequiredQuestionCountForQuizType(
  connection: Pick<typeof pool, "query">,
  quizType: QuizType,
): Promise<number> {
  const whereClause = quizType === "quiz_about_plan"
    ? "knowledge_point LIKE '%生产计划%'"
    : "knowledge_point LIKE '%模型%' AND knowledge_point NOT LIKE '%生产计划%'";
  const [[row]] = await connection.query<AvailableQuestionCountRow[]>(
    `SELECT COUNT(*) AS available_count FROM question_bank WHERE ${whereClause}`,
  );
  const availableCount = Number(row?.available_count ?? 0);
  return Math.min(QUIZ_QUESTION_TARGET_COUNT, availableCount);
}

function isDeadlockError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const maybeError = error as {
    code?: string;
    errno?: number;
    sqlState?: string;
    message?: string;
  };

  return (
    maybeError.code === "ER_LOCK_DEADLOCK" ||
    maybeError.errno === 1213 ||
    maybeError.sqlState === "40001" ||
    maybeError.message?.includes("Deadlock found") === true
  );
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function shuffleArray<T>(arr: T[]): T[] {
  const shuffled = [...arr];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j]!, shuffled[i]!];
  }
  return shuffled;
}

export async function getQuestionsByKnowledge(knowledgePattern: string, count = 4) {
  const [idRows] = await pool.query<QuestionIdRow[]>(
    `SELECT question_id FROM question_bank WHERE knowledge_point LIKE ?`,
    [`%${knowledgePattern}%`],
  );

  if (idRows.length === 0) {
    return null;
  }

  const allIds = idRows.map((row) => row.question_id);
  const selectedIds = shuffleArray(allIds).slice(0, Math.min(count, allIds.length));

  const placeholders = selectedIds.map(() => "?").join(",");
  const [rows] = await pool.query<QuestionRow[]>(
    `SELECT qb.question_id,
            qb.knowledge_point,
            qb.question_type,
            qb.question_text,
            qb.options,
            qb.creator_id,
            creator.full_name AS creator_name
       FROM question_bank qb
  LEFT JOIN users creator ON creator.user_id = qb.creator_id
      WHERE qb.question_id IN (${placeholders})`,
    selectedIds,
  );

  return rows.map((row) => normalizeQuestionRow(row, false));
}

export interface SubmittedAnswer {
  question_id: number;
  submitted_answer: string[];
}

export async function submitAnswers(
  experimentId: number,
  studentId: number,
  submittedAnswers: SubmittedAnswer[],
): Promise<{ error: string; status: 400 | 403 | 404 } | { results: QuizAnswerDetail[] }> {
  const normalizedAnswers = [...submittedAnswers].sort(
    (left, right) => left.question_id - right.question_id,
  );
  const questionIds = normalizedAnswers.map((a) => a.question_id);
  if (questionIds.length === 0) {
    return { error: "未提供答案", status: 400 };
  }
  if (new Set(questionIds).size !== questionIds.length) {
    return { error: "测验提交包含重复题目", status: 400 };
  }

  for (let attempt = 0; attempt < QUIZ_SUBMIT_MAX_ATTEMPTS; attempt += 1) {
    const connection = await pool.getConnection();
    try {
      await connection.query("SET TRANSACTION ISOLATION LEVEL READ COMMITTED");
      await connection.beginTransaction();

      const [[experimentOwnership]] = await connection.query<ExperimentOwnerRow[]>(
        `SELECT student_id, status FROM experiment_status WHERE experiment_id = ? LIMIT 1 FOR UPDATE`,
        [experimentId],
      );

      if (!experimentOwnership) {
        await connection.rollback();
        return { error: "实验不存在", status: 404 };
      }

      if (experimentOwnership.student_id !== studentId) {
        await connection.rollback();
        return { error: "权限不足：该实验不属于当前用户", status: 403 };
      }

      if (experimentOwnership.status === "Completed") {
        await connection.rollback();
        return { error: "已完成的实验不允许继续提交测试答案", status: 400 };
      }

      const placeholders = questionIds.map(() => "?").join(",");
      const [questionRows] = await connection.query<QuestionAnswerRow[]>(
        `SELECT question_id, correct_answers, knowledge_point,
                question_type,
                question_text,
                options
           FROM question_bank
          WHERE question_id IN (${placeholders})`,
        questionIds,
      );

      if (questionRows.length !== questionIds.length) {
        await connection.rollback();
        return { error: "测验题目不存在或不可用于提交", status: 400 };
      }

      const correctAnswersMap = new Map<number, { row: QuestionAnswerRow; answers: string[]; quizType: QuizType }>();
      for (const row of questionRows) {
        const knowledge = row.knowledge_point || "";
        correctAnswersMap.set(row.question_id, {
          row,
          answers: parseStringArrayField(row.correct_answers),
          quizType: getQuizTypeFromKnowledgePoint(knowledge),
        });
      }

      const submittedQuestionCountByType = new Map<QuizType, number>();
      for (const submitted of normalizedAnswers) {
        const questionData = correctAnswersMap.get(submitted.question_id);
        if (!questionData) continue;
        submittedQuestionCountByType.set(
          questionData.quizType,
          (submittedQuestionCountByType.get(questionData.quizType) ?? 0) + 1,
        );
      }

      for (const [quizType, submittedCount] of submittedQuestionCountByType) {
        const requiredCount = await getRequiredQuestionCountForQuizType(connection, quizType);
        if (requiredCount === 0) {
          await connection.rollback();
          return { error: "测验题目不存在或不可用于提交", status: 400 };
        }
        if (submittedCount !== requiredCount) {
          await connection.rollback();
          const quizName = quizType === "quiz_about_plan" ? "生产计划测验" : "预测模型测验";
          return { error: `${quizName}必须一次提交 ${requiredCount} 道题`, status: 400 };
        }
      }

      const insertValues: unknown[][] = [];
      const submittedQuizTypes = new Set<QuizType>();
      const resultsForFrontend: QuizAnswerDetail[] = [];

      for (const submitted of normalizedAnswers) {
        const questionData = correctAnswersMap.get(submitted.question_id)!;

        const isCorrect =
          JSON.stringify([...submitted.submitted_answer].sort()) ===
          JSON.stringify([...questionData.answers].sort());

        const quizType = questionData.quizType;
        const questionSnapshot = normalizeQuestionSnapshot(questionData.row);
        submittedQuizTypes.add(quizType);

        insertValues.push([
          experimentId,
          studentId,
          submitted.question_id,
          quizType,
          JSON.stringify(submitted.submitted_answer),
          JSON.stringify(questionSnapshot),
          isCorrect,
        ]);

        resultsForFrontend.push(
          buildAnswerDetailFromSnapshot(questionSnapshot, quizType, submitted.submitted_answer, isCorrect),
        );
      }

      if (submittedQuizTypes.size > 0) {
        const types = Array.from(submittedQuizTypes).sort();
        const typePlaceholders = types.map(() => "?").join(",");
        await connection.query(
          `DELETE FROM quiz_answers
            WHERE experiment_id = ? AND student_id = ?
              AND quiz_type IN (${typePlaceholders})`,
          [experimentId, studentId, ...types],
        );
      }

      if (insertValues.length > 0) {
        await connection.query(
          `INSERT INTO quiz_answers
             (experiment_id, student_id, question_id, quiz_type, submitted_answer, question_snapshot, is_correct)
           VALUES ?`,
          [insertValues],
        );
      }

      if (submittedQuizTypes.size > 0) {
        const [[stats]] = await connection.query<StatsRow[]>(
          `SELECT
             SUM(is_correct) AS correct_count,
             COUNT(*) AS total_count
           FROM quiz_answers
          WHERE experiment_id = ? AND student_id = ?
            AND quiz_type IN ('quiz_about_model', 'quiz_about_plan')`,
          [experimentId, studentId],
        );

        const totalCount = Number(stats?.total_count ?? 0);
        const correctCount = Number(stats?.correct_count ?? 0);
        const knowledgeScore = totalCount > 0 ? Number(((correctCount / totalCount) * 100).toFixed(2)) : 0;

        await connection.query(
          `UPDATE experiment_grade
              SET knowledge_test = ?
            WHERE experiment_id = ? AND student_id = ?`,
          [knowledgeScore, experimentId, studentId],
        );

        const completedModelQuiz = submittedQuizTypes.has("quiz_about_model") ? 1 : 0;
        const completedPlanQuiz = submittedQuizTypes.has("quiz_about_plan") ? 1 : 0;
        await connection.query(
          `UPDATE experiment_status
              SET quiz_about_model_completed = CASE
                    WHEN ? = 1 THEN 1
                    ELSE quiz_about_model_completed
                  END,
                  quiz_about_plan_completed = CASE
                    WHEN ? = 1 THEN 1
                    ELSE quiz_about_plan_completed
                  END,
                  status = CASE
                    WHEN status = 'Not Started' THEN 'In Progress'
                    ELSE status
                  END,
                  last_activity_at = NOW(),
                  start_time = COALESCE(start_time, NOW())
            WHERE experiment_id = ? AND student_id = ?`,
          [completedModelQuiz, completedPlanQuiz, experimentId, studentId],
        );
      }

      await connection.commit();
      return { results: resultsForFrontend };
    } catch (error) {
      await connection.rollback().catch(() => undefined);
      if (isDeadlockError(error) && attempt < QUIZ_SUBMIT_MAX_ATTEMPTS - 1) {
        logger.warn(
          { attempt: attempt + 1, experimentId, studentId },
          "Quiz submission deadlock detected, retrying transaction",
        );
        await sleep(QUIZ_SUBMIT_RETRY_DELAYS_MS[attempt] ?? 200);
        continue;
      }
      throw error;
    } finally {
      connection.release();
    }
  }

  throw new Error("Quiz submission retries exhausted unexpectedly");
}

export async function getSubmittedQuizResults(
  experimentId: number,
  studentId: number,
  kind: QuizKind,
): Promise<{ error: string; status: 403 | 404 } | { results: QuizAnswerDetail[] }> {
  const [[experiment]] = await pool.query<ExperimentAccessRow[]>(
    `SELECT student_id FROM experiment_status WHERE experiment_id = ? LIMIT 1`,
    [experimentId],
  );

  if (!experiment) {
    return { error: "实验不存在", status: 404 };
  }

  if (experiment.student_id !== studentId) {
    return { error: "权限不足：该实验不属于当前用户", status: 403 };
  }

  const quizType = getQuizTypeFromKind(kind);
  const [rows] = await pool.query<QuizAnswerResultRow[]>(
    `SELECT qa.answer_id,
            qa.question_id,
            qa.quiz_type,
            qa.submitted_answer,
            qa.is_correct,
            qa.question_snapshot,
            qb.knowledge_point,
            qb.question_type,
            qb.question_text,
            qb.options,
            qb.correct_answers
       FROM quiz_answers qa
  LEFT JOIN question_bank qb ON qb.question_id = qa.question_id
      WHERE qa.experiment_id = ?
        AND qa.student_id = ?
        AND qa.quiz_type = ?
      ORDER BY qa.answer_id ASC`,
    [experimentId, studentId, quizType],
  );

  return { results: rows.map(normalizeStoredAnswerRow) };
}
