import type { RowDataPacket, ResultSetHeader } from "mysql2";
import { pool } from "../../config";
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
  correct_answers: string;
  knowledge_point: string | null;
}

interface ExperimentOwnerRow extends RowDataPacket {
  student_id: number;
  status: string;
}

interface StatsRow extends RowDataPacket {
  correct_count: number;
  total_count: number;
}

type QuizType = "quiz_about_model" | "quiz_about_plan";

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

export interface AnswerResult {
  question_id: number;
  is_correct: boolean;
  correct_answers: string[];
}

export async function submitAnswers(
  experimentId: number,
  studentId: number,
  submittedAnswers: SubmittedAnswer[],
): Promise<{ error: string; status: 400 | 403 | 404 } | { results: AnswerResult[] }> {
  const questionIds = submittedAnswers.map((a) => a.question_id);
  if (questionIds.length === 0) {
    return { error: "未提供答案", status: 400 };
  }

  const connection = await pool.getConnection();
  try {
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
      `SELECT question_id, correct_answers, knowledge_point
         FROM question_bank
        WHERE question_id IN (${placeholders})`,
      questionIds,
    );

    const correctAnswersMap = new Map<number, { answers: string[]; knowledge: string }>();
    for (const row of questionRows) {
      correctAnswersMap.set(row.question_id, {
        answers: Array.isArray(row.correct_answers)
          ? row.correct_answers
          : JSON.parse(row.correct_answers),
        knowledge: row.knowledge_point || "",
      });
    }

    const insertValues: unknown[][] = [];
    const submittedQuizTypes = new Set<QuizType>();
    const resultsForFrontend: AnswerResult[] = [];

    for (const submitted of submittedAnswers) {
      const questionData = correctAnswersMap.get(submitted.question_id);
      if (!questionData) continue;

      const isCorrect =
        JSON.stringify([...submitted.submitted_answer].sort()) ===
        JSON.stringify([...questionData.answers].sort());

      const quizType: QuizType = questionData.knowledge.includes("生产计划")
        ? "quiz_about_plan"
        : "quiz_about_model";
      submittedQuizTypes.add(quizType);

      insertValues.push([
        experimentId,
        studentId,
        submitted.question_id,
        quizType,
        JSON.stringify(submitted.submitted_answer),
        isCorrect,
      ]);

      resultsForFrontend.push({
        question_id: submitted.question_id,
        is_correct: isCorrect,
        correct_answers: questionData.answers,
      });
    }

    if (submittedQuizTypes.size > 0) {
      const types = Array.from(submittedQuizTypes);
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
           (experiment_id, student_id, question_id, quiz_type, submitted_answer, is_correct)
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
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}
