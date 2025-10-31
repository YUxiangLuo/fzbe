import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";

const parseJsonField = (value: unknown) => {
  if (value === null || value === undefined) return value;
  if (typeof value === "string") {
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  }
  return value;
};

const normalizeQuestionRow = (row: any, includeAnswers = true) => {
  const parsedOptions = parseJsonField(row.options);
  const normalized: Record<string, any> = {
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
};

const singleAnswerSchema = z.object({
  question_id: z.number().int().positive(),
  submitted_answer: z.array(z.string().trim().min(1)).min(1),
});

const batchSubmitAnswersSchema = z.object({
  experiment_id: z.number().int().positive(),
  answers: z.array(singleAnswerSchema).min(1, "At least one answer must be provided"),
});

export default (pool: Pool, JWT_SECRET: string) => {
  const quizzesApi = new Hono();

  quizzesApi.use("*", jwt({ secret: JWT_SECRET }));

  // GET /api/quiz/model-questions - 获取模型相关题目
  quizzesApi.get("/api/quiz/model-questions", async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Forbidden" }, 403);
      }

      const sql = `
        SELECT qb.question_id,
               qb.knowledge_point,
               qb.question_type,
               qb.question_text,
               qb.options,
               qb.creator_id,
               creator.full_name AS creator_name
          FROM question_bank qb
     LEFT JOIN users creator ON creator.user_id = qb.creator_id
         WHERE qb.knowledge_point LIKE '%模型%'
      ORDER BY RAND()
         LIMIT 4
      `;

      const [rows]: any[] = await pool.query(sql);

      if (rows.length === 0) {
        return c.json({ error: "No model-related questions available" }, 404);
      }

      const normalized = rows.map((row: any) => normalizeQuestionRow(row, false));
      return c.json(normalized);

    } catch (error) {
      console.error("Error fetching random model questions:", error);
      throw error;
    }
  });

  // GET /api/quiz/plan-questions - 获取生产计划题目
  quizzesApi.get("/api/quiz/plan-questions", async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Forbidden" }, 403);
      }

      const sql = `
        SELECT qb.question_id,
               qb.knowledge_point,
               qb.question_type,
               qb.question_text,
               qb.options,
               qb.creator_id,
               creator.full_name AS creator_name
          FROM question_bank qb
     LEFT JOIN users creator ON creator.user_id = qb.creator_id
         WHERE qb.knowledge_point LIKE '%生产计划%'
      ORDER BY RAND()
         LIMIT 4
      `;

      const [rows]: any[] = await pool.query(sql);

      if (rows.length === 0) {
        return c.json({ error: "No plan-related questions available" }, 404);
      }

      const normalized = rows.map((row: any) => normalizeQuestionRow(row, false));
      return c.json(normalized);

    } catch (error) {
      console.error("Error fetching random plan questions:", error);
      throw error;
    }
  });

  // POST /api/quiz/answers - 提交答案
  quizzesApi.post(
    "/api/quiz/answers",
    zValidator("json", batchSubmitAnswersSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Student") {
          return c.json({ error: "Forbidden" }, 403);
        }
        const studentId = Number.parseInt(payload.sub, 10);

        const { experiment_id, answers: submittedAnswers } = c.req.valid("json");

        const questionIds = submittedAnswers.map(a => a.question_id);
        if (questionIds.length === 0) {
          return c.json({ error: "No answers provided" }, 400);
        }

        // 1. Fetch all questions' data in one go
        const placeholders = questionIds.map(() => '?').join(',');
        const [questionRows]: any[] = await pool.query(
          `SELECT question_id, correct_answers, knowledge_point FROM question_bank WHERE question_id IN (${placeholders})`,
          questionIds
        );

        // 2. Create a lookup map for easy access
        const correctAnswersMap = new Map<number, { answers: string[], knowledge: string }>();
        for (const row of questionRows) {
          correctAnswersMap.set(row.question_id, {
            answers: JSON.parse(row.correct_answers),
            knowledge: row.knowledge_point || ''
          });
        }

        const insertValues: any[] = [];
        const resultsForFrontend: any[] = [];

        // 3. Process each submitted answer
        for (const submitted of submittedAnswers) {
          const questionData = correctAnswersMap.get(submitted.question_id);
          if (!questionData) {
            continue; // Skip if the question doesn't exist in the DB
          }

          const isCorrect =
            JSON.stringify([...submitted.submitted_answer].sort()) ===
            JSON.stringify([...questionData.answers].sort());

          // Infer quiz_type from knowledge_point
          let quiz_type: 'quiz_about_model' | 'quiz_about_plan' = 'quiz_about_model';
          if (questionData.knowledge.includes('生产计划')) {
            quiz_type = 'quiz_about_plan';
          }

          insertValues.push([
            experiment_id,
            studentId,
            submitted.question_id,
            quiz_type,
            JSON.stringify(submitted.submitted_answer),
            isCorrect,
          ]);

          resultsForFrontend.push({
            question_id: submitted.question_id,
            is_correct: isCorrect,
            correct_answers: questionData.answers,
          });
        }

        // 4. Batch insert all processed answers
        if (insertValues.length > 0) {
          await pool.query(
            `INSERT INTO quiz_answers
               (experiment_id, student_id, question_id, quiz_type, submitted_answer, is_correct)
             VALUES ?`,
            [insertValues]
          );
        }

        return c.json(resultsForFrontend, 201);

      } catch (error) {
        console.error("Error submitting batch quiz answers:", error);
        throw error;
      }
    },
  );

  return quizzesApi;
};
