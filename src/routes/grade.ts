import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";
import puppeteer from "puppeteer";
import path from "path";
import { mkdir } from "node:fs/promises";


const optionsSchema = z.union([
  z.record(z.string(), z.string()),
  z.array(z.string().min(1)),
]);

const createQuestionSchema = z
  .object({
    knowledge_point: z.string().trim().min(1).optional(),
    question_type: z.enum(["Single Choice", "Multiple Choice", "True/False"]),
    question_text: z.string().trim().min(1),
    options: optionsSchema.optional(),
    correct_answers: z.array(z.string().trim().min(1)).min(1),
  })
  .refine((data) => {
    if (data.question_type === "Single Choice") {
      return data.correct_answers.length === 1;
    }
    return true;
  }, {
    message: "Single Choice questions must have exactly one correct answer",
    path: ["correct_answers"],
  })
  .refine((data) => {
    if (data.question_type === "True/False") {
      return data.correct_answers.every((ans) => ans === "正确" || ans === "错误");
    }
    return true;
  }, {
    message: "True/False questions must use '正确' 或 '错误' 作为答案",
    path: ["correct_answers"],
  })
  .refine((data) => {
    if (data.question_type !== "True/False") {
      return data.options !== undefined;
    }
    return true;
  }, {
    message: "Choice questions must include options",
    path: ["options"],
  });

const updateQuestionSchema = z
  .object({
    knowledge_point: z.string().trim().min(1).optional(),
    question_type: z.enum(["Single Choice", "Multiple Choice", "True/False"]).optional(),
    question_text: z.string().trim().min(1).optional(),
    options: optionsSchema.optional(),
    correct_answers: z.array(z.string().trim().min(1)).min(1).optional(),
  })
  .refine(
    (data) =>
      data.knowledge_point !== undefined ||
      data.question_type !== undefined ||
      data.question_text !== undefined ||
      data.options !== undefined ||
      data.correct_answers !== undefined,
    {
      message: "At least one field must be provided",
    },
  )
  .refine(
    (data) => {
      if (data.question_type === "Single Choice" && data.correct_answers !== undefined) {
        return data.correct_answers.length === 1;
      }
      return true;
    },
    {
      message: "Single Choice questions must have exactly one correct answer",
      path: ["correct_answers"],
    },
  )
  .refine(
    (data) => {
      if (data.question_type === "True/False") {
        const answers = data.correct_answers;
        if (answers && !answers.every((ans) => ans === "正确" || ans === "错误")) {
          return false;
        }
      }
      return true;
    },
    {
      message: "True/False questions must use '正确' 或 '错误' 作为答案",
      path: ["correct_answers"],
    },
  );

const questionIdParamSchema = z.object({
  questionId: z.string().regex(/^\d+$/, { message: "questionId must be a number" }),
});

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

export default (pool: Pool, JWT_SECRET: string) => {
  const gradeApi = new Hono();

  gradeApi.use("*", jwt({ secret: JWT_SECRET }));

  gradeApi.get("/question-bank", async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin" && payload.role !== "Teacher") {
        return c.json({ error: "Forbidden" }, 403);
      }

      const [questions] = await pool.query(
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

      const normalized = (questions as any[]).map((row) => normalizeQuestionRow(row, true));
      return c.json(normalized);
    } catch (error) {
      console.error("Error fetching question bank:", error);
      throw error;
    }
  });

  gradeApi.post(
    "/question-bank",
    zValidator("json", createQuestionSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin" && payload.role !== "Teacher") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { knowledge_point, question_type, question_text, options, correct_answers } =
          c.req.valid("json");

        let optionsToStore = options;
        if (question_type === "True/False" && !optionsToStore) {
          optionsToStore = ["正确", "错误"];
        }

        const [result]: any = await pool.query(
          `INSERT INTO question_bank
             (knowledge_point, question_type, question_text, options, correct_answers, creator_id)
           VALUES (?, ?, ?, ?, ?, ?)`,
          [
            knowledge_point || null,
            question_type,
            question_text,
            optionsToStore ? JSON.stringify(optionsToStore) : null,
            JSON.stringify(correct_answers),
            payload.sub,
          ],
        );

        const insertedId = result.insertId;

        const [rows] = await pool.query(
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
          [insertedId],
        );

        const createdQuestion = (rows as any[])[0];
        return c.json(normalizeQuestionRow(createdQuestion, true), 201);
      } catch (error) {
        console.error("Error creating question bank item:", error);
        throw error;
      }
    },
  );

  gradeApi.put(
    "/question-bank/:questionId",
    zValidator("param", questionIdParamSchema),
    zValidator("json", updateQuestionSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin" && payload.role !== "Teacher") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { questionId } = c.req.valid("param");
        const questionIdNumber = Number.parseInt(questionId, 10);
        const updatePayload = c.req.valid("json");

        const fields: string[] = [];
        const values: any[] = [];

        if (updatePayload.knowledge_point !== undefined) {
          fields.push("knowledge_point = ?");
          values.push(updatePayload.knowledge_point);
        }
        if (updatePayload.question_type !== undefined) {
          fields.push("question_type = ?");
          values.push(updatePayload.question_type);
        }
        if (updatePayload.question_text !== undefined) {
          fields.push("question_text = ?");
          values.push(updatePayload.question_text);
        }
        if (updatePayload.options !== undefined) {
          fields.push("options = ?");
          values.push(JSON.stringify(updatePayload.options));
        } else if (updatePayload.question_type === "True/False") {
          fields.push("options = ?");
          values.push(JSON.stringify(["正确", "错误"]));
        }
        if (updatePayload.correct_answers !== undefined) {
          fields.push("correct_answers = ?");
          values.push(JSON.stringify(updatePayload.correct_answers));
        }

        if (fields.length === 0) {
          return c.json({ error: "No fields to update" }, 400);
        }

        values.push(questionIdNumber);

        const sql = `UPDATE question_bank SET ${fields.join(", ")} WHERE question_id = ?`;
        const [result]: any = await pool.query(sql, values);

        if (result.affectedRows === 0) {
          return c.json({ error: "Question not found" }, 404);
        }

        const [rows] = await pool.query(
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
          [questionIdNumber],
        );

        if ((rows as any[]).length === 0) {
          return c.json({ error: "Question not found" }, 404);
        }

        return c.json(normalizeQuestionRow((rows as any[])[0], true));
      } catch (error) {
        console.error("Error updating question bank item:", error);
        throw error;
      }
    },
  );

  gradeApi.delete(
    "/question-bank/:questionId",
    zValidator("param", questionIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin" && payload.role !== "Teacher") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { questionId } = c.req.valid("param");
        const questionIdNumber = Number.parseInt(questionId, 10);

        const [result]: any = await pool.query(
          "DELETE FROM question_bank WHERE question_id = ?",
          [questionIdNumber],
        );

        if (result.affectedRows === 0) {
          return c.json({ error: "Question not found" }, 404);
        }

        return c.json({ message: "Question deleted successfully" });
      } catch (error) {
        console.error("Error deleting question bank item:", error);
        throw error;
      }
    },
  );

  // 学生: 获取4条关于“模型”的随机题目
  gradeApi.get("/quiz/model-questions", async (c) => {
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

      // Use the existing helper, ensuring answers are not included
      const normalized = rows.map((row: any) => normalizeQuestionRow(row, false));
      return c.json(normalized);

    } catch (error) {
      console.error("Error fetching random model questions:", error);
      throw error;
    }
  });

  // 学生: 获取4条关于“生产计划”的随机题目
  gradeApi.get("/quiz/plan-questions", async (c) => {
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

  const singleAnswerSchema = z.object({
    question_id: z.number().int().positive(),
    submitted_answer: z.array(z.string().trim().min(1)).min(1),
  });

  const batchSubmitAnswersSchema = z.object({
    experiment_id: z.number().int().positive(),
    answers: z.array(singleAnswerSchema).min(1, "At least one answer must be provided"),
  });

  gradeApi.post(
    "/quiz/answers",
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

  const submitReportSchema = z.object({
    experiment_id: z.number().int().positive(),
    report_content: z.string().trim().min(1).refine((val) => {
      try {
        // Attempt to decode to see if it's valid Base64
        Buffer.from(val, 'base64');
        return true;
      } catch (e) {
        return false;
      }
    }, { message: "report_content must be a valid Base64 encoded string" }),
  });

  gradeApi.post(
    "/reports",
    zValidator("json", submitReportSchema),
    async (c) => {
      let browser;
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Student") {
          return c.json({ error: "Forbidden" }, 403);
        }
        const studentId = Number.parseInt(payload.sub, 10);

        const { experiment_id, report_content: base64Content } = c.req.valid("json");

        const report_content = Buffer.from(base64Content, "base64").toString("utf-8");

        // --- PDF Generation Logic ---
        const reportsDir = path.resolve(import.meta.dir, "../../uploads", "reports");
        await mkdir(reportsDir, { recursive: true });

        const uniqueFilename = `report-exp${experiment_id}-student${studentId}-${Date.now()}.pdf`;
        const pdfPath = path.join(reportsDir, uniqueFilename);

        browser = await puppeteer.launch({
          args: ["--no-sandbox", "--disable-setuid-sandbox"],
        });
        const page = await browser.newPage();
        await page.setContent(report_content, { waitUntil: "networkidle0" });
        await page.pdf({ path: pdfPath, format: "A4" });
        // --- End of PDF Generation ---

        // --- Database Logic ---
        const sql = `
          INSERT INTO experiment_reports (experiment_id, student_id, report_content, pdf_file_path, submitted_at)
          VALUES (?, ?, ?, ?, NOW())
          ON DUPLICATE KEY UPDATE
            report_content = VALUES(report_content),
            pdf_file_path = VALUES(pdf_file_path),
            submitted_at = NOW()
        `;

        const [result]: any = await pool.query(sql, [
          experiment_id,
          studentId,
          report_content,
          pdfPath,
        ]);

        const reportId = result.insertId;

        if (reportId === 0 && result.affectedRows > 0) {
          const [rows]: any = await pool.query(
            "SELECT report_id FROM experiment_reports WHERE experiment_id = ? AND student_id = ?",
            [experiment_id, studentId],
          );
          if (rows.length > 0) {
            return c.json({
              message: "Report updated successfully",
              report_id: rows[0].report_id,
              pdf_path: pdfPath,
            });
          }
        }

        return c.json(
          {
            message: "Report submitted successfully",
            report_id: reportId,
            pdf_path: pdfPath,
          },
          201,
        );
      } catch (error) {
        console.error("Error submitting experiment report:", error);
        return c.json(
          { error: "An internal server error occurred while processing the report." },
          500,
        );
      } finally {
        if (browser) {
          await browser.close();
        }
      }
    },
  );

  return gradeApi;
};
