import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";

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
  const questionBankApi = new Hono();

  questionBankApi.use("*", jwt({ secret: JWT_SECRET }));

  // GET /api/v1/question-bank/questions - 获取所有题目
  questionBankApi.get("/api/v1/question-bank/questions", async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (
        payload.role !== "Admin" &&
        payload.role !== "Teacher" &&
        payload.role !== "Assistant"
      ) {
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

  // POST /api/v1/question-bank/questions - 创建新题目
  questionBankApi.post(
    "/api/v1/question-bank/questions",
    zValidator("json", createQuestionSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (
          payload.role !== "Admin" &&
          payload.role !== "Teacher" &&
          payload.role !== "Assistant"
        ) {
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

  // PUT /api/v1/question-bank/questions/:questionId - 更新题目
  questionBankApi.put(
    "/api/v1/question-bank/questions/:questionId",
    zValidator("param", questionIdParamSchema),
    zValidator("json", updateQuestionSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (
          payload.role !== "Admin" &&
          payload.role !== "Teacher" &&
          payload.role !== "Assistant"
        ) {
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

  // DELETE /api/v1/question-bank/questions/:questionId - 删除题目
  questionBankApi.delete(
    "/api/v1/question-bank/questions/:questionId",
    zValidator("param", questionIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (
          payload.role !== "Admin" &&
          payload.role !== "Teacher" &&
          payload.role !== "Assistant"
        ) {
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

  return questionBankApi;
};
