import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireTeacherOrAssistant } from "../../middleware";
import {
  createQuestionSchema,
  updateQuestionSchema,
  questionIdParamSchema,
} from "./question-bank.schema";
import {
  getAllQuestions,
  createQuestion,
  updateQuestion,
  deleteQuestion,
} from "./question-bank.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import type { JwtPayload } from "../../types";

const questionBankController = new Hono();

questionBankController.get(
  "/api/v1/question-bank/questions",
  jwtAuth,
  requireTeacherOrAssistant,
  async (c) => {
    try {
      const questions = await getAllQuestions();
      return c.json(questions);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取题库失败");
    }
  },
);

questionBankController.post(
  "/api/v1/question-bank/questions",
  jwtAuth,
  requireTeacherOrAssistant,
  zValidator("json", createQuestionSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const input = c.req.valid("json");
      const question = await createQuestion(input, payload.sub);
      return c.json(question, 201);
    } catch (error) {
      return handleUnexpectedError(c, error, "创建题目失败");
    }
  },
);

questionBankController.put(
  "/api/v1/question-bank/questions/:questionId",
  jwtAuth,
  requireTeacherOrAssistant,
  zValidator("param", questionIdParamSchema),
  zValidator("json", updateQuestionSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { questionId } = c.req.valid("param");
      const input = c.req.valid("json");
      const result = await updateQuestion(parseInt(questionId, 10), input, payload.sub, payload.role);

      if ("error" in result) {
        if (result.error === "not_found") {
          return c.json({ error: "题目不存在" }, 404);
        }
        if (result.error === "forbidden") {
          return c.json({ error: "权限不足" }, 403);
        }
        if (result.error === "invalid_answers") {
          return c.json({ error: "正确答案必须来自题目选项" }, 400);
        }
        return c.json({ error: "没有需要更新的字段" }, 400);
      }

      return c.json(result.data);
    } catch (error) {
      return handleUnexpectedError(c, error, "更新题目失败");
    }
  },
);

questionBankController.delete(
  "/api/v1/question-bank/questions/:questionId",
  jwtAuth,
  requireTeacherOrAssistant,
  zValidator("param", questionIdParamSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { questionId } = c.req.valid("param");
      const result = await deleteQuestion(parseInt(questionId, 10), payload.sub, payload.role);

      if ("error" in result) {
        if (result.error === "forbidden") {
          return c.json({ error: "权限不足" }, 403);
        }
        return c.json({ error: "题目不存在" }, 404);
      }

      return c.json({ message: "题目删除成功" });
    } catch (error) {
      return handleUnexpectedError(c, error, "删除题目失败");
    }
  },
);

export default questionBankController;
