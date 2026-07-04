import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireStudent } from "../../middleware";
import { batchSubmitAnswersSchema, quizKindParamSchema, quizResultsQuerySchema } from "./quizzes.schema";
import { getQuestionsByKnowledge, getSubmittedQuizResults, submitAnswers } from "./quizzes.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import type { JwtPayload } from "../../types";

const quizzesController = new Hono();

quizzesController.get(
  "/api/v1/quizzes/model/questions",
  jwtAuth,
  requireStudent,
  async (c) => {
    try {
      const questions = await getQuestionsByKnowledge("模型", 4);
      if (!questions) {
        return c.json({ error: "没有可用的模型相关题目" }, 404);
      }
      return c.json(questions);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取模型测试题目失败");
    }
  },
);

quizzesController.get(
  "/api/v1/quizzes/plan/questions",
  jwtAuth,
  requireStudent,
  async (c) => {
    try {
      const questions = await getQuestionsByKnowledge("生产计划", 4);
      if (!questions) {
        return c.json({ error: "没有可用的生产计划相关题目" }, 404);
      }
      return c.json(questions);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取生产计划测试题目失败");
    }
  },
);

quizzesController.get(
  "/api/v1/quizzes/:kind/results",
  jwtAuth,
  requireStudent,
  zValidator("param", quizKindParamSchema),
  zValidator("query", quizResultsQuerySchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { kind } = c.req.valid("param");
      const { experiment_id } = c.req.valid("query");

      const result = await getSubmittedQuizResults(experiment_id, payload.sub, kind);
      if ("error" in result) {
        return c.json({ error: result.error }, result.status);
      }

      return c.json(result.results);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取测试答题结果失败");
    }
  },
);

quizzesController.post(
  "/api/v1/quizzes/answers",
  jwtAuth,
  requireStudent,
  zValidator("json", batchSubmitAnswersSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { experiment_id, answers } = c.req.valid("json");

      const result = await submitAnswers(experiment_id, payload.sub, answers);

      if ("error" in result) {
        return c.json({ error: result.error }, result.status);
      }

      return c.json(result.results, 201);
    } catch (error) {
      return handleUnexpectedError(c, error, "提交测试答案失败");
    }
  },
);

export default quizzesController;
