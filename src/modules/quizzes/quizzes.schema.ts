import { z } from "zod";

const singleAnswerSchema = z.object({
  question_id: z.number().int().positive(),
  submitted_answer: z.array(z.string().trim().min(1)).min(1),
});

export const batchSubmitAnswersSchema = z.object({
  experiment_id: z.number().int().positive(),
  answers: z.array(singleAnswerSchema).min(1, "至少需要提供一个答案"),
});

export const quizKindParamSchema = z.object({
  kind: z.enum(["model", "plan"]),
});

export const quizResultsQuerySchema = z.object({
  experiment_id: z.coerce.number().int().positive(),
});

export type BatchSubmitAnswersInput = z.infer<typeof batchSubmitAnswersSchema>;
export type QuizKindParam = z.infer<typeof quizKindParamSchema>;
export type QuizResultsQuery = z.infer<typeof quizResultsQuerySchema>;
