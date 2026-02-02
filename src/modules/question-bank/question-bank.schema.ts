import { z } from "zod";

const optionsSchema = z.union([
  z.record(z.string(), z.string()),
  z.array(z.string().min(1)),
]);

export const createQuestionSchema = z
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
    message: "单选题必须有且仅有一个正确答案",
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
    message: "选择题必须包含选项",
    path: ["options"],
  });

export const updateQuestionSchema = z
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
    { message: "至少需要提供一个字段" },
  )
  .refine(
    (data) => {
      if (data.question_type === "Single Choice" && data.correct_answers !== undefined) {
        return data.correct_answers.length === 1;
      }
      return true;
    },
    { message: "单选题必须有且仅有一个正确答案", path: ["correct_answers"] },
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
    { message: "True/False questions must use '正确' 或 '错误' 作为答案", path: ["correct_answers"] },
  );

export const questionIdParamSchema = z.object({
  questionId: z.string().regex(/^\d+$/, { message: "questionId 必须是数字" }),
});

export type CreateQuestionInput = z.infer<typeof createQuestionSchema>;
export type UpdateQuestionInput = z.infer<typeof updateQuestionSchema>;
export type QuestionIdParam = z.infer<typeof questionIdParamSchema>;
