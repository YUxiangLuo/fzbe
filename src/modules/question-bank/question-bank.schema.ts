import { z } from "zod";

const optionsSchema = z.union([
  z.record(z.string(), z.string()),
  z.array(z.string().min(1)),
]);

type QuestionType = "Single Choice" | "Multiple Choice" | "True/False";
type QuestionOptions = z.infer<typeof optionsSchema> | undefined;

const TRUE_FALSE_ANSWERS = new Set(["正确", "错误"]);

function normalizeOptionValues(options: QuestionOptions): string[] {
  if (!options) return [];
  const rawValues = Array.isArray(options) ? options : Object.values(options);
  return rawValues.map((value) => value.trim()).filter(Boolean);
}

function normalizeOptionKeys(options: QuestionOptions): string[] {
  if (!options || Array.isArray(options)) return [];
  return Object.keys(options).map((key) => key.trim()).filter(Boolean);
}

export function hasValidQuestionAnswerMapping(input: {
  question_type: QuestionType;
  options?: QuestionOptions;
  correct_answers: string[];
}): boolean {
  const answers = input.correct_answers.map((answer) => answer.trim()).filter(Boolean);

  if (input.question_type === "True/False") {
    return answers.every((answer) => TRUE_FALSE_ANSWERS.has(answer));
  }

  const allowedAnswers = new Set([
    ...normalizeOptionValues(input.options),
    ...normalizeOptionKeys(input.options),
  ]);
  if (allowedAnswers.size === 0) return false;

  return answers.every((answer) => allowedAnswers.has(answer));
}

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
  })
  .refine((data) => hasValidQuestionAnswerMapping(data), {
    message: "正确答案必须来自题目选项",
    path: ["correct_answers"],
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
