import { describe, test, expect } from "bun:test";
import {
  createQuestionSchema,
  updateQuestionSchema,
  hasValidQuestionAnswerMapping,
} from "../../../src/modules/question-bank/question-bank.schema";

describe("Question Bank Schemas", () => {
  describe("createQuestionSchema", () => {
    test("should accept valid single choice question", () => {
      const result = createQuestionSchema.safeParse({
        question_text: "测试单选题",
        question_type: "Single Choice",
        knowledge_point: "MPS基本原理",
        options: ["选项甲", "选项乙"],
        correct_answers: ["选项甲"],
      });

      expect(result.success).toBe(true);
    });

    test("should reject single choice answers outside options", () => {
      const result = createQuestionSchema.safeParse({
        question_text: "非法单选题",
        question_type: "Single Choice",
        knowledge_point: "MPS基本原理",
        options: ["选项甲", "选项乙"],
        correct_answers: ["不存在的答案"],
      });

      expect(result.success).toBe(false);
    });

    test("should reject multiple choice answers outside options", () => {
      const result = createQuestionSchema.safeParse({
        question_text: "非法多选题",
        question_type: "Multiple Choice",
        knowledge_point: "产能规划",
        options: ["选项甲", "选项乙", "选项丙"],
        correct_answers: ["选项甲", "错误答案"],
      });

      expect(result.success).toBe(false);
    });

    test("should accept record option keys as correct answers", () => {
      const result = createQuestionSchema.safeParse({
        question_text: "键值格式单选题",
        question_type: "Single Choice",
        knowledge_point: "预测模型-ARIMA模型",
        options: { A: "选项甲", B: "选项乙" },
        correct_answers: ["A"],
      });

      expect(result.success).toBe(true);
    });
  });

  describe("updateQuestionSchema", () => {
    test("should allow updating options only", () => {
      const result = updateQuestionSchema.safeParse({
        options: ["选项甲", "选项乙"],
      });

      expect(result.success).toBe(true);
    });

    test("should allow editing selection answers using option keys", () => {
      const result = updateQuestionSchema.safeParse({
        question_type: "Multiple Choice",
        options: { A: "选项甲", B: "选项乙", C: "选项丙" },
        correct_answers: ["A", "C"],
      });

      expect(result.success).toBe(true);
    });
  });

  describe("hasValidQuestionAnswerMapping", () => {
    test("should accept true false canonical answers", () => {
      expect(hasValidQuestionAnswerMapping({
        question_type: "True/False",
        correct_answers: ["正确"],
      })).toBe(true);
    });

    test("should reject when selection question has no options", () => {
      expect(hasValidQuestionAnswerMapping({
        question_type: "Single Choice",
        correct_answers: ["选项甲"],
      })).toBe(false);
    });

    test("should accept record option keys for selection questions", () => {
      expect(hasValidQuestionAnswerMapping({
        question_type: "Single Choice",
        options: { A: "选项甲", B: "选项乙" },
        correct_answers: ["A"],
      })).toBe(true);
    });
  });
});
