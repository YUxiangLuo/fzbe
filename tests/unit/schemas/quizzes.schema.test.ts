import { describe, test, expect } from "bun:test";
import { batchSubmitAnswersSchema } from "../../../src/modules/quizzes/quizzes.schema";

describe("Quizzes Schemas", () => {
  describe("batchSubmitAnswersSchema", () => {
    test("should accept valid batch submission", () => {
      const result = batchSubmitAnswersSchema.safeParse({
        experiment_id: 1,
        answers: [
          { question_id: 1, submitted_answer: ["A"] },
          { question_id: 2, submitted_answer: ["B", "C"] },
        ],
      });
      expect(result.success).toBe(true);
    });

    test("should reject empty answers array", () => {
      const result = batchSubmitAnswersSchema.safeParse({
        experiment_id: 1,
        answers: [],
      });
      expect(result.success).toBe(false);
    });

    test("should reject missing experiment_id", () => {
      const result = batchSubmitAnswersSchema.safeParse({
        answers: [{ question_id: 1, submitted_answer: ["A"] }],
      });
      expect(result.success).toBe(false);
    });

    test("should reject negative experiment_id", () => {
      const result = batchSubmitAnswersSchema.safeParse({
        experiment_id: -1,
        answers: [{ question_id: 1, submitted_answer: ["A"] }],
      });
      expect(result.success).toBe(false);
    });

    test("should reject empty submitted_answer array", () => {
      const result = batchSubmitAnswersSchema.safeParse({
        experiment_id: 1,
        answers: [{ question_id: 1, submitted_answer: [] }],
      });
      expect(result.success).toBe(false);
    });

    test("should reject empty string in submitted_answer", () => {
      const result = batchSubmitAnswersSchema.safeParse({
        experiment_id: 1,
        answers: [{ question_id: 1, submitted_answer: [""] }],
      });
      expect(result.success).toBe(false);
    });

    test("should accept multiple answers", () => {
      const answers = Array.from({ length: 20 }, (_, i) => ({
        question_id: i + 1,
        submitted_answer: ["Option A"],
      }));
      const result = batchSubmitAnswersSchema.safeParse({
        experiment_id: 1,
        answers,
      });
      expect(result.success).toBe(true);
    });

    test("should reject negative question_id", () => {
      const result = batchSubmitAnswersSchema.safeParse({
        experiment_id: 1,
        answers: [{ question_id: -1, submitted_answer: ["A"] }],
      });
      expect(result.success).toBe(false);
    });
  });
});
