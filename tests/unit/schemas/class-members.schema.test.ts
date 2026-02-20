import { describe, test, expect } from "bun:test";
import {
  classIdParamSchema,
  classAssistantParamSchema,
  classStudentParamSchema,
  createStudentSchema,
  addAssistantSchema,
  joinClassSchema,
  enrollStudentSchema,
} from "../../../src/modules/class-members/class-members.schema";

describe("Class Members Schemas", () => {
  describe("createStudentSchema", () => {
    test("should accept valid student payload", () => {
      const result = createStudentSchema.safeParse({
        username: "2024123456",
        full_name: "张三",
        email: "2024123456@student.edu",
        phone_number: "13800138000",
        password: "password123",
      });
      expect(result.success).toBe(true);
    });

    test("should reject username shorter than 8 digits", () => {
      const result = createStudentSchema.safeParse({
        username: "1234567",
        full_name: "张三",
      });
      expect(result.success).toBe(false);
    });

    test("should reject non-numeric username", () => {
      const result = createStudentSchema.safeParse({
        username: "student01",
        full_name: "张三",
      });
      expect(result.success).toBe(false);
    });

    test("should reject invalid full_name characters", () => {
      const result = createStudentSchema.safeParse({
        username: "2024123456",
        full_name: "张三@",
      });
      expect(result.success).toBe(false);
    });

    test("should reject invalid phone number format", () => {
      const result = createStudentSchema.safeParse({
        username: "2024123456",
        full_name: "张三",
        phone_number: "1234abc",
      });
      expect(result.success).toBe(false);
    });
  });

  describe("param schemas", () => {
    test("classIdParamSchema should accept numeric classId", () => {
      expect(classIdParamSchema.safeParse({ classId: "1" }).success).toBe(true);
    });

    test("classAssistantParamSchema should reject non-numeric assistantId", () => {
      expect(classAssistantParamSchema.safeParse({ classId: "1", assistantId: "A12" }).success).toBe(false);
    });

    test("classStudentParamSchema should reject non-numeric studentId", () => {
      expect(classStudentParamSchema.safeParse({ classId: "1", studentId: "S12" }).success).toBe(false);
    });
  });

  describe("other payload schemas", () => {
    test("addAssistantSchema should accept positive assistant_id", () => {
      expect(addAssistantSchema.safeParse({ assistant_id: 1 }).success).toBe(true);
    });

    test("joinClassSchema should accept optional class_code", () => {
      expect(joinClassSchema.safeParse({ class_code: "C-001" }).success).toBe(true);
      expect(joinClassSchema.safeParse({}).success).toBe(true);
    });

    test("enrollStudentSchema should reject non-positive student_id", () => {
      expect(enrollStudentSchema.safeParse({ student_id: 0 }).success).toBe(false);
    });
  });
});

