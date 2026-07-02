import { describe, test, expect } from "bun:test";
import {
  createClassSchema,
  updateClassSchema,
  academicTermQuerySchema,
  classIdParamSchema,
  teacherAssistantParamSchema,
} from "../../../src/modules/classes/classes.schema";

const validAcademicTerm = {
  academic_year_start: 2025,
  semester: 1,
};

describe("Classes Schemas", () => {
  describe("createClassSchema", () => {
    test("should accept valid class data", () => {
      const result = createClassSchema.safeParse({
        class_name: "Math 101",
        class_code: "MATH101",
        ...validAcademicTerm,
      });
      expect(result.success).toBe(true);
    });

    test("should reject missing academic term", () => {
      const result = createClassSchema.safeParse({
        class_name: "Math 101",
        class_code: "MATH101",
      });
      expect(result.success).toBe(false);
    });

    test("should reject partial academic term", () => {
      const result = createClassSchema.safeParse({
        class_name: "Math 101",
        class_code: "MATH101",
        academic_year_start: 2025,
      });
      expect(result.success).toBe(false);
    });

    test("should reject empty class_name", () => {
      const result = createClassSchema.safeParse({
        class_name: "",
        class_code: "MATH101",
        ...validAcademicTerm,
      });
      expect(result.success).toBe(false);
    });

    test("should accept empty class_code (optional field)", () => {
      const result = createClassSchema.safeParse({
        class_name: "Math 101",
        class_code: "",
        ...validAcademicTerm,
      });
      expect(result.success).toBe(true);
    });

    test("should accept missing class_code", () => {
      const result = createClassSchema.safeParse({
        class_name: "Math 101",
        ...validAcademicTerm,
      });
      expect(result.success).toBe(true);
    });

    test("should accept Chinese class name", () => {
      const result = createClassSchema.safeParse({
        class_name: "高等数学",
        class_code: "MATH101",
        ...validAcademicTerm,
      });
      expect(result.success).toBe(true);
    });

    test("should accept class code with numbers and letters", () => {
      const result = createClassSchema.safeParse({
        class_name: "Physics",
        class_code: "PHY2024A",
        ...validAcademicTerm,
      });
      expect(result.success).toBe(true);
    });
  });

  describe("academicTermQuerySchema", () => {
    test("should accept a complete academic term filter", () => {
      const result = academicTermQuerySchema.safeParse({
        academic_year_start: "2026",
        semester: "2",
      });
      expect(result).toMatchObject({
        success: true,
        data: { academic_year_start: 2026, semester: 2 },
      });
    });

    test("should accept an omitted academic term filter for unfiltered endpoints", () => {
      const result = academicTermQuerySchema.safeParse({});
      expect(result.success).toBe(true);
    });

    test("should reject partial academic term query parameters", () => {
      const result = academicTermQuerySchema.safeParse({
        academic_year_start: "2026",
      });
      expect(result.success).toBe(false);
    });

    test("should reject semesters outside the supported range", () => {
      const result = academicTermQuerySchema.safeParse({
        academic_year_start: "2026",
        semester: "3",
      });
      expect(result.success).toBe(false);
    });
  });

  describe("updateClassSchema", () => {
    test("should accept partial update with class_name only", () => {
      const result = updateClassSchema.safeParse({
        class_name: "Updated Name",
      });
      expect(result.success).toBe(true);
    });

    test("should accept partial update with class_code only", () => {
      const result = updateClassSchema.safeParse({
        class_code: "NEW101",
      });
      expect(result.success).toBe(true);
    });

    test("should reject empty object", () => {
      const result = updateClassSchema.safeParse({});
      expect(result.success).toBe(false);
    });

    test("should accept both fields", () => {
      const result = updateClassSchema.safeParse({
        class_name: "New Name",
        class_code: "NEW101",
      });
      expect(result.success).toBe(true);
    });

    test("should reject empty class_name when provided", () => {
      const result = updateClassSchema.safeParse({
        class_name: "",
      });
      expect(result.success).toBe(false);
    });
  });

  describe("classIdParamSchema", () => {
    test("should accept numeric string", () => {
      const result = classIdParamSchema.safeParse({ classId: "123" });
      expect(result.success).toBe(true);
    });

    test("should reject non-numeric string", () => {
      const result = classIdParamSchema.safeParse({ classId: "abc" });
      expect(result.success).toBe(false);
    });

    test("should reject mixed alphanumeric", () => {
      const result = classIdParamSchema.safeParse({ classId: "123abc" });
      expect(result.success).toBe(false);
    });

    test("should reject negative numbers", () => {
      const result = classIdParamSchema.safeParse({ classId: "-1" });
      expect(result.success).toBe(false);
    });

    test("should accept zero", () => {
      const result = classIdParamSchema.safeParse({ classId: "0" });
      expect(result.success).toBe(true);
    });
  });

  describe("teacherAssistantParamSchema", () => {
    test("should accept numeric teacherId and assistantId", () => {
      const result = teacherAssistantParamSchema.safeParse({
        teacherId: "101",
        assistantId: "201",
      });
      expect(result.success).toBe(true);
    });

    test("should reject non-numeric teacherId", () => {
      const result = teacherAssistantParamSchema.safeParse({
        teacherId: "teacher-1",
        assistantId: "201",
      });
      expect(result.success).toBe(false);
    });

    test("should reject non-numeric assistantId", () => {
      const result = teacherAssistantParamSchema.safeParse({
        teacherId: "101",
        assistantId: "assistant-1",
      });
      expect(result.success).toBe(false);
    });
  });
});
