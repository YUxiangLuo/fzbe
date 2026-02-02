import { describe, test, expect } from "bun:test";
import {
  classIdParamSchema,
  reportIdParamSchema,
  gradeReportSchema,
  updateGradeWeightsSchema,
  EXP_FLOW_FIELDS,
} from "../../../src/modules/class-grades/class-grades.schema";

describe("Class Grades Schemas", () => {
  describe("classIdParamSchema", () => {
    test("should accept valid numeric string", () => {
      const result = classIdParamSchema.safeParse({ classId: "123" });
      expect(result.success).toBe(true);
    });

    test("should reject non-numeric string", () => {
      const result = classIdParamSchema.safeParse({ classId: "abc" });
      expect(result.success).toBe(false);
    });
  });

  describe("reportIdParamSchema", () => {
    test("should accept valid numeric string", () => {
      const result = reportIdParamSchema.safeParse({ reportId: "456" });
      expect(result.success).toBe(true);
    });

    test("should reject non-numeric string", () => {
      const result = reportIdParamSchema.safeParse({ reportId: "xyz" });
      expect(result.success).toBe(false);
    });
  });

  describe("gradeReportSchema", () => {
    test("should accept grade only", () => {
      const result = gradeReportSchema.safeParse({ grade: 85 });
      expect(result.success).toBe(true);
    });

    test("should accept feedback only", () => {
      const result = gradeReportSchema.safeParse({ feedback: "Good work!" });
      expect(result.success).toBe(true);
    });

    test("should accept status only", () => {
      const result = gradeReportSchema.safeParse({ status: "graded" });
      expect(result.success).toBe(true);
    });

    test("should accept rejected status", () => {
      const result = gradeReportSchema.safeParse({ status: "rejected" });
      expect(result.success).toBe(true);
    });

    test("should reject invalid status", () => {
      const result = gradeReportSchema.safeParse({ status: "pending" });
      expect(result.success).toBe(false);
    });

    test("should reject grade below 0", () => {
      const result = gradeReportSchema.safeParse({ grade: -1 });
      expect(result.success).toBe(false);
    });

    test("should reject grade above 100", () => {
      const result = gradeReportSchema.safeParse({ grade: 101 });
      expect(result.success).toBe(false);
    });

    test("should accept experiment_grade fields", () => {
      const result = gradeReportSchema.safeParse({
        experiment_grade: {
          exp_flow_demand_data_preparation: 90,
          model_quality: 85,
        },
      });
      expect(result.success).toBe(true);
    });

    test("should reject empty object", () => {
      const result = gradeReportSchema.safeParse({});
      expect(result.success).toBe(false);
    });

    test("should accept combined fields", () => {
      const result = gradeReportSchema.safeParse({
        grade: 88,
        feedback: "Excellent analysis",
        status: "graded",
      });
      expect(result.success).toBe(true);
    });

    test("should trim empty feedback to null", () => {
      const result = gradeReportSchema.safeParse({
        grade: 80,
        feedback: "   ",
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.feedback).toBeNull();
      }
    });
  });

  describe("updateGradeWeightsSchema", () => {
    test("should accept single weight field", () => {
      const result = updateGradeWeightsSchema.safeParse({ exp_flow_weight: 40 });
      expect(result.success).toBe(true);
    });

    test("should accept all weight fields", () => {
      const result = updateGradeWeightsSchema.safeParse({
        exp_flow_weight: 40,
        model_quality_weight: 20,
        knowledge_test_weight: 20,
        report_quality_weight: 20,
      });
      expect(result.success).toBe(true);
    });

    test("should reject weight below 0", () => {
      const result = updateGradeWeightsSchema.safeParse({ exp_flow_weight: -1 });
      expect(result.success).toBe(false);
    });

    test("should reject weight above 100", () => {
      const result = updateGradeWeightsSchema.safeParse({ exp_flow_weight: 101 });
      expect(result.success).toBe(false);
    });

    test("should reject empty object", () => {
      const result = updateGradeWeightsSchema.safeParse({});
      expect(result.success).toBe(false);
    });
  });

  describe("EXP_FLOW_FIELDS constant", () => {
    test("should have 9 experiment flow fields", () => {
      expect(EXP_FLOW_FIELDS).toHaveLength(9);
    });

    test("should include all expected fields", () => {
      expect(EXP_FLOW_FIELDS).toContain("exp_flow_demand_data_preparation");
      expect(EXP_FLOW_FIELDS).toContain("exp_flow_demand_descriptive_stats");
      expect(EXP_FLOW_FIELDS).toContain("exp_flow_demand_model_selection");
      expect(EXP_FLOW_FIELDS).toContain("exp_flow_report_submission");
    });
  });
});
