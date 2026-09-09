import { describe, test, expect } from "bun:test";
import {
  classIdParamSchema,
  reportIdParamSchema,
  gradeReportSchema,
  updateGradeWeightsSchema,
  EXP_FLOW_FIELDS,
} from "../../../src/modules/class-grades/class-grades.schema";

const VALID_REVIEW_REVISION = "a".repeat(64);

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
    test("should require the report status observed by the reviewer", () => {
      expect(gradeReportSchema.safeParse({
        expected_review_revision: VALID_REVIEW_REVISION,
        grade: 85,
      }).success).toBe(false);
      expect(gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
      }).success).toBe(false);
    });

    test("should require a valid review revision", () => {
      expect(gradeReportSchema.safeParse({
        expected_status: "submitted",
        grade: 85,
      }).success).toBe(false);
      expect(gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: "not-a-revision",
        grade: 85,
      }).success).toBe(false);
    });

    test("should accept grade only after identifying the observed status", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        grade: 85,
      });
      expect(result.success).toBe(true);
    });

    test("should accept feedback only", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        feedback: "Good work!",
      });
      expect(result.success).toBe(true);
    });

    test("should reject graded status without score", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        status: "graded",
      });
      expect(result.success).toBe(false);
    });

    test("should reject rejected status without a reason", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        status: "rejected",
      });
      expect(result.success).toBe(false);
    });

    test("should reject invalid status", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        status: "pending",
      });
      expect(result.success).toBe(false);
    });

    test("should reject an invalid expected status", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "rejected",
        expected_review_revision: VALID_REVIEW_REVISION,
        feedback: "No mutation is allowed from a rejected report",
      });
      expect(result.success).toBe(false);
    });

    test("should reject grade below 0", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        grade: -1,
      });
      expect(result.success).toBe(false);
    });

    test("should reject grade above 100", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        grade: 101,
      });
      expect(result.success).toBe(false);
    });

    test("should keep the report grade non-nullable", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "graded",
        expected_review_revision: VALID_REVIEW_REVISION,
        grade: null,
      });
      expect(result.success).toBe(false);
    });

    test("should accept numeric and null experiment_grade fields", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "graded",
        expected_review_revision: VALID_REVIEW_REVISION,
        experiment_grade: {
          exp_flow_demand_data_preparation: 90,
          model_quality: null,
        },
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.experiment_grade?.model_quality).toBeNull();
      }
    });

    test("should reject empty object", () => {
      const result = gradeReportSchema.safeParse({});
      expect(result.success).toBe(false);
    });

    test("should accept combined fields", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        grade: 88,
        feedback: "Excellent analysis",
        status: "graded",
      });
      expect(result.success).toBe(true);
    });

    test("should trim empty feedback to null", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "graded",
        expected_review_revision: VALID_REVIEW_REVISION,
        grade: 80,
        feedback: "   ",
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.feedback).toBeNull();
      }
    });

    test("should reject graded status without report grade", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        status: "graded",
        feedback: "Only feedback",
        experiment_grade: {
          model_quality: 93,
        },
      });
      expect(result.success).toBe(false);
    });

    test("should accept rejected status without report grade", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        status: "rejected",
        feedback: "Need revision",
      });
      expect(result.success).toBe(true);
    });

    test("should reject a rejection claiming it was opened from graded", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "graded",
        expected_review_revision: VALID_REVIEW_REVISION,
        status: "rejected",
        feedback: "A graded report cannot be rejected through this API",
      });
      expect(result.success).toBe(false);
    });

    test("should reject blank rejection reasons after trimming", () => {
      const result = gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        status: "rejected",
        feedback: "   ",
      });
      expect(result.success).toBe(false);
    });

    test("should reject scores mixed into a rejection command", () => {
      expect(gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        status: "rejected",
        feedback: "Need revision",
        grade: 0,
      }).success).toBe(false);

      expect(gradeReportSchema.safeParse({
        expected_status: "submitted",
        expected_review_revision: VALID_REVIEW_REVISION,
        status: "rejected",
        feedback: "Need revision",
        experiment_grade: { model_quality: null },
      }).success).toBe(false);
    });
  });

  describe("updateGradeWeightsSchema", () => {
    const expected_weights = {
      ...Object.fromEntries(EXP_FLOW_FIELDS.map((field, i) => [field, i === 0 ? 100 : 0])),
      exp_flow_weight: 40, model_quality_weight: 20, knowledge_test_weight: 20, report_quality_weight: 20,
    };
    test("should accept single weight field", () => {
      const result = updateGradeWeightsSchema.safeParse({ exp_flow_weight: 40, expected_weights });
      expect(result.success).toBe(true);
    });

    test("should accept all weight fields", () => {
      const result = updateGradeWeightsSchema.safeParse({
        expected_weights,
        exp_flow_weight: 40,
        model_quality_weight: 20,
        knowledge_test_weight: 20,
        report_quality_weight: 20,
      });
      expect(result.success).toBe(true);
    });

    test("should reject weight below 0", () => {
      const result = updateGradeWeightsSchema.safeParse({ exp_flow_weight: -1, expected_weights });
      expect(result.success).toBe(false);
    });

    test("should reject weight above 100", () => {
      const result = updateGradeWeightsSchema.safeParse({ exp_flow_weight: 101, expected_weights });
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
