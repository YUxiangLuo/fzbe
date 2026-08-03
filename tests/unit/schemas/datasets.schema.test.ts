import { describe, test, expect } from "bun:test";
import {
  datasetIdParamSchema,
  industryParamSchema,
  industryAndCompanyParamSchema,
  industryCompanyProductParamSchema,
  datasetMetadataSchema,
  updateDatasetSchema,
} from "../../../src/modules/datasets/datasets.schema";

describe("Datasets Schemas", () => {
  describe("datasetIdParamSchema", () => {
    test("should accept valid numeric string", () => {
      const result = datasetIdParamSchema.safeParse({ datasetId: "123" });
      expect(result.success).toBe(true);
    });

    test("should reject non-numeric string", () => {
      const result = datasetIdParamSchema.safeParse({ datasetId: "abc" });
      expect(result.success).toBe(false);
    });

    test("should reject empty string", () => {
      const result = datasetIdParamSchema.safeParse({ datasetId: "" });
      expect(result.success).toBe(false);
    });
  });

  describe("industryParamSchema", () => {
    test("should accept valid industry name", () => {
      const result = industryParamSchema.safeParse({ industry: "Manufacturing" });
      expect(result.success).toBe(true);
    });

    test("should reject empty industry", () => {
      const result = industryParamSchema.safeParse({ industry: "" });
      expect(result.success).toBe(false);
    });
  });

  describe("industryAndCompanyParamSchema", () => {
    test("should accept valid industry and company", () => {
      const result = industryAndCompanyParamSchema.safeParse({
        industry: "Retail",
        company: "ABC Corp",
      });
      expect(result.success).toBe(true);
    });

    test("should reject empty company", () => {
      const result = industryAndCompanyParamSchema.safeParse({
        industry: "Retail",
        company: "",
      });
      expect(result.success).toBe(false);
    });
  });

  describe("industryCompanyProductParamSchema", () => {
    test("should accept all three params", () => {
      const result = industryCompanyProductParamSchema.safeParse({
        industry: "Tech",
        company: "XYZ Inc",
        product: "Widget",
      });
      expect(result.success).toBe(true);
    });

    test("should reject empty product", () => {
      const result = industryCompanyProductParamSchema.safeParse({
        industry: "Tech",
        company: "XYZ Inc",
        product: "",
      });
      expect(result.success).toBe(false);
    });
  });

  describe("updateDatasetSchema", () => {
    test("should accept data_name update", () => {
      const result = updateDatasetSchema.safeParse({
        data_name: "New Dataset Name",
      });
      expect(result.success).toBe(true);
    });

    test("should accept description update", () => {
      const result = updateDatasetSchema.safeParse({
        description: "New description",
      });
      expect(result.success).toBe(true);
    });

    test("should accept both fields", () => {
      const result = updateDatasetSchema.safeParse({
        data_name: "Updated Name",
        description: "Updated description",
      });
      expect(result.success).toBe(true);
    });

    test("should reject empty data_name when provided", () => {
      const result = updateDatasetSchema.safeParse({
        data_name: "",
      });
      expect(result.success).toBe(false);
    });

    test("should accept empty object (no required fields)", () => {
      const result = updateDatasetSchema.safeParse({});
      expect(result.success).toBe(true);
    });

    test("should reject names and descriptions beyond the admin UI contract", () => {
      expect(updateDatasetSchema.safeParse({ data_name: "x" }).success).toBe(false);
      expect(updateDatasetSchema.safeParse({ data_name: "x".repeat(101) }).success).toBe(false);
      expect(updateDatasetSchema.safeParse({ description: "x".repeat(201) }).success).toBe(false);
    });

    test("should trim valid upload metadata", () => {
      const result = datasetMetadataSchema.safeParse({
        data_name: "  销售数据  ",
        description: "  备注  ",
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toEqual({ data_name: "销售数据", description: "备注" });
      }
    });
  });
});
