import { describe, expect, test } from "bun:test";
import {
  manualMetadataSchema,
  updateManualSchema,
} from "../../../src/modules/manuals/manuals.schema";

describe("Manuals Schemas", () => {
  test("accepts and trims metadata within the admin UI limits", () => {
    const result = manualMetadataSchema.safeParse({
      file_name: "  实验手册  ",
      description: "  使用说明  ",
    });
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data).toEqual({ file_name: "实验手册", description: "使用说明" });
    }
  });

  test("rejects too-short or overlong metadata", () => {
    expect(manualMetadataSchema.safeParse({ file_name: "x" }).success).toBe(false);
    expect(manualMetadataSchema.safeParse({ file_name: "x".repeat(101) }).success).toBe(false);
    expect(manualMetadataSchema.safeParse({
      file_name: "有效名称",
      description: "x".repeat(201),
    }).success).toBe(false);
  });

  test("keeps status-only updates valid", () => {
    expect(updateManualSchema.safeParse({ is_active: true }).success).toBe(true);
  });
});
