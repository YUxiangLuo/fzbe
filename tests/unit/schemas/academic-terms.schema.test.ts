import { describe, expect, test } from "bun:test";
import {
  academicTermSchema,
  termIdParamSchema,
  updateAcademicTermSchema,
} from "../../../src/modules/academic-terms/academic-terms.schema";

describe("Academic terms schemas", () => {
  test("accepts a valid academic term and trims the academic year", () => {
    const result = academicTermSchema.parse({
      academic_year: " 2026-2027 ",
      semester: 1,
      is_active: true,
    });

    expect(result).toEqual({
      academic_year: "2026-2027",
      semester: 1,
      is_active: true,
    });
  });

  test("rejects semesters outside the supported range", () => {
    const result = academicTermSchema.safeParse({
      academic_year: "2026-2027",
      semester: 3,
    });

    expect(result.success).toBe(false);
  });

  test("requires at least one field when updating", () => {
    const result = updateAcademicTermSchema.safeParse({});

    expect(result.success).toBe(false);
  });

  test("validates term id route params as positive integer strings", () => {
    expect(termIdParamSchema.safeParse({ termId: "12" }).success).toBe(true);
    expect(termIdParamSchema.safeParse({ termId: "current" }).success).toBe(false);
    expect(termIdParamSchema.safeParse({ termId: "-1" }).success).toBe(false);
  });
});
