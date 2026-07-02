import { z } from "zod";

export const classIdParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, { message: "classId 必须是数字" }),
});

export const teacherIdParamSchema = z.object({
  teacherId: z.string().regex(/^\d+$/, { message: "teacherId 必须是数字" }),
});

export const assistantIdParamSchema = z.object({
  assistantId: z.string().regex(/^\d+$/, { message: "assistantId 必须是数字" }),
});

export const teacherAssistantParamSchema = z.object({
  teacherId: z.string().regex(/^\d+$/, { message: "teacherId 必须是数字" }),
  assistantId: z.string().regex(/^\d+$/, { message: "assistantId 必须是数字" }),
});

export const academicYearStartSchema = z.coerce
  .number()
  .int()
  .min(2000, "学年起始年份不能早于 2000")
  .max(2100, "学年起始年份不能晚于 2100");

export const semesterSchema = z.coerce
  .number()
  .int()
  .min(1, "学期只能为 1 或 2")
  .max(2, "学期只能为 1 或 2");

export const academicTermQuerySchema = z
  .object({
    academic_year_start: academicYearStartSchema.optional(),
    semester: semesterSchema.optional(),
  })
  .refine(
    (data) =>
      (data.academic_year_start === undefined && data.semester === undefined) ||
      (data.academic_year_start !== undefined && data.semester !== undefined),
    { message: "academic_year_start 和 semester 必须同时提供" },
  );

export const createClassSchema = z
  .object({
    class_name: z.string().min(1),
    class_code: z.string().optional(),
    academic_year_start: academicYearStartSchema.optional(),
    semester: semesterSchema.optional(),
  })
  .refine(
    (data) =>
      (data.academic_year_start === undefined && data.semester === undefined) ||
      (data.academic_year_start !== undefined && data.semester !== undefined),
    { message: "academic_year_start 和 semester 必须同时提供" },
  );

export const updateClassSchema = z
  .object({
    class_name: z.string().min(1).optional(),
    class_code: z.string().optional(),
    academic_year_start: academicYearStartSchema.optional(),
    semester: semesterSchema.optional(),
  })
  .refine(
    (data) =>
      (data.academic_year_start === undefined && data.semester === undefined) ||
      (data.academic_year_start !== undefined && data.semester !== undefined),
    { message: "academic_year_start 和 semester 必须同时提供" },
  )
  .refine(
    (data) =>
      data.class_name !== undefined ||
      data.class_code !== undefined ||
      data.academic_year_start !== undefined ||
      data.semester !== undefined,
    {
      message: "至少需要提供一个字段 (class_name、class_code、academic_year_start 或 semester)",
    },
  );

export type ClassIdParam = z.infer<typeof classIdParamSchema>;
export type TeacherIdParam = z.infer<typeof teacherIdParamSchema>;
export type AssistantIdParam = z.infer<typeof assistantIdParamSchema>;
export type TeacherAssistantParam = z.infer<typeof teacherAssistantParamSchema>;
export type AcademicTermQuery = z.infer<typeof academicTermQuerySchema>;
export type CreateClassInput = z.infer<typeof createClassSchema>;
export type UpdateClassInput = z.infer<typeof updateClassSchema>;
