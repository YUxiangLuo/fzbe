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

export const createClassSchema = z.object({
  class_name: z.string().min(1),
  class_code: z.string().optional(),
});

export const updateClassSchema = z
  .object({
    class_name: z.string().min(1).optional(),
    class_code: z.string().optional(),
  })
  .refine((data) => data.class_name !== undefined || data.class_code !== undefined, {
    message: "至少需要提供一个字段 (class_name 或 class_code)",
  });

export type ClassIdParam = z.infer<typeof classIdParamSchema>;
export type TeacherIdParam = z.infer<typeof teacherIdParamSchema>;
export type AssistantIdParam = z.infer<typeof assistantIdParamSchema>;
export type CreateClassInput = z.infer<typeof createClassSchema>;
export type UpdateClassInput = z.infer<typeof updateClassSchema>;
