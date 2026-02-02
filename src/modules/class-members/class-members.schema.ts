import { z } from "zod";

export const classIdParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, "classId 必须是数字"),
});

export const classAssistantParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, "classId 必须是数字"),
  assistantId: z.string().regex(/^\d+$/, "assistantId 必须是数字"),
});

export const classStudentParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, "classId 必须是数字"),
  studentId: z.string().regex(/^\d+$/, "studentId 必须是数字"),
});

export const createStudentSchema = z.object({
  username: z.string().min(1, "学号不能为空"),
  full_name: z.string().min(1, "姓名不能为空"),
  email: z.string().regex(/^[^\s@]+@[^\s@]+\.[^\s@]+$/, "无效的邮箱格式").optional(),
  phone_number: z.string().optional(),
  password: z.string().min(6, "密码必须至少6位").optional(),
});

export const addAssistantSchema = z.object({
  assistant_id: z.number().int().min(1),
});

export const joinClassSchema = z.object({
  class_code: z.string().optional(),
});

export const enrollStudentSchema = z.object({
  student_id: z.number().int().positive("学生 ID 必须是正整数"),
});

export type ClassIdParam = z.infer<typeof classIdParamSchema>;
export type ClassAssistantParam = z.infer<typeof classAssistantParamSchema>;
export type ClassStudentParam = z.infer<typeof classStudentParamSchema>;
export type CreateStudentInput = z.infer<typeof createStudentSchema>;
export type AddAssistantInput = z.infer<typeof addAssistantSchema>;
export type JoinClassInput = z.infer<typeof joinClassSchema>;
export type EnrollStudentInput = z.infer<typeof enrollStudentSchema>;
