import { z } from "zod";

export const registerSchema = z.object({
  username: z.string().min(3),
  password: z.string().min(6, { message: "密码长度至少为6位" }),
  full_name: z.string().min(1),
  email: z.string().email(),
  role: z.enum(["Student", "Teacher", "Assistant", "Admin"]).default("Student"),
  phone_number: z.string().optional(),
});

export const createTeacherSchema = z.object({
  username: z.string().min(3),
  full_name: z.string().min(1),
  email: z.string().email().optional(),
  phone: z.string().length(11, "手机号必须为11位").regex(/^1\d{10}$/, "手机号必须以1开头的11位数字").optional(),
  password: z.string().min(6, "密码必须至少6位").optional(),
});

export const createAssistantSingleSchema = z.object({
  username: z.string().min(3),
  full_name: z.string().min(1),
  email: z.string().email().optional(),
  phone: z.string().length(11, "手机号必须为11位").regex(/^1\d{10}$/, "手机号必须以1开头的11位数字").optional(),
  password: z.string().min(6, "密码必须至少6位").optional(),
});

export const createAssistantSchema = registerSchema
  .extend({ class_ids: z.array(z.number().int().min(1)).min(1, "至少需要一个班级 ID") })
  .omit({ role: true });

export const updateUserSchema = z.object({
  full_name: z.string().min(1).optional(),
  phone_number: z.string().min(1).optional(),
  email: z.string().email("Invalid email format").optional(),
}).refine((data) => data.full_name || data.phone_number || data.email, {
  message: "至少需要提供一个字段 (full_name, phone_number, 或 email)",
});

export const changePasswordSchema = z.object({
  currentPassword: z.string().min(1, { message: "当前密码不能为空" }),
  newPassword: z.string().min(6, { message: "新密码长度至少为6位" }),
});

export const adminChangePasswordSchema = z.object({
  newPassword: z.string().min(6, { message: "新密码长度至少为6位" }),
});

export const resetPasswordSchema = z.object({
  newPassword: z.string().min(6, { message: "新密码长度至少为6位" }).optional(),
});

export const userSearchQuerySchema = z.object({
  q: z.string().trim().min(1, { message: "搜索关键词不能为空" }),
  page: z.coerce.number().int().min(1).optional().default(1),
  limit: z.coerce.number().int().min(1).max(100).optional().default(10),
});

export const userIdParamSchema = z.object({ userId: z.string().regex(/^\d+$/, { message: "userId 必须是数字" }) });
export const teacherIdParamSchema = z.object({ teacherId: z.string().regex(/^\d+$/, { message: "teacherId 必须是数字" }) });
export const paginationQuerySchema = z.object({
  page: z.coerce.number().int().min(1).optional().default(1),
  limit: z.coerce.number().int().min(1).max(100).optional().default(10),
});

export const roleParamSchema = z.object({
  role: z.enum(["Student", "Teacher", "Assistant"], "无效的角色。必须是 Student, Teacher, 或 Assistant"),
});

export const studentSelfRegisterSchema = z.object({
  username: z.string().min(8, "学生用户名至少为8位数字").regex(/^\d+$/, "学生用户名只能包含数字"),
  password: z.string().min(6, "密码长度至少为6位"),
  name: z.string().min(1, "姓名不能为空").regex(/^[\u4e00-\u9fa5a-zA-Z\s]+$/, "姓名只能包含中文字符、英文字母和空格"),
  email: z.string().regex(/^[^\s@]+@[^\s@]+\.[^\s@]+$/, "无效的邮箱格式"),
});

export const teacherAssistantSelfRegisterSchema = z.object({
  username: z.string().min(3, "用户名至少为3个字符").regex(/^[a-zA-Z0-9]+$/, "用户名只能包含字母和数字"),
  password: z.string().min(6, "密码长度至少为6位"),
  name: z.string().min(1, "姓名不能为空").regex(/^[\u4e00-\u9fa5a-zA-Z\s]+$/, "姓名只能包含中文字符、英文字母和空格"),
  email: z.string().regex(/^[^\s@]+@[^\s@]+\.[^\s@]+$/, "无效的邮箱格式"),
  phone: z.string().length(11, "手机号必须是11位数字").regex(/^\d{11}$/, "手机号只能包含数字"),
});

export type RegisterInput = z.infer<typeof registerSchema>;
export type CreateTeacherInput = z.infer<typeof createTeacherSchema>;
export type CreateAssistantSingleInput = z.infer<typeof createAssistantSingleSchema>;
export type CreateAssistantInput = z.infer<typeof createAssistantSchema>;
export type UpdateUserInput = z.infer<typeof updateUserSchema>;
export type ChangePasswordInput = z.infer<typeof changePasswordSchema>;
export type AdminChangePasswordInput = z.infer<typeof adminChangePasswordSchema>;
export type ResetPasswordInput = z.infer<typeof resetPasswordSchema>;
export type UserSearchQuery = z.infer<typeof userSearchQuerySchema>;
export type PaginationQuery = z.infer<typeof paginationQuerySchema>;
