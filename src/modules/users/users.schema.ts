import { z } from "zod";

const normalizeOptionalPhone = (value: unknown): unknown => {
  if (value === null || value === undefined) return undefined;
  if (typeof value !== "string") return value;
  const normalized = value.trim();
  return normalized === "" ? undefined : normalized;
};

const normalizeNullablePhone = (value: unknown): unknown => {
  if (value === null || value === undefined) return value;
  if (typeof value !== "string") return value;
  const normalized = value.trim();
  return normalized === "" ? null : normalized;
};

const optionalPhoneNumberSchema = z.preprocess(
  normalizeOptionalPhone,
  z.string().regex(/^\d{6,15}$/, "手机号必须为6到15位数字").optional(),
);

const optionalChineseMobileSchema = z.preprocess(
  normalizeOptionalPhone,
  z.string().regex(/^1\d{10}$/, "手机号必须以1开头且为11位数字").optional(),
);

const nullablePhoneNumberSchema = z.preprocess(
  normalizeNullablePhone,
  z.string().regex(/^\d{6,15}$/, "手机号必须为6到15位数字").nullable().optional(),
);

export const registerSchema = z.object({
  username: z.string().min(3),
  password: z.string().min(6, { message: "密码长度至少为6位" }),
  full_name: z.string().min(1),
  email: z.string().email(),
  role: z.enum(["Student", "Teacher", "Assistant", "Admin"]).default("Student"),
  phone_number: optionalPhoneNumberSchema,
});

export const createTeacherSchema = z.object({
  username: z.string().min(3),
  full_name: z.string().min(1),
  email: z.string().email().optional(),
  phone: optionalChineseMobileSchema,
  password: z.string().min(6, "密码必须至少6位").optional(),
});

export const createAssistantSingleSchema = z.object({
  username: z.string().min(3),
  full_name: z.string().min(1),
  email: z.string().email().optional(),
  phone: optionalChineseMobileSchema,
  password: z.string().min(6, "密码必须至少6位").optional(),
});

export const createAssistantSchema = registerSchema
  .extend({ class_ids: z.array(z.number().int().min(1)).min(1, "至少需要一个班级 ID") })
  .omit({ role: true });

export const updateUserSchema = z.object({
  full_name: z.string().min(1).optional(),
  phone_number: nullablePhoneNumberSchema,
  email: z.string().email("Invalid email format").optional(),
}).refine((data) => Object.values(data).some((value) => value !== undefined), {
  message: "至少需要提供一个字段 (full_name, phone_number, 或 email)",
});

export const changePasswordSchema = z.object({
  currentPassword: z.string().min(1, { message: "当前密码不能为空" }),
  newPassword: z.string().min(6, { message: "新密码长度至少为6位" }),
});

export const adminChangePasswordSchema = z.object({
  newPassword: z.string().min(6, { message: "新密码长度至少为6位" }),
});

export const resetPasswordSchema = z
  .object({
    newPassword: z.string().min(6, { message: "新密码长度至少为6位" }).optional(),
    /** @deprecated Use `newPassword` instead. Kept for backward compatibility with older clients. */
    new_password: z.string().min(6, { message: "新密码长度至少为6位" }).optional(),
  })
  .refine(
    ({ newPassword, new_password }) =>
      !newPassword || !new_password || newPassword === new_password,
    { message: "newPassword 与 new_password 不一致，请只传其中一个" },
  )
  .transform(({ newPassword, new_password }) => ({
    newPassword: newPassword ?? new_password,
  }));

export const userSearchQuerySchema = z.object({
  q: z.string().trim().min(1, { message: "搜索关键词不能为空" }),
  page: z.coerce.number().int().min(1).optional().default(1),
  limit: z.coerce.number().int().min(1).max(100).optional().default(10),
});

export const bulkDeleteUsersSchema = z.object({
  userIds: z.array(z.number().int().positive("用户 ID 必须为正整数")).min(1, "至少选择一个用户").max(100, "单次最多删除 100 个用户")
    .refine((userIds) => new Set(userIds).size === userIds.length, { message: "用户 ID 不能重复" }),
});

export const userIdParamSchema = z.object({ userId: z.string().regex(/^\d+$/, { message: "userId 必须是数字" }) });
export const teacherIdParamSchema = z.object({ teacherId: z.string().regex(/^\d+$/, { message: "teacherId 必须是数字" }) });
export const paginationQuerySchema = z.object({
  page: z.coerce.number().int().min(1).optional().default(1),
  limit: z.coerce.number().int().min(1).max(100).optional().default(10),
});

export const roleParamSchema = z.object({
  role: z.enum(["Student"], "无效的角色。仅允许 Student 自助注册"),
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
  phone: z.string().trim().length(11, "手机号必须是11位数字").regex(/^\d{11}$/, "手机号只能包含数字"),
});

export type RegisterInput = z.infer<typeof registerSchema>;
export type CreateTeacherInput = z.infer<typeof createTeacherSchema>;
export type CreateAssistantSingleInput = z.infer<typeof createAssistantSingleSchema>;
export type CreateAssistantInput = z.infer<typeof createAssistantSchema>;
export type UpdateUserInput = z.infer<typeof updateUserSchema>;
export type ChangePasswordInput = z.infer<typeof changePasswordSchema>;
export type AdminChangePasswordInput = z.infer<typeof adminChangePasswordSchema>;
export type BulkDeleteUsersInput = z.infer<typeof bulkDeleteUsersSchema>;
export type ResetPasswordInput = z.infer<typeof resetPasswordSchema>;
export type UserSearchQuery = z.infer<typeof userSearchQuerySchema>;
export type PaginationQuery = z.infer<typeof paginationQuerySchema>;
