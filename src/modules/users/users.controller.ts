import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth } from "../../middleware";
import { handleUnexpectedError } from "../../utils/error-handler";
import type { JwtPayload } from "../../types";
import {
  roleParamSchema, createAssistantSingleSchema, userSearchQuerySchema, createAssistantSchema, createTeacherSchema,
  teacherIdParamSchema, paginationQuerySchema, updateUserSchema, changePasswordSchema, userIdParamSchema,
  adminChangePasswordSchema, resetPasswordSchema, bulkDeleteUsersSchema
} from "./users.schema";
import {
  registerSelfUser, createAssistantSingle, searchUsers, createAssistantWithClasses, createTeacher,
  batchCreateTeachers, batchCreateAssistants, getTeacherAssistants, getAllUsers, getAllAssistants,
  getUnenrolledStudents, getCurrentUser, updateCurrentUser, changePassword, getUserById, adminChangePassword,
  resetStudentPassword, deleteUser, batchDeleteUsers
} from "./users.service";
import { academicTermQuerySchema } from "../classes/classes.schema";

const usersController = new Hono();

usersController.post("/api/v1/users/register/:role", zValidator("param", roleParamSchema), async (c) => {
  try {
    const { role } = c.req.valid("param");
    const body = await c.req.json();
    const result = await registerSelfUser(role, body);
    if ("error" in result) {
      if (result.error === "validation") return c.json({ error: "验证失败", details: result.details }, 400);
      if (result.error === "dup_username") return c.json({ error: "用户名已存在" }, 409);
      if (result.error === "dup_email") return c.json({ error: "邮箱已存在" }, 409);
      if (result.error === "dup_phone") return c.json({ error: "手机号已存在" }, 409);
    }
    const roleNameMap: Record<string, string> = {
      Student: "学生",
      Teacher: "教师",
      Assistant: "助教",
    };
    const roleName = roleNameMap[role] ?? role;
    return c.json({ ...result.data, message: `${roleName}账号创建成功` }, 201);
  } catch (error) { return handleUnexpectedError(c, error, "注册失败"); }
});

usersController.post("/api/v1/users/assistants", jwtAuth, zValidator("json", createAssistantSingleSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (payload.role !== "Admin") return c.json({ error: "权限不足" }, 403);
    const input = c.req.valid("json");
    const result = await createAssistantSingle(input);
    if ("error" in result) {
      if (result.error === "dup_username") return c.json({ error: "用户名已存在" }, 409);
      if (result.error === "dup_email") return c.json({ error: "邮箱已存在" }, 409);
      if (result.error === "dup_phone") return c.json({ error: "手机号已存在" }, 409);
    }
    return c.json(result.data, 201);
  } catch (error) { return handleUnexpectedError(c, error, "创建助教失败"); }
});

usersController.get("/api/v1/users/search", jwtAuth, zValidator("query", userSearchQuerySchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (payload.role !== "Admin") return c.json({ error: "权限不足" }, 403);
    const query = c.req.valid("query");
    const result = await searchUsers(query);
    if ("error" in result) return c.json({ error: "搜索关键词不能为空" }, 400);
    return c.json(result.data);
  } catch (error) { return handleUnexpectedError(c, error, "搜索用户失败"); }
});

usersController.post("/api/v1/assistants", jwtAuth, zValidator("json", createAssistantSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (payload.role !== "Admin" && payload.role !== "Teacher") return c.json({ error: "权限不足" }, 403);
    const input = c.req.valid("json");
    const result = await createAssistantWithClasses(input, payload.sub, payload.role);
    if ("error" in result) {
      if (result.error === "invalid_classes") return c.json({ error: "一个或多个班级 ID 无效" }, 400);
      if (result.error === "not_class_owner") return c.json({ error: `禁止将助教分配给您不拥有的班级 (ID: ${result.classId})` }, 403);
      if (result.error === "dup_username") return c.json({ error: "用户名已存在" }, 409);
      if (result.error === "dup_email") return c.json({ error: "邮箱已存在" }, 409);
      if (result.error === "dup_phone") return c.json({ error: "手机号已存在" }, 409);
    }
    return c.json(result.data, 201);
  } catch (error) { return handleUnexpectedError(c, error, "创建助教失败"); }
});

usersController.post("/api/v1/users/teachers", jwtAuth, zValidator("json", createTeacherSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (payload.role !== "Admin") return c.json({ error: "权限不足" }, 403);
    const input = c.req.valid("json");
    const result = await createTeacher(input);
    if ("error" in result) {
      if (result.error === "dup_username") return c.json({ error: "用户名已存在" }, 409);
      if (result.error === "dup_email") return c.json({ error: "邮箱已存在" }, 409);
      if (result.error === "dup_phone") return c.json({ error: "手机号已存在" }, 409);
    }
    return c.json(result.data, 201);
  } catch (error) { return handleUnexpectedError(c, error, "创建教师失败"); }
});

usersController.post("/api/v1/users/teachers/batch", jwtAuth, async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (payload.role !== "Admin") return c.json({ error: "权限不足" }, 403);
    const body = await c.req.parseBody();
    const file = body["file"];
    const password = body["password"];
    if (!(file instanceof File)) return c.json({ error: "CSV 文件缺失或类型错误" }, 400);
    const result = await batchCreateTeachers(file, password);
    if ("error" in result) {
      if (result.error === "invalid_password_type") return c.json({ error: "密码必须是字符串类型" }, 400);
      if (result.error === "password_too_short") return c.json({ error: "统一密码必须至少6位" }, 400);
      if (result.error === "invalid_file_type") return c.json({ error: "仅支持上传 CSV 文件" }, 400);
      if (result.error === "file_too_large") return c.json({ error: "CSV 文件大小不能超过 5MB" }, 413);
      if (result.error === "invalid_rows") return c.json({ error: "CSV 文件包含无效记录", invalid_rows: result.invalid_rows }, 400);
      if (result.error === "no_valid_records") return c.json({ error: "CSV 文件未包含有效的教师记录" }, 400);
      if (result.error === "conflicts") return c.json({ error: "部分教师信息已存在，批量创建已取消", conflicts: result.conflicts }, 409);
    }
    return c.json(result.data, 201);
  } catch (error) { return handleUnexpectedError(c, error, "批量创建教师失败"); }
});

usersController.post("/api/v1/users/assistants/batch", jwtAuth, async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (payload.role !== "Admin") return c.json({ error: "权限不足" }, 403);
    const body = await c.req.parseBody();
    const file = body["file"];
    const password = body["password"];
    if (!(file instanceof File)) return c.json({ error: "CSV 文件缺失或类型错误" }, 400);
    const result = await batchCreateAssistants(file, password);
    if ("error" in result) {
      if (result.error === "invalid_password_type") return c.json({ error: "密码必须是字符串类型" }, 400);
      if (result.error === "password_too_short") return c.json({ error: "统一密码必须至少6位" }, 400);
      if (result.error === "invalid_file_type") return c.json({ error: "仅支持上传 CSV 文件" }, 400);
      if (result.error === "file_too_large") return c.json({ error: "CSV 文件大小不能超过 5MB" }, 413);
      if (result.error === "invalid_rows") return c.json({ error: "CSV 文件包含无效记录", invalid_rows: result.invalid_rows }, 400);
      if (result.error === "no_valid_records") return c.json({ error: "CSV 文件未包含有效的助教记录" }, 400);
      if (result.error === "conflicts") return c.json({ error: "部分助教信息已存在，批量创建已取消", conflicts: result.conflicts }, 409);
    }
    return c.json(result.data, 201);
  } catch (error) { return handleUnexpectedError(c, error, "批量创建助教失败"); }
});

usersController.get(
  "/api/v1/teachers/:teacherId/assistants",
  jwtAuth,
  zValidator("param", teacherIdParamSchema),
  zValidator("query", academicTermQuerySchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { teacherId } = c.req.valid("param");
      const termFilter = c.req.valid("query");
      const teacherIdNumber = parseInt(teacherId, 10);
      if (payload.role !== "Admin" && payload.sub !== teacherIdNumber) return c.json({ error: "权限不足" }, 403);
      const result = await getTeacherAssistants(teacherIdNumber, termFilter);
      if ("error" in result) return c.json({ error: "目标用户不是教师" }, 404);
      return c.json(result.data);
    } catch (error) { return handleUnexpectedError(c, error, "获取教师的助教失败"); }
  },
);

usersController.get("/api/v1/users", jwtAuth, zValidator("query", paginationQuerySchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (payload.role !== "Admin") return c.json({ error: "权限不足" }, 403);
    const query = c.req.valid("query");
    const result = await getAllUsers(query);
    return c.json(result);
  } catch (error) { return handleUnexpectedError(c, error, "获取用户列表失败"); }
});

usersController.get("/api/v1/assistants", jwtAuth, async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (payload.role !== "Admin" && payload.role !== "Teacher") return c.json({ error: "权限不足" }, 403);
    const assistants = await getAllAssistants();
    return c.json(assistants);
  } catch (error) { return handleUnexpectedError(c, error, "获取助教列表失败"); }
});

usersController.get("/api/v1/students/unenrolled", jwtAuth, async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (!["Admin", "Teacher", "Assistant"].includes(payload.role)) return c.json({ error: "权限不足" }, 403);
    const students = await getUnenrolledStudents();
    return c.json(students);
  } catch (error) { return handleUnexpectedError(c, error, "获取未选课学生失败"); }
});

usersController.get("/api/v1/users/me", jwtAuth, async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const result = await getCurrentUser(payload.sub);
    if ("error" in result) return c.json({ error: "用户不存在" }, 404);
    return c.json(result.data);
  } catch (error) { return handleUnexpectedError(c, error, "获取当前用户信息失败"); }
});

usersController.put("/api/v1/users/me", jwtAuth, zValidator("json", updateUserSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const input = c.req.valid("json");
    const result = await updateCurrentUser(payload.sub, input);
    if ("error" in result) {
      if (result.error === "dup_username") return c.json({ error: "用户名已存在" }, 409);
      if (result.error === "dup_email") return c.json({ error: "邮箱已存在" }, 409);
      if (result.error === "dup_phone") return c.json({ error: "手机号已存在" }, 409);
    }
    return c.json(result.data);
  } catch (error) { return handleUnexpectedError(c, error, "更新用户信息失败"); }
});

usersController.put("/api/v1/users/me/password", jwtAuth, zValidator("json", changePasswordSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { currentPassword, newPassword } = c.req.valid("json");
    const result = await changePassword(payload.sub, currentPassword, newPassword);
    if ("error" in result) {
      if (result.error === "not_found") return c.json({ error: "用户不存在" }, 404);
      if (result.error === "wrong_password") return c.json({ error: "当前密码错误" }, 401);
    }
    return c.json({ message: "密码更新成功" });
  } catch (error) { return handleUnexpectedError(c, error, "修改密码失败"); }
});

usersController.get("/api/v1/users/:userId", jwtAuth, zValidator("param", userIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { userId } = c.req.valid("param");
    const targetUserId = parseInt(userId, 10);
    if (payload.role !== "Admin" && payload.sub !== targetUserId) return c.json({ error: "权限不足" }, 403);
    const result = await getUserById(targetUserId);
    if ("error" in result) return c.json({ error: "用户不存在" }, 404);
    return c.json(result.data);
  } catch (error) { return handleUnexpectedError(c, error, "获取用户信息失败"); }
});

usersController.put("/api/v1/users/:userId/password", jwtAuth, zValidator("param", userIdParamSchema), zValidator("json", adminChangePasswordSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (payload.role !== "Admin") return c.json({ error: "权限不足" }, 403);
    const { userId } = c.req.valid("param");
    const targetUserId = parseInt(userId, 10);
    if (payload.sub === targetUserId) return c.json({ error: "管理员应使用 /users/me/password 路由修改自己的密码" }, 400);
    const { newPassword } = c.req.valid("json");
    const result = await adminChangePassword(targetUserId, newPassword);
    if ("error" in result) return c.json({ error: "用户不存在" }, 404);
    return c.json({ message: "用户密码更新成功" });
  } catch (error) { return handleUnexpectedError(c, error, "强制修改密码失败"); }
});

usersController.post("/api/v1/users/:userId/reset-password", jwtAuth, zValidator("param", userIdParamSchema), zValidator("json", resetPasswordSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (!["Admin", "Teacher", "Assistant"].includes(payload.role)) return c.json({ error: "权限不足" }, 403);
    const { userId } = c.req.valid("param");
    const targetUserId = parseInt(userId, 10);
    const input = c.req.valid("json");
    const result = await resetStudentPassword(targetUserId, payload.sub, payload.role, input);
    if ("error" in result) {
      if (result.error === "not_found") return c.json({ error: "用户不存在" }, 404);
      if (result.error === "not_student") return c.json({ error: "只能重置学生的密码" }, 400);
      if (result.error === "not_your_student") return c.json({ error: "权限不足：该学生不在您的班级中" }, 403);
      if (result.error === "not_assigned_student") return c.json({ error: "权限不足：该学生未分配给您的班级" }, 403);
    }
    const message = result.isCustomPassword ? "密码更新成功" : "密码已成功重置为默认密码";
    return c.json({ message });
  } catch (error) { return handleUnexpectedError(c, error, "重置密码失败"); }
});

usersController.delete("/api/v1/users/:userId", jwtAuth, zValidator("param", userIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (payload.role !== "Admin") return c.json({ error: "权限不足" }, 403);
    const { userId } = c.req.valid("param");
    const targetUserId = parseInt(userId, 10);
    const result = await deleteUser(targetUserId, payload.sub);
    if ("error" in result) {
      if (result.error === "self_delete") return c.json({ error: "管理员不能删除自己" }, 400);
      if (result.error === "not_found") return c.json({ error: "用户不存在" }, 404);
    }
    return c.json({ message: "用户删除成功", cleanup: result.cleanup });
  } catch (error) { return handleUnexpectedError(c, error, "删除用户失败"); }
});

usersController.delete("/api/v1/users", jwtAuth, zValidator("json", bulkDeleteUsersSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (payload.role !== "Admin") return c.json({ error: "权限不足" }, 403);
    const { userIds } = c.req.valid("json");
    const result = await batchDeleteUsers(userIds, payload.sub);
    if ("error" in result) {
      if (result.error === "self_delete") return c.json({ error: "批量删除中不能包含当前管理员账户" }, 400);
      if (result.error === "not_found") return c.json({ error: "部分用户不存在", missingUserIds: result.missingUserIds }, 404);
    }
    return c.json({ message: "批量删除用户成功", deletedCount: result.deletedCount, cleanup: result.cleanup });
  } catch (error) { return handleUnexpectedError(c, error, "批量删除用户失败"); }
});

export default usersController;
