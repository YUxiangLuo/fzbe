import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireStudent } from "../../middleware";
import {
  classIdParamSchema, classAssistantParamSchema, classStudentParamSchema,
  createStudentSchema, addAssistantSchema, joinClassSchema, enrollStudentSchema,
} from "./class-members.schema";
import {
  createStudentInClass, enrollExistingStudent, getClassStudents, removeStudentFromClass,
  studentJoinClass, getJoinableClasses, addAssistantToClass, getClassAssistants, removeAssistantFromClass,
} from "./class-members.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import type { JwtPayload } from "../../types";

const classMembersController = new Hono();

classMembersController.post(
  "/api/v1/classes/:classId/students",
  jwtAuth,
  zValidator("param", classIdParamSchema),
  zValidator("json", createStudentSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      if (!["Admin", "Teacher", "Assistant"].includes(payload.role)) return c.json({ error: "权限不足" }, 403);

      const { classId } = c.req.valid("param");
      const input = c.req.valid("json");
      const result = await createStudentInClass(parseInt(classId, 10), payload.sub, payload.role, input);

      if ("error" in result) {
        if (result.error === "not_found") return c.json({ error: "班级不存在" }, 404);
        if (result.error === "forbidden") return c.json({ error: "权限不足" }, 403);
        if (result.error === "dup_username") return c.json({ error: "学号已存在" }, 409);
        if (result.error === "dup_email") return c.json({ error: "邮箱已存在" }, 409);
        return c.json({ error: "学生创建冲突" }, 409);
      }
      return c.json(result.data, 201);
    } catch (error) {
      return handleUnexpectedError(c, error, "创建学生并加入班级失败");
    }
  },
);

classMembersController.post(
  "/api/v1/classes/:classId/enroll",
  jwtAuth,
  zValidator("param", classIdParamSchema),
  zValidator("json", enrollStudentSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      if (!["Teacher", "Assistant", "Admin"].includes(payload.role)) return c.json({ error: "权限不足" }, 403);

      const { classId } = c.req.valid("param");
      const { student_id } = c.req.valid("json");
      const result = await enrollExistingStudent(parseInt(classId, 10), student_id, payload.sub, payload.role);

      if ("error" in result) {
        if (result.error === "class_not_found") return c.json({ error: "班级不存在" }, 404);
        if (result.error === "forbidden") return c.json({ error: "权限不足：您只能将学生添加到您自己的班级" }, 403);
        if (result.error === "student_not_found") return c.json({ error: "学生不存在" }, 404);
        if (result.error === "not_student") return c.json({ error: "用户不是学生" }, 400);
        if (result.error === "already_enrolled") {
          return c.json({ error: "学生已加入班级", message: "该学生已加入其他班级", current_class: result.currentClass }, 409);
        }
        return c.json({ error: "学生已加入该班级" }, 409);
      }
      return c.json({ message: "Student enrolled successfully", student: { user_id: result.data.user_id, full_name: result.data.full_name }, class_id: result.data.class_id }, 201);
    } catch (error) {
      return handleUnexpectedError(c, error, "添加学生失败");
    }
  },
);

classMembersController.get(
  "/api/v1/classes/:classId/students",
  jwtAuth,
  zValidator("param", classIdParamSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { classId } = c.req.valid("param");
      const result = await getClassStudents(parseInt(classId, 10), payload.sub, payload.role);

      if ("error" in result) {
        if (result.error === "not_found") return c.json({ error: "班级不存在" }, 404);
        return c.json({ error: "权限不足" }, 403);
      }
      return c.json(result.data);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取班级学生失败");
    }
  },
);

classMembersController.delete(
  "/api/v1/classes/:classId/students/:studentId",
  jwtAuth,
  zValidator("param", classStudentParamSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { classId, studentId } = c.req.valid("param");
      const result = await removeStudentFromClass(parseInt(classId, 10), parseInt(studentId, 10), payload.sub, payload.role);

      if ("error" in result) {
        if (result.error === "class_not_found") return c.json({ error: "班级不存在" }, 404);
        if (result.error === "forbidden") return c.json({ error: "权限不足" }, 403);
        if (result.error === "user_not_found") return c.json({ error: "用户不存在" }, 404);
        if (result.error === "not_student") return c.json({ error: "用户不是学生" }, 400);
        return c.json({ error: "未找到加入记录" }, 404);
      }
      return c.json({ message: "学生已从班级移除" });
    } catch (error) {
      return handleUnexpectedError(c, error, "从班级移除学生失败");
    }
  },
);

classMembersController.post(
  "/api/v1/classes/:classId/join",
  jwtAuth,
  requireStudent,
  zValidator("param", classIdParamSchema),
  zValidator("json", joinClassSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { classId } = c.req.valid("param");
      const { class_code } = c.req.valid("json");
      const result = await studentJoinClass(parseInt(classId, 10), payload.sub, class_code);

      if ("error" in result) {
        if (result.error === "already_enrolled") {
          return c.json({ error: "已加入班级", message: "您已加入其他班级。请先退出当前班级再加入新班级。", current_class: result.currentClass }, 409);
        }
        if (result.error === "class_not_found") return c.json({ error: "班级不存在" }, 404);
        if (result.error === "code_required") return c.json({ error: "该班级需要班级代码" }, 400);
        if (result.error === "invalid_code") return c.json({ error: "班级代码无效" }, 403);
      }
      return c.json({ message: "成功加入班级", class: result.data }, 201);
    } catch (error) {
      return handleUnexpectedError(c, error, "加入班级失败");
    }
  },
);

classMembersController.get("/api/v1/classes/joinable", jwtAuth, requireStudent, async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const result = await getJoinableClasses(payload.sub);

    if ("error" in result) {
      return c.json({ error: "已加入班级", message: "您已加入班级。请先退出再加入新班级。", current_class_id: result.currentClassId }, 409);
    }
    return c.json(result.data);
  } catch (error) {
    return handleUnexpectedError(c, error, "加载可加入班级列表失败");
  }
});

classMembersController.post(
  "/api/v1/classes/:classId/assistants",
  jwtAuth,
  zValidator("param", classIdParamSchema),
  zValidator("json", addAssistantSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { classId } = c.req.valid("param");
      const { assistant_id } = c.req.valid("json");
      const result = await addAssistantToClass(parseInt(classId, 10), assistant_id, payload.sub, payload.role);

      if ("error" in result) {
        if (result.error === "class_not_found") return c.json({ error: "班级不存在" }, 404);
        if (result.error === "forbidden") return c.json({ error: "权限不足" }, 403);
        if (result.error === "not_assistant") return c.json({ error: "用户不是助教" }, 400);
        return c.json({ error: "助教已分配给该班级" }, 409);
      }
      return c.json({ message: "助教添加成功" }, 201);
    } catch (error) {
      return handleUnexpectedError(c, error, "添加助教失败");
    }
  },
);

classMembersController.get(
  "/api/v1/classes/:classId/assistants",
  jwtAuth,
  zValidator("param", classIdParamSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { classId } = c.req.valid("param");
      const result = await getClassAssistants(parseInt(classId, 10), payload.sub, payload.role);

      if ("error" in result) {
        if (result.error === "not_found") return c.json({ error: "班级不存在" }, 404);
        return c.json({ error: "权限不足" }, 403);
      }
      return c.json(result.data);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取班级助教失败");
    }
  },
);

classMembersController.delete(
  "/api/v1/classes/:classId/assistants/:assistantId",
  jwtAuth,
  zValidator("param", classAssistantParamSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { classId, assistantId } = c.req.valid("param");
      const result = await removeAssistantFromClass(parseInt(classId, 10), parseInt(assistantId, 10), payload.sub, payload.role);

      if ("error" in result) {
        if (result.error === "class_not_found") return c.json({ error: "班级不存在" }, 404);
        if (result.error === "forbidden") return c.json({ error: "权限不足" }, 403);
        return c.json({ error: "未找到分配记录" }, 404);
      }
      return c.json({ message: "助教移除成功" });
    } catch (error) {
      return handleUnexpectedError(c, error, "移除助教失败");
    }
  },
);

export default classMembersController;
