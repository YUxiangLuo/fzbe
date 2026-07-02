import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireTeacher, requireAdmin } from "../../middleware";
import {
  academicTermQuerySchema,
  classIdParamSchema,
  teacherIdParamSchema,
  assistantIdParamSchema,
  teacherAssistantParamSchema,
  createClassSchema,
  updateClassSchema,
} from "./classes.schema";
import {
  createClass,
  getAllClasses,
  getClassById,
  updateClass,
  deleteClass,
  getClassesByTeacher,
  getClassesByTeacherAssistant,
  getClassesByAssistant,
  FileValidationError,
} from "./classes.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import type { JwtPayload } from "../../types";

const classesController = new Hono();

const normalizeOptionalBodyValue = (value: unknown) => {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

classesController.post("/api/v1/classes", jwtAuth, requireTeacher, async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const body = await c.req.parseBody();
    const className = body["class_name"];
    const classCode = body["class_code"];
    // Backward-compatible field support: FE currently uses `students_file`.
    const studentListFile = body["student_list"] ?? body["students_file"];

    if (!className || typeof className !== "string") {
      return c.json({ error: "班级名称不能为空" }, 400);
    }

    const validation = createClassSchema.safeParse({
      class_name: typeof className === "string" ? className : undefined,
      class_code: typeof classCode === "string" ? classCode : undefined,
      academic_year_start: normalizeOptionalBodyValue(body["academic_year_start"]),
      semester: normalizeOptionalBodyValue(body["semester"]),
    });
    if (!validation.success) {
      return c.json({ error: validation.error.issues[0]?.message ?? "班级参数无效" }, 400);
    }

    const input = validation.data;
    const result = await createClass(
      input.class_name,
      input.class_code ?? null,
      payload.sub,
      studentListFile instanceof File ? studentListFile : undefined,
      input.academic_year_start,
      input.semester,
    );

    if ("error" in result) {
      const response: Record<string, unknown> = { error: result.error };
      if (result.details) Object.assign(response, result.details);
      return c.json(response, result.status);
    }

    return c.json(result.data, 201);
  } catch (error) {
    if (error instanceof FileValidationError) {
      return c.json({ error: error.message }, 400);
    }
    return handleUnexpectedError(c, error, "创建班级失败");
  }
});

classesController.get("/api/v1/classes", jwtAuth, requireAdmin, async (c) => {
  try {
    const classes = await getAllClasses();
    return c.json(classes);
  } catch (error) {
    return handleUnexpectedError(c, error, "获取班级列表失败");
  }
});

classesController.get(
  "/api/v1/classes/:classId",
  jwtAuth,
  zValidator("param", classIdParamSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { classId } = c.req.valid("param");
      const result = await getClassById(parseInt(classId, 10), payload.sub, payload.role);

      if ("error" in result) {
        if (result.error === "not_found") return c.json({ error: "班级不存在" }, 404);
        return c.json({ error: "权限不足" }, 403);
      }

      return c.json(result.data);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取班级详情失败");
    }
  },
);

classesController.put(
  "/api/v1/classes/:classId",
  jwtAuth,
  zValidator("param", classIdParamSchema),
  zValidator("json", updateClassSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      if (payload.role !== "Admin" && payload.role !== "Teacher") {
        return c.json({ error: "权限不足" }, 403);
      }
      const { classId } = c.req.valid("param");
      const input = c.req.valid("json");
      const result = await updateClass(parseInt(classId, 10), payload.sub, payload.role, input);

      if ("error" in result) {
        if (result.error === "not_found") return c.json({ error: "班级不存在" }, 404);
        if (result.error === "forbidden") return c.json({ error: "权限不足" }, 403);
        if (result.error === "duplicate") return c.json({ error: "班级代码已存在" }, 409);
        return c.json({ error: "没有需要更新的字段" }, 400);
      }

      return c.json(result.data);
    } catch (error) {
      return handleUnexpectedError(c, error, "更新班级失败");
    }
  },
);

classesController.delete(
  "/api/v1/classes/:classId",
  jwtAuth,
  zValidator("param", classIdParamSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { classId } = c.req.valid("param");
      const result = await deleteClass(parseInt(classId, 10), payload.sub, payload.role);

      if ("error" in result) {
        if (result.error === "not_found") return c.json({ error: "班级不存在" }, 404);
        return c.json({ error: "权限不足" }, 403);
      }

      return c.json(result.data);
    } catch (error) {
      return handleUnexpectedError(c, error, "删除班级失败");
    }
  },
);

classesController.get(
  "/api/v1/teachers/:teacherId/classes",
  jwtAuth,
  zValidator("param", teacherIdParamSchema),
  zValidator("query", academicTermQuerySchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { teacherId } = c.req.valid("param");
      const termFilter = c.req.valid("query");
      const result = await getClassesByTeacher(parseInt(teacherId, 10), payload.sub, payload.role, termFilter);

      if ("error" in result) {
        if (result.error === "forbidden") return c.json({ error: "权限不足" }, 403);
        return c.json({ error: "目标用户不是教师" }, 404);
      }

      return c.json(result.data);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取教师班级失败");
    }
  },
);

classesController.get(
  "/api/v1/teachers/:teacherId/assistants/:assistantId/classes",
  jwtAuth,
  zValidator("param", teacherAssistantParamSchema),
  zValidator("query", academicTermQuerySchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { teacherId, assistantId } = c.req.valid("param");
      const termFilter = c.req.valid("query");
      const result = await getClassesByTeacherAssistant(
        parseInt(teacherId, 10),
        parseInt(assistantId, 10),
        payload.sub,
        payload.role,
        termFilter,
      );

      if ("error" in result) {
        if (result.error === "forbidden") return c.json({ error: "权限不足" }, 403);
        if (result.error === "not_teacher") return c.json({ error: "目标用户不是教师" }, 404);
        return c.json({ error: "目标用户不是助教" }, 404);
      }

      return c.json(result.data);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取教师助教班级失败");
    }
  },
);

classesController.get(
  "/api/v1/assistants/:assistantId/classes",
  jwtAuth,
  zValidator("param", assistantIdParamSchema),
  zValidator("query", academicTermQuerySchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const { assistantId } = c.req.valid("param");
      const termFilter = c.req.valid("query");
      const result = await getClassesByAssistant(parseInt(assistantId, 10), payload.sub, payload.role, termFilter);

      if ("error" in result) {
        if (result.error === "forbidden") return c.json({ error: "权限不足" }, 403);
        return c.json({ error: "目标用户不是助教" }, 404);
      }

      return c.json(result.data);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取助教班级失败");
    }
  },
);

export default classesController;
