import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireAdmin } from "../../middleware";
import { handleUnexpectedError } from "../../utils/error-handler";
import {
  academicTermSchema,
  termIdParamSchema,
  updateAcademicTermSchema,
} from "./academic-terms.schema";
import {
  createAcademicTerm,
  deleteAcademicTerm,
  getCurrentAcademicTerm,
  listAcademicTerms,
  updateAcademicTerm,
} from "./academic-terms.service";

const academicTermsController = new Hono();

academicTermsController.get("/api/v1/academic-terms", jwtAuth, async (c) => {
  try {
    const terms = await listAcademicTerms();
    return c.json(terms);
  } catch (error) {
    return handleUnexpectedError(c, error, "获取学年学期列表失败");
  }
});

academicTermsController.get("/api/v1/academic-terms/current", jwtAuth, async (c) => {
  try {
    const term = await getCurrentAcademicTerm();
    if (!term) return c.json({ error: "当前学期未设置" }, 404);
    return c.json(term);
  } catch (error) {
    return handleUnexpectedError(c, error, "获取当前学期失败");
  }
});

academicTermsController.post(
  "/api/v1/academic-terms",
  jwtAuth,
  requireAdmin,
  zValidator("json", academicTermSchema),
  async (c) => {
    try {
      const input = c.req.valid("json");
      const result = await createAcademicTerm(input);
      if ("error" in result) return c.json({ error: "该学年学期已存在" }, 409);
      return c.json(result.data, 201);
    } catch (error) {
      return handleUnexpectedError(c, error, "创建学年学期失败");
    }
  },
);

academicTermsController.put(
  "/api/v1/academic-terms/:termId",
  jwtAuth,
  requireAdmin,
  zValidator("param", termIdParamSchema),
  zValidator("json", updateAcademicTermSchema),
  async (c) => {
    try {
      const { termId } = c.req.valid("param");
      const input = c.req.valid("json");
      const result = await updateAcademicTerm(parseInt(termId, 10), input);

      if ("error" in result) {
        if (result.error === "not_found") return c.json({ error: "学期不存在" }, 404);
        if (result.error === "cannot_deactivate_current") {
          return c.json({ error: "不能直接取消当前学期，请将其他学期设为当前学期" }, 400);
        }
        return c.json({ error: "该学年学期已存在" }, 409);
      }

      return c.json(result.data);
    } catch (error) {
      return handleUnexpectedError(c, error, "更新学年学期失败");
    }
  },
);

academicTermsController.delete(
  "/api/v1/academic-terms/:termId",
  jwtAuth,
  requireAdmin,
  zValidator("param", termIdParamSchema),
  async (c) => {
    try {
      const { termId } = c.req.valid("param");
      const result = await deleteAcademicTerm(parseInt(termId, 10));

      if ("error" in result) {
        if (result.error === "not_found") return c.json({ error: "学期不存在" }, 404);
        if (result.error === "active_term") return c.json({ error: "不能删除当前学期" }, 400);
        return c.json({ error: "该学期下已有班级，不能删除" }, 409);
      }

      return c.json(result.data);
    } catch (error) {
      return handleUnexpectedError(c, error, "删除学年学期失败");
    }
  },
);

export default academicTermsController;
