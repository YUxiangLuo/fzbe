import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireAdmin } from "../../middleware";
import { manualIdParamSchema, manualMetadataSchema, updateManualSchema } from "./manuals.schema";
import {
  uploadManual,
  getAllManuals,
  getActiveManual,
  updateManual,
  deleteManual,
  FileValidationError,
} from "./manuals.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import type { JwtPayload } from "../../types";

const manualsController = new Hono();

manualsController.post(
  "/api/v1/manuals",
  jwtAuth,
  requireAdmin,
  async (c) => {
    try {
      const payload = c.get("jwtPayload") as JwtPayload;
      const body = await c.req.parseBody();
      const file = body["file"];
      const fileName = body["file_name"];
      const description = body["description"];

      if (!(file instanceof File)) {
        return c.json({ error: "必须上传文件" }, 400);
      }
      if (!fileName || typeof fileName !== "string") {
        return c.json({ error: "必须提供文件名" }, 400);
      }
      if (description !== undefined && typeof description !== "string") {
        return c.json({ error: "备注格式不正确" }, 400);
      }

      const metadata = manualMetadataSchema.safeParse({
        file_name: fileName,
        description,
      });
      if (!metadata.success) {
        return c.json({ error: metadata.error.issues[0]?.message ?? "手册信息格式不正确" }, 400);
      }

      const result = await uploadManual({
        file,
        fileName: metadata.data.file_name,
        description: metadata.data.description,
        uploaderId: payload.sub,
      });

      return c.json(result, 201);
    } catch (error) {
      if (error instanceof FileValidationError) {
        return c.json({ error: error.message }, 400);
      }
      return handleUnexpectedError(c, error, "上传手册失败");
    }
  },
);

manualsController.get(
  "/api/v1/manuals",
  jwtAuth,
  requireAdmin,
  async (c) => {
    try {
      const manuals = await getAllManuals();
      return c.json(manuals);
    } catch (error) {
      return handleUnexpectedError(c, error, "获取手册列表失败");
    }
  },
);

manualsController.get("/api/v1/manuals/active", jwtAuth, async (c) => {
  try {
    const manual = await getActiveManual();
    if (!manual) {
      return c.json({ error: "未找到激活的手册" }, 404);
    }
    return c.json(manual);
  } catch (error) {
    return handleUnexpectedError(c, error, "获取激活手册失败");
  }
});

manualsController.put(
  "/api/v1/manuals/:manualId",
  jwtAuth,
  requireAdmin,
  zValidator("param", manualIdParamSchema),
  zValidator("json", updateManualSchema),
  async (c) => {
    try {
      const { manualId } = c.req.valid("param");
      const input = c.req.valid("json");

      const result = await updateManual(parseInt(manualId, 10), input);

      if ("error" in result) {
        if (result.error === "not_found") {
          return c.json({ error: "手册不存在" }, 404);
        }
        return c.json({ error: result.error }, 400);
      }

      return c.json(result.data);
    } catch (error) {
      return handleUnexpectedError(c, error, "更新手册失败");
    }
  },
);

manualsController.delete(
  "/api/v1/manuals/:manualId",
  jwtAuth,
  requireAdmin,
  zValidator("param", manualIdParamSchema),
  async (c) => {
    try {
      const { manualId } = c.req.valid("param");
      const result = await deleteManual(parseInt(manualId, 10));

      if ("error" in result) {
        if (result.error === "not_found") {
          return c.json({ error: "手册不存在" }, 404);
        }
        return c.json({ error: "手册文件无法安全清理，手册未删除" }, 500);
      }

      return c.json({
        message: result.cleanupPending
          ? "手册已删除；隔离文件将在后续清理"
          : "手册及文件删除成功",
        cleanup_pending: result.cleanupPending,
      });
    } catch (error) {
      return handleUnexpectedError(c, error, "删除手册失败");
    }
  },
);

export default manualsController;
