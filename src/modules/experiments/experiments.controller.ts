import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireStudent } from "../../middleware";
import { experimentIdParamSchema, createExperimentSchema, retryExperimentSchema, updateExperimentSchema, experimentEventSchema } from "./experiments.schema";
import { getActiveExperiment, createExperiment, retryRejectedExperiment, getAllExperiments, deleteExperiment, updateExperiment, createExperimentEvent } from "./experiments.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import type { JwtPayload } from "../../types";

const experimentsController = new Hono();

experimentsController.get("/api/v1/students/me/experiment-runs/active", jwtAuth, requireStudent, async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const experiment = await getActiveExperiment(payload.sub);
    if (!experiment) return c.json({ error: "未找到实验状态" }, 404);
    return c.json(experiment);
  } catch (error) {
    return handleUnexpectedError(c, error, "获取实验状态失败");
  }
});

experimentsController.post("/api/v1/students/me/experiment-runs", jwtAuth, requireStudent, zValidator("json", createExperimentSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const data = c.req.valid("json");
    const result = await createExperiment(payload.sub, data);
    if ("error" in result) {
      if (result.error === "student_not_found") {
        return c.json({ error: "学生不存在", code: "STUDENT_NOT_FOUND" }, 404);
      }
      return c.json({
        error: "已经存在正式实验；被驳回后请使用重做入口",
        code: "EXPERIMENT_ALREADY_EXISTS",
        current: result.data,
      }, 409);
    }
    return c.json(result.data, 201);
  } catch (error) {
    return handleUnexpectedError(c, error, "创建实验状态失败");
  }
});

experimentsController.post("/api/v1/students/me/experiment-runs/retry", jwtAuth, requireStudent, zValidator("json", retryExperimentSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const input = c.req.valid("json");
    const result = await retryRejectedExperiment(payload.sub, input);

    if ("error" in result) {
      if (result.error === "student_not_found") {
        return c.json({ error: "学生不存在", code: "STUDENT_NOT_FOUND" }, 404);
      }
      if (result.error === "report_not_found") {
        return c.json({ error: "未找到属于您的驳回报告", code: "REJECTED_REPORT_NOT_FOUND" }, 404);
      }
      if (result.error === "report_not_rejected") {
        return c.json({ error: "只有已驳回报告可以发起重做", code: "REPORT_NOT_REJECTED" }, 409);
      }
      if (result.error === "parent_not_completed") {
        return c.json({ error: "被驳回报告对应的实验尚未完成", code: "REJECTED_EXPERIMENT_NOT_COMPLETED" }, 409);
      }
      if (result.error === "report_not_latest") {
        return c.json({
          error: "该驳回报告不是当前正式实验",
          code: "REJECTED_REPORT_NOT_LATEST",
          current: result.data,
        }, 409);
      }
      return c.json({ error: "重做实验关联异常", code: "INVALID_RETRY_LINEAGE" }, 409);
    }

    return result.created
      ? c.json(result.data, 201)
      : c.json(result.data, 200);
  } catch (error) {
    return handleUnexpectedError(c, error, "创建重做实验失败");
  }
});

experimentsController.get("/api/v1/students/me/experiment-runs", jwtAuth, requireStudent, async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const experiments = await getAllExperiments(payload.sub);
    return c.json(experiments);
  } catch (error) {
    return handleUnexpectedError(c, error, "加载实验列表失败");
  }
});

experimentsController.delete("/api/v1/students/me/experiment-runs/:experimentId", jwtAuth, requireStudent, zValidator("param", experimentIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { experimentId } = c.req.valid("param");
    const result = await deleteExperiment(parseInt(experimentId, 10), payload.sub);

    if ("error" in result) {
      if (result.error === "not_found") return c.json({ error: "实验不存在" }, 404);
      return c.json({ error: "已完成的实验无法删除" }, 400);
    }
    return c.json({ message: "实验删除成功" });
  } catch (error) {
    return handleUnexpectedError(c, error, "删除实验失败");
  }
});

experimentsController.on(["PUT", "PATCH"], "/api/v1/experiment-runs/:experimentId", jwtAuth, requireStudent, zValidator("param", experimentIdParamSchema), zValidator("json", updateExperimentSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { experimentId } = c.req.valid("param");
    const body = c.req.valid("json");
    const result = await updateExperiment(parseInt(experimentId, 10), payload.sub, body);

    if ("error" in result) {
      if (result.error === "not_found") return c.json({ error: "未找到实验状态" }, 404);
      if (result.error === "forbidden") return c.json({ error: "权限不足" }, 403);
      if (result.error === "completed") return c.json({ error: "已完成的实验不允许继续修改" }, 400);
      if (result.error === "conflict") {
        return c.json({ error: "实验状态已被其他页面更新，请基于最新状态重试", current: result.data }, 409);
      }
      if (result.error === "invalid_best_model") {
        return c.json({ error: "只能选择已完成训练且仍然有效的模型" }, 400);
      }
      if (result.error === "invalid_production_plan") {
        return c.json({
          error: `生产计划数据无效：${result.message}`,
          code: result.code,
        }, 400);
      }
      if (result.error === "no_fields") return c.json({ error: "未提供更新字段" }, 400);
      if (result.error === "invalid_current_step") return c.json({ error: "current_step 不能为空" }, 400);
      return c.json({ error: "highest_completed_step 不能为空" }, 400);
    }
    return c.json(result.data);
  } catch (error) {
    return handleUnexpectedError(c, error, "更新实验状态失败");
  }
});

experimentsController.post("/api/v1/experiment-runs/:experimentId/events", jwtAuth, requireStudent, zValidator("param", experimentIdParamSchema), zValidator("json", experimentEventSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { experimentId } = c.req.valid("param");
    const input = c.req.valid("json");
    const result = await createExperimentEvent(parseInt(experimentId, 10), payload.sub, input);

    if ("error" in result) {
      if (result.error === "not_found") return c.json({ error: "实验不存在" }, 404);
      if (result.error === "completed") return c.json({ error: "已完成的实验不允许继续写入步骤事件" }, 400);
      return c.json({ error: "权限不足：您不是该实验的所有者" }, 403);
    }
    return c.json({ message: "事件创建成功", event_id: result.data.event_id }, 201);
  } catch (error) {
    return handleUnexpectedError(c, error, "创建实验步骤事件失败");
  }
});

export default experimentsController;
