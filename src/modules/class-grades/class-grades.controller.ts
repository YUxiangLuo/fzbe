import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth } from "../../middleware";
import { classIdParamSchema, teacherIdParamSchema, assistantIdParamSchema, reportIdParamSchema, gradeReportSchema, updateGradeWeightsSchema } from "./class-grades.schema";
import { checkClassAccess, getClassReports, gradeReport, getClassExperimentRuns, getGradingPolicy, updateGradingPolicy, getGradeSummaries, getTeacherClassGradeSummaries, getAssistantClassGradeSummaries, exportReportArchive, exportGradeCsv, exportReportCsv, getExperimentEvents } from "./class-grades.service";
import { handleUnexpectedError } from "../../utils/error-handler";
import { resolveAcademicTermId } from "../academic-terms";
import type { JwtPayload } from "../../types";

const classGradesController = new Hono();

async function parseTermFilter(rawTermId: string | undefined | null) {
  const result = await resolveAcademicTermId(rawTermId);
  if ("error" in result) return result;
  return { termId: result.termId };
}

classGradesController.get("/api/v1/classes/:classId/reports", jwtAuth, zValidator("param", classIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { classId } = c.req.valid("param");
    const access = await checkClassAccess(parseInt(classId, 10), payload.sub, payload.role);
    if ("error" in access) return c.json({ error: access.error === "not_found" ? "班级不存在" : "权限不足" }, access.error === "not_found" ? 404 : 403);
    const reports = await getClassReports(parseInt(classId, 10));
    return c.json(reports);
  } catch (error) { return handleUnexpectedError(c, error, "获取班级报告失败"); }
});

classGradesController.put("/api/v1/reports/:reportId", jwtAuth, zValidator("param", reportIdParamSchema), zValidator("json", gradeReportSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (!["Teacher", "Admin", "Assistant"].includes(payload.role)) return c.json({ error: "权限不足" }, 403);
    const { reportId } = c.req.valid("param");
    const input = c.req.valid("json");
    const result = await gradeReport(parseInt(reportId, 10), payload.sub, payload.role, input);
    if ("error" in result) {
      if (result.error === "not_found") return c.json({ error: "报告不存在" }, 404);
      if (result.error === "forbidden") return c.json({ error: "权限不足" }, 403);
      return c.json({ error: "成绩记录不存在" }, 404);
    }
    return c.json(result.data);
  } catch (error) { return handleUnexpectedError(c, error, "批改报告失败"); }
});

classGradesController.get("/api/v1/classes/:classId/experiment-runs", jwtAuth, zValidator("param", classIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { classId } = c.req.valid("param");
    const access = await checkClassAccess(parseInt(classId, 10), payload.sub, payload.role);
    if ("error" in access) return c.json({ error: access.error === "not_found" ? "班级不存在" : "权限不足" }, access.error === "not_found" ? 404 : 403);
    const experiments = await getClassExperimentRuns(parseInt(classId, 10));
    return c.json(experiments);
  } catch (error) { return handleUnexpectedError(c, error, "获取实验状态失败"); }
});

classGradesController.get("/api/v1/classes/:classId/grading-policy", jwtAuth, zValidator("param", classIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { classId } = c.req.valid("param");
    const access = await checkClassAccess(parseInt(classId, 10), payload.sub, payload.role);
    if ("error" in access) return c.json({ error: access.error === "not_found" ? "班级不存在" : "权限不足" }, access.error === "not_found" ? 404 : 403);
    const weights = await getGradingPolicy(parseInt(classId, 10));
    if (!weights) return c.json({ error: "未找到该班级的成绩权重" }, 404);
    return c.json(weights);
  } catch (error) { return handleUnexpectedError(c, error, "获取成绩权重失败"); }
});

classGradesController.put("/api/v1/classes/:classId/grading-policy", jwtAuth, zValidator("param", classIdParamSchema), zValidator("json", updateGradeWeightsSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    if (!["Teacher", "Admin", "Assistant"].includes(payload.role)) return c.json({ error: "权限不足" }, 403);
    const { classId } = c.req.valid("param");
    const access = await checkClassAccess(parseInt(classId, 10), payload.sub, payload.role);
    if ("error" in access) return c.json({ error: access.error === "not_found" ? "班级不存在" : "权限不足" }, access.error === "not_found" ? 404 : 403);
    const input = c.req.valid("json");
    const result = await updateGradingPolicy(parseInt(classId, 10), input);
    if ("error" in result) {
      if (result.error === "invalid_weight_total") {
        return c.json({ error: "顶层权重和流程细节权重总和必须分别为 100%" }, 400);
      }
      return c.json({ error: "未找到该班级的成绩权重" }, 404);
    }
    return c.json(result.data);
  } catch (error) { return handleUnexpectedError(c, error, "更新成绩权重失败"); }
});

classGradesController.get("/api/v1/classes/:classId/grade-summaries", jwtAuth, zValidator("param", classIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { classId } = c.req.valid("param");
    const access = await checkClassAccess(parseInt(classId, 10), payload.sub, payload.role);
    if ("error" in access) return c.json({ error: access.error === "not_found" ? "班级不存在" : "权限不足" }, access.error === "not_found" ? 404 : 403);
    const result = await getGradeSummaries(parseInt(classId, 10));
    if ("error" in result) return c.json({ error: "该班级未配置成绩权重" }, 404);
    return c.json(result.data);
  } catch (error) { return handleUnexpectedError(c, error, "获取最新成绩失败"); }
});

classGradesController.get("/api/v1/teachers/:teacherId/grade-summaries", jwtAuth, zValidator("param", teacherIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { teacherId } = c.req.valid("param");
    const termFilter = await parseTermFilter(c.req.query("term_id"));
    if ("error" in termFilter) {
      if (termFilter.error === "no_current_term") return c.json({ error: "当前学期未设置" }, 404);
      return c.json({ error: "学期不存在" }, 400);
    }
    const result = await getTeacherClassGradeSummaries(parseInt(teacherId, 10), payload.sub, payload.role, termFilter.termId);
    if ("error" in result) {
      if (result.error === "forbidden") return c.json({ error: "权限不足" }, 403);
      return c.json({ error: "目标用户不是教师" }, 404);
    }
    return c.json(result.data);
  } catch (error) { return handleUnexpectedError(c, error, "获取班级成绩汇总失败"); }
});

classGradesController.get("/api/v1/assistants/:assistantId/grade-summaries", jwtAuth, zValidator("param", assistantIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { assistantId } = c.req.valid("param");
    const termFilter = await parseTermFilter(c.req.query("term_id"));
    if ("error" in termFilter) {
      if (termFilter.error === "no_current_term") return c.json({ error: "当前学期未设置" }, 404);
      return c.json({ error: "学期不存在" }, 400);
    }
    const result = await getAssistantClassGradeSummaries(parseInt(assistantId, 10), payload.sub, payload.role, termFilter.termId);
    if ("error" in result) {
      if (result.error === "forbidden") return c.json({ error: "权限不足" }, 403);
      return c.json({ error: "目标用户不是助教" }, 404);
    }
    return c.json(result.data);
  } catch (error) { return handleUnexpectedError(c, error, "获取助教班级成绩汇总失败"); }
});

classGradesController.get("/api/v1/classes/:classId/report-archive", jwtAuth, zValidator("param", classIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { classId } = c.req.valid("param");
    const access = await checkClassAccess(parseInt(classId, 10), payload.sub, payload.role);
    if ("error" in access) return c.json({ error: access.error === "not_found" ? "班级不存在" : "权限不足" }, access.error === "not_found" ? 404 : 403);
    const result = await exportReportArchive(parseInt(classId, 10));
    if ("error" in result) {
      if (result.error === "no_reports") return c.json({ error: "未找到该班级的有效报告（可能是因为学生尚未提交或报告已被驳回）" }, 404);
      return c.json({ error: "没有可下载的报告文件" }, 404);
    }
    return c.json(result.data);
  } catch (error) { return handleUnexpectedError(c, error, "导出报告失败"); }
});

classGradesController.get("/api/v1/classes/:classId/grade-export.csv", jwtAuth, zValidator("param", classIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { classId } = c.req.valid("param");
    const access = await checkClassAccess(parseInt(classId, 10), payload.sub, payload.role);
    if ("error" in access) return c.json({ error: access.error === "not_found" ? "班级不存在" : "权限不足" }, access.error === "not_found" ? 404 : 403);
    const result = await exportGradeCsv(parseInt(classId, 10));
    if ("error" in result) {
      if (result.error === "no_weights") return c.json({ error: "该班级未配置成绩权重" }, 404);
      return c.json({ error: "该班级未找到已加入的学生" }, 404);
    }
    return c.json(result.data);
  } catch (error) { return handleUnexpectedError(c, error, "导出成绩失败"); }
});

classGradesController.get("/api/v1/classes/:classId/report-export.csv", jwtAuth, zValidator("param", classIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { classId } = c.req.valid("param");
    const access = await checkClassAccess(parseInt(classId, 10), payload.sub, payload.role);
    if ("error" in access) return c.json({ error: access.error === "not_found" ? "班级不存在" : "权限不足" }, access.error === "not_found" ? 404 : 403);
    const result = await exportReportCsv(parseInt(classId, 10));
    if ("error" in result) return c.json({ error: "该班级未找到已加入的学生" }, 404);
    return c.json(result.data);
  } catch (error) { return handleUnexpectedError(c, error, "导出报告CSV失败"); }
});

classGradesController.get("/api/v1/classes/:classId/experiment-events", jwtAuth, zValidator("param", classIdParamSchema), async (c) => {
  try {
    const payload = c.get("jwtPayload") as JwtPayload;
    const { classId } = c.req.valid("param");
    const access = await checkClassAccess(parseInt(classId, 10), payload.sub, payload.role);
    if ("error" in access) return c.json({ error: access.error === "not_found" ? "班级不存在" : "权限不足" }, access.error === "not_found" ? 404 : 403);
    const events = await getExperimentEvents(parseInt(classId, 10));
    return c.json(events);
  } catch (error) { return handleUnexpectedError(c, error, "获取实验步骤事件失败"); }
});

export default classGradesController;
