import { z } from "zod";

export const classIdParamSchema = z.object({
  classId: z.string().regex(/^\d+$/, { message: "classId 必须是数字" }),
});

export const teacherIdParamSchema = z.object({
  teacherId: z.string().regex(/^\d+$/, { message: "teacherId 必须是数字" }),
});

export const assistantIdParamSchema = z.object({
  assistantId: z.string().regex(/^\d+$/, { message: "assistantId 必须是数字" }),
});

export const reportIdParamSchema = z.object({
  reportId: z.string().regex(/^\d+$/, { message: "reportId 必须是数字" }),
});

export const EXP_FLOW_FIELDS = [
  "exp_flow_demand_data_preparation", "exp_flow_demand_descriptive_stats", "exp_flow_demand_model_selection",
  "exp_flow_demand_generate_results", "exp_flow_production_inventory_calc", "exp_flow_production_service_level",
  "exp_flow_production_variable_calc", "exp_flow_production_plan_creation", "exp_flow_report_submission",
] as const;

export const EXPERIMENT_GRADE_FIELDS = [...EXP_FLOW_FIELDS, "model_quality"] as const;
export const EXPERIMENT_GRADE_MUTABLE_FIELDS = [...EXPERIMENT_GRADE_FIELDS] as const;

const createOptionalGradeValue = () =>
  z.number().min(0, { message: "分数必须在 0 到 100 之间" }).max(100, { message: "分数必须在 0 到 100 之间" }).optional();

const optionalGradeValue = createOptionalGradeValue();
const optionalFeedbackValue = z.string().transform((value) => { const trimmed = value.trim(); return trimmed.length ? trimmed : null; }).optional();
const createOptionalWeightValue = () =>
  z.number().min(0, { message: "权重必须在 0 到 100 之间" }).max(100, { message: "权重必须在 0 到 100 之间" }).optional();

const experimentGradeFieldSchemas = EXPERIMENT_GRADE_MUTABLE_FIELDS.reduce((acc, field) => {
  acc[field] = createOptionalGradeValue();
  return acc;
}, {} as Record<(typeof EXPERIMENT_GRADE_MUTABLE_FIELDS)[number], ReturnType<typeof createOptionalGradeValue>>);

const experimentGradeUpdateSchema = z.object(experimentGradeFieldSchemas).superRefine((data, ctx) => {
  const hasValue = EXPERIMENT_GRADE_MUTABLE_FIELDS.some((field) => data[field] !== undefined);
  if (!hasValue) ctx.addIssue({ code: z.ZodIssueCode.custom, message: "至少需要提供一个实验成绩字段" });
}).optional();

export const gradeReportSchema = z.object({
  grade: optionalGradeValue,
  feedback: optionalFeedbackValue,
  status: z.enum(["graded", "rejected"]).optional(),
  experiment_grade: experimentGradeUpdateSchema,
}).refine((data) => data.grade !== undefined || data.feedback !== undefined || data.status !== undefined || data.experiment_grade !== undefined, {
  message: "至少需要提供分数、反馈、状态或实验成绩中的一项",
});

const expFlowWeightFieldSchemas = EXP_FLOW_FIELDS.reduce((acc, field) => {
  acc[field] = createOptionalWeightValue();
  return acc;
}, {} as Record<(typeof EXP_FLOW_FIELDS)[number], ReturnType<typeof createOptionalWeightValue>>);

export const updateGradeWeightsSchema = z.object({
  exp_flow_weight: createOptionalWeightValue(),
  ...expFlowWeightFieldSchemas,
  model_quality_weight: createOptionalWeightValue(),
  knowledge_test_weight: createOptionalWeightValue(),
  report_quality_weight: createOptionalWeightValue(),
}).refine((data) => Object.values(data).some((v) => v !== undefined), { message: "至少需要提供一个权重字段" });

export type ClassIdParam = z.infer<typeof classIdParamSchema>;
export type TeacherIdParam = z.infer<typeof teacherIdParamSchema>;
export type AssistantIdParam = z.infer<typeof assistantIdParamSchema>;
export type ReportIdParam = z.infer<typeof reportIdParamSchema>;
export type GradeReportInput = z.infer<typeof gradeReportSchema>;
export type UpdateGradeWeightsInput = z.infer<typeof updateGradeWeightsSchema>;
