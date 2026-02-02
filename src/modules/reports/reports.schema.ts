import { z } from "zod";

export const experimentIdParamSchema = z.object({
  experimentId: z.string().regex(/^\d+$/, { message: "experimentId 必须是数字" }),
});

export const submitReportSchema = z.object({
  report_content: z.string().trim().min(1),
});

export type ExperimentIdParam = z.infer<typeof experimentIdParamSchema>;
export type SubmitReportInput = z.infer<typeof submitReportSchema>;
