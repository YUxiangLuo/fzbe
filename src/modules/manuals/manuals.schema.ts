import { z } from "zod";

export const manualIdParamSchema = z.object({
  manualId: z.string().regex(/^\d+$/, { message: "manualId 必须是数字" }),
});

export const manualMetadataSchema = z.object({
  file_name: z.string().trim().min(2, "手册名称至少需要2个字符").max(100, "手册名称不能超过100个字符"),
  description: z.string().trim().max(200, "备注不能超过200个字符").optional(),
});

export const updateManualSchema = manualMetadataSchema.partial().extend({
  is_active: z.boolean().optional(),
});

export type ManualIdParam = z.infer<typeof manualIdParamSchema>;
export type UpdateManualInput = z.infer<typeof updateManualSchema>;
