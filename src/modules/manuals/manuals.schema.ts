import { z } from "zod";

export const manualIdParamSchema = z.object({
  manualId: z.string().regex(/^\d+$/, { message: "manualId 必须是数字" }),
});

export const updateManualSchema = z.object({
  file_name: z.string().min(1).optional(),
  description: z.string().optional(),
  is_active: z.boolean().optional(),
});

export type ManualIdParam = z.infer<typeof manualIdParamSchema>;
export type UpdateManualInput = z.infer<typeof updateManualSchema>;
