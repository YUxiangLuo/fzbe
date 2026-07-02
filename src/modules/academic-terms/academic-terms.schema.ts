import { z } from "zod";

export const termIdParamSchema = z.object({
  termId: z.string().regex(/^\d+$/, { message: "termId 必须是数字" }),
});

export const academicTermSchema = z.object({
  academic_year: z.string().trim().min(1, "学年不能为空").max(20, "学年不能超过20个字符"),
  semester: z.number().int().refine((value) => value === 1 || value === 2, {
    message: "学期必须是 1 或 2",
  }),
  is_active: z.boolean().optional(),
});

export const updateAcademicTermSchema = academicTermSchema.partial().refine(
  (data) =>
    data.academic_year !== undefined ||
    data.semester !== undefined ||
    data.is_active !== undefined,
  { message: "至少需要提供一个字段" },
);

export type TermIdParam = z.infer<typeof termIdParamSchema>;
export type AcademicTermInput = z.infer<typeof academicTermSchema>;
export type UpdateAcademicTermInput = z.infer<typeof updateAcademicTermSchema>;
