import { z } from "zod";

export const industryParamSchema = z.object({
  industry: z.string().min(1),
});

export const industryAndCompanyParamSchema = z.object({
  industry: z.string().min(1),
  company: z.string().min(1),
});

export const industryCompanyProductParamSchema = z.object({
  industry: z.string().min(1),
  company: z.string().min(1),
  product: z.string().min(1),
});

export const datasetIdParamSchema = z.object({
  datasetId: z.string().regex(/^\d+$/, { message: "datasetId 必须是数字" }),
});

export const datasetMetadataSchema = z.object({
  data_name: z.string().trim().min(2, "数据集名称至少需要2个字符").max(100, "数据集名称不能超过100个字符"),
  description: z.string().trim().max(200, "备注不能超过200个字符").optional(),
});

export const updateDatasetSchema = datasetMetadataSchema.partial();

export type IndustryParam = z.infer<typeof industryParamSchema>;
export type IndustryAndCompanyParam = z.infer<typeof industryAndCompanyParamSchema>;
export type IndustryCompanyProductParam = z.infer<typeof industryCompanyProductParamSchema>;
export type DatasetIdParam = z.infer<typeof datasetIdParamSchema>;
export type UpdateDatasetInput = z.infer<typeof updateDatasetSchema>;
