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

export const updateDatasetSchema = z.object({
  data_name: z.string().min(1).optional(),
  description: z.string().optional(),
});

export type IndustryParam = z.infer<typeof industryParamSchema>;
export type IndustryAndCompanyParam = z.infer<typeof industryAndCompanyParamSchema>;
export type IndustryCompanyProductParam = z.infer<typeof industryCompanyProductParamSchema>;
export type DatasetIdParam = z.infer<typeof datasetIdParamSchema>;
export type UpdateDatasetInput = z.infer<typeof updateDatasetSchema>;
