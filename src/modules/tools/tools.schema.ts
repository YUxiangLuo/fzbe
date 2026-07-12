import { z } from "zod";

export const adfPointSchema = z.object({
  month: z.string().trim().min(1, { message: "月份不能为空" }),
  sales: z.coerce.number().refine((value) => Number.isFinite(value), {
    message: "销售额必须是有限数字",
  }),
});

export const adfRequestSchema = z.object({
  series: z.array(adfPointSchema).min(4, {
    message: "series 至少需要包含 4 条数据点才能进行可靠的 ADF 检验",
  }).max(5000, { message: "series 最多允许 5000 条数据点" }),
});

export type AdfRequest = z.infer<typeof adfRequestSchema>;
