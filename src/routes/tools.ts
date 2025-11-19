import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";
import { runAdf } from "../utils";
import type { AdfInputPoint } from "../utils";
import { handleUnexpectedError } from "../utils/error-handler";

const adfPointSchema = z.object({
  month: z
    .string()
    .trim()
    .min(1, { message: "月份不能为空" }),
  sales: z
    .coerce.number()
    .refine((value) => Number.isFinite(value), {
      message: "销售额必须是有限数字",
    }),
});

const adfRequestSchema = z.object({
  series: z.array(adfPointSchema).min(4, {
    message: "series 至少需要包含 4 条数据点才能进行可靠的 ADF 检验",
  }),
});

export default (_pool: Pool, JWT_SECRET: string) => {
  const toolsApi = new Hono();

  toolsApi.post(
    "/api/v1/tools/adf",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", adfRequestSchema),
    async (c) => {
      const { series } = c.req.valid("json") as { series: AdfInputPoint[] };

      try {
        const result = await runAdf(series);
        return c.json({ result });
      } catch (error) {
        return handleUnexpectedError(c, error, "ADF 检测失败");
      }
    },
  );

  return toolsApi;
};
