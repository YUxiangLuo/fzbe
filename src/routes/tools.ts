import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";
import { runAdf } from "../utils";
import type { AdfInputPoint } from "../utils";

const adfPointSchema = z.object({
  month: z
    .string()
    .trim()
    .min(1, { message: "month is required" }),
  sales: z
    .coerce.number()
    .refine((value) => Number.isFinite(value), {
      message: "sales must be a finite number",
    }),
});

const adfRequestSchema = z.object({
  series: z.array(adfPointSchema).min(3, {
    message: "series 至少需要包含 3 条数据点",
  }),
});

export default (_pool: Pool, JWT_SECRET: string) => {
  const toolsApi = new Hono();

  toolsApi.use("*", jwt({ secret: JWT_SECRET }));

  toolsApi.post(
    "/api/v1/tools/adf",
    zValidator("json", adfRequestSchema),
    async (c) => {
      const { series } = c.req.valid("json") as { series: AdfInputPoint[] };

      try {
        const result = await runAdf(series);
        return c.json({ result });
      } catch (error) {
        console.error("ADF calculation failed:", error);
        const message =
          error instanceof Error ? error.message : "Unknown error occurred";
        return c.json({ error: "ADF 检验失败", details: message }, 500);
      }
    },
  );

  return toolsApi;
};
