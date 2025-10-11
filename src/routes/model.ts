import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import path from "path";
import fs from "fs";
import type { Pool } from "mysql2/promise";
import csv from "csv-parser";

// Zod schema for the moving average training request
const trainMovingAverageSchema = z
  .object({
    selected_industry: z.string().min(1),
    selected_company: z.string().min(1),
    selected_product: z.string().min(1),
    data_window_train_start_index: z.coerce
      .number()
      .int()
      .min(0, "train start index must be >= 0"),
    data_window_train_end_index: z.coerce
      .number()
      .int()
      .min(0, "train end index must be >= 0"),
    data_window_evaluate_start_index: z.coerce
      .number()
      .int()
      .min(0, "evaluate start index must be >= 0"),
    data_window_evaluate_end_index: z.coerce
      .number()
      .int()
      .min(0, "evaluate end index must be >= 0"),
    moving_average_window: z.coerce
      .number()
      .int()
      .min(1, "moving average window must be at least 1"),
  })
  .refine(
    (data) =>
      data.data_window_train_start_index <= data.data_window_train_end_index,
    {
      message: "train_start_index must be <= train_end_index",
      path: ["data_window_train_end_index"],
    },
  )
  .refine(
    (data) =>
      data.data_window_evaluate_start_index <=
      data.data_window_evaluate_end_index,
    {
      message: "evaluate_start_index must be <= evaluate_end_index",
      path: ["data_window_evaluate_end_index"],
    },
  );

export default (pool: Pool, JWT_SECRET: string) => {
  const modelApi = new Hono();

  modelApi.use("*", jwt({ secret: JWT_SECRET }));

  modelApi.get("/model/hello", (c) => {
    return c.json({ message: "Model service placeholder" });
  });

  // New route for training moving average model
  modelApi.post(
    "/model/moving-average/train",
    zValidator("json", trainMovingAverageSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Only students can train models" }, 403);
      }
      const studentId = payload.sub;

      const body = c.req.valid("json");
      const {
        selected_industry,
        selected_company,
        selected_product,
        data_window_train_start_index,
        data_window_train_end_index,
        data_window_evaluate_start_index,
        data_window_evaluate_end_index,
        moving_average_window,
      } = body;

      // Get the file path for the specific product segment from the request body
      const [segmentRows]: any[] = await pool.query(
        "SELECT file_path FROM dataset_segments WHERE industry_name = ? AND company_name = ? AND product_name = ?",
        [selected_industry, selected_company, selected_product],
      );

      if (segmentRows.length === 0) {
        return c.json({ error: "Dataset segment file not found" }, 404);
      }
      const productCsvPath = segmentRows[0].file_path;

      // Read the entire product CSV into memory
      const allData: Record<string, string>[] = [];
      await new Promise<void>((resolve, reject) => {
        fs.createReadStream(productCsvPath)
          .pipe(csv({ mapHeaders: ({ header }: { header: string }) => header.replace(/^\uFEFF/, "").trim() }))
          .on("data", (row: Record<string, string>) => allData.push(row))
          .on("end", resolve)
          .on("error", reject);
      });

 

      // Validate window indices and split the data
      if (
        data_window_train_end_index >= allData.length ||
        data_window_evaluate_end_index >= allData.length
      ) {
        return c.json({ error: "Data window indices are out of bounds for the dataset.", dataset_size: allData.length }, 400);
      }

      const trainData = allData.slice(
        data_window_train_start_index,
        data_window_train_end_index + 1,
      );
      const evaluateData = allData.slice(
        data_window_evaluate_start_index,
        data_window_evaluate_end_index + 1,
      );

      // Create temporary CSV files for training and evaluation
      const outputDir = path.resolve(import.meta.dir, `../../model_csv/${studentId}/ma`);
      await fs.promises.mkdir(outputDir, { recursive: true });

      const trainCsvPath = path.join(outputDir, "train.csv");
      const evaluateCsvPath = path.join(outputDir, "evaluate.csv");

      const toCsvString = (data: Record<string, string>[]): string => {
        if (data.length === 0) return "";
        const headers = Object.keys(data[0] ?? {});
        const headerRow = headers.join(",");
        const bodyRows = data.map((row) => headers.map((header) => `"${row[header]}"`).join(","));
        return [headerRow, ...bodyRows].join("\n");
      };

      await Bun.write(trainCsvPath, toCsvString(trainData));
      await Bun.write(evaluateCsvPath, toCsvString(evaluateData));

      // Return a fake success response with the old structure for frontend compatibility
      return c.json({
        status: "success",
        results: {
          metrics: {
            rmse: 0.5 + Math.random() * 0.5, // Fake metric
            mae: 0.5 + Math.random() * 0.5,  // Fake metric
            r2: 0.1 + Math.random() * 0.5,   // Fake metric
          },
          window: moving_average_window,
        },
      });
    },
  );

  return modelApi;
};
