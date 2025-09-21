import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import { parse, stringify } from "yaml";
import path from "path";
import fs from "fs";
import type { Pool } from "mysql2/promise";

// Zod schema for the moving average training request
const trainMovingAverageSchema = z.object({
  selected_industry: z.string(),
  selected_company: z.string(),
  selected_product: z.string(),
  movingAverage: z.object({
    window: z.number().int().min(1, "Window size must be at least 1"),
  }),
  data_window: z.object({
    trainStartIndex: z.number().int(),
    trainEndIndex: z.number().int(),
    predictStartIndex: z.number().int(),
    predictEndIndex: z.number().int(),
  }),
});

// Infer the TypeScript type from the Zod schema
type TrainMovingAverageBody = z.infer<typeof trainMovingAverageSchema>;

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
      const window = body.movingAverage.window;

      // --- Start of TODO implementation ---

      // 1. Query the database for the dataset segment information
      const [segmentRows]: any[] = await pool.query(
        "SELECT file_path, start_row, end_row FROM dataset_segments WHERE industry_name = ? AND company_name = ? AND product_name = ?",
        [body.selected_industry, body.selected_company, body.selected_product],
      );

      if (segmentRows.length === 0) {
        return c.json({ error: "Dataset segment not found" }, 404);
      }
      const segment = segmentRows[0];
      const sourceCsvPath = segment.file_path;
      // Assuming start_row and end_row are 1-based line numbers.
      // We need to read lines from start_row to end_row inclusive.
      const startRow = segment.start_row;
      const endRow = segment.end_row;

      // 2. Extract the segment from the source CSV
      const sourceCsvContent = await Bun.file(sourceCsvPath).text();
      const lines = sourceCsvContent.split("\n");
      const header = lines[0];
      // Slice the data rows. Adjust for 0-based index and header.
      const segmentLines = lines.slice(startRow, endRow + 1);
      const newCsvContent = [header, ...segmentLines].join("\n");

      // 3. Save the new CSV file
      const outputDir = "./tmp/seg_csv";
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }
      const newCsvPath = `${outputDir}/ma_${studentId}.csv`;
      await Bun.write(newCsvPath, newCsvContent);

      // --- End of TODO implementation ---

      const ma_config_template = fs.readFileSync(
        "./py/config/ma_config.yaml",
        "utf8",
      );
      const ma_config = parse(ma_config_template);
      // Use the newly created CSV path
      ma_config.dataset.files[0] = "." + newCsvPath;
      ma_config.dataset.window_size = window;

      const ma_config_file = "./tmp/config/ma_" + studentId + ".yaml";
      await Bun.write(ma_config_file, stringify(ma_config));

      const res =
        await Bun.$`bash -c "cd py && source venv/bin/activate && python run.py --config .${ma_config_file}"`.quiet();

      const res_text = res.text();
      const match = res_text.match(/^\[test\].*$/m);
      if (match) {
        console.log(match[0]); // [test] {'RMSE': 0.5912415385246277, 'MAE': 0.5878775715827942, 'R2': -37.21657943725586}
        const metrics = JSON.parse(
          match[0].replace(/^\[test\]/, "").replaceAll(/'/g, '"'),
        );
        return c.json({
          status: "success",
          results: {
            metrics: {
              rmse: metrics.RMSE,
              mae: metrics.MAE,
              r2: metrics.R2,
            },
            window: window,
          },
        });
      } else {
        return c.json({ error: "Internal Server Error" }, 500);
      }
    },
  );

  return modelApi;
};
