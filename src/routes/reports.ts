import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";
import puppeteer from "puppeteer";
import path from "path";
import { mkdir } from "node:fs/promises";

const submitReportSchema = z.object({
  experiment_id: z.number().int().positive(),
  report_content: z.string().trim().min(1).refine((val) => {
    try {
      // Attempt to decode to see if it's valid Base64
      Buffer.from(val, 'base64');
      return true;
    } catch (e) {
      return false;
    }
  }, { message: "report_content must be a valid Base64 encoded string" }),
});

export default (pool: Pool, JWT_SECRET: string) => {
  const reportsApi = new Hono();

  reportsApi.use("*", jwt({ secret: JWT_SECRET }));

  // POST /api/reports - 提交实验报告
  reportsApi.post(
    "/api/reports",
    zValidator("json", submitReportSchema),
    async (c) => {
      let browser;
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Student") {
          return c.json({ error: "Forbidden" }, 403);
        }
        const studentId = Number.parseInt(payload.sub, 10);

        const { experiment_id, report_content: base64Content } = c.req.valid("json");

        const report_content = Buffer.from(base64Content, "base64").toString("utf-8");

        // --- PDF Generation Logic ---
        const reportsDir = path.resolve(import.meta.dir, "../../uploads", "reports");
        await mkdir(reportsDir, { recursive: true });

        const uniqueFilename = `report-exp${experiment_id}-student${studentId}-${Date.now()}.pdf`;
        const pdfPath = path.join(reportsDir, uniqueFilename);

        browser = await puppeteer.launch({
          headless: true,
          args: ["--no-sandbox", "--disable-setuid-sandbox"],
        });
        const page = await browser.newPage();
        await page.setContent(report_content, { waitUntil: "networkidle0" });
        await page.pdf({ path: pdfPath, format: "A4" });
        // --- End of PDF Generation ---

        // --- Database Logic ---
        const connection = await pool.getConnection();
        try {
          await connection.beginTransaction();

          const reportSql = `
            INSERT INTO experiment_reports (experiment_id, student_id, report_content, pdf_file_path, submitted_at)
            VALUES (?, ?, ?, ?, NOW())
            ON DUPLICATE KEY UPDATE
              report_content = VALUES(report_content),
              pdf_file_path = VALUES(pdf_file_path),
              submitted_at = NOW()
          `;

          const [reportResult]: any = await connection.query(reportSql, [
            experiment_id,
            studentId,
            report_content,
            pdfPath,
          ]);

          // Now, create the initial grade entry
          const gradeSql = `
            INSERT INTO experiment_grade (
              experiment_id, student_id,
              exp_flow,
              exp_flow_demand_data_preparation, exp_flow_demand_descriptive_stats,
              exp_flow_demand_model_selection, exp_flow_demand_generate_results,
              exp_flow_production_inventory_calc, exp_flow_production_service_level,
              exp_flow_production_variable_calc, exp_flow_production_plan_creation,
              exp_flow_report_submission,
              model_quality
            )
            VALUES (?, ?, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100)
            ON DUPLICATE KEY UPDATE
              exp_flow = 100,
              exp_flow_demand_data_preparation = 100, exp_flow_demand_descriptive_stats = 100,
              exp_flow_demand_model_selection = 100, exp_flow_demand_generate_results = 100,
              exp_flow_production_inventory_calc = 100, exp_flow_production_service_level = 100,
              exp_flow_production_variable_calc = 100, exp_flow_production_plan_creation = 100,
              exp_flow_report_submission = 100,
              model_quality = 100,
              -- Reset other fields in case of re-submission
              knowledge_test = 0,
              report_quality = 0,
              grader_id = NULL
          `;

          await connection.query(gradeSql, [experiment_id, studentId]);

          await connection.commit();

          let reportId = reportResult.insertId;
          if (reportId === 0 && reportResult.affectedRows > 0) {
            const [rows]: any = await connection.query(
              "SELECT report_id FROM experiment_reports WHERE experiment_id = ? AND student_id = ?",
              [experiment_id, studentId],
            );
            if (rows.length > 0) {
              reportId = rows[0].report_id;
              return c.json({
                message: "Report updated successfully",
                report_id: reportId,
                pdf_path: pdfPath,
              });
            }
          }

          return c.json(
            {
              message: "Report submitted successfully",
              report_id: reportId,
              pdf_path: pdfPath,
            },
            201,
          );
        } catch (dbError) {
          await connection.rollback();
          throw dbError; // Re-throw to be caught by the outer try-catch
        } finally {
          connection.release();
        }
      } catch (error) {
        console.error("Error submitting experiment report:", error);
        return c.json(
          { error: "An internal server error occurred while processing the report." },
          500,
        );
      } finally {
        if (browser) {
          await browser.close();
        }
      }
    },
  );

  return reportsApi;
};
