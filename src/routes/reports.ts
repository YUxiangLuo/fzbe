import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";
import { marked } from "marked";
import puppeteer from "puppeteer-core";
import path from "path";
import { mkdir } from "node:fs/promises";
import { handleUnexpectedError } from "../utils/error-handler";
import { sanitizeHTML, HTMLSanitizationError } from "../utils/html-sanitizer";

const experimentIdParamSchema = z.object({
  experimentId: z.string().regex(/^\d+$/, { message: "experimentId must be a number" }),
});

const submitReportSchema = z.object({
  report_content: z.string().trim().min(1),
});

async function markdownToSanitizedHtml(markdown: string): Promise<string> {
  const parsed = await marked.parse(markdown);

  if (typeof parsed !== "string") {
    throw new HTMLSanitizationError("Failed to convert Markdown to HTML");
  }

  return sanitizeHTML(parsed);
}

function buildPdfDocument(content: string): string {
  return `<!DOCTYPE html>
  <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>Experiment Report</title>
      <style>
        :root {
          color-scheme: light;
        }
        body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
          margin: 0;
          padding: 40px;
          line-height: 1.6;
          color: #1f2933;
          background-color: #ffffff;
        }
        .report-content h1,
        .report-content h2,
        .report-content h3,
        .report-content h4,
        .report-content h5,
        .report-content h6 {
          margin-top: 1.8rem;
          margin-bottom: 0.6rem;
        }
        .report-content pre {
          background-color: #f8fafc;
          padding: 14px;
          border-radius: 6px;
          overflow-x: auto;
          font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
          font-size: 0.9rem;
        }
        .report-content code {
          font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
          background: #f1f5f9;
          padding: 2px 4px;
          border-radius: 4px;
        }
        .report-content table {
          border-collapse: collapse;
          width: 100%;
          margin: 1.2rem 0;
        }
        .report-content th,
        .report-content td {
          border: 1px solid #cbd5f5;
          padding: 8px 10px;
          text-align: left;
        }
        .report-content ul,
        .report-content ol {
          padding-left: 1.4rem;
        }
        .report-content blockquote {
          border-left: 4px solid #94a3b8;
          padding-left: 12px;
          color: #475569;
          margin: 1rem 0;
        }
        .report-content hr {
          border: none;
          border-top: 1px solid #e2e8f0;
          margin: 2rem 0;
        }
      </style>
    </head>
    <body>
      <main class="report-content">${content}</main>
    </body>
  </html>`;
}

export default (pool: Pool, JWT_SECRET: string) => {
  const reportsApi = new Hono();

  const authMiddleware = jwt({ secret: JWT_SECRET });

  // POST /api/v1/experiment-runs/:experimentId/report - 提交实验报告
  reportsApi.post(
    "/api/v1/experiment-runs/:experimentId/report",
    authMiddleware,
    zValidator("param", experimentIdParamSchema),
    zValidator("json", submitReportSchema),
    async (c) => {
      let browser;
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Student") {
          return c.json({ error: "Forbidden" }, 403);
        }
        const studentId = Number.parseInt(payload.sub, 10);

        const { experimentId } = c.req.valid("param");
        const experiment_id = Number.parseInt(experimentId, 10);
        const { report_content: markdownContent } = c.req.valid("json");

        const [[ownership]]: any[] = await pool.query(
          `SELECT student_id
             FROM experiment_status
            WHERE experiment_id = ?
            LIMIT 1`,
          [experiment_id],
        );

        if (!ownership) {
          return c.json({ error: "Experiment not found" }, 404);
        }

        if (Number.parseInt(String(ownership.student_id), 10) !== studentId) {
          return c.json(
            { error: "Forbidden: Experiment does not belong to the current user" },
            403,
          );
        }

        // Convert Markdown to sanitized HTML to prevent XSS/SSRF
        let sanitizedReportContent: string;
        try {
          sanitizedReportContent = await markdownToSanitizedHtml(markdownContent);
        } catch (error) {
          if (error instanceof HTMLSanitizationError) {
            return c.json(
              { error: "Invalid report content: " + error.message },
              400
            );
          }
          throw error;
        }

        const htmlDocument = buildPdfDocument(sanitizedReportContent);

        // --- PDF Generation Logic ---
        const reportsDir = path.resolve(import.meta.dir, "../../uploads", "reports");
        await mkdir(reportsDir, { recursive: true });

        const uniqueFilename = `report-exp${experiment_id}-student${studentId}-${Date.now()}.pdf`;
        const pdfPath = path.join(reportsDir, uniqueFilename);

        browser = await puppeteer.launch({
          executablePath: "/usr/bin/chromium",
          headless: true,
          args: [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-web-security", // Prevent CORS issues but safe since content is sanitized
            "--disable-features=IsolateOrigins,site-per-process", // For PDF generation
          ],
        });
        const page = await browser.newPage();

        // Set Content Security Policy to prevent any script execution
        await page.setExtraHTTPHeaders({
          "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline';"
        });

        // Disable JavaScript execution as an extra security layer
        await page.setJavaScriptEnabled(false);

        await page.setContent(htmlDocument, {
          waitUntil: "domcontentloaded", // Use domcontentloaded instead of networkidle0 to prevent SSRF
          timeout: 10000 // 10 second timeout
        });

        await page.pdf({
          path: pdfPath,
          format: "A4",
          printBackground: true,
        });
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
            sanitizedReportContent,
            pdfPath,
          ]);



          // 提交报告成功则实验流程所有步骤都100分
          const EXP_FLOW_GRADE_COLUMNS = [
            "exp_flow_demand_data_preparation",
            "exp_flow_demand_descriptive_stats",
            "exp_flow_demand_model_selection",
            "exp_flow_demand_generate_results",
            "exp_flow_production_inventory_calc",
            "exp_flow_production_service_level",
            "exp_flow_production_variable_calc",
            "exp_flow_production_plan_creation",
            "exp_flow_report_submission",
          ] as const;
          const EXP_FLOW_GRADE_UPDATE_SET = EXP_FLOW_GRADE_COLUMNS.map(
            (col) => `${col} = ?`,
          ).join(", ");
          const fullValues = EXP_FLOW_GRADE_COLUMNS.map(() => 100);
          await connection.query(
            `UPDATE experiment_grade SET ${EXP_FLOW_GRADE_UPDATE_SET} WHERE experiment_id = ? AND student_id = ?`,
            [...fullValues, experiment_id, studentId],
          );

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
          return handleUnexpectedError(
            c,
            dbError,
            "An internal server error occurred while processing the report.",
          );
        } finally {
          connection.release();
        }
      } catch (error) {
        return handleUnexpectedError(
          c,
          error,
          "An internal server error occurred while processing the report.",
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
