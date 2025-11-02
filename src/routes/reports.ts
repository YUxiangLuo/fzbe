import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";
import puppeteer from "puppeteer";
import path from "path";
import { mkdir } from "node:fs/promises";

const experimentIdParamSchema = z.object({
  experimentId: z.string().regex(/^\d+$/, { message: "experimentId must be a number" }),
});

const submitReportSchema = z.object({
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
        const { report_content: base64Content } = c.req.valid("json");

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
