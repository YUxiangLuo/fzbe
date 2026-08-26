import { describe, expect, test } from "bun:test";
import { mkdtemp, readdir, rm, utimes, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { EXP_FLOW_FIELDS } from "../../../src/modules/class-grades/class-grades.schema";
import {
  allocateReportArchiveEntryFilename,
  buildGradeCsvLines,
  buildGeneratedExportFilename,
  buildReportCsvLines,
  buildCsvContent,
  cleanupExpiredExportFiles,
  computeExperimentScores,
  createZipArchive,
  encodeCsvCell,
  formatCsvDateTime,
  isReportPdfAvailable,
  parseClassIdFromExportFilename,
  resolveStoredReportPdfPath,
  validateGradeWeightTotals,
} from "../../../src/modules/class-grades/class-grades.service";
import { FILESYSTEM_FILENAME_MAX_BYTES } from "../../../src/utils/file-validator";

const defaultWeights: Record<string, number> = {
  exp_flow_demand_data_preparation: 5,
  exp_flow_demand_descriptive_stats: 5,
  exp_flow_demand_model_selection: 10,
  exp_flow_demand_generate_results: 10,
  exp_flow_production_inventory_calc: 10,
  exp_flow_production_service_level: 5,
  exp_flow_production_variable_calc: 5,
  exp_flow_production_plan_creation: 15,
  exp_flow_report_submission: 35,
  exp_flow_weight: 40,
  knowledge_test_weight: 20,
  model_quality_weight: 20,
  report_quality_weight: 20,
};

describe("buildCsvContent", () => {
  test("adds UTF-8 BOM, CRLF line endings, and a final newline", () => {
    const content = buildCsvContent(["a,b", "1,2"]);

    expect(content.startsWith("\uFEFF")).toBe(true);
    expect(content).toBe("\uFEFFa,b\r\n1,2\r\n");
  });
});

describe("encodeCsvCell", () => {
  test("keeps spreadsheet formula prefixes as quoted text", () => {
    const dangerousValues = [
      "=1+1",
      "+SUM(1,1)",
      "-2+3",
      "@SUM(1,1)",
      "  =1+1",
      "\t=1+1",
      "\r=1+1",
      "\n=1+1",
      "\t普通文本",
      "\r普通文本",
      "\n普通文本",
      "\u0000普通文本",
      "＝1+1",
      "＋1+1",
      "－1+1",
      "＠SUM(1,1)",
    ];

    for (const value of dangerousValues) {
      expect(encodeCsvCell(value)).toBe(`"\t${value.replace(/"/g, '""')}"`);
    }
  });

  test("quotes commas, quotes, LF, and standalone CR while preserving ordinary text", () => {
    expect(encodeCsvCell("普通文本")).toBe("普通文本");
    expect(encodeCsvCell('包含,逗号和"引号"')).toBe('"包含,逗号和""引号"""');
    expect(encodeCsvCell("第一行\n第二行")).toBe('"第一行\n第二行"');
    expect(encodeCsvCell("第一段\r第二段")).toBe('"第一段\r第二段"');
  });
});

describe("formatCsvDateTime", () => {
  test("preserves database wall-clock strings without applying a timezone conversion", () => {
    expect(formatCsvDateTime("2026-01-11 10:40:00")).toBe("2026-01-11 10:40:00");
  });

  test("formats Date instances as the same UTC instant regardless of the server timezone", () => {
    expect(formatCsvDateTime(new Date("2026-01-11T10:40:05.000Z"))).toBe("2026-01-11 10:40:05");
  });

  test("keeps invalid values inspectable and leaves empty values blank", () => {
    expect(formatCsvDateTime(null)).toBe("");
    expect(formatCsvDateTime("不是日期")).toBe("不是日期");
  });
});

describe("cleanupExpiredExportFiles", () => {
  test("removes only expired generated export files", async () => {
    const exportsDir = await mkdtemp(path.join(os.tmpdir(), "fangzhen-export-cleanup-test-"));
    const classId = Date.now();
    const oldExport = path.join(exportsDir, buildGeneratedExportFilename(
      classId,
      "reports",
      "zip",
      1000,
      "00000000-0000-4000-8000-000000000001",
    ));
    const freshExport = path.join(exportsDir, buildGeneratedExportFilename(
      classId,
      "grades",
      "csv",
      2000,
      "00000000-0000-4000-8000-000000000002",
    ));
    const legacyExport = path.join(exportsDir, `class-${classId}-reports-999.zip`);
    const unrelated = path.join(exportsDir, `manual-${classId}.csv`);

    await writeFile(oldExport, "old");
    await writeFile(freshExport, "fresh");
    await writeFile(legacyExport, "legacy");
    await writeFile(unrelated, "manual");
    await utimes(oldExport, new Date("2026-07-04T00:00:00.000Z"), new Date("2026-07-04T00:00:00.000Z"));
    await utimes(freshExport, new Date("2026-07-05T00:00:00.000Z"), new Date("2026-07-05T00:00:00.000Z"));
    await utimes(legacyExport, new Date("2026-07-04T00:00:00.000Z"), new Date("2026-07-04T00:00:00.000Z"));
    await utimes(unrelated, new Date("2026-07-04T00:00:00.000Z"), new Date("2026-07-04T00:00:00.000Z"));

    try {
      const result = await cleanupExpiredExportFiles({
        directory: exportsDir,
        now: new Date("2026-07-05T00:00:00.000Z"),
        maxAgeMs: 60_000,
      });

      const remaining = await readdir(exportsDir);
      expect(result.deletedEntries).toBeGreaterThanOrEqual(1);
      expect(remaining).not.toContain(path.basename(oldExport));
      expect(remaining).not.toContain(path.basename(legacyExport));
      expect(remaining).toContain(path.basename(freshExport));
      expect(remaining).toContain(path.basename(unrelated));
    } finally {
      await rm(exportsDir, { recursive: true, force: true });
    }
  });
});

describe("createZipArchive", () => {
  test("creates a ZIP without invoking a host zip executable", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "fangzhen-zip-export-test-"));
    const archivePath = path.join(directory, "reports.zip");
    const firstName = "学生甲_student_1_experiment_2.pdf";
    const secondName = "学生乙_student_3_experiment_4.pdf";
    try {
      await writeFile(path.join(directory, firstName), "first-pdf");
      await writeFile(path.join(directory, secondName), "second-pdf");

      await createZipArchive(archivePath, [firstName, secondName], directory);

      const archive = Buffer.from(await Bun.file(archivePath).arrayBuffer());
      expect(archive.subarray(0, 2).toString("ascii")).toBe("PK");
      expect(archive.toString("utf8")).toContain(firstName);
      expect(archive.toString("utf8")).toContain(secondName);
      expect(archive.includes(Buffer.from("PK\u0005\u0006", "binary"))).toBeTrue();
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("rejects archive entry paths instead of escaping the working directory", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "fangzhen-zip-export-path-test-"));
    try {
      await expect(createZipArchive(
        path.join(directory, "reports.zip"),
        ["../outside.pdf"],
        directory,
      )).rejects.toThrow("报告归档条目名称无效");
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});

describe("generated export filenames", () => {
  test("stays unique for exports created in the same millisecond", () => {
    const first = buildGeneratedExportFilename(42, "reports", "zip", 1234567890);
    const second = buildGeneratedExportFilename(42, "reports", "zip", 1234567890);

    expect(first).not.toBe(second);
    expect(parseClassIdFromExportFilename(first)).toBe(42);
    expect(parseClassIdFromExportFilename(second)).toBe(42);
  });

  test("parses only generated report/grade export shapes", () => {
    const token = "00000000-0000-4000-8000-000000000003";
    expect(parseClassIdFromExportFilename(
      buildGeneratedExportFilename(88, "reports", "csv", 1234, token),
    )).toBe(88);
    expect(parseClassIdFromExportFilename(
      buildGeneratedExportFilename(88, "grades", "csv", 1234, token),
    )).toBe(88);
    // Keep the pre-UUID deployment shape readable until retention removes it.
    expect(parseClassIdFromExportFilename("class-88-reports-1234.zip")).toBe(88);
    expect(parseClassIdFromExportFilename("class-88-grades-1234.csv")).toBe(88);
    expect(parseClassIdFromExportFilename(
      `class-88-grades-1234-${token}.zip`,
    )).toBeNull();
  });
});

describe("report archive entry filenames", () => {
  test("bounds a 255-digit username by UTF-8 bytes while preserving identity and extension", () => {
    const usedNames = new Set<string>();
    const filename = allocateReportArchiveEntryFilename({
      username: "9".repeat(255),
      studentId: 42,
      experimentId: 314,
    }, usedNames);

    expect(Buffer.byteLength(filename, "utf8")).toBeLessThanOrEqual(
      FILESYSTEM_FILENAME_MAX_BYTES,
    );
    expect(filename).toEndWith("_student_42_experiment_314.pdf");
    expect(path.basename(filename)).toBe(filename);
    expect(usedNames.has(filename)).toBeTrue();
  });

  test("does not split long multibyte usernames and allocates bounded collision names", () => {
    const usedNames = new Set<string>();
    const identity = {
      username: "学生姓名".repeat(100),
      studentId: 88,
      experimentId: 99,
    };
    const first = allocateReportArchiveEntryFilename(identity, usedNames);
    const second = allocateReportArchiveEntryFilename(identity, usedNames);

    expect(Buffer.byteLength(first, "utf8")).toBeLessThanOrEqual(
      FILESYSTEM_FILENAME_MAX_BYTES,
    );
    expect(Buffer.byteLength(second, "utf8")).toBeLessThanOrEqual(
      FILESYSTEM_FILENAME_MAX_BYTES,
    );
    expect(first).toEndWith("_student_88_experiment_99.pdf");
    expect(second).toEndWith("_student_88_experiment_99_1.pdf");
    expect(first).not.toBe(second);
    expect(Buffer.from(first).toString("utf8")).toBe(first);
  });

  test("removes traversal syntax from the user-derived component", () => {
    const filename = allocateReportArchiveEntryFilename({
      username: "../../../../etc/passwd",
      studentId: 7,
      experimentId: 11,
    }, new Set());

    expect(filename).not.toContain("/");
    expect(filename).not.toContain("\\");
    expect(filename).not.toContain("..");
    expect(path.basename(filename)).toBe(filename);
    expect(filename).toEndWith("_student_7_experiment_11.pdf");
  });
});

describe("isReportPdfAvailable", () => {
  test("returns true only when the report PDF exists on disk", async () => {
    const reportsDir = await mkdtemp(path.join(os.tmpdir(), "fangzhen-report-file-test-"));
    const existingReport = path.join(reportsDir, "report.pdf");
    const missingReport = path.join(reportsDir, "missing.pdf");

    await writeFile(existingReport, "%PDF-1.4\n%%EOF\n");

    try {
      await expect(isReportPdfAvailable(existingReport, { reportsDir })).resolves.toBe(true);
      await expect(isReportPdfAvailable(missingReport, { reportsDir })).resolves.toBe(false);
      await expect(isReportPdfAvailable(null, { reportsDir })).resolves.toBe(false);
    } finally {
      await rm(reportsDir, { recursive: true, force: true });
    }
  });

  test("fails closed for a live absolute path outside the managed reports root", async () => {
    const reportsDir = await mkdtemp(path.join(os.tmpdir(), "fangzhen-report-root-test-"));
    const outsideDir = await mkdtemp(path.join(os.tmpdir(), "fangzhen-report-outside-test-"));
    const outsideReport = path.join(outsideDir, "secret.pdf");
    await writeFile(outsideReport, "%PDF-1.4 outside\n");

    try {
      await expect(resolveStoredReportPdfPath(outsideReport, { reportsDir })).resolves.toBeNull();
      await expect(isReportPdfAvailable(outsideReport, { reportsDir })).resolves.toBe(false);
    } finally {
      await Promise.all([
        rm(reportsDir, { recursive: true, force: true }),
        rm(outsideDir, { recursive: true, force: true }),
      ]);
    }
  });

  test("does not alias a stale outside absolute path to a same-basename managed file", async () => {
    const reportsDir = await mkdtemp(path.join(os.tmpdir(), "fangzhen-served-report-test-"));
    const staleReportPath = path.join(os.tmpdir(), `stale-fangzhen-${Date.now()}`, "served-report.pdf");
    const servedReportPath = path.join(reportsDir, path.basename(staleReportPath));

    await writeFile(servedReportPath, "%PDF-1.4\n%%EOF\n");

    try {
      await expect(resolveStoredReportPdfPath(staleReportPath, { reportsDir }))
        .resolves.toBeNull();
      await expect(isReportPdfAvailable(staleReportPath, { reportsDir })).resolves.toBe(false);
    } finally {
      await rm(reportsDir, { recursive: true, force: true });
    }
  });
});

describe("buildGradeCsvLines", () => {
  test("includes escaped report feedback in grade exports", () => {
    const row: Record<string, unknown> = {
      username: "S001",
      full_name: "张三",
      experiment_id: 11,
      knowledge_test: 90,
      model_quality: 80,
      report_quality: 70,
      report_status: "graded",
      report_feedback: '整体良好，注意"库存",继续改进',
    };
    for (const field of EXP_FLOW_FIELDS) row[field] = 100;

    const [header, gradeRow] = buildGradeCsvLines([row], defaultWeights);

    expect(header.split(",")).toContain("报告评语");
    expect(gradeRow).toContain('"整体良好，注意""库存"",继续改进"');
  });

  test("keeps report feedback for rejected reports while omitting scores", () => {
    const row: Record<string, unknown> = {
      username: "S002",
      full_name: "李四",
      experiment_id: 12,
      knowledge_test: 90,
      model_quality: 80,
      report_quality: 70,
      report_status: "rejected",
      report_feedback: "请补充关键图表后重新提交",
    };
    for (const field of EXP_FLOW_FIELDS) row[field] = 100;

    const [, gradeRow] = buildGradeCsvLines([row], defaultWeights);

    expect(gradeRow.endsWith(",已驳回,请补充关键图表后重新提交,")).toBe(true);
  });

  test("neutralizes formula-like student names and feedback in the complete grade row", () => {
    const row: Record<string, unknown> = {
      username: "=1+1",
      full_name: "＋SUM(1,1)",
      experiment_id: 13,
      knowledge_test: 90,
      model_quality: 80,
      report_quality: 70,
      report_status: "graded",
      report_feedback: "@SUM(1,1)",
    };
    for (const field of EXP_FLOW_FIELDS) row[field] = 100;

    const [, gradeRow] = buildGradeCsvLines([row], defaultWeights);

    expect(gradeRow).toStartWith('"\t=1+1","\t＋SUM(1,1)",13,');
    expect(gradeRow).toContain(',已评分,"\t@SUM(1,1)",');
  });

  test("does not export stale report or final scores before a report is graded", () => {
    const row: Record<string, unknown> = {
      username: "S003",
      full_name: "王五",
      experiment_id: 13,
      knowledge_test: 90,
      model_quality: 80,
      report_quality: 99,
      report_status: "submitted",
      report_feedback: null,
    };
    for (const field of EXP_FLOW_FIELDS) row[field] = 100;

    const [, gradeRow] = buildGradeCsvLines([row], defaultWeights);
    const cells = gradeRow.split(",");

    expect(cells).toContain("待评分");
    expect(cells.slice(3, 16).every((cell) => cell === "")).toBeTrue();
    expect(cells.at(-1)).toBe("");
    expect(gradeRow).not.toContain("99.00");
  });

  test("exports a retry without a report as redoing with no official scores", () => {
    const row: Record<string, unknown> = {
      username: "S004",
      full_name: "赵六",
      experiment_id: 14,
      knowledge_test: 100,
      model_quality: 100,
      report_quality: 100,
      report_status: "redo_in_progress",
      report_feedback: "必须完整重做实验",
    };
    for (const field of EXP_FLOW_FIELDS) row[field] = 100;

    const [, gradeRow] = buildGradeCsvLines([row], defaultWeights);
    const cells = gradeRow.split(",");

    expect(cells.slice(3, 16).every((cell) => cell === "")).toBeTrue();
    expect(cells[16]).toBe("重做中");
    expect(cells[17]).toBe("必须完整重做实验");
    expect(cells[18]).toBe("");
  });
});

describe("buildReportCsvLines", () => {
  test("only exports report grades for graded reports and preserves real zero", () => {
    const [, gradedRow, submittedRow, rejectedRow] = buildReportCsvLines([
      {
        username: "S001",
        full_name: "零分学生",
        experiment_id: 1,
        report_id: 10,
        status: "graded",
        submitted_at_text: "2026-08-02 10:00:00",
        grade: 0,
        grader_name: "教师",
        feedback: "正式零分",
      },
      {
        username: "S002",
        full_name: "待评学生",
        experiment_id: 2,
        report_id: 20,
        status: "submitted",
        submitted_at_text: "2026-08-02 11:00:00",
        grade: 88,
      },
      {
        username: "S003",
        full_name: "驳回学生",
        experiment_id: 3,
        report_id: 30,
        status: "rejected",
        submitted_at_text: "2026-08-02 12:00:00",
        grade: 99,
        grader_name: "教师",
        feedback: "请重做",
      },
    ]);

    expect(gradedRow?.split(",")[6]).toBe("0.00");
    expect(submittedRow?.split(",")[6]).toBe("");
    expect(rejectedRow?.split(",")[6]).toBe("");
  });

  test("exports retry state with its parent rejection context but no current report", () => {
    const [header, retryRow] = buildReportCsvLines([{
      username: "S004",
      full_name: "重做学生",
      experiment_id: 4,
      report_id: null,
      status: "redo_in_progress",
      submitted_at: null,
      grade: 100,
      grader_name: "驳回教师",
      feedback: "缺少关键实验步骤",
    }]);
    const cells = retryRow?.split(",") ?? [];

    expect(header).toBe("学号,姓名,实验ID,报告ID,报告状态,提交时间,报告得分,评阅人,评语");
    expect(cells).toEqual([
      "S004",
      "重做学生",
      "4",
      "",
      "重做中",
      "",
      "",
      "驳回教师",
      "缺少关键实验步骤",
    ]);
  });
});

describe("class grade score calculation", () => {
  test("uses class weights to calculate experiment-flow and final scores", () => {
    const row: Record<string, unknown> = {
      experiment_id: 1,
      knowledge_test: 89,
      model_quality: 83,
      report_quality: 81,
    };
    for (const field of EXP_FLOW_FIELDS) row[field] = 100;

    const result = computeExperimentScores(row, defaultWeights);

    expect(result.expFlowScore).toBeCloseTo(100, 5);
    expect(result.finalScore).toBeCloseTo(90.6, 5);
    expect(result.finalScoreComponents.exp_flow.weighted_score).toBeCloseTo(40, 5);
    expect(result.finalScoreComponents.knowledge_test.weighted_score).toBeCloseTo(17.8, 5);
    expect(result.finalScoreComponents.model_quality.weighted_score).toBeCloseTo(16.6, 5);
    expect(result.finalScoreComponents.report_quality.weighted_score).toBeCloseTo(16.2, 5);
  });

  test("does not produce scores when a student has no experiment", () => {
    const row: Record<string, unknown> = {
      experiment_id: null,
      knowledge_test: 100,
      model_quality: 100,
      report_quality: 100,
    };
    for (const field of EXP_FLOW_FIELDS) row[field] = 100;

    const result = computeExperimentScores(row, defaultWeights);

    expect(result.expFlowScore).toBeNull();
    expect(result.finalScore).toBeNull();
  });

  test("preserves real zero scores and ignores non-finite component values", () => {
    const row: Record<string, unknown> = {
      experiment_id: 1,
      knowledge_test: 0,
      model_quality: "not-a-score",
      report_quality: 0,
    };
    for (const field of EXP_FLOW_FIELDS) row[field] = 0;
    row.exp_flow_demand_data_preparation = Number.POSITIVE_INFINITY;

    const result = computeExperimentScores(row, defaultWeights);

    expect(result.expFlowScore).toBe(0);
    expect(result.finalScore).toBe(0);
    expect(result.finalScoreComponents.knowledge_test.weighted_score).toBe(0);
    expect(result.finalScoreComponents.model_quality.weighted_score).toBeNull();
    expect(result.finalScoreComponents.report_quality.weighted_score).toBe(0);
  });

  test("returns partial weighted totals without renormalizing missing dimensions", () => {
    const row: Record<string, unknown> = {
      experiment_id: 1,
      knowledge_test: 100,
      model_quality: null,
      report_quality: null,
    };
    for (const field of EXP_FLOW_FIELDS) row[field] = null;

    const result = computeExperimentScores(row, defaultWeights);

    expect(result.expFlowScore).toBeNull();
    expect(result.finalScore).toBe(20);
  });
});

describe("validateGradeWeightTotals", () => {
  test("accepts the default class grading weights", () => {
    const result = validateGradeWeightTotals(defaultWeights);

    expect(result.valid).toBe(true);
    expect(result.topLevelTotal).toBe(100);
    expect(result.flowDetailTotal).toBe(100);
  });

  test("rejects top-level weights that do not sum to 100", () => {
    const result = validateGradeWeightTotals({
      ...defaultWeights,
      exp_flow_weight: 50,
    });

    expect(result.valid).toBe(false);
    expect(result.topLevelValid).toBe(false);
    expect(result.flowDetailValid).toBe(true);
  });

  test("rejects experiment-flow detail weights that do not sum to 100", () => {
    const result = validateGradeWeightTotals({
      ...defaultWeights,
      exp_flow_report_submission: 30,
    });

    expect(result.valid).toBe(false);
    expect(result.topLevelValid).toBe(true);
    expect(result.flowDetailValid).toBe(false);
  });

  test("rejects non-finite totals but tolerates harmless floating-point noise", () => {
    expect(validateGradeWeightTotals({
      ...defaultWeights,
      report_quality_weight: Number.NaN,
    }).valid).toBeFalse();

    const closeToOneHundred = validateGradeWeightTotals({
      ...defaultWeights,
      exp_flow_weight: 39.99999,
      knowledge_test_weight: 20.00001,
    });
    expect(closeToOneHundred.valid).toBeTrue();
  });
});
