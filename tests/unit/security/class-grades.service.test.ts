import { describe, expect, test } from "bun:test";

import { EXP_FLOW_FIELDS } from "../../../src/modules/class-grades/class-grades.schema";
import {
  buildGradeCsvLines,
  buildCsvContent,
  computeExperimentScores,
  validateGradeWeightTotals,
} from "../../../src/modules/class-grades/class-grades.service";

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
});
