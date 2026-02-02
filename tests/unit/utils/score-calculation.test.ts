import { describe, test, expect } from "bun:test";
import { computeExperimentScores } from "../../../src/modules/class-grades/class-grades.service";

describe("Score Calculation", () => {
  const defaultWeights = {
    exp_flow_demand_data_preparation: 10,
    exp_flow_demand_descriptive_stats: 10,
    exp_flow_demand_model_selection: 10,
    exp_flow_demand_generate_results: 10,
    exp_flow_production_inventory_calc: 10,
    exp_flow_production_service_level: 10,
    exp_flow_production_variable_calc: 10,
    exp_flow_production_plan_creation: 10,
    exp_flow_report_submission: 20,
    exp_flow_weight: 40,
    model_quality_weight: 20,
    knowledge_test_weight: 20,
    report_quality_weight: 20,
  };

  describe("computeExperimentScores", () => {
    test("should return null scores when no experiment exists", () => {
      const row = { experiment_id: null };
      const result = computeExperimentScores(row, defaultWeights);

      expect(result.expFlowScore).toBeNull();
      expect(result.finalScore).toBeNull();
    });

    test("should calculate exp flow score from weighted components", () => {
      const row = {
        experiment_id: 1,
        exp_flow_demand_data_preparation: 100,
        exp_flow_demand_descriptive_stats: 100,
        exp_flow_demand_model_selection: 100,
        exp_flow_demand_generate_results: 100,
        exp_flow_production_inventory_calc: 100,
        exp_flow_production_service_level: 100,
        exp_flow_production_variable_calc: 100,
        exp_flow_production_plan_creation: 100,
        exp_flow_report_submission: 100,
      };

      const result = computeExperimentScores(row, defaultWeights);

      // All 100 with weights summing to 100 should give 100
      expect(result.expFlowScore).toBe(100);
    });

    test("should calculate partial exp flow score", () => {
      const row = {
        experiment_id: 1,
        exp_flow_demand_data_preparation: 50,
        exp_flow_demand_descriptive_stats: 50,
        // rest are null/undefined
      };

      const result = computeExperimentScores(row, defaultWeights);

      // 50 * 0.1 + 50 * 0.1 = 10
      expect(result.expFlowScore).toBe(10);
    });

    test("should calculate final score from all components", () => {
      const row = {
        experiment_id: 1,
        exp_flow_demand_data_preparation: 100,
        exp_flow_demand_descriptive_stats: 100,
        exp_flow_demand_model_selection: 100,
        exp_flow_demand_generate_results: 100,
        exp_flow_production_inventory_calc: 100,
        exp_flow_production_service_level: 100,
        exp_flow_production_variable_calc: 100,
        exp_flow_production_plan_creation: 100,
        exp_flow_report_submission: 100,
        knowledge_test: 100,
        model_quality: 100,
        report_quality: 100,
      };

      const result = computeExperimentScores(row, defaultWeights);

      // exp_flow: 100 * 0.4 = 40
      // knowledge: 100 * 0.2 = 20
      // model: 100 * 0.2 = 20
      // report: 100 * 0.2 = 20
      // total: 100
      expect(result.finalScore).toBe(100);
    });

    test("should handle mixed scores", () => {
      const row = {
        experiment_id: 1,
        exp_flow_demand_data_preparation: 80,
        exp_flow_demand_descriptive_stats: 80,
        exp_flow_demand_model_selection: 80,
        exp_flow_demand_generate_results: 80,
        exp_flow_production_inventory_calc: 80,
        exp_flow_production_service_level: 80,
        exp_flow_production_variable_calc: 80,
        exp_flow_production_plan_creation: 80,
        exp_flow_report_submission: 80,
        knowledge_test: 90,
        model_quality: 85,
        report_quality: 95,
      };

      const result = computeExperimentScores(row, defaultWeights);

      // exp_flow: 80
      // final: 80*0.4 + 90*0.2 + 85*0.2 + 95*0.2 = 32 + 18 + 17 + 19 = 86
      expect(result.expFlowScore).toBe(80);
      expect(result.finalScore).toBe(86);
    });

    test("should return breakdown components", () => {
      const row = {
        experiment_id: 1,
        exp_flow_demand_data_preparation: 100,
        knowledge_test: 80,
        model_quality: 90,
        report_quality: 70,
      };

      const result = computeExperimentScores(row, defaultWeights);

      expect(result.expFlowComponents).toHaveLength(9);
      expect(result.expFlowComponents[0]).toMatchObject({
        field: "exp_flow_demand_data_preparation",
        score: 100,
        weight: 10,
        weighted_score: 10,
      });

      expect(result.finalScoreComponents.knowledge_test).toMatchObject({
        score: 80,
        weight: 20,
        weighted_score: 16,
      });
    });

    test("should handle null component scores in breakdown", () => {
      const row = {
        experiment_id: 1,
        exp_flow_demand_data_preparation: 100,
        knowledge_test: null,
      };

      const result = computeExperimentScores(row, defaultWeights);

      expect(result.finalScoreComponents.knowledge_test.score).toBeNull();
      expect(result.finalScoreComponents.knowledge_test.weighted_score).toBeNull();
    });

    test("should handle zero weights", () => {
      const zeroWeights = { ...defaultWeights, knowledge_test_weight: 0 };
      const row = {
        experiment_id: 1,
        knowledge_test: 100,
      };

      const result = computeExperimentScores(row, zeroWeights);

      expect(result.finalScoreComponents.knowledge_test.weighted_score).toBe(0);
    });
  });
});
