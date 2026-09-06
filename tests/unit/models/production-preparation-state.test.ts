import { describe, expect, test } from "bun:test";
import {
  buildProductionPreparationPayload,
  expandProductionPreparationPlan,
  hasValidProductionStepProgress,
  parseProductionCompletedStepIds,
  parseProductionPreparationStepPlan,
  type ProductionPreparationSessionRow,
} from "../../../src/modules/models/production-preparation-state";

const ends = [
  { id: "initialize", label: "确认生产准备材料" },
  { id: "finalize", label: "组装并校验生产产物" },
];
const plan = [ends[0]!, { id: "fit_model", label: "重拟合生产模型" }, ends[1]!];

describe("production preparation step protocol", () => {
  test("accepts JSON and decoded columns but rejects malformed progress", () => {
    expect(parseProductionPreparationStepPlan(JSON.stringify(plan))).toEqual(plan);
    expect(parseProductionCompletedStepIds('["initialize"]')).toEqual(["initialize"]);
    for (const value of [null, '{bad', {}, [null], [{ id: 'fit_model', label: '' }]]) {
      expect(parseProductionPreparationStepPlan(value)).toBeNull();
    }
    for (const value of [null, '{bad', {}, [''], [1]]) {
      expect(parseProductionCompletedStepIds(value)).toBeNull();
    }
  });

  test("only accepts a contiguous completed prefix and a fully completed terminal plan", () => {
    expect(hasValidProductionStepProgress({ step_plan: plan, completed_step_ids: ['initialize'], status: 'ready' })).toBe(true);
    for (const completed of [['fit_model'], ['initialize', 'finalize'], ['initialize', 'initialize']]) {
      expect(hasValidProductionStepProgress({ step_plan: plan, completed_step_ids: completed, status: 'ready' })).toBe(false);
    }
    const all = plan.map(step => step.id);
    expect(hasValidProductionStepProgress({ step_plan: plan, completed_step_ids: all, status: 'completed' })).toBe(true);
    expect(hasValidProductionStepProgress({ step_plan: plan, completed_step_ids: all, status: 'running' })).toBe(false);
    expect(hasValidProductionStepProgress({ step_plan: plan, completed_step_ids: ['initialize'], status: 'completed' })).toBe(false);
    expect(hasValidProductionStepProgress({ step_plan: [ends[0], plan[1], plan[1], ends[1]], completed_step_ids: [], status: 'ready' })).toBe(false);
  });

  test("allows baseline-only boosting to finalize without fabricating work", () => {
    expect(expandProductionPreparationPlan(ends, { work_items: [], direct_finalize: true }, 'boosting')).toEqual(ends);
    expect(() => expandProductionPreparationPlan(ends, { work_items: [], direct_finalize: true }, 'ma')).toThrow();
    expect(expandProductionPreparationPlan(ends, {
      work_items: [{ id: 'fit_stage_2_lstm' }, { id: 'fit_member_arima' }],
    }, 'boosting').map(step => step.label)).toEqual([
      '确认生产准备材料', '重拟合第 2 阶段：LSTM', '重拟合成员：ARIMA', '组装并校验生产产物',
    ]);
  });

  test("keeps resume responses and completed artifact metadata distinct", () => {
    const row = {
      session_id: 'session-1', status: 'failed', completed_step_ids: '["initialize"]',
      step_plan: JSON.stringify(plan), error_message: 'retryable failure',
      forecast_steps: 6, production_revision: 'revision',
    } as ProductionPreparationSessionRow;
    expect(buildProductionPreparationPayload(row).results).toEqual({
      session_id: 'session-1', session_status: 'failed', completed_step_ids: ['initialize'],
      total_steps: 3, next_step: plan[1], error_message: 'retryable failure',
    });
    expect(buildProductionPreparationPayload({ ...row, status: 'completed', completed_step_ids: plan.map(step => step.id) }, true).results)
      .toMatchObject({ next_step: null, preparation_token: 'revision', prepared_forecast_steps: 6, reused_existing: true });
    expect(buildProductionPreparationPayload({ ...row, status: 'superseded' }).results)
      .toMatchObject({ next_step: null, session_status: 'superseded' });
  });
});
