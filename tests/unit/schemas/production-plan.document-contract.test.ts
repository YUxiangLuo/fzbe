import { describe, test, expect } from "bun:test";
import {
  PRODUCTION_SERVICE_LEVEL_PAIRS,
  buildCanonicalProductionMps,
  buildProductionCapacityScenarios,
  resolveProductionServiceLevelPair,
  validateCompletedProductionPlan,
} from "../../../src/modules/experiments/experiments.schema";
import { issueProductionForecastEvidence } from "../../../src/modules/experiments/production-forecast-evidence";

/**
 * 本文件按《虚拟仿真系统需求确认文档-最终版20241226》第五章“生产计划的制定”
 * 的名词与公式逐条锁定服务端 MPS 复算规则。字段与文档名词的对应关系：
 *
 *   demand_forecast      需求量（需求预测结果）
 *   safety_stock         安全库存
 *   forecast_quantity    预测量 = 需求预测结果 + 安全库存            （5.4）
 *   planned_production   投入量 = 预测量 - 上月库存量 + 上月缺货量   （5.5）
 *   beginning_inventory  上月库存量
 *   production_output    产出量 = 上月投入量                         （5.2，提前期 1 个月）
 *   ending_inventory     库存量
 *   stockout             缺货量（欠交，连续缺货时含上期未补部分）
 *   service_level        服务水平 = 1 - 缺货量 / 实际需求量           （5.3）
 *
 * 文档第一个月按 5.1.3 标准化：产出量 = 需求量，库存量 = 缺货量 = 0，
 * 预测量不含安全库存，投入量 = 预测量。文档的表格是本公式在
 * 期初库存 = 0、产能不受限 时的特例。
 */

// 充足产能：保证“产出量 = 上月投入量”不被产能截断，以便单独验证文档公式。
const 充足产能 = 1_000_000;

const 四期预测 = [
  { prediction: 100, std_dev: 10 },
  { prediction: 80, std_dev: 10 },
  { prediction: 120, std_dev: 10 },
  { prediction: 90, std_dev: 10 },
];

const 文档参数下的计划表 = (forecasts = 四期预测, zScore = 1.65) =>
  buildCanonicalProductionMps({
    forecasts,
    zScore,
    initialInventory: 0,
    capacity: 充足产能,
  });

const 余额 = (row: { ending_inventory: number; stockout: number }) =>
  row.ending_inventory - row.stockout;

const 完整预测点 = (point: { prediction: number; std_dev: number }) => ({
  ...point,
  upper_error_p99: 2.33 * point.std_dev,
  uncertainty_source: "fallback",
  calibration_mean_error: null,
  calibration_count: null,
  interval_lower: point.prediction - 1.96 * point.std_dev,
  interval_upper: point.prediction + 1.96 * point.std_dev,
  interval_level: 0.95,
  interval_kind: "fallback_normal_approximation",
  coverage_claim: "none",
});

const 构造进行中快照 = (overrides: Record<string, unknown> = {}) => {
  const forecasts = 四期预测.map(完整预测点);
  const capacity = buildProductionCapacityScenarios(forecasts, 1.65).abundant;
  const mps = buildCanonicalProductionMps({
    forecasts,
    zScore: 1.65,
    initialInventory: 0,
    capacity,
  });
  return {
    experiment_id: 1,
    student_id: 2,
    status: "In Progress",
    selected_best_model: "ma",
    production_plan_completed: true,
    production_forecast_periods: forecasts.length,
    production_initial_inventory: 0,
    production_target_service_level: 0.95,
    production_safety_stock_z_score: 1.65,
    production_forecast_results: forecasts,
    production_forecast_evidence: issueProductionForecastEvidence({
      experiment_id: 1,
      student_id: 2,
      model_type: "ma",
      run_id: "doc-contract-run",
      production_revision: "doc-contract-revision",
      forecast_steps: forecasts.length,
      forecasts,
    }),
    production_mps_table: mps,
    production_capacity_mode: "scenario",
    production_capacity_scenario: "abundant",
    production_capacity: capacity,
    production_custom_capacity: null,
    ...overrides,
  };
};

describe("生产计划（MPS）与需求文档第五章公式的契约", () => {
  describe("5.1.3 标准化第一个月", () => {
    test("期初库存为 0 时，第一个月产出量等于需求量，库存量与缺货量均为 0", () => {
      const [第一月] = 文档参数下的计划表();
      expect(第一月).toMatchObject({
        period: 1,
        period_label: "期 1",
        demand_forecast: 100,
        beginning_inventory: 0,
        production_output: 100,
        ending_inventory: 0,
        stockout: 0,
        service_level: 1,
      });
    });

    test("第一个月预测量不含安全库存，投入量直接等于预测量", () => {
      const [第一月] = 文档参数下的计划表(四期预测, 2.33);
      expect(第一月!.safety_stock).toBe(0);
      expect(第一月!.forecast_quantity).toBe(第一月!.demand_forecast);
      expect(第一月!.planned_production).toBe(第一月!.forecast_quantity);
    });

    test("期初库存不为 0 时按通式处理：第一个月库存量等于期初库存，并可直接用于满足需求", () => {
      const rows = buildCanonicalProductionMps({
        forecasts: 四期预测,
        zScore: 1.65,
        initialInventory: 30,
        capacity: 充足产能,
      });
      expect(rows[0]!.beginning_inventory).toBe(30);
      expect(rows[0]!.production_output).toBe(100);
      expect(rows[0]!.ending_inventory).toBe(30);
      // 投入量 = 预测量 - 上月库存量 + 上月缺货量，其中“上月库存量”为期初库存
      expect(rows[0]!.planned_production).toBe(100 - 30);
      expect(rows[1]!.production_output).toBe(70);
      expect(rows[1]!.ending_inventory).toBe(30 + 70 - 80);
    });
  });

  describe("5.2 库存变量参数计算", () => {
    test("产能充足时，第二个月起产出量等于上月投入量", () => {
      const rows = 文档参数下的计划表();
      for (let i = 1; i < rows.length; i += 1) {
        expect(rows[i]!.production_output).toBe(rows[i - 1]!.planned_production);
      }
    });

    test("第二个月套用文档公式：库存量 = 产出量 - 实际需求量，缺货量 = 实际需求量 - 产出量", () => {
      const rows = 文档参数下的计划表();
      const 第二月 = rows[1]!;
      expect(第二月.beginning_inventory).toBe(0);
      expect(第二月.ending_inventory).toBe(
        Math.max(0, 第二月.production_output - 第二月.demand_forecast),
      );
      expect(第二月.stockout).toBe(
        Math.max(0, 第二月.demand_forecast - 第二月.production_output),
      );
    });

    test("通式：余额 = 上月余额 + 产出量 - 需求量，库存量与缺货量为余额的正负部分", () => {
      const rows = 文档参数下的计划表();
      rows.forEach((row, i) => {
        const 上月余额 = i === 0 ? 0 : 余额(rows[i - 1]!);
        expect(余额(row)).toBe(上月余额 + row.production_output - row.demand_forecast);
        expect(row.ending_inventory).toBeGreaterThanOrEqual(0);
        expect(row.stockout).toBeGreaterThanOrEqual(0);
        expect(row.ending_inventory === 0 || row.stockout === 0).toBeTrue();
        expect(row.beginning_inventory).toBe(Math.max(0, 上月余额));
      });
    });

    test("物料守恒：期初库存 + Σ产出量 - Σ需求量 = 期末余额（上月缺货由本期产出优先补发）", () => {
      for (const initialInventory of [0, 30]) {
        for (const capacity of [充足产能, 105, 90]) {
          const rows = buildCanonicalProductionMps({
            forecasts: [...四期预测, { prediction: 90, std_dev: 10 }],
            zScore: 1.65,
            initialInventory,
            capacity,
          });
          const Σ产出 = rows.reduce((s, r) => s + r.production_output, 0);
          const Σ需求 = rows.reduce((s, r) => s + r.demand_forecast, 0);
          expect(initialInventory + Σ产出 - Σ需求).toBe(余额(rows.at(-1)!));
        }
      }
    });

    test("连续缺货时，缺货量包含上期未补部分", () => {
      const rows = buildCanonicalProductionMps({
        forecasts: [
          { prediction: 100, std_dev: 0 },
          { prediction: 110, std_dev: 0 }, // 产出 100，缺货 10
          { prediction: 120, std_dev: 0 }, // 投入 120，产出 110，缺货 10 + 10
        ],
        zScore: 1.65,
        initialInventory: 0,
        capacity: 充足产能,
      });
      expect(rows[1]!.stockout).toBe(10);
      expect(rows[2]!.production_output).toBe(110);
      expect(rows[2]!.stockout).toBe(10 + (120 - 110));
    });
  });

  describe("5.3 服务水平参数计算", () => {
    test("服务水平 = 1 - 缺货量 / 实际需求量，文档示例 1000 需求、900 产出得 0.9", () => {
      const rows = 文档参数下的计划表([
        { prediction: 900, std_dev: 0 },
        { prediction: 1000, std_dev: 0 },
      ]);
      const 第二月 = rows[1]!;
      expect(第二月.production_output).toBe(900);
      expect(第二月.demand_forecast).toBe(1000);
      expect(第二月.stockout).toBe(100);
      expect(第二月.service_level).toBeCloseTo(0.9, 12);
    });

    test("每一期都满足服务水平 = 1 - 缺货量 / 需求量，并在 0 处截断", () => {
      const rows = buildCanonicalProductionMps({
        forecasts: [
          { prediction: 100, std_dev: 0 },
          { prediction: 400, std_dev: 0 }, // 缺货 300
          { prediction: 100, std_dev: 0 }, // 产出 400，累计缺货 300 + 100 - 400 = 0
          { prediction: 10, std_dev: 0 },
        ],
        zScore: 1.65,
        initialInventory: 0,
        capacity: 50, // 产能极低，使累计缺货超过本期需求
      });
      for (const row of rows) {
        const 原始值 = row.demand_forecast > 0 ? 1 - row.stockout / row.demand_forecast : 1;
        expect(row.service_level).toBeCloseTo(Math.max(0, 原始值), 12);
        expect(row.service_level).toBeGreaterThanOrEqual(0);
        expect(row.service_level).toBeLessThanOrEqual(1);
      }
      expect(rows.some((row) => row.stockout > row.demand_forecast)).toBeTrue();
    });

    test("目标服务水平只接受 0.9 / 0.95 / 0.99 三档，并与安全系数一一对应", () => {
      expect(PRODUCTION_SERVICE_LEVEL_PAIRS.map((pair) => pair.serviceLevel))
        .toEqual([0.9, 0.95, 0.99]);
      for (const pair of PRODUCTION_SERVICE_LEVEL_PAIRS) {
        expect(resolveProductionServiceLevelPair(pair.serviceLevel, pair.zScore)).toEqual(pair);
        expect(resolveProductionServiceLevelPair(String(pair.serviceLevel), String(pair.zScore)))
          .toEqual(pair);
      }
      expect(resolveProductionServiceLevelPair(0.9, 1.65)).toBeNull();
      expect(resolveProductionServiceLevelPair(0.8, 0.84)).toBeNull();
      expect(resolveProductionServiceLevelPair(null, 1.65)).toBeNull();
    });
  });

  describe("5.4 计算预测量", () => {
    test("第二个月起安全库存 = 安全系数 × σ（取整），预测量 = 需求预测结果 + 安全库存", () => {
      const rows = 文档参数下的计划表();
      for (let i = 1; i < rows.length; i += 1) {
        expect(rows[i]!.safety_stock).toBe(Math.round(1.65 * 四期预测[i]!.std_dev));
        expect(rows[i]!.forecast_quantity).toBe(rows[i]!.demand_forecast + rows[i]!.safety_stock);
      }
    });

    test("文档示例：需求预测 1000、安全库存 200 时预测量为 1200", () => {
      const rows = 文档参数下的计划表([
        { prediction: 1000, std_dev: 0 },
        { prediction: 1000, std_dev: 200 / 1.65 },
      ]);
      expect(rows[1]!.safety_stock).toBe(200);
      expect(rows[1]!.forecast_quantity).toBe(1200);
    });

    test("目标服务水平越高，安全库存与预测量越大", () => {
      const 安全库存 = PRODUCTION_SERVICE_LEVEL_PAIRS.map((pair) =>
        文档参数下的计划表(四期预测, pair.zScore)[1]!.safety_stock,
      );
      expect(安全库存).toEqual([...安全库存].sort((a, b) => a - b));
      expect(new Set(安全库存).size).toBe(安全库存.length);
    });
  });

  describe("5.5 计算投入量", () => {
    test("投入量 = 预测量 - 上月库存量 + 上月缺货量", () => {
      const rows = buildCanonicalProductionMps({
        forecasts: [...四期预测, { prediction: 90, std_dev: 10 }],
        zScore: 1.65,
        initialInventory: 0,
        capacity: 105,
      });
      for (let i = 1; i < rows.length; i += 1) {
        const 上月 = rows[i - 1]!;
        expect(rows[i]!.beginning_inventory).toBe(上月.ending_inventory);
        expect(rows[i]!.planned_production).toBe(
          Math.max(0, rows[i]!.forecast_quantity - 上月.ending_inventory + 上月.stockout),
        );
      }
      expect(rows.some((row) => row.stockout > 0)).toBeTrue();
    });

    test("文档示例：预测量 1200、上月库存 100 得 1100；上月缺货 50 得 1250", () => {
      const 有库存 = 文档参数下的计划表([
        { prediction: 1000, std_dev: 0 },
        { prediction: 900, std_dev: 0 }, // 产出 1000，需求 900 → 库存 100
        { prediction: 1000, std_dev: 200 / 1.65 }, // 预测量 1200
      ]);
      expect(有库存[1]!.ending_inventory).toBe(100);
      expect(有库存[2]!.forecast_quantity).toBe(1200);
      expect(有库存[2]!.planned_production).toBe(1200 - 100);

      const 有缺货 = 文档参数下的计划表([
        { prediction: 1000, std_dev: 0 },
        { prediction: 1050, std_dev: 0 }, // 产出 1000，需求 1050 → 缺货 50
        { prediction: 1000, std_dev: 200 / 1.65 }, // 预测量 1200
      ]);
      expect(有缺货[1]!.stockout).toBe(50);
      expect(有缺货[2]!.planned_production).toBe(1200 + 50);
    });

    test("投入量不为负：上月库存超过预测量时投入量截断为 0", () => {
      const rows = 文档参数下的计划表([
        { prediction: 500, std_dev: 0 },
        { prediction: 10, std_dev: 0 }, // 产出 500，需求 10 → 库存 490
        { prediction: 20, std_dev: 0 }, // 预测量 20 - 490 < 0
      ]);
      expect(rows[2]!.planned_production).toBe(0);
    });

    test("产能缺口形成缺货，并在下期投入量中补回", () => {
      const rows = buildCanonicalProductionMps({
        forecasts: 四期预测,
        zScore: 1.65,
        initialInventory: 0,
        capacity: 99,
      });
      expect(rows[0]!.planned_production).toBe(100);
      expect(rows[1]!.production_output).toBe(99);
      expect(rows[1]!.ending_inventory).toBe(99 - 80);
      expect(rows[2]!.production_output).toBe(Math.min(rows[1]!.planned_production, 99));
      expect(rows[2]!.stockout).toBe(120 - 19 - rows[2]!.production_output);
      expect(rows[3]!.planned_production).toBe(
        rows[3]!.forecast_quantity - 0 + rows[2]!.stockout,
      );
    });
  });

  describe("5.6 完整计划表生成", () => {
    test("计划表行数与预测期数一致，期序号与期标签连续", () => {
      for (const periods of [2, 3, 6, 12]) {
        const forecasts = Array.from({ length: periods }, (_, i) => ({
          prediction: 100 + i,
          std_dev: 5,
        }));
        const rows = 文档参数下的计划表(forecasts, 1.28);
        expect(rows).toHaveLength(periods);
        rows.forEach((row, i) => {
          expect(row.period).toBe(i + 1);
          expect(row.period_label).toBe(`期 ${i + 1}`);
        });
      }
    });

    test("整张计划表的每一期都同时满足文档各步公式（产能充足、期初库存为 0）", () => {
      const forecasts = Array.from({ length: 12 }, (_, i) => ({
        prediction: 100 + 40 * Math.sin(i),
        std_dev: 8 + i,
      }));
      const rows = 文档参数下的计划表(forecasts, 1.28);
      rows.forEach((row, i) => {
        expect(row.demand_forecast).toBe(Math.round(forecasts[i]!.prediction));
        expect(row.forecast_quantity).toBe(row.demand_forecast + row.safety_stock);
        if (i === 0) {
          expect(row.safety_stock).toBe(0);
          expect(row.production_output).toBe(row.demand_forecast);
          expect(row.ending_inventory).toBe(0);
          expect(row.stockout).toBe(0);
          expect(row.planned_production).toBe(row.forecast_quantity);
          return;
        }
        const 上月 = rows[i - 1]!;
        expect(row.safety_stock).toBe(Math.round(1.28 * forecasts[i]!.std_dev));
        expect(row.production_output).toBe(上月.planned_production);
        expect(row.planned_production).toBe(
          Math.max(0, row.forecast_quantity - 上月.ending_inventory + 上月.stockout),
        );
        expect(余额(row)).toBe(余额(上月) + row.production_output - row.demand_forecast);
        expect(row.service_level).toBeCloseTo(
          Math.max(0, row.demand_forecast > 0 ? 1 - row.stockout / row.demand_forecast : 1),
          12,
        );
      });
    });

    test("学生提交的计划表只要有一个单元格偏离文档公式即被拒绝", () => {
      const snapshot = 构造进行中快照();
      expect(validateCompletedProductionPlan(snapshot).ok).toBeTrue();

      const 字段 = [
        "safety_stock",
        "forecast_quantity",
        "planned_production",
        "production_output",
        "ending_inventory",
        "stockout",
        "service_level",
      ] as const;
      for (const field of 字段) {
        const tampered = snapshot.production_mps_table.map((row) => ({ ...row }));
        tampered[2]![field] = field === "service_level"
          ? Math.min(1, tampered[2]![field] + 0.01)
          : tampered[2]![field] + 1;
        const result = validateCompletedProductionPlan({
          ...snapshot,
          production_mps_table: tampered,
        });
        expect(result).toMatchObject({ ok: false, code: "PRODUCTION_PLAN_MISMATCH" });
        if (!result.ok) expect(result.message).toContain(field);
      }
    });
  });

  describe("产能场景（文档未定义公式，锁定现有约定）", () => {
    test("紧张 / 正常 / 充足产能分别为平均负荷的 90% / 100% / max(峰值, 110%)", () => {
      const loads = 四期预测.map((point, i) =>
        Math.round(point.prediction) + (i === 0 ? 0 : Math.round(1.65 * point.std_dev)));
      const average = Math.round(loads.reduce((a, b) => a + b, 0) / loads.length);
      expect(buildProductionCapacityScenarios(四期预测, 1.65)).toEqual({
        tight: Math.round(average * 0.9),
        normal: average,
        abundant: Math.max(Math.max(...loads), Math.round(average * 1.1)),
      });
    });
  });
});
