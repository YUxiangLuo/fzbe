import type { RowDataPacket } from "mysql2/promise";
import { pool } from "../../config";
import { assertExperimentModelReadAccess } from "./model-context";
import { queryWithTimeout } from "./model-db-guard";
import type { ModelType } from "./models.schema";

interface ActiveObservabilityRow extends RowDataPacket {
  run_id: string;
  model_type: ModelType;
  parameters: unknown;
  metrics: unknown;
}

const parseRecord = (value: unknown): Record<string, unknown> => {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value !== "string" || value.length === 0) {
    return {};
  }
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : {};
  } catch {
    return {};
  }
};

export const buildModelObservabilityMap = (
  rows: readonly ActiveObservabilityRow[],
): Partial<Record<ModelType, Record<string, unknown>>> => Object.fromEntries(
  rows.flatMap((row) => {
    const parameters = parseRecord(row.parameters);
    const observability = parseRecord(parameters.observability);
    if (Object.keys(observability).length === 0) return [];
    const metrics = parseRecord(row.metrics);
    return [[row.model_type, {
      ...observability,
      ...(observability.outer_interval_evaluation == null && metrics.interval_evaluation
        ? { outer_interval_evaluation: metrics.interval_evaluation }
        : {}),
    }]];
  }),
);

export async function getExperimentModelObservability(
  experimentId: number,
  studentId: number,
) {
  await assertExperimentModelReadAccess(experimentId, studentId);
  const [rows] = await queryWithTimeout<ActiveObservabilityRow[]>(
    pool,
    `SELECT run_id, model_type, parameters, metrics
       FROM model_training_runs
      WHERE experiment_id = ?
        AND student_id = ?
        AND is_active = true
      ORDER BY activated_at ASC, run_id ASC`,
    [experimentId, studentId],
  );
  return { models: buildModelObservabilityMap(rows) };
}
