import path from "node:path";
import { access, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { normalizeSalesValue, normalizeYearMonth } from "../src/modules/datasets/dataset-values";
import {
  computeBaseModelArtifactManifestSha256,
} from "../src/modules/models/base-model-artifact-digest";
import { runPythonModel } from "../src/modules/models/model-python-runner";
import { hashDatasetSnapshot } from "../src/modules/models/model-runtime-identity";
import { readCsvRows } from "../src/utils/csv-processor";

export interface ParallelMaTrainingTemplate {
  artifactPath: string;
  artifactSha256: string;
  datasetRevision: string;
  metrics: Record<string, unknown> & {
    rmse: number;
    mae: number;
    r2: number;
  };
  parameters: {
    moving_average_window: number;
  };
  temporaryDirectory: string;
}

const asRecord = (value: unknown, label: string): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
};

const requireFiniteMetric = (
  metrics: Record<string, unknown>,
  name: "rmse" | "mae" | "r2",
) => {
  const value = metrics[name];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`parallel MA fixture returned an invalid ${name} metric`);
  }
};

/**
 * Keep this normalization byte-for-byte compatible with loadDatasetFromDb.
 * The active training run revision is calculated from the resulting JSON, so a
 * seed-specific shortcut here would make production preparation reject the run.
 */
export function normalizeParallelModelDatasetRows(
  rows: readonly Record<string, string>[],
): Record<string, unknown>[] {
  return rows.map((row, index) => {
    const separateTime = normalizeYearMonth(row["年份"], row["月份"]);
    const combinedMonthMatch = String(row["月份"] ?? "")
      .trim()
      .match(/^(\d{4})-(0?[1-9]|1[0-2])$/);
    const combinedTime = combinedMonthMatch
      ? normalizeYearMonth(combinedMonthMatch[1], combinedMonthMatch[2])
      : null;
    const time = separateTime ?? combinedTime;
    if (!time) {
      throw new Error(`parallel dataset row ${index + 1} has an invalid year/month`);
    }

    const sales = normalizeSalesValue(row["销售数量"]);
    if (!sales) {
      throw new Error(`parallel dataset row ${index + 1} has an invalid sales value`);
    }

    const normalizedRow = {
      ...row,
      "销售数量": sales.canonical,
    };
    return separateTime
      ? { ...normalizedRow, "年份": time.year, "月份": time.month }
      : { ...normalizedRow, "月份": `${time.year}-${time.month}` };
  });
}

export async function prepareParallelMaTrainingTemplate(input: {
  datasetFilePath: string;
  trainStartIndex: number;
  trainEndIndex: number;
  evaluateStartIndex: number;
  evaluateEndIndex: number;
  movingAverageWindow: number;
}): Promise<ParallelMaTrainingTemplate> {
  const rows = await readCsvRows(input.datasetFilePath);
  const normalizedRows = normalizeParallelModelDatasetRows(rows);
  const datasetJson = JSON.stringify(normalizedRows);
  const temporaryDirectory = await mkdtemp(
    path.join(tmpdir(), "fz-parallel-ma-fixture-"),
  );
  const datasetSnapshotPath = path.join(temporaryDirectory, "dataset.json");
  const artifactPath = path.join(temporaryDirectory, "ma.pkl");

  try {
    await Bun.write(datasetSnapshotPath, datasetJson);
    const result = asRecord(await runPythonModel("ma", {
      dataFile: datasetSnapshotPath,
      trainStart: input.trainStartIndex,
      trainEnd: input.trainEndIndex,
      evalStart: input.evaluateStartIndex,
      evalEnd: input.evaluateEndIndex,
      moving_average_window: input.movingAverageWindow,
      saveModel: artifactPath,
      pretty: true,
    }), "parallel MA fixture result");
    const metrics = asRecord(result.metrics, "parallel MA fixture metrics");
    requireFiniteMetric(metrics, "rmse");
    requireFiniteMetric(metrics, "mae");
    requireFiniteMetric(metrics, "r2");
    await access(artifactPath);
    const artifactSha256 = await computeBaseModelArtifactManifestSha256(
      "ma",
      artifactPath,
    );

    return {
      artifactPath,
      artifactSha256,
      datasetRevision: hashDatasetSnapshot(datasetJson),
      metrics: metrics as ParallelMaTrainingTemplate["metrics"],
      parameters: {
        moving_average_window: input.movingAverageWindow,
      },
      temporaryDirectory,
    };
  } catch (error) {
    await rm(temporaryDirectory, { recursive: true, force: true });
    throw error;
  }
}

export async function disposeParallelMaTrainingTemplate(
  template: ParallelMaTrainingTemplate,
) {
  await rm(template.temporaryDirectory, { recursive: true, force: true });
}
