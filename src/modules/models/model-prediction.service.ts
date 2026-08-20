import { TIMEOUTS } from "../../utils/constants";
import type { ModelType } from "./models.schema";
import {
  ModelArtifactError,
  ModelConflictError,
} from "./model-errors";
import {
  withModelOperationContext,
  withModelOperationLease,
  type ModelOperationLease,
} from "./model-operation-lease";
import { runPythonModel } from "./model-python-runner";
import {
  applyPlanningUncertaintyFloor,
  normalizeDemandPredictionPoints,
} from "./model-results";
import { getActiveModelTrainingRun } from "./model-run-store";
import { loadDatasetFromDb } from "./model-datasets";
import {
  invalidateProductionArtifactForRun,
  verifyBaseModelArtifactIntegrity,
  verifyProductionBaseArtifactForRun,
} from "./model-artifacts";
import {
  type ExperimentModelContextRow,
  assertExperimentModelAccess,
  assertModelReadyForProduction,
  getExperimentTrainingConfig,
  requireActiveModelTrainingRun,
} from "./model-context";
import { validateActiveRunAgainstSnapshot } from "./model-runtime-identity";
import { issueProductionForecastEvidence } from "../experiments/production-forecast-evidence";

const predictExperimentModelWithLease = async (
  modelType: ModelType,
  studentId: number,
  experimentId: number,
  forecastSteps: number,
  preparationToken: string,
  experiment: ExperimentModelContextRow,
  lease: ModelOperationLease,
) => {
  const activeRun = await requireActiveModelTrainingRun(experimentId, studentId, modelType);
  if (
    activeRun.production_revision !== preparationToken
    || activeRun.production_artifact_forecast_steps === null
    || Number(activeRun.production_artifact_forecast_steps) < forecastSteps
  ) {
    throw new ModelConflictError(
      "生产预测准备已失效或预测期数不足，请重新准备生产模型",
      "PRODUCTION_PREPARATION_STALE",
    );
  }
  const productionModelPath = activeRun.production_artifact_path;
  if (!productionModelPath) {
    throw new ModelConflictError(
      `未找到该实验已准备的 ${modelType.toUpperCase()} 生产预测模型`,
      "PRODUCTION_PREPARATION_REQUIRED",
    );
  }
  let verifiedProductionSha256: string | null = null;
  try {
    verifiedProductionSha256 = await verifyProductionBaseArtifactForRun(
      modelType,
      activeRun,
      lease.terminationSignal,
    );
  } catch (error) {
    if (!(error instanceof ModelArtifactError)) throw error;
    await invalidateProductionArtifactForRun(activeRun);
    throw new ModelConflictError(
      `${modelType.toUpperCase()} 生产模型产物已失效，请重新准备生产模型`,
      "PRODUCTION_PREPARATION_REQUIRED",
      { required_action: "prepare" },
    );
  }

  const trainingConfig = getExperimentTrainingConfig(experiment);
  const allData = await loadDatasetFromDb(
    trainingConfig.selected_industry,
    trainingConfig.selected_company,
    trainingConfig.selected_product,
    lease.terminationSignal,
  );
  validateActiveRunAgainstSnapshot(activeRun, JSON.stringify(allData));

  const pythonParams = {
    loadModel: productionModelPath,
    forecastSteps,
    pretty: true,
  };

  let result: Record<string, unknown>;
  try {
    result = (await runPythonModel(
      modelType,
      pythonParams,
      TIMEOUTS.MODEL_PREDICTION,
      undefined,
      lease.terminationSignal,
    )) as Record<string, unknown>;
  } catch (error) {
    if (!(error instanceof ModelArtifactError)) throw error;
    const currentRun = await getActiveModelTrainingRun(experimentId, studentId, modelType);
    if (
      currentRun?.run_id !== activeRun.run_id
      || currentRun.production_revision !== preparationToken
    ) {
      throw new ModelConflictError(
        "预测期间模型版本已变化，请重新准备生产模型",
        "PRODUCTION_PREPARATION_STALE",
      );
    }
    await invalidateProductionArtifactForRun(activeRun);
    throw new ModelConflictError(
      `${modelType.toUpperCase()} 生产模型产物无法加载，请重新准备生产模型`,
      "PRODUCTION_PREPARATION_REQUIRED",
      { required_action: "prepare" },
    );
  }
  lease.throwIfTerminated();
  const currentRun = await getActiveModelTrainingRun(experimentId, studentId, modelType);
  if (
    currentRun?.run_id !== activeRun.run_id
    || currentRun.production_revision !== preparationToken
  ) {
    throw new ModelConflictError(
      "预测期间模型版本已变化，请重新准备生产模型",
      "PRODUCTION_PREPARATION_STALE",
    );
  }
  if (verifiedProductionSha256) {
    try {
      await verifyBaseModelArtifactIntegrity(
        modelType,
        productionModelPath,
        verifiedProductionSha256,
        lease.terminationSignal,
      );
    } catch (error) {
      if (!(error instanceof ModelArtifactError)) throw error;
      await invalidateProductionArtifactForRun(activeRun);
      throw new ModelConflictError(
        `${modelType.toUpperCase()} 生产模型产物在预测期间发生变化，请重新准备生产模型`,
        "PRODUCTION_PREPARATION_REQUIRED",
        { required_action: "prepare" },
      );
    }
  }
  await lease.verifyOwned();

  let predictions: ReturnType<typeof normalizeDemandPredictionPoints>;
  try {
    const normalizationOptions = {
      requireStdDev: true,
      requireUpperErrorP99: true,
      requireCalibrationDiagnostics: true,
      requireIntervalMetadata: true,
    } as const;
    predictions = normalizeDemandPredictionPoints(
      result.predictions,
      forecastSteps,
      normalizationOptions,
    );
    predictions = applyPlanningUncertaintyFloor(predictions, activeRun.metrics);
    predictions = normalizeDemandPredictionPoints(
      predictions,
      forecastSteps,
      normalizationOptions,
    );
  } catch (error) {
    if (!(error instanceof Error) || !error.message.startsWith("[PYTHON_DATA]")) {
      throw error;
    }
    await invalidateProductionArtifactForRun(activeRun);
    throw new ModelConflictError(
      `${modelType.toUpperCase()} 生产模型返回了无效预测结果，请重新准备生产模型`,
      "PRODUCTION_PREPARATION_REQUIRED",
      { required_action: "prepare" },
    );
  }

  const metadata = Object.fromEntries(
    ["method_name", "forecast_strategy", "implementation_notes"]
      .filter((key) => result[key] !== undefined)
      .map((key) => [key, result[key]]),
  );

  return {
    status: "success",
    results: {
      predictions,
      forecast_evidence: issueProductionForecastEvidence({
        experiment_id: experimentId,
        student_id: studentId,
        model_type: modelType,
        run_id: activeRun.run_id,
        production_revision: preparationToken,
        forecast_steps: forecastSteps,
        forecasts: predictions,
      }),
      ...metadata,
    },
  };
};

export async function predictExperimentModel(
  modelType: ModelType,
  studentId: number,
  experimentId: number,
  forecastSteps: number,
  preparationToken: string,
) {
  return await withModelOperationContext("prediction", async (operationContext) => {
    const experiment = await assertExperimentModelAccess(experimentId, studentId);
    assertModelReadyForProduction(experiment, modelType);
    operationContext.throwIfTerminated();

    return await withModelOperationLease(
      operationContext,
      { execution: { experimentId, modelType } },
      async (lease) => await predictExperimentModelWithLease(
        modelType,
        studentId,
        experimentId,
        forecastSteps,
        preparationToken,
        experiment,
        lease,
      ),
    );
  });
}
