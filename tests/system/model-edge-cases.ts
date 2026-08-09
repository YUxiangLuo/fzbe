import mysql from "mysql2/promise";
import { spawn } from "node:child_process";
import path from "node:path";
import { resolvePythonExecutable } from "../../src/utils/python";
import { assertE2eDatabaseResetAllowed } from "../../scripts/test-database-reset-guard";

type Layer = "python" | "api" | "all";
type Profile = "full";
type ModelType = "ma" | "es" | "arima" | "lstm" | "weighted_avg" | "stacking" | "boosting";

interface CliOptions {
  help: boolean;
  layer: Layer;
  profile: Profile;
  json: boolean;
}

interface ApiResponse<T = unknown> {
  status: number;
  payload: T;
}

type AppLike = {
  request: (input: string | Request, init?: RequestInit) => Response | Promise<Response>;
};

const ROOT = process.cwd();
const PYTHON_EDGE_SCRIPT = path.join(ROOT, "py", "test", "model_edge_cases.py");
const STUDENT_USERNAME = process.env.E2E_STUDENT_USERNAME ?? "20240002";
const STUDENT_PASSWORD = process.env.E2E_STUDENT_PASSWORD ?? "StudentE2E!234";
const SHIYAN_INDUSTRY = process.env.E2E_SHIYAN_INDUSTRY ?? "E2E智能制造业";
const SHIYAN_COMPANY = process.env.E2E_SHIYAN_COMPANY ?? "E2E样本企业A";
const SHIYAN_PRIMARY_PRODUCT = process.env.E2E_SHIYAN_PRODUCT_PRIMARY ?? "智能传感器A型";
const TRAIN_WINDOW = {
  trainStart: 0,
  trainEnd: 27,
  evalStart: 28,
  evalEnd: 35,
};
const BASE_MODELS: Array<Extract<ModelType, "ma" | "es" | "arima" | "lstm">> = ["ma", "es", "arima", "lstm"];
const ENSEMBLE_MODELS: Array<Extract<ModelType, "weighted_avg" | "stacking" | "boosting">> = [
  "weighted_avg",
  "stacking",
  "boosting",
];
function parseOptions(): CliOptions {
  const options: CliOptions = {
    help: false,
    layer: "all",
    profile: "full",
    json: false,
  };

  for (const arg of process.argv.slice(2)) {
    if (arg === "--help" || arg === "-h") options.help = true;
    else if (arg === "--json") options.json = true;
    else if (arg.startsWith("--layer=")) {
      const value = arg.slice("--layer=".length);
      if (!["python", "api", "all"].includes(value)) {
        throw new Error(`Invalid --layer value: ${value}`);
      }
      options.layer = value as Layer;
    } else if (arg.startsWith("--profile=")) {
      const value = arg.slice("--profile=".length);
      if (value !== "full") {
        throw new Error(`Invalid --profile value: ${value}`);
      }
      options.profile = value as Profile;
    } else {
      throw new Error(`Unknown option: ${arg}`);
    }
  }

  return options;
}

function runCommand(command: string, args: string[], label: string, env: Record<string, string | undefined> = {}) {
  return new Promise<void>((resolve, reject) => {
    console.error(`[model-edge] ${label}`);
    const child = spawn(command, args, {
      cwd: ROOT,
      stdio: "inherit",
      env: { ...process.env, ...env },
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${label} exited with code ${code}`));
      }
    });
  });
}

async function runPythonLayer(options: CliOptions) {
  await runCommand(
    process.env.PYTHON ?? resolvePythonExecutable(),
    [PYTHON_EDGE_SCRIPT, "--profile", options.profile, ...(options.json ? ["--json"] : [])],
    "python model edge cases",
  );
  return { status: "ok" };
}

async function resetAndSeedDatabase() {
  await runCommand("bun", ["run", "scripts/db-reset.ts", "--force"], "reset database for model edge API checks");
  await runCommand("bun", ["run", "scripts/e2e-seed-shiyan-fixtures.ts"], "seed shiyan model edge fixtures");
}

async function resetStudentPassword() {
  const conn = await mysql.createConnection({
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD ?? "",
    database: process.env.DB_DATABASE,
    charset: "utf8mb4",
  });
  try {
    const passwordHash = await Bun.password.hash(STUDENT_PASSWORD, {
      algorithm: "bcrypt",
      cost: 10,
    });
    const [result] = await conn.query<mysql.ResultSetHeader>(
      "UPDATE users SET password_hash = ? WHERE username = ? AND role = 'Student'",
      [passwordHash, STUDENT_USERNAME],
    );
    if (result.affectedRows !== 1) {
      throw new Error(`Could not reset password for student ${STUDENT_USERNAME}`);
    }
  } finally {
    await conn.end();
  }
}

function buildUrl(pathname: string) {
  return `http://model-edge.local${pathname}`;
}

function unwrapApiPayload(payload: unknown) {
  if (
    typeof payload === "object" &&
    payload !== null &&
    !Array.isArray(payload) &&
    "data" in payload
  ) {
    return (payload as { data: unknown }).data;
  }
  return payload;
}

async function requestApp<T>(
  app: AppLike,
  pathname: string,
  options: { method?: string; token?: string; body?: unknown } = {},
): Promise<ApiResponse<T>> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (options.token) {
    headers.Authorization = `Bearer ${options.token}`;
  }
  const response = await app.request(buildUrl(pathname), {
    method: options.method ?? "GET",
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });
  const rawText = await response.text();
  const payload = rawText ? unwrapApiPayload(JSON.parse(rawText)) : null;
  return { status: response.status, payload: payload as T };
}

function assertStatus<T>(response: ApiResponse<T>, expectedStatus: number, label: string): T {
  if (response.status !== expectedStatus) {
    throw new Error(`${label} expected HTTP ${expectedStatus}, got ${response.status}: ${JSON.stringify(response.payload)}`);
  }
  return response.payload;
}

function assertFiniteNumber(value: unknown, label: string) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number, got ${String(value)}`);
  }
}

function assertTrainingPayload(modelType: ModelType, payload: any) {
  if (payload?.status !== "success" || typeof payload.results !== "object" || payload.results === null) {
    throw new Error(`${modelType} training did not return success results: ${JSON.stringify(payload)}`);
  }
  const results = payload.results;
  for (const metricKey of ["rmse", "mae", "r2"]) {
    assertFiniteNumber(results.metrics?.[metricKey], `${modelType}.metrics.${metricKey}`);
  }
  if (!Array.isArray(results.eval_y_true) || !Array.isArray(results.eval_predictions)) {
    throw new Error(`${modelType} training missing eval arrays`);
  }
  if (results.eval_y_true.length !== 8 || results.eval_predictions.length !== 8) {
    throw new Error(`${modelType} training eval length mismatch`);
  }
  const mape = results.metrics?.mape;
  const everyActualIsZero = results.eval_y_true.every(
    (value: unknown) => typeof value === "number" && Math.abs(value) <= 1e-12,
  );
  if (everyActualIsZero) {
    if (mape !== null) {
      throw new Error(`${modelType}.metrics.mape must be null when every actual value is zero`);
    }
  } else {
    assertFiniteNumber(mape, `${modelType}.metrics.mape`);
  }
  for (const [index, prediction] of results.eval_predictions.entries()) {
    assertFiniteNumber(prediction, `${modelType}.eval_predictions[${index}]`);
    if (prediction < 0) {
      throw new Error(`${modelType}.eval_predictions[${index}] must be non-negative`);
    }
  }
  if (!Array.isArray(results.evaluate_months) || results.evaluate_months.length !== 8) {
    throw new Error(`${modelType} training missing evaluate_months`);
  }
  if (typeof results.method_name !== "string" || typeof results.forecast_strategy !== "string") {
    throw new Error(`${modelType} training missing method metadata`);
  }
  if (!Array.isArray(results.implementation_notes) || results.implementation_notes.length === 0) {
    throw new Error(`${modelType} training missing implementation_notes`);
  }

  if (modelType === "arima" && typeof results.best_order !== "object") {
    throw new Error("arima training missing best_order");
  }
  if (modelType === "lstm" && typeof results.feature_types !== "object") {
    throw new Error("lstm training missing feature_types");
  }
  if (modelType === "weighted_avg") {
    if (!Array.isArray(results.model_names) || !Array.isArray(results.weights)) {
      throw new Error("weighted_avg training missing model_names/weights");
    }
  }
  if (modelType === "stacking" && typeof results.meta_model !== "object") {
    throw new Error("stacking training missing meta_model");
  }
  if (modelType === "boosting" && !Array.isArray(results.model_names)) {
    throw new Error("boosting training missing model_names");
  }
}

function assertPredictionPayload(modelType: ModelType, payload: any, expectedLength: number) {
  if (payload?.status !== "success" || typeof payload.results !== "object" || payload.results === null) {
    throw new Error(`${modelType} prediction did not return success results: ${JSON.stringify(payload)}`);
  }
  const results = payload.results;
  if (!Array.isArray(results.predictions) || results.predictions.length !== expectedLength) {
    throw new Error(`${modelType} prediction length mismatch`);
  }
  for (const [index, point] of results.predictions.entries()) {
    if (!point || typeof point !== "object" || Array.isArray(point)) {
      throw new Error(`${modelType}.predictions[${index}] must be an object`);
    }
    const requiredFields = [
      "prediction",
      "std_dev",
      "uncertainty_source",
      "upper_error_p99",
      "calibration_source",
      "calibration_mean_error",
      "calibration_count",
      "bias_diagnostic_source",
      "interval_lower",
      "interval_upper",
      "interval_level",
      "interval_kind",
      "coverage_claim",
    ];
    for (const field of requiredFields) {
      if (!Object.prototype.hasOwnProperty.call(point, field)) {
        throw new Error(`${modelType}.predictions[${index}] missing ${field}`);
      }
    }
    assertFiniteNumber(point.prediction, `${modelType}.predictions[${index}].prediction`);
    assertFiniteNumber(point.std_dev, `${modelType}.predictions[${index}].std_dev`);
    assertFiniteNumber(point.upper_error_p99, `${modelType}.predictions[${index}].upper_error_p99`);
    assertFiniteNumber(point.interval_lower, `${modelType}.predictions[${index}].interval_lower`);
    assertFiniteNumber(point.interval_upper, `${modelType}.predictions[${index}].interval_upper`);
    assertFiniteNumber(point.interval_level, `${modelType}.predictions[${index}].interval_level`);
    if (point.std_dev < 0) {
      throw new Error(`${modelType}.predictions[${index}].std_dev must be non-negative`);
    }
    if (point.upper_error_p99 < 0) {
      throw new Error(`${modelType}.predictions[${index}].upper_error_p99 must be non-negative`);
    }
    if (point.prediction < 0) {
      throw new Error(`${modelType}.predictions[${index}].prediction must be non-negative`);
    }
    if (point.interval_lower > point.interval_upper) {
      throw new Error(`${modelType}.predictions[${index}] interval bounds are reversed`);
    }
    if (Math.abs(point.interval_level - 0.95) > 1e-9) {
      throw new Error(`${modelType}.predictions[${index}].interval_level must equal 0.95`);
    }
    if (!["model", "empirical", "fallback"].includes(point.uncertainty_source)) {
      throw new Error(`${modelType}.predictions[${index}].uncertainty_source is invalid`);
    }
    if (
      typeof point.calibration_source !== "string"
      || point.calibration_source.length === 0
    ) {
      throw new Error(`${modelType}.predictions[${index}].calibration_source is invalid`);
    }
    const calibrationIsNull = (
      point.calibration_mean_error === null
      && point.calibration_count === null
    );
    const calibrationIsValid = (
      typeof point.calibration_mean_error === "number"
      && Number.isFinite(point.calibration_mean_error)
      && Number.isInteger(point.calibration_count)
      && point.calibration_count > 0
    );
    if (!calibrationIsNull && !calibrationIsValid) {
      throw new Error(`${modelType}.predictions[${index}] calibration diagnostics are invalid`);
    }
    if (
      point.bias_diagnostic_source !== null
      && (
        typeof point.bias_diagnostic_source !== "string"
        || point.bias_diagnostic_source.length === 0
      )
    ) {
      throw new Error(`${modelType}.predictions[${index}].bias_diagnostic_source is invalid`);
    }
    if (
      point.uncertainty_reason !== undefined
      && (
        typeof point.uncertainty_reason !== "string"
        || point.uncertainty_reason.length === 0
      )
    ) {
      throw new Error(`${modelType}.predictions[${index}].uncertainty_reason is invalid`);
    }
    const intervalKinds = [
      "model_prediction_interval",
      "empirical_residual_quantile",
      "normal_approximation",
      "fallback_normal_approximation",
    ];
    if (!intervalKinds.includes(point.interval_kind)) {
      throw new Error(`${modelType}.predictions[${index}].interval_kind is invalid`);
    }
    const expectedCoverage = point.interval_kind === "model_prediction_interval"
      ? "model_nominal"
      : "none";
    if (point.coverage_claim !== expectedCoverage) {
      throw new Error(
        `${modelType}.predictions[${index}].coverage_claim must be ${expectedCoverage}`,
      );
    }
  }
  if (typeof results.method_name !== "string" || typeof results.forecast_strategy !== "string") {
    throw new Error(`${modelType} prediction missing method metadata`);
  }
  if (!Array.isArray(results.implementation_notes) || results.implementation_notes.length === 0) {
    throw new Error(`${modelType} prediction missing implementation_notes`);
  }
}

function baseExperimentPatch() {
  return {
    selected_industry: SHIYAN_INDUSTRY,
    selected_company: SHIYAN_COMPANY,
    selected_product: SHIYAN_PRIMARY_PRODUCT,
    data_window_train_start_index: TRAIN_WINDOW.trainStart,
    data_window_train_end_index: TRAIN_WINDOW.trainEnd,
    data_window_evaluate_start_index: TRAIN_WINDOW.evalStart,
    data_window_evaluate_end_index: TRAIN_WINDOW.evalEnd,
    current_step: 5,
    highest_completed_step: 4,
  };
}

function trainingBody(experimentId: number, modelType: ModelType) {
  const base = {
    experiment_id: experimentId,
    selected_industry: SHIYAN_INDUSTRY,
    selected_company: SHIYAN_COMPANY,
    selected_product: SHIYAN_PRIMARY_PRODUCT,
    data_window_train_start_index: TRAIN_WINDOW.trainStart,
    data_window_train_end_index: TRAIN_WINDOW.trainEnd,
    data_window_evaluate_start_index: TRAIN_WINDOW.evalStart,
    data_window_evaluate_end_index: TRAIN_WINDOW.evalEnd,
  };
  switch (modelType) {
    case "ma":
      return { ...base, moving_average_window: 6 };
    case "es":
      return { ...base, exponential_smoothing_alpha: 0.35 };
    case "arima":
      return { ...base, arimaD: 1 };
    case "lstm":
      return {
        ...base,
        lstmNormalization: "minmax",
        lstmTargetFeature: "销售数量",
        lstmFeatures: "价格指数,促销投入,产能利用率",
      };
    case "weighted_avg":
    case "stacking":
      return {
        ...base,
        models: BASE_MODELS.join(","),
        moving_average_window: 6,
        exponential_smoothing_alpha: 0.35,
        arimaD: 1,
        lstmNormalization: "minmax",
        lstmTargetFeature: "销售数量",
        lstmFeatures: "价格指数,促销投入,产能利用率",
      };
    case "boosting":
      return {
        ...base,
        models: BASE_MODELS.join(","),
        moving_average_window: 6,
        exponential_smoothing_alpha: 0.35,
        arimaD: 1,
        lstmNormalization: "minmax",
        lstmTargetFeature: "销售数量",
        lstmFeatures: "价格指数,促销投入,产能利用率",
      };
  }
}

async function trainGuidedModel(
  app: AppLike,
  token: string,
  experimentId: number,
  modelType: ModelType,
) {
  let session = assertStatus<any>(
    await requestApp(app, `/api/v1/models/${modelType}/guided-training/sessions`, {
      method: "POST",
      token,
      body: trainingBody(experimentId, modelType),
    }),
    200,
    `${modelType} guided session creation`,
  );

  for (let stepCount = 0; stepCount < 64 && session.status !== "completed"; stepCount += 1) {
    if (session.status === "failed" || session.status === "superseded") {
      throw new Error(`${modelType} guided session became ${session.status}: ${JSON.stringify(session)}`);
    }
    if (typeof session.next_step_id !== "string" || typeof session.session_id !== "string") {
      throw new Error(`${modelType} guided session has no runnable next step: ${JSON.stringify(session)}`);
    }
    session = assertStatus<any>(
      await requestApp(
        app,
        `/api/v1/models/${modelType}/guided-training/sessions/${session.session_id}/steps/${session.next_step_id}/run`,
        { method: "POST", token, body: {} },
      ),
      200,
      `${modelType} guided step ${session.next_step_id}`,
    );
  }

  if (session.status !== "completed" || !session.result) {
    throw new Error(`${modelType} guided training did not complete: ${JSON.stringify(session)}`);
  }
  return session.result;
}

async function prepareProductionModel(
  app: AppLike,
  token: string,
  experimentId: number,
  modelType: ModelType,
  forecastSteps: number,
) {
  let response = assertStatus<any>(
    await requestApp(app, `/api/v1/models/${modelType}/production-preparation/sessions`, {
      method: "POST",
      token,
      body: { experiment_id: experimentId, forecast_steps: forecastSteps },
    }),
    200,
    `${modelType} production preparation session creation`,
  );

  for (let stepCount = 0; stepCount < 64; stepCount += 1) {
    const session = response?.results;
    if (session?.session_status === "completed") {
      if (typeof session.preparation_token !== "string") {
        throw new Error(`${modelType} completed production session has no token: ${JSON.stringify(response)}`);
      }
      return session;
    }
    if (session?.session_status === "failed" || session?.session_status === "superseded") {
      throw new Error(
        `${modelType} production session became ${session.session_status}: ${JSON.stringify(response)}`,
      );
    }
    if (typeof session?.session_id !== "string" || typeof session?.next_step?.id !== "string") {
      throw new Error(`${modelType} production session has no runnable next step: ${JSON.stringify(response)}`);
    }
    response = assertStatus<any>(
      await requestApp(
        app,
        `/api/v1/models/${modelType}/production-preparation/sessions/${session.session_id}`
          + `/steps/${session.next_step.id}/run`,
        { method: "POST", token, body: {} },
      ),
      200,
      `${modelType} production preparation step ${session.next_step.id}`,
    );
  }

  throw new Error(`${modelType} production preparation exceeded the step limit: ${JSON.stringify(response)}`);
}

async function login(app: AppLike) {
  const payload = assertStatus<any>(
    await requestApp(app, "/api/v1/sessions", {
      method: "POST",
      body: {
        username: STUDENT_USERNAME,
        password: STUDENT_PASSWORD,
        role: "Student",
      },
    }),
    200,
    "student login",
  );
  if (typeof payload.token !== "string") {
    throw new Error(`student login missing token: ${JSON.stringify(payload)}`);
  }
  return payload.token;
}

async function createModelExperiment(
  app: AppLike,
  token: string,
) {
  const created = assertStatus<any>(
    await requestApp(app, "/api/v1/students/me/experiment-runs", {
      method: "POST",
      token,
      body: {},
    }),
    201,
    "create experiment",
  );
  const experimentId = Number(created.experiment_id);
  if (!Number.isInteger(experimentId)) {
    throw new Error(`create experiment missing experiment_id: ${JSON.stringify(created)}`);
  }
  assertStatus(
    await requestApp(app, `/api/v1/experiment-runs/${experimentId}`, {
      method: "PATCH",
      token,
      body: {
        ...baseExperimentPatch(),
        expected_version: Number(created.state_version ?? 0),
      },
    }),
    200,
    "prepare model stage experiment",
  );
  return experimentId;
}

async function runApiLayer() {
  assertE2eDatabaseResetAllowed();
  await resetAndSeedDatabase();
  await resetStudentPassword();
  const [{ app }, { pool }] = await Promise.all([
    import("../../src/index"),
    import("../../src/config"),
  ]);
  const token = await login(app);
  const experimentId = await createModelExperiment(app, token);
  const trainedModels: ModelType[] = [];

  const badModelType = await requestApp(app, "/api/v1/models/not-real/guided-training/sessions", {
    method: "POST",
    token,
    body: { experiment_id: experimentId },
  });
  assertStatus(badModelType, 400, "invalid model type should be rejected");

  const badSchema = await requestApp(app, "/api/v1/models/ma/guided-training/sessions", {
    method: "POST",
    token,
    body: {
      experiment_id: experimentId,
      moving_average_window: 1,
    },
  });
  assertStatus(badSchema, 400, "invalid MA schema should be rejected");

  const notReadyPrepare = await requestApp(app, "/api/v1/models/ma/production-preparation/sessions", {
    method: "POST",
    token,
    body: { experiment_id: experimentId, forecast_steps: 3 },
  });
  assertStatus(notReadyPrepare, 400, "production session should reject incomplete models");

  for (const modelType of BASE_MODELS) {
    console.error(`[model-edge] api train: ${modelType}`);
    const training = await trainGuidedModel(app, token, experimentId, modelType);
    assertTrainingPayload(modelType, training);
    trainedModels.push(modelType);
  }

  for (const modelType of ENSEMBLE_MODELS) {
    console.error(`[model-edge] api train: ${modelType}`);
    const training = await trainGuidedModel(app, token, experimentId, modelType);
    assertTrainingPayload(modelType, training);
    trainedModels.push(modelType);
  }

  for (const modelType of trainedModels) {
    console.error(`[model-edge] api production predict: ${modelType}`);
    const preparation = await prepareProductionModel(app, token, experimentId, modelType, 3);
    const prediction = assertStatus<any>(
      await requestApp(app, `/api/v1/models/${modelType}/predict`, {
        method: "POST",
        token,
        body: {
          experiment_id: experimentId,
          forecast_steps: 3,
          preparation_token: preparation.preparation_token,
        },
      }),
      200,
      `${modelType} predict`,
    );
    assertPredictionPayload(modelType, prediction, 3);
  }

  const expandedHorizonModels: ModelType[] = ["lstm", ...ENSEMBLE_MODELS];
  for (const modelType of expandedHorizonModels) {
    console.error(`[model-edge] api expand production horizon: ${modelType}`);
    const preparation = await prepareProductionModel(app, token, experimentId, modelType, 8);
    const prediction = assertStatus<any>(
      await requestApp(app, `/api/v1/models/${modelType}/predict`, {
        method: "POST",
        token,
        body: {
          experiment_id: experimentId,
          forecast_steps: 8,
          preparation_token: preparation.preparation_token,
        },
      }),
      200,
      `${modelType} expanded-horizon predict`,
    );
    assertPredictionPayload(modelType, prediction, 8);
  }

  const invalidPredict = await requestApp(app, "/api/v1/models/ma/predict", {
    method: "POST",
    token,
    body: { experiment_id: experimentId, forecast_steps: 0 },
  });
  assertStatus(invalidPredict, 400, "predict should reject forecast_steps <= 0");

  await pool.end();
  return {
    status: "ok",
    experiment_id: experimentId,
    trained_models: trainedModels,
    expanded_horizon_models: expandedHorizonModels,
  };
}

async function main() {
  const options = parseOptions();
  if (options.help) {
    console.log(`Usage: bun run test:system:model-edge[:python|:api]

Options:
  --layer=python|api|all  Select the execution layer (default: all)
  --profile=full         Run the complete seven-algorithm matrix
  --json                 Forward JSON output mode to the Python layer
  --help, -h             Show this help

The API layer resets and reseeds the configured test database.
It requires E2E_ALLOW_DB_RESET=1 and a disposable database named
fangzhen_e2e, fz_e2e_*, fz_pr_* or fz_test_*.`);
    return;
  }

  if (options.layer === "api" || options.layer === "all") {
    assertE2eDatabaseResetAllowed();
  }

  const startedAt = Date.now();
  const summary: Record<string, unknown> = {
    profile: options.profile,
    layer: options.layer,
    started_at: new Date(startedAt).toISOString(),
  };

  if (options.layer === "python" || options.layer === "all") {
    summary.python = await runPythonLayer(options);
  }
  if (options.layer === "api" || options.layer === "all") {
    summary.api = await runApiLayer();
  }

  summary.duration_ms = Date.now() - startedAt;
  summary.status = "ok";
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((error) => {
  console.error("[model-edge] failed");
  console.error(error);
  process.exit(1);
});
