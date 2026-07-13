import mysql from "mysql2/promise";
import { spawn } from "node:child_process";
import path from "node:path";
import { resolvePythonExecutable } from "../src/utils/python";

type Layer = "python" | "api" | "all";
type Profile = "full";
type ModelType = "ma" | "es" | "arima" | "lstm" | "weighted_avg" | "stacking" | "boosting";

interface CliOptions {
  layer: Layer;
  profile: Profile;
  json: boolean;
  keepArtifacts: boolean;
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
    layer: "all",
    profile: "full",
    json: false,
    keepArtifacts: false,
  };

  for (const arg of process.argv.slice(2)) {
    if (arg === "--json") options.json = true;
    else if (arg === "--keep-artifacts") options.keepArtifacts = true;
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
  for (const metricKey of ["rmse", "mae", "mape", "r2"]) {
    assertFiniteNumber(results.metrics?.[metricKey], `${modelType}.metrics.${metricKey}`);
  }
  if (!Array.isArray(results.eval_y_true) || !Array.isArray(results.eval_predictions)) {
    throw new Error(`${modelType} training missing eval arrays`);
  }
  if (results.eval_y_true.length !== 8 || results.eval_predictions.length !== 8) {
    throw new Error(`${modelType} training eval length mismatch`);
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
    assertFiniteNumber(point.prediction, `${modelType}.predictions[${index}].prediction`);
    assertFiniteNumber(point.std_dev, `${modelType}.predictions[${index}].std_dev`);
    if (point.std_dev < 0) {
      throw new Error(`${modelType}.predictions[${index}].std_dev must be non-negative`);
    }
    if (point.prediction < 0) {
      throw new Error(`${modelType}.predictions[${index}].prediction must be non-negative`);
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
  await resetAndSeedDatabase();
  await resetStudentPassword();
  const [{ app }, { pool }] = await Promise.all([
    import("../src/index"),
    import("../src/config"),
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

  const notReadyExperimentId = await createModelExperiment(app, token);
  const notReadyPrepare = await requestApp(app, "/api/v1/models/ma/prepare-production", {
    method: "POST",
    token,
    body: { experiment_id: notReadyExperimentId, forecast_steps: 3 },
  });
  assertStatus(notReadyPrepare, 400, "prepare-production should reject incomplete models");

  for (const modelType of trainedModels) {
    console.error(`[model-edge] api production predict: ${modelType}`);
    const preparation = assertStatus<any>(
      await requestApp(app, `/api/v1/models/${modelType}/prepare-production`, {
        method: "POST",
        token,
        body: { experiment_id: experimentId, forecast_steps: 3 },
      }),
      200,
      `${modelType} prepare-production`,
    );
    const prediction = assertStatus<any>(
      await requestApp(app, `/api/v1/models/${modelType}/predict`, {
        method: "POST",
        token,
        body: {
          experiment_id: experimentId,
          forecast_steps: 3,
          preparation_token: preparation.results?.preparation_token,
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
    const preparation = assertStatus<any>(
      await requestApp(app, `/api/v1/models/${modelType}/prepare-production`, {
        method: "POST",
        token,
        body: { experiment_id: experimentId, forecast_steps: 8 },
      }),
      200,
      `${modelType} expanded-horizon prepare-production`,
    );
    const prediction = assertStatus<any>(
      await requestApp(app, `/api/v1/models/${modelType}/predict`, {
        method: "POST",
        token,
        body: {
          experiment_id: experimentId,
          forecast_steps: 8,
          preparation_token: preparation.results?.preparation_token,
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
