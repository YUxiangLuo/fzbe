import path from "path";
import { TIMEOUTS } from "../../utils/constants";
import { PY_DIR } from "../../utils/python";

export const TRAINING_TIMEOUT = TIMEOUTS.MODEL_TRAINING;
export const MODEL_ORCHESTRATION_HARD_CAP = TIMEOUTS.MODEL_ORCHESTRATION_HARD_CAP;
export const GUIDED_TRAINING_PROTOCOL_VERSION = 3;
export const PRODUCTION_PREPARATION_PROTOCOL_VERSION = 1;
export const DEFAULT_PRODUCTION_FORECAST_STEPS = 6;
export const MIN_TRAINING_POINTS = 8;
export const MIN_EVALUATION_POINTS = 2;
export const GUIDED_DATASET_FILE_NAME = "dataset.json";
export const PRODUCTION_PREPARATION_ROOT = path.join(PY_DIR, "production_preparation");
