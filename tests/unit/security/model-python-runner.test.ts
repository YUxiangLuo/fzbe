import { describe, expect, test } from "bun:test";
import { TIMEOUTS } from "../../../src/utils/constants";
import {
  buildPythonModelProcessEnv,
  resolvePythonProcessHardCapMs,
} from "../../../src/modules/models/model-python-runner";

describe("model Python runner hard caps", () => {
  test("reserves orchestration cleanup time around every production refit child", () => {
    expect(resolvePythonProcessHardCapMs("lstm", "production_prepare_model.py")).toBe(
      TIMEOUTS.MODEL_ORCHESTRATION_HARD_CAP,
    );
    expect(resolvePythonProcessHardCapMs("ma", "production_prepare_model.py")).toBe(
      TIMEOUTS.MODEL_ORCHESTRATION_HARD_CAP,
    );
  });

  test("keeps direct base-model processes at the immutable execution cap", () => {
    expect(resolvePythonProcessHardCapMs("lstm", "guided_model.py")).toBe(
      TIMEOUTS.MODEL_EXECUTION_HARD_CAP,
    );
    expect(resolvePythonProcessHardCapMs("weighted_avg", "guided_ensemble_model.py")).toBe(
      TIMEOUTS.MODEL_ORCHESTRATION_HARD_CAP,
    );
  });

  test("keeps server-launched ensemble children inside the global job budget", () => {
    const previous = process.env.FZ_ENSEMBLE_MAX_WORKERS;
    process.env.FZ_ENSEMBLE_MAX_WORKERS = "8";
    try {
      expect(buildPythonModelProcessEnv().FZ_ENSEMBLE_MAX_WORKERS).toBe("1");
    } finally {
      if (previous === undefined) delete process.env.FZ_ENSEMBLE_MAX_WORKERS;
      else process.env.FZ_ENSEMBLE_MAX_WORKERS = previous;
    }
  });
});
