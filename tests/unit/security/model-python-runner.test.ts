import { describe, expect, test } from "bun:test";
import { TIMEOUTS } from "../../../src/utils/constants";
import { resolvePythonProcessHardCapMs } from "../../../src/modules/models/model-python-runner";

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
});
