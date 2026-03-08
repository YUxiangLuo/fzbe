import { describe, expect, test } from "bun:test";
import {
  buildExperimentScopedModelPath,
  buildModelExecutionLockName,
  buildModelSlotLockName,
} from "../../../src/modules/models/model-runtime";

describe("model runtime helpers", () => {
  test("builds experiment-scoped model artifact paths", () => {
    const result = buildExperimentScopedModelPath(42, 7, "lstm");

    expect(result).toContain("trained_models");
    expect(result.endsWith("exp42-student7-lstm.pkl")).toBe(true);
  });

  test("builds execution lock names scoped by experiment and model", () => {
    expect(buildModelExecutionLockName(42, "weighted_avg")).toBe(
      "fangzhen-model:exp:42:weighted_avg",
    );
  });

  test("builds slot lock names scoped by slot number", () => {
    expect(buildModelSlotLockName(2)).toBe("fangzhen-model:slot:2");
  });
});