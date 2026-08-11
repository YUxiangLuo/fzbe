import { describe, expect, test } from "bun:test";
import { buildExperimentScopedModelPath } from "../../../src/modules/models/model-runtime";

describe("model runtime helpers", () => {
  test("builds experiment-scoped model artifact paths", () => {
    const result = buildExperimentScopedModelPath(42, 7, "lstm");

    expect(result).toContain("trained_models");
    expect(result.endsWith("exp42-student7-lstm.pkl")).toBe(true);
  });
});
