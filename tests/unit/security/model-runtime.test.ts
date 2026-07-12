import { describe, expect, test } from "bun:test";
import {
  buildVersionedModelArtifactPath,
  buildModelExecutionLockName,
  buildModelSlotLockName,
} from "../../../src/modules/models/model-runtime";

describe("model runtime helpers", () => {
  test("builds immutable versioned model artifact paths", () => {
    const result = buildVersionedModelArtifactPath(42, 7, "lstm", "revision-1");

    expect(result).toContain("trained_models");
    expect(result.endsWith("exp42-student7-lstm-backtest-revision-1.pkl")).toBe(true);
  });

  test("builds one execution lock name for the whole experiment", () => {
    expect(buildModelExecutionLockName(42, "weighted_avg")).toBe(
      "fangzhen-model:exp:42",
    );
  });

  test("builds slot lock names scoped by slot number", () => {
    expect(buildModelSlotLockName(2)).toBe("fangzhen-model:slot:2");
  });
});
