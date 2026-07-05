import { describe, expect, test } from "bun:test";
import {
  hasLegacyGuidedTrainingCleanupTimerDisable,
  isRuntimeDiskCleanupTimerDisabled,
  shouldSkipGuidedTrainingCleanupForTrigger,
} from "../../../src/utils/runtime-cleanup-config";

describe("runtime cleanup configuration", () => {
  test("uses DISABLE_RUNTIME_DISK_CLEANUP_TIMER as the only whole-timer kill switch", () => {
    expect(isRuntimeDiskCleanupTimerDisabled({
      DISABLE_RUNTIME_DISK_CLEANUP_TIMER: "1",
      DISABLE_GUIDED_TRAINING_CLEANUP_TIMER: "1",
    })).toBeTrue();

    expect(isRuntimeDiskCleanupTimerDisabled({
      DISABLE_GUIDED_TRAINING_CLEANUP_TIMER: "1",
    })).toBeFalse();
  });

  test("keeps the legacy guided-training variable scoped to interval guided cleanup", () => {
    const env = { DISABLE_GUIDED_TRAINING_CLEANUP_TIMER: "1" };

    expect(hasLegacyGuidedTrainingCleanupTimerDisable(env)).toBeTrue();
    expect(shouldSkipGuidedTrainingCleanupForTrigger("interval", env)).toBeTrue();
    expect(shouldSkipGuidedTrainingCleanupForTrigger("startup", env)).toBeFalse();
    expect(shouldSkipGuidedTrainingCleanupForTrigger("manual", env)).toBeFalse();
  });
});
