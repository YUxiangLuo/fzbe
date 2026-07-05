type EnvLike = Record<string, string | undefined>;

export function isRuntimeDiskCleanupTimerDisabled(env: EnvLike = process.env): boolean {
  return env.DISABLE_RUNTIME_DISK_CLEANUP_TIMER === "1";
}

export function hasLegacyGuidedTrainingCleanupTimerDisable(env: EnvLike = process.env): boolean {
  return env.DISABLE_GUIDED_TRAINING_CLEANUP_TIMER === "1";
}

export function shouldSkipGuidedTrainingCleanupForTrigger(
  trigger: string,
  env: EnvLike = process.env,
): boolean {
  return trigger === "interval" && hasLegacyGuidedTrainingCleanupTimerDisable(env);
}
