import { afterEach, describe, expect, mock, test } from "bun:test";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { ModelArtifactError } from "../../../src/modules/models/model-errors";

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const PY_DIR = path.resolve(import.meta.dir, "../../../py");
const pythonModulePath = path.join(srcRoot, "utils", "python.ts");
const modelArtifactsModulePath = path.join(
  srcRoot,
  "modules",
  "models",
  "model-artifacts.ts",
);
const projectPythonCandidates = process.platform === "win32"
  ? [
    path.join(PY_DIR, "venv", "Scripts", "python.exe"),
    path.join(PY_DIR, ".venv", "Scripts", "python.exe"),
  ]
  : [
    path.join(PY_DIR, "venv", "bin", "python"),
    path.join(PY_DIR, ".venv", "bin", "python"),
  ];
// These tests need only Python's standard-library pickle module. Injecting a
// host interpreter keeps the unit gate independent of the production model
// venv, while production code continues to require that controlled runtime.
const fixturePythonExecutable = [
  Bun.which("python3"),
  Bun.which("python"),
  ...projectPythonCandidates,
].find((candidate): candidate is string => {
  if (!candidate || !fs.existsSync(candidate)) return false;
  try {
    return Bun.spawnSync([candidate, "-c", "import pickle"]).exitCode === 0;
  } catch {
    // Windows Store aliases can exist without launching an interpreter.
    return false;
  }
});

if (!fixturePythonExecutable) {
  throw new Error("A Python interpreter is required for artifact fixture tests");
}

mock.module(pythonModulePath, () => ({
  PY_DIR,
  resolvePythonExecutable: () => fixturePythonExecutable,
}));

const {
  computeEnsembleArtifactSha256,
  verifyBaseModelArtifactIntegrity,
} = await import(`${modelArtifactsModulePath}?boosting-baseline-test`);

const createdArtifacts: string[] = [];

const writePickleState = (state: Record<string, unknown>) => {
  const modelPath = path.join(
    PY_DIR,
    "trained_models",
    `artifact-baseline-test-${crypto.randomUUID()}.pkl`,
  );
  fs.mkdirSync(path.dirname(modelPath), { recursive: true });
  const completed = Bun.spawnSync([
    fixturePythonExecutable,
    "-c",
    [
      "import json",
      "import pickle",
      "import sys",
      "with open(sys.argv[1], 'wb') as handle:",
      "    pickle.dump(json.loads(sys.argv[2]), handle, protocol=pickle.HIGHEST_PROTOCOL)",
    ].join("\n"),
    modelPath,
    JSON.stringify(state),
  ]);
  if (completed.exitCode !== 0) {
    throw new Error(new TextDecoder().decode(completed.stderr));
  }
  createdArtifacts.push(modelPath);
  return modelPath;
};

const baselineOnlyBoostingState = (baseline: Record<string, unknown>) => ({
  model_type: "boosting",
  boosting_variant: "residual",
  model_chain: [],
  stage_coefficients: [],
  base_model_states: {},
  baseline,
  uncertainty: {
    calibration_residuals: [1, -1, 2, -2],
    calibration_source: "post_selection_calibration",
    reference_series: [10, 12, 11, 13],
  },
});

afterEach(() => {
  for (const artifact of createdArtifacts.splice(0)) {
    fs.rmSync(artifact, { force: true });
  }
});

describe("baseline-only Boosting artifact integrity", () => {
  test("hashes and semantically validates a valid root-only artifact", async () => {
    const modelPath = writePickleState(
      baselineOnlyBoostingState({ kind: "naive", last_value: 40 }),
    );

    const digest = await computeEnsembleArtifactSha256("boosting", modelPath);
    expect(digest).toMatch(/^[a-f\d]{64}$/);
    await expect(
      verifyBaseModelArtifactIntegrity("boosting", modelPath, null),
    ).resolves.toEqual({
      sha256: digest,
      legacyChecksum: true,
    });
  });

  test("does not permit other ensemble types to use the root-only exception", async () => {
    const modelPath = writePickleState({
      ...baselineOnlyBoostingState({ kind: "mean", training_mean: 20 }),
      model_type: "weighted_avg",
    });

    await expect(
      computeEnsembleArtifactSha256("weighted_avg", modelPath),
    ).rejects.toBeInstanceOf(ModelArtifactError);
  });

  test("rejects a root-only Boosting artifact with an invalid persisted baseline", async () => {
    const modelPath = writePickleState(
      baselineOnlyBoostingState({ kind: "naive" }),
    );

    await expect(
      computeEnsembleArtifactSha256("boosting", modelPath),
    ).rejects.toBeInstanceOf(ModelArtifactError);
  });

  test("rejects a root-only legacy zero baseline because it has no predictive member", async () => {
    const modelPath = writePickleState(
      baselineOnlyBoostingState({ kind: "zero" }),
    );

    await expect(
      computeEnsembleArtifactSha256("boosting", modelPath),
    ).rejects.toBeInstanceOf(ModelArtifactError);
  });
});
