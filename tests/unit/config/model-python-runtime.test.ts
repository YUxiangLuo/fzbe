import { afterEach, describe, expect, test } from "bun:test";
import { buildPythonModelProcessEnv } from "../../../src/modules/models/model-python-runner";
import { PYTHON_RUNTIME_CONFIG } from "../../../src/utils/constants";

const inheritedThreadVariables = [
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "VECLIB_MAXIMUM_THREADS",
  "NUMEXPR_NUM_THREADS",
  "TF_NUM_INTRAOP_THREADS",
  "TF_NUM_INTEROP_THREADS",
] as const;

const originalValues = Object.fromEntries(
  inheritedThreadVariables.map((name) => [name, process.env[name]]),
);

afterEach(() => {
  for (const name of inheritedThreadVariables) {
    const originalValue = originalValues[name];
    if (originalValue === undefined) {
      delete process.env[name];
    } else {
      process.env[name] = originalValue;
    }
  }
});

describe("Python model process runtime", () => {
  test("overrides inherited native thread counts with the bounded model configuration", () => {
    for (const name of inheritedThreadVariables) {
      process.env[name] = "64";
    }

    const modelThreads = String(PYTHON_RUNTIME_CONFIG.MODEL_PROCESS_THREADS);
    const tfInterOpThreads = String(PYTHON_RUNTIME_CONFIG.TF_INTER_OP_THREADS);
    expect(buildPythonModelProcessEnv()).toMatchObject({
      OMP_NUM_THREADS: modelThreads,
      OPENBLAS_NUM_THREADS: modelThreads,
      MKL_NUM_THREADS: modelThreads,
      VECLIB_MAXIMUM_THREADS: modelThreads,
      NUMEXPR_NUM_THREADS: modelThreads,
      TF_NUM_INTRAOP_THREADS: modelThreads,
      TF_NUM_INTEROP_THREADS: tfInterOpThreads,
    });
  });
});
