import { afterEach, beforeEach, expect, mock, test } from "bun:test";

mock.module("../../../src/utils/constants", () => ({
  PYTHON_RUNTIME_CONFIG: { MODEL_PROCESS_THREADS: 2, TF_INTER_OP_THREADS: 1 },
}));
const { buildPythonModelProcessEnv } = await import("../../../src/utils/python-process-env");
const keys = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
  "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "TF_CPP_MIN_LOG_LEVEL"];
let saved: Array<string | undefined>;
beforeEach(() => {
  saved = keys.map(key => process.env[key]);
  keys.forEach(key => { delete process.env[key]; });
});
afterEach(() => {
  keys.forEach((key, index) => {
    if (saved[index] === undefined) delete process.env[key];
    else process.env[key] = saved[index];
  });
});

test("product thread budgets work without native environment variables supplied by CI", () => {
  expect(buildPythonModelProcessEnv()).toEqual({
    OMP_NUM_THREADS: "2", OPENBLAS_NUM_THREADS: "2", MKL_NUM_THREADS: "2", VECLIB_MAXIMUM_THREADS: "2",
    NUMEXPR_NUM_THREADS: "2", TF_NUM_INTRAOP_THREADS: "2", TF_NUM_INTEROP_THREADS: "1", TF_CPP_MIN_LOG_LEVEL: "3",
  });
});

test("explicit native environment overrides remain authoritative", () => {
  process.env.OPENBLAS_NUM_THREADS = "3";
  process.env.TF_NUM_INTEROP_THREADS = "4";
  expect(buildPythonModelProcessEnv()).toMatchObject({
    OMP_NUM_THREADS: "2", OPENBLAS_NUM_THREADS: "3", TF_NUM_INTEROP_THREADS: "4",
  });
});
