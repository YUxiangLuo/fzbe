import { describe, expect, test } from "bun:test";
import {
  AdfValidationError,
  parseAdfResultPayload,
} from "../../../src/utils/adf";

const validResult = {
  diff_order: 1,
  statistic: -3.84,
  p_value: 0.018,
  used_lags: 2,
  n_obs: 11,
  stationary: true,
  critical_values: { "1%": -3.75, "5%": -2.99, "10%": -2.64 },
};

describe("ADF Python result parser", () => {
  test("preserves lag and observation metadata", () => {
    expect(parseAdfResultPayload([validResult])).toEqual([validResult]);
  });

  test("rejects a successful row that omits transport fields", () => {
    const { used_lags: _omitted, ...missingUsedLags } = validResult;
    expect(() => parseAdfResultPayload([missingUsedLags])).toThrow(
      "[PYTHON_DATA] ADF result 0 has invalid fields",
    );
  });

  test("maps an all-error Python result to the existing small-data validation error", () => {
    expect(() => parseAdfResultPayload([{ diff_order: 0, error: "too short" }]))
      .toThrow(AdfValidationError);
  });

  test("rejects duplicate differencing orders", () => {
    expect(() => parseAdfResultPayload([validResult, validResult])).toThrow(
      "[PYTHON_DATA] ADF result 1 repeats a differencing order",
    );
  });
});
