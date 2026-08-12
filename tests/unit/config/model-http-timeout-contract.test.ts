import { describe, expect, test } from "bun:test";
import rawContract from "../../../contracts/model-http-timeouts.json";
import {
  MODEL_HTTP_TIMEOUT_CONTRACT,
  parseModelHttpTimeoutContract,
} from "../../../src/config/model-http-timeout-contract";

const cloneContract = () => structuredClone(rawContract) as typeof rawContract;

describe("model HTTP timeout contract", () => {
  test("loads the committed versioned contract and preserves the transport ordering", () => {
    expect(MODEL_HTTP_TIMEOUT_CONTRACT.schemaVersion).toBe(1);
    expect(MODEL_HTTP_TIMEOUT_CONTRACT.policyVersion).toBe(1);

    const { model, adf } = MODEL_HTTP_TIMEOUT_CONTRACT;
    const cleanupMs =
      model.processTerminationGraceDefaultMs + model.responseReserveMs;
    expect(model.executionHardCapMs).toBeLessThanOrEqual(model.orchestrationHardCapMs);
    expect(model.orchestrationHardCapMs).toBeLessThanOrEqual(
      model.operationDeadlineHardCapMs,
    );
    expect(model.operationDeadlineHardCapMs + cleanupMs).toBeLessThanOrEqual(
      model.nginxReadTimeoutMs,
    );
    expect(model.nginxReadTimeoutMs).toBeLessThan(model.browserRequestTimeoutMs);
    expect(adf.executionTimeoutMs + cleanupMs).toBeLessThanOrEqual(
      adf.nginxReadTimeoutMs,
    );
    expect(adf.nginxReadTimeoutMs).toBeLessThan(adf.browserRequestTimeoutMs);
  });

  test("rejects unsupported versions, unknown keys, and invalid integers", () => {
    expect(() => parseModelHttpTimeoutContract({
      ...cloneContract(),
      policyVersion: 2,
    })).toThrow("expected 1");

    const unknownKey = { ...cloneContract(), unexpected: true };
    expect(() => parseModelHttpTimeoutContract(unknownKey)).toThrow(
      "Unrecognized key",
    );

    const invalidInteger = cloneContract();
    invalidInteger.model.browserRequestTimeoutMs = Number.MAX_SAFE_INTEGER + 1;
    expect(() => parseModelHttpTimeoutContract(invalidInteger)).toThrow(
      "Invalid model HTTP timeout contract",
    );

    const nonPositive = cloneContract();
    nonPositive.adf.executionTimeoutMs = 0;
    expect(() => parseModelHttpTimeoutContract(nonPositive)).toThrow(
      "Too small",
    );
  });

  test("rejects transport budgets that cannot contain backend cleanup", () => {
    const modelTooShort = cloneContract();
    modelTooShort.model.nginxReadTimeoutMs = 670_000;
    expect(() => parseModelHttpTimeoutContract(modelTooShort)).toThrow(
      "must cover the operation deadline",
    );

    const browserTooShort = cloneContract();
    browserTooShort.model.browserRequestTimeoutMs =
      browserTooShort.model.nginxReadTimeoutMs;
    expect(() => parseModelHttpTimeoutContract(browserTooShort)).toThrow(
      "must be greater than model.nginxReadTimeoutMs",
    );

    const adfTooShort = cloneContract();
    adfTooShort.adf.nginxReadTimeoutMs = 40_000;
    expect(() => parseModelHttpTimeoutContract(adfTooShort)).toThrow(
      "must cover ADF execution",
    );
  });
});
