import { z } from "zod";
import rawModelHttpTimeoutContract from "../../contracts/model-http-timeouts.json";

const positiveSafeInteger = z.number().int().positive().max(Number.MAX_SAFE_INTEGER);

const modelHttpTimeoutContractSchema = z.object({
  schemaVersion: z.literal(1),
  policyVersion: z.literal(1),
  model: z.object({
    executionHardCapMs: positiveSafeInteger,
    orchestrationHardCapMs: positiveSafeInteger,
    operationDeadlineHardCapMs: positiveSafeInteger,
    processTerminationGraceDefaultMs: positiveSafeInteger,
    responseReserveMs: positiveSafeInteger,
    nginxReadTimeoutMs: positiveSafeInteger,
    browserRequestTimeoutMs: positiveSafeInteger,
    metadataBrowserRequestTimeoutMs: positiveSafeInteger,
  }).strict(),
  adf: z.object({
    executionTimeoutMs: positiveSafeInteger,
    nginxReadTimeoutMs: positiveSafeInteger,
    browserRequestTimeoutMs: positiveSafeInteger,
  }).strict(),
}).strict().superRefine((contract, context) => {
  const { model, adf } = contract;

  if (model.executionHardCapMs > model.orchestrationHardCapMs) {
    context.addIssue({
      code: "custom",
      path: ["model", "orchestrationHardCapMs"],
      message: "must be >= model.executionHardCapMs",
    });
  }
  if (model.orchestrationHardCapMs > model.operationDeadlineHardCapMs) {
    context.addIssue({
      code: "custom",
      path: ["model", "operationDeadlineHardCapMs"],
      message: "must be >= model.orchestrationHardCapMs",
    });
  }

  const cleanupBudgetMs =
    model.processTerminationGraceDefaultMs + model.responseReserveMs;
  if (model.operationDeadlineHardCapMs + cleanupBudgetMs > model.nginxReadTimeoutMs) {
    context.addIssue({
      code: "custom",
      path: ["model", "nginxReadTimeoutMs"],
      message: "must cover the operation deadline, termination grace, and response reserve",
    });
  }
  if (model.nginxReadTimeoutMs >= model.browserRequestTimeoutMs) {
    context.addIssue({
      code: "custom",
      path: ["model", "browserRequestTimeoutMs"],
      message: "must be greater than model.nginxReadTimeoutMs",
    });
  }
  if (model.metadataBrowserRequestTimeoutMs >= model.browserRequestTimeoutMs) {
    context.addIssue({
      code: "custom",
      path: ["model", "metadataBrowserRequestTimeoutMs"],
      message: "must be less than model.browserRequestTimeoutMs",
    });
  }

  if (adf.executionTimeoutMs + cleanupBudgetMs > adf.nginxReadTimeoutMs) {
    context.addIssue({
      code: "custom",
      path: ["adf", "nginxReadTimeoutMs"],
      message: "must cover ADF execution, termination grace, and response reserve",
    });
  }
  if (adf.nginxReadTimeoutMs >= adf.browserRequestTimeoutMs) {
    context.addIssue({
      code: "custom",
      path: ["adf", "browserRequestTimeoutMs"],
      message: "must be greater than adf.nginxReadTimeoutMs",
    });
  }
});

export type ModelHttpTimeoutContract = z.infer<typeof modelHttpTimeoutContractSchema>;

export function parseModelHttpTimeoutContract(value: unknown): ModelHttpTimeoutContract {
  const result = modelHttpTimeoutContractSchema.safeParse(value);
  if (!result.success) {
    const details = result.error.issues
      .map((issue) => `${issue.path.join(".") || "contract"}: ${issue.message}`)
      .join("; ");
    throw new Error(`Invalid model HTTP timeout contract: ${details}`);
  }
  return result.data;
}

export const MODEL_HTTP_TIMEOUT_CONTRACT = parseModelHttpTimeoutContract(
  rawModelHttpTimeoutContract,
);
