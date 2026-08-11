import { runAdf, type AdfInputPoint, type AdfResult } from "../../utils/adf";
import { TIMEOUTS } from "../../utils/constants";
import {
  withModelOperationContext,
  withModelOperationLease,
} from "../models/model-operation-lease";

export async function performAdfTest(series: AdfInputPoint[]): Promise<AdfResult> {
  return await withModelOperationContext("adf", async (context) =>
    await withModelOperationLease(context, {}, async (lease) => {
      lease.throwIfTerminated();
      const result = await runAdf(series, lease.terminationSignal);
      lease.throwIfTerminated();
      return result;
    }), TIMEOUTS.ADF_OPERATION);
}
