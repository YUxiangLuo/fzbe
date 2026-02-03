import { runAdf, type AdfInputPoint, type AdfResult } from "../../utils/adf";

export async function performAdfTest(series: AdfInputPoint[]): Promise<AdfResult> {
  return runAdf(series);
}
