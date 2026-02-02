import { runAdf, type AdfInputPoint, type AdfResult } from "../../utils";

export async function performAdfTest(series: AdfInputPoint[]): Promise<AdfResult> {
  return runAdf(series);
}
