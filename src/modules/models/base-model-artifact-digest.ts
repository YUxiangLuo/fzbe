import crypto from "node:crypto";
import fs from "node:fs";

export type BaseArtifactModelType = "ma" | "es" | "arima" | "lstm";

const BASE_ARTIFACT_DIGEST_PROTOCOL = "fz-base-model-artifact-v1";

export const getLstmSidecarPath = (modelPath: string) =>
  modelPath.replace(/\.pkl$/, "_keras_model.h5");

const getTerminationError = (signal: AbortSignal) =>
  signal.reason instanceof Error
    ? signal.reason
    : new Error("辅助 Python 任务已终止");

export const hashFileContents = async (
  filePath: string,
  terminationSignal?: AbortSignal,
) => await new Promise<string>((resolve, reject) => {
  if (terminationSignal?.aborted) {
    reject(getTerminationError(terminationSignal));
    return;
  }

  const digest = crypto.createHash("sha256");
  const stream = fs.createReadStream(filePath);
  const onAbort = () => stream.destroy(getTerminationError(terminationSignal!));
  terminationSignal?.addEventListener("abort", onAbort, { once: true });
  const cleanup = () =>
    terminationSignal?.removeEventListener("abort", onAbort);

  stream.on("data", (chunk) => digest.update(chunk));
  stream.once("error", (error) => {
    cleanup();
    reject(error);
  });
  stream.once("end", () => {
    cleanup();
    resolve(digest.digest("hex"));
  });
});

export const computeBaseModelArtifactManifestSha256 = async (
  modelType: BaseArtifactModelType,
  modelPath: string,
  terminationSignal?: AbortSignal,
) => {
  const artifacts = [{ role: "state", filePath: modelPath }];
  if (modelType === "lstm") {
    artifacts.push({
      role: "keras",
      filePath: getLstmSidecarPath(modelPath),
    });
  }

  const manifest = crypto.createHash("sha256");
  manifest.update(BASE_ARTIFACT_DIGEST_PROTOCOL);
  manifest.update("\0");
  manifest.update(modelType);
  for (const artifact of artifacts) {
    if (terminationSignal?.aborted) {
      throw getTerminationError(terminationSignal);
    }
    manifest.update("\0");
    manifest.update(artifact.role);
    manifest.update("\0");
    manifest.update(
      await hashFileContents(artifact.filePath, terminationSignal),
    );
  }
  return manifest.digest("hex");
};
