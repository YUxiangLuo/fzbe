import crypto from "node:crypto";
import path from "node:path";
import {
  mkdtemp,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { describe, expect, test } from "bun:test";
import {
  computeBaseModelArtifactManifestSha256,
  getLstmSidecarPath,
} from "../../../src/modules/models/base-model-artifact-digest";

const hashBytes = (value: string) =>
  crypto.createHash("sha256").update(value).digest("hex");

const expectedManifest = (
  modelType: "ma" | "lstm",
  artifacts: Array<{ role: string; contents: string }>,
) => {
  const digest = crypto.createHash("sha256");
  digest.update("fz-base-model-artifact-v1");
  digest.update("\0");
  digest.update(modelType);
  for (const artifact of artifacts) {
    digest.update("\0");
    digest.update(artifact.role);
    digest.update("\0");
    digest.update(hashBytes(artifact.contents));
  }
  return digest.digest("hex");
};

describe("base model artifact digest", () => {
  test("keeps the pure helper byte-compatible with the production manifest protocol", async () => {
    const temporaryDirectory = await mkdtemp(
      path.join(tmpdir(), "fz-artifact-digest-"),
    );
    const maPath = path.join(temporaryDirectory, "ma.pkl");
    const lstmPath = path.join(temporaryDirectory, "lstm.pkl");
    const lstmSidecarPath = getLstmSidecarPath(lstmPath);

    try {
      await Promise.all([
        writeFile(maPath, "ma-state"),
        writeFile(lstmPath, "lstm-state"),
        writeFile(lstmSidecarPath, "keras-weights"),
      ]);

      await expect(
        computeBaseModelArtifactManifestSha256("ma", maPath),
      ).resolves.toBe(expectedManifest("ma", [
        { role: "state", contents: "ma-state" },
      ]));
      await expect(
        computeBaseModelArtifactManifestSha256("lstm", lstmPath),
      ).resolves.toBe(expectedManifest("lstm", [
        { role: "state", contents: "lstm-state" },
        { role: "keras", contents: "keras-weights" },
      ]));
    } finally {
      await rm(temporaryDirectory, { recursive: true, force: true });
    }
  });

  test("honors an already-aborted integrity check", async () => {
    const controller = new AbortController();
    const reason = new Error("digest cancelled");
    controller.abort(reason);

    await expect(
      computeBaseModelArtifactManifestSha256(
        "ma",
        path.join(tmpdir(), "unused.pkl"),
        controller.signal,
      ),
    ).rejects.toBe(reason);
  });

  test("changes the manifest when either the state or LSTM sidecar changes", async () => {
    const temporaryDirectory = await mkdtemp(
      path.join(tmpdir(), "fz-artifact-mutation-"),
    );
    const lstmPath = path.join(temporaryDirectory, "lstm.pkl");
    const sidecarPath = getLstmSidecarPath(lstmPath);

    try {
      await writeFile(lstmPath, "state-v1");
      await writeFile(sidecarPath, "weights-v1");
      const original = await computeBaseModelArtifactManifestSha256("lstm", lstmPath);
      expect(await computeBaseModelArtifactManifestSha256("lstm", lstmPath)).toBe(original);

      await writeFile(lstmPath, "state-v2");
      const stateChanged = await computeBaseModelArtifactManifestSha256("lstm", lstmPath);
      expect(stateChanged).not.toBe(original);

      await writeFile(lstmPath, "state-v1");
      await writeFile(sidecarPath, "weights-v2");
      const sidecarChanged = await computeBaseModelArtifactManifestSha256("lstm", lstmPath);
      expect(sidecarChanged).not.toBe(original);
      expect(sidecarChanged).not.toBe(stateChanged);
    } finally {
      await rm(temporaryDirectory, { recursive: true, force: true });
    }
  });

  test("fails closed when a required state or LSTM sidecar file is missing", async () => {
    const temporaryDirectory = await mkdtemp(
      path.join(tmpdir(), "fz-artifact-missing-"),
    );
    const missingState = path.join(temporaryDirectory, "missing.pkl");
    const lstmPath = path.join(temporaryDirectory, "lstm.pkl");
    await writeFile(lstmPath, "state-only");

    try {
      await expect(
        computeBaseModelArtifactManifestSha256("ma", missingState),
      ).rejects.toMatchObject({ code: "ENOENT" });
      await expect(
        computeBaseModelArtifactManifestSha256("lstm", lstmPath),
      ).rejects.toMatchObject({ code: "ENOENT" });
    } finally {
      await rm(temporaryDirectory, { recursive: true, force: true });
    }
  });

  test("derives the LSTM sidecar next to the terminal pickle filename", () => {
    expect(getLstmSidecarPath("/tmp/models/archive.pkl/model.pkl"))
      .toBe("/tmp/models/archive.pkl/model_keras_model.h5");
    expect(getLstmSidecarPath("relative-model.pkl"))
      .toBe("relative-model_keras_model.h5");
  });
});
