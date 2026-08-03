import path from "node:path";
import { env } from "../config/env";
import { CLEANUP_RETENTION } from "./constants";
import { cleanupDirectoryEntriesOlderThan } from "./disk-cleanup";
import { ensureDatabaseManagedStorageRoot } from "./managed-storage";

const uploadsRoot = path.resolve(import.meta.dir, "../../uploads");
const tempUploadsRoot = path.join(uploadsRoot, "temp");

export async function cleanupStaleUploadTempFiles(
  options: { now?: Date; maxAgeMs?: number } = {},
) {
  const tempUploadsDir = await ensureDatabaseManagedStorageRoot(tempUploadsRoot, env);
  return cleanupDirectoryEntriesOlderThan(
    tempUploadsDir,
    {
    now: options.now,
    maxAgeMs: options.maxAgeMs ?? CLEANUP_RETENTION.TEMP_FILES_MS,
    entryKind: "file",
    },
  );
}
