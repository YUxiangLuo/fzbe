import path from "node:path";
import { CLEANUP_RETENTION } from "./constants";
import { cleanupDirectoryEntriesOlderThan } from "./disk-cleanup";

const uploadsRoot = path.resolve(import.meta.dir, "../../uploads");
const tempUploadsDir = path.join(uploadsRoot, "temp");

export async function cleanupStaleUploadTempFiles(
  options: { now?: Date; maxAgeMs?: number } = {},
) {
  return cleanupDirectoryEntriesOlderThan(tempUploadsDir, {
    now: options.now,
    maxAgeMs: options.maxAgeMs ?? CLEANUP_RETENTION.TEMP_FILES_MS,
    entryKind: "file",
  });
}
