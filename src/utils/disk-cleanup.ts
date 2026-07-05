import path from "node:path";
import { lstat, readdir, rm } from "node:fs/promises";

export type CleanupEntryKind = "file" | "directory" | "any";

export interface DirectoryAgeCleanupResult {
  scannedEntries: number;
  deletedEntries: number;
  failedEntries: number;
}

export interface DirectoryAgeCleanupOptions {
  now?: Date;
  maxAgeMs: number;
  entryKind?: CleanupEntryKind;
  namePattern?: RegExp;
}

const isExpectedEntryKind = (entryKind: CleanupEntryKind, stats: Awaited<ReturnType<typeof lstat>>) => {
  if (entryKind === "any") return true;
  if (entryKind === "file") return stats.isFile();
  return stats.isDirectory();
};

export async function cleanupDirectoryEntriesOlderThan(
  directory: string,
  options: DirectoryAgeCleanupOptions,
): Promise<DirectoryAgeCleanupResult> {
  const nowMs = (options.now ?? new Date()).getTime();
  const entryKind = options.entryKind ?? "file";
  const result: DirectoryAgeCleanupResult = {
    scannedEntries: 0,
    deletedEntries: 0,
    failedEntries: 0,
  };

  let names: string[];
  try {
    names = await readdir(directory);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return result;
    }
    throw error;
  }

  for (const name of names) {
    if (options.namePattern && !options.namePattern.test(name)) {
      continue;
    }

    result.scannedEntries += 1;
    const entryPath = path.join(directory, name);
    try {
      const stats = await lstat(entryPath);
      if (!isExpectedEntryKind(entryKind, stats)) {
        continue;
      }

      if (nowMs - stats.mtimeMs < options.maxAgeMs) {
        continue;
      }

      await rm(entryPath, { recursive: stats.isDirectory(), force: true });
      result.deletedEntries += 1;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        result.failedEntries += 1;
      }
    }
  }

  return result;
}
