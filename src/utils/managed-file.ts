import { lstat, realpath } from "node:fs/promises";
import path from "node:path";
import {
  getDatabaseManagedStorageRoot,
  isDisposableStorageNamespaceEnabled,
  isPathInsideRoot,
  resolveStoredManagedPath,
  type ManagedStorageEnvironment,
  type ResolvedStoredManagedPath,
} from "./managed-storage";

export interface ResolveExistingManagedFileOptions {
  storedPath: string;
  managedRoot: string;
  environment: ManagedStorageEnvironment;
  legacyFallbackDirectory?: string;
  enforceDisposableNamespace?: boolean;
}

/**
 * Resolves a DB path and then validates every managed path component without
 * following symbolic links. This is suitable for serving/parsing a regular
 * managed file; destructive flows still use the staged-deletion validator.
 */
export async function resolveExistingManagedFile(
  options: ResolveExistingManagedFileOptions,
): Promise<ResolvedStoredManagedPath | null> {
  const managedRoot = path.resolve(options.managedRoot);
  const resolved = await resolveStoredManagedPath(options);
  if (!resolved.exists) return null;

  const databaseRoot = getDatabaseManagedStorageRoot(managedRoot, options.environment);
  const relativeDatabaseRoot = path.relative(managedRoot, databaseRoot);
  const relativeFile = path.relative(managedRoot, resolved.absolutePath);
  if (
    !relativeFile
    || relativeFile === ".."
    || relativeFile.startsWith(`..${path.sep}`)
    || path.isAbsolute(relativeFile)
  ) {
    throw new Error("受管文件位于受管根目录之外");
  }

  const directoryComponents = path.dirname(relativeFile)
    .split(path.sep)
    .filter((component) => component && component !== ".");
  let current = managedRoot;
  for (const component of ["", ...directoryComponents]) {
    current = component ? path.join(current, component) : current;
    const stats = await lstat(current);
    if (!stats.isDirectory() || stats.isSymbolicLink()) {
      throw new Error(`受管目录包含符号链接或不是目录: ${current}`);
    }
  }

  const fileStats = await lstat(resolved.absolutePath);
  if (!fileStats.isFile() || fileStats.isSymbolicLink()) {
    throw new Error("受管路径不是常规文件");
  }

  const containmentRoot = (
    options.enforceDisposableNamespace
    && isDisposableStorageNamespaceEnabled(options.environment)
  ) ? databaseRoot : managedRoot;
  const [canonicalContainmentRoot, canonicalFilePath] = await Promise.all([
    realpath(containmentRoot),
    realpath(resolved.absolutePath),
  ]);
  if (!isPathInsideRoot(canonicalFilePath, canonicalContainmentRoot)) {
    throw new Error("受管文件通过符号链接越出当前数据库目录");
  }

  // Keep the relative database-root calculation live in this shared helper so
  // malformed future namespace layouts cannot silently become accepted.
  if (
    isDisposableStorageNamespaceEnabled(options.environment)
    && relativeDatabaseRoot.startsWith(`..${path.sep}`)
  ) {
    throw new Error("数据库存储命名空间位于受管根目录之外");
  }
  return resolved;
}
