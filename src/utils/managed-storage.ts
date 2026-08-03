import { createHash } from "node:crypto";
import path from "node:path";
import { lstat, mkdir, realpath } from "node:fs/promises";

const DISPOSABLE_DATABASE_PATTERN =
  /^(?:fangzhen_e2e(?:_[a-z0-9][a-z0-9_]*)?|fz_(?:e2e|pr|test|student_dogfood)_[a-z0-9][a-z0-9_]*)$/i;

export type ManagedStoragePathErrorCode =
  | "invalid_path"
  | "outside_existing"
  | "legacy_unscoped";

export class ManagedStoragePathError extends Error {
  constructor(
    public readonly code: ManagedStoragePathErrorCode,
    message: string,
  ) {
    super(message);
    this.name = "ManagedStoragePathError";
  }
}

export interface ManagedStorageEnvironment {
  DB_DATABASE: string;
  DB_HOST: string;
  /** mysql2 uses 3306 when the port is omitted; hash that effective value. */
  DB_PORT?: number;
  E2E_ALLOW_DB_RESET?: boolean | string;
}

export function isDisposableDatabaseName(databaseName: string): boolean {
  return DISPOSABLE_DATABASE_PATTERN.test(databaseName.trim());
}

export function isPathInsideRoot(candidatePath: string, rootPath: string): boolean {
  const relative = path.relative(path.resolve(rootPath), path.resolve(candidatePath));
  return (
    relative.length > 0
    && relative !== ".."
    && !relative.startsWith(`..${path.sep}`)
    && !path.isAbsolute(relative)
  );
}

function normalizedStorageIdentity(environment: ManagedStorageEnvironment) {
  const databaseName = environment.DB_DATABASE;
  const databaseHost = environment.DB_HOST;
  const databasePort = Number(environment.DB_PORT ?? 3306);
  if (
    databaseName !== databaseName.trim()
    || databaseHost !== databaseHost.trim()
    || !databaseName
    || !databaseHost
    || !Number.isInteger(databasePort)
    || databasePort <= 0
    || databasePort > 65535
  ) {
    throw new ManagedStoragePathError("invalid_path", "数据库存储身份格式无效");
  }
  return { databaseName, databaseHost, databasePort };
}

export function isDisposableStorageNamespaceEnabled(
  environment: ManagedStorageEnvironment,
): boolean {
  const { databaseName } = normalizedStorageIdentity(environment);
  const resetAllowed = (
    environment.E2E_ALLOW_DB_RESET === true
    || environment.E2E_ALLOW_DB_RESET === "1"
  );
  return resetAllowed && isDisposableDatabaseName(databaseName);
}

export function databaseStorageNamespace(
  environment: ManagedStorageEnvironment,
): string {
  const { databaseName, databaseHost, databasePort } = normalizedStorageIdentity(environment);
  const identity = JSON.stringify([
    databaseHost.trim().toLowerCase(),
    databasePort,
    databaseName.trim(),
  ]);
  return createHash("sha256").update(identity).digest("hex").slice(0, 16);
}

export function getDatabaseManagedStorageRoot(
  managedRoot: string,
  environment: ManagedStorageEnvironment,
): string {
  const resolvedRoot = path.resolve(managedRoot);
  if (!isDisposableStorageNamespaceEnabled(environment)) return resolvedRoot;
  return path.join(
    resolvedRoot,
    ".db",
    databaseStorageNamespace(environment),
  );
}

async function ensureRealDirectory(directory: string, create: boolean): Promise<void> {
  let stats: Awaited<ReturnType<typeof lstat>>;
  try {
    stats = await lstat(directory);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT" || !create) throw error;
    try {
      await mkdir(directory);
    } catch (mkdirError) {
      if ((mkdirError as NodeJS.ErrnoException).code !== "EEXIST") throw mkdirError;
    }
    stats = await lstat(directory);
  }
  if (!stats.isDirectory() || stats.isSymbolicLink()) {
    throw new ManagedStoragePathError(
      "invalid_path",
      `受管存储路径包含符号链接或不是目录: ${directory}`,
    );
  }
}

/**
 * Safely creates/validates category/.db/<identity> one component at a time.
 * The category parent (uploads) must already exist as a real directory, so a
 * symlink cannot be followed before validation gets a chance to fail closed.
 */
export async function ensureDatabaseManagedStorageRoot(
  managedRoot: string,
  environment: ManagedStorageEnvironment,
): Promise<string> {
  const categoryRoot = path.resolve(managedRoot);
  await ensureRealDirectory(path.dirname(categoryRoot), false);
  await ensureRealDirectory(categoryRoot, true);
  if (!isDisposableStorageNamespaceEnabled(environment)) return categoryRoot;

  const namespaceParent = path.join(categoryRoot, ".db");
  const databaseRoot = getDatabaseManagedStorageRoot(categoryRoot, environment);
  await ensureRealDirectory(namespaceParent, true);
  await ensureRealDirectory(databaseRoot, true);

  const [canonicalCategory, canonicalDatabaseRoot] = await Promise.all([
    realpath(categoryRoot),
    realpath(databaseRoot),
  ]);
  if (!isPathInsideRoot(canonicalDatabaseRoot, canonicalCategory)) {
    throw new ManagedStoragePathError(
      "invalid_path",
      "数据库存储命名空间越出受管分类目录",
    );
  }
  return databaseRoot;
}

async function pathExists(candidatePath: string): Promise<boolean> {
  try {
    await lstat(candidatePath);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

export interface ResolveStoredManagedPathOptions {
  storedPath: string;
  managedRoot: string;
  environment: ManagedStorageEnvironment;
  legacyFallbackDirectory?: string;
  /** Enforced for destructive operations; reads may serve known legacy files. */
  enforceDisposableNamespace?: boolean;
}

export interface ResolvedStoredManagedPath {
  absolutePath: string;
  exists: boolean;
  databaseScoped: boolean;
  usedLegacyFallback: boolean;
}

/**
 * Resolves a DB-stored managed path without ever following an absolute path
 * outside the current managed root. Absolute paths retain their ownership
 * identity even after the file disappears; only legacy relative values may
 * fall back by basename for compatibility.
 */
export async function resolveStoredManagedPath(
  options: ResolveStoredManagedPathOptions,
): Promise<ResolvedStoredManagedPath> {
  const storedPath = options.storedPath.trim();
  if (!storedPath) {
    throw new ManagedStoragePathError("invalid_path", "受管文件路径为空");
  }

  const managedRoot = path.resolve(options.managedRoot);
  const fallbackDirectory = path.resolve(options.legacyFallbackDirectory ?? managedRoot);
  if (fallbackDirectory !== managedRoot && !isPathInsideRoot(fallbackDirectory, managedRoot)) {
    throw new ManagedStoragePathError("invalid_path", "旧路径回退目录位于受管根目录之外");
  }

  let absolutePath: string;
  let usedLegacyFallback = false;
  if (path.isAbsolute(storedPath)) {
    const resolvedStoredPath = path.resolve(storedPath);
    if (isPathInsideRoot(resolvedStoredPath, managedRoot)) {
      absolutePath = resolvedStoredPath;
    } else {
      // An absolute path carries an ownership identity, not merely a display
      // basename. Falling back after it goes stale can alias an unrelated
      // current file with the same name (cross-owner/cross-database confused
      // deputy). Only legacy relative values may use basename fallback.
      throw new ManagedStoragePathError(
        "outside_existing",
        `拒绝访问受管根目录之外的绝对路径: ${resolvedStoredPath}`,
      );
    }
  } else {
    absolutePath = path.resolve(fallbackDirectory, path.basename(storedPath));
    usedLegacyFallback = true;
  }

  if (!isPathInsideRoot(absolutePath, managedRoot)) {
    throw new ManagedStoragePathError("invalid_path", "解析后的文件路径位于受管根目录之外");
  }

  const exists = await pathExists(absolutePath);
  const databaseRoot = getDatabaseManagedStorageRoot(managedRoot, options.environment);
  const databaseScoped = (
    !isDisposableStorageNamespaceEnabled(options.environment)
    || isPathInsideRoot(absolutePath, databaseRoot)
  );
  if (options.enforceDisposableNamespace && exists && !databaseScoped) {
    throw new ManagedStoragePathError(
      "legacy_unscoped",
      `拒绝在共享目录中清理未按数据库隔离的旧文件: ${absolutePath}`,
    );
  }

  return { absolutePath, exists, databaseScoped, usedLegacyFallback };
}
