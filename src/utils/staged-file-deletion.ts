import path from "node:path";
import {
  link,
  lstat,
  mkdir,
  open,
  readdir,
  readFile,
  realpath,
  rename,
  rm,
  unlink,
} from "node:fs/promises";

const JOURNAL_DIRECTORY_NAME = ".deletion-journal";
const JOURNAL_VERSION = 1 as const;
const TOMBSTONE_PREFIX = ".deleting-";
const UUID_V4_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const JOURNAL_TEMP_PATTERN = /^\.[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.tmp$/;
const STALE_JOURNAL_TEMP_AGE_MS = 24 * 60 * 60 * 1000;
const MAX_JOURNAL_BYTES = 1024 * 1024;

export interface ManagedDeletionMetadata extends Record<string, unknown> {
  databaseName: string;
  kind: string;
  entityIds: number[];
}

export interface StagedManagedPath {
  originalPath: string;
  stagedPath: string;
  journalPath?: string;
}

export type ManagedDeletionOwnershipMatch = "all" | "none" | "partial";

export function classifyManagedDeletionOwnership(
  entries: Array<Pick<StagedManagedPath, "originalPath">>,
  ownerReferencedPaths: Iterable<string>,
): ManagedDeletionOwnershipMatch {
  const references = new Set(
    [...ownerReferencedPaths].map((candidatePath) => path.resolve(candidatePath)),
  );
  const originals = [...new Set(
    entries.map((entry) => path.resolve(entry.originalPath)),
  )];
  const matched = originals.filter((originalPath) => references.has(originalPath)).length;
  if (matched === 0) return "none";
  if (matched === originals.length) return "all";
  return "partial";
}

export interface ManagedDeletionJournal {
  version: typeof JOURNAL_VERSION;
  operationId: string;
  createdAt: string;
  metadata: ManagedDeletionMetadata;
  entries: Array<Pick<StagedManagedPath, "originalPath" | "stagedPath">>;
}

export interface ManagedDeletionReconcileResult {
  scannedJournals: number;
  restoredJournals: number;
  purgedJournals: number;
  skippedJournals: number;
  failedJournals: number;
  failedJournalNames: string[];
}

export type ManagedDeletionReconcileAction = "restore" | "purge" | "skip";

export interface ManagedDeletionReconcileResolution {
  action: ManagedDeletionReconcileAction;
  /**
   * Optional database-row lease release hook. Reconciliation keeps the row
   * locks until the filesystem restore/purge decision has completed.
   */
  release?: () => Promise<void>;
}

export class ManagedPathDeletionError extends Error {
  override readonly cause: unknown;

  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = "ManagedPathDeletionError";
    this.cause = cause;
  }
}

export async function fsyncDirectory(directoryPath: string): Promise<void> {
  const handle = await open(directoryPath, "r");
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
}

export interface ManagedRestoreFileOperations {
  createHardLink: (existingPath: string, newPath: string) => Promise<void>;
  fsyncDirectory: (directoryPath: string) => Promise<void>;
  unlink: (candidatePath: string) => Promise<void>;
}

const DEFAULT_RESTORE_FILE_OPERATIONS: ManagedRestoreFileOperations = {
  createHardLink: link,
  fsyncDirectory,
  unlink,
};

export async function fsyncParentDirectories(candidatePaths: string[]): Promise<void> {
  const directories = [...new Set(candidatePaths.map((candidatePath) => (
    path.dirname(path.resolve(candidatePath))
  )))].sort();
  for (const directory of directories) {
    await fsyncDirectory(directory);
  }
}

export function isPathStrictlyInside(candidatePath: string, rootPath: string): boolean {
  const resolvedRoot = path.resolve(rootPath);
  const resolvedCandidate = path.resolve(candidatePath);
  const relative = path.relative(resolvedRoot, resolvedCandidate);
  return (
    relative.length > 0
    && relative !== ".."
    && !relative.startsWith(`..${path.sep}`)
    && !path.isAbsolute(relative)
  );
}

function assertLexicallyManagedPath(candidatePath: string, rootPath: string): string {
  const resolvedCandidate = path.resolve(candidatePath);
  if (!isPathStrictlyInside(resolvedCandidate, rootPath)) {
    throw new ManagedPathDeletionError(`拒绝删除受管目录之外的路径: ${resolvedCandidate}`);
  }

  const relative = path.relative(path.resolve(rootPath), resolvedCandidate);
  if (relative === JOURNAL_DIRECTORY_NAME || relative.startsWith(`${JOURNAL_DIRECTORY_NAME}${path.sep}`)) {
    throw new ManagedPathDeletionError(`拒绝删除文件隔离日志目录: ${resolvedCandidate}`);
  }
  return resolvedCandidate;
}

async function assertNoSymlinkAncestor(candidatePath: string, rootPath: string): Promise<string> {
  const resolvedCandidate = assertLexicallyManagedPath(candidatePath, rootPath);
  const [resolvedRealRoot, resolvedRealParent] = await Promise.all([
    realpath(path.resolve(rootPath)),
    realpath(path.dirname(resolvedCandidate)),
  ]);
  const relativeParent = path.relative(resolvedRealRoot, resolvedRealParent);
  const parentIsManaged = (
    relativeParent === ""
    || (
      relativeParent !== ".."
      && !relativeParent.startsWith(`..${path.sep}`)
      && !path.isAbsolute(relativeParent)
    )
  );
  if (!parentIsManaged) {
    throw new ManagedPathDeletionError(`拒绝通过符号链接删除受管目录之外的路径: ${resolvedCandidate}`);
  }
  return resolvedCandidate;
}

function buildStagedPath(originalPath: string, operationId: string, index: number): string {
  // Keep the tombstone independent of the original basename. A valid uploaded
  // filename may already be close to Linux NAME_MAX (255 bytes).
  return path.join(path.dirname(originalPath), `${TOMBSTONE_PREFIX}${operationId}-${index}`);
}

function journalDirectory(rootPath: string): string {
  return path.join(path.resolve(rootPath), JOURNAL_DIRECTORY_NAME);
}

function journalPathFor(rootPath: string, operationId: string): string {
  return path.join(journalDirectory(rootPath), `${operationId}.json`);
}

async function assertManagedJournalDirectory(rootPath: string): Promise<string> {
  const directory = journalDirectory(rootPath);
  const stats = await lstat(directory);
  if (!stats.isDirectory() || stats.isSymbolicLink()) {
    throw new ManagedPathDeletionError(`文件隔离日志路径必须是受管目录内的真实目录: ${directory}`);
  }
  const [resolvedRealRoot, resolvedRealDirectory] = await Promise.all([
    realpath(path.resolve(rootPath)),
    realpath(directory),
  ]);
  const relative = path.relative(resolvedRealRoot, resolvedRealDirectory);
  if (
    relative.length === 0
    || relative === ".."
    || relative.startsWith(`..${path.sep}`)
    || path.isAbsolute(relative)
  ) {
    throw new ManagedPathDeletionError(`文件隔离日志目录通过符号链接逃逸受管根目录: ${directory}`);
  }
  return directory;
}

function isValidManagedDeletionMetadata(value: unknown): value is ManagedDeletionMetadata {
  if (!value || typeof value !== "object") return false;
  const metadata = value as Partial<ManagedDeletionMetadata>;
  return (
    typeof metadata.databaseName === "string"
    && metadata.databaseName.trim().length > 0
    && typeof metadata.kind === "string"
    && metadata.kind.trim().length > 0
    && Array.isArray(metadata.entityIds)
    && metadata.entityIds.length > 0
    && metadata.entityIds.every((entityId) => Number.isInteger(entityId) && entityId > 0)
  );
}

function isCanonicalIsoTimestamp(value: string): boolean {
  const timestamp = new Date(value);
  return Number.isFinite(timestamp.getTime()) && timestamp.toISOString() === value;
}

async function pathKind(candidatePath: string): Promise<"present" | "missing"> {
  try {
    await lstat(candidatePath);
    return "present";
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return "missing";
    throw error;
  }
}

type ExistingRegularFile = Awaited<ReturnType<typeof lstat>>;

async function regularFileState(
  candidatePath: string,
): Promise<{ state: "missing" } | { state: "present"; stats: ExistingRegularFile }> {
  let stats: ExistingRegularFile;
  try {
    stats = await lstat(candidatePath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { state: "missing" };
    throw error;
  }
  if (!stats.isFile() || stats.isSymbolicLink()) {
    throw new ManagedPathDeletionError(`受管删除协议只允许普通文件: ${candidatePath}`);
  }
  return { state: "present", stats };
}

function isSameFileIdentity(left: ExistingRegularFile, right: ExistingRegularFile): boolean {
  return left.dev === right.dev && left.ino === right.ino;
}

async function writeJournalAtomically(
  rootPath: string,
  journal: ManagedDeletionJournal,
): Promise<string> {
  const directory = journalDirectory(rootPath);
  const targetPath = journalPathFor(rootPath, journal.operationId);
  const temporaryPath = path.join(directory, `.${journal.operationId}.tmp`);
  let handle: Awaited<ReturnType<typeof open>> | null = null;
  let journalDirectoryValidated = false;
  try {
    await mkdir(directory, { recursive: true });
    // Persist first creation of .deletion-journal in the managed root before
    // relying on the journal as the recovery authority.
    await fsyncDirectory(path.resolve(rootPath));
    await assertManagedJournalDirectory(rootPath);
    journalDirectoryValidated = true;
    handle = await open(temporaryPath, "wx", 0o600);
    await handle.writeFile(`${JSON.stringify(journal)}\n`, "utf8");
    await handle.sync();
    await handle.close();
    handle = null;
    await rename(temporaryPath, targetPath);
    // Persist the directory entry as well as the journal contents on Linux.
    await fsyncDirectory(directory);
    return targetPath;
  } catch (error) {
    await handle?.close().catch(() => {});
    if (journalDirectoryValidated) {
      await rm(temporaryPath, { force: true }).catch(() => {});
      await fsyncDirectory(directory).catch(() => {});
    }
    if (error instanceof ManagedPathDeletionError) throw error;
    throw new ManagedPathDeletionError("写入文件隔离日志失败", error);
  }
}

function journalPaths(entries: StagedManagedPath[]): string[] {
  return [...new Set(entries.flatMap((entry) => entry.journalPath ? [entry.journalPath] : []))];
}

async function removeJournals(entries: StagedManagedPath[]): Promise<string[]> {
  const failures: string[] = [];
  const paths = journalPaths(entries);
  for (const journalPath of paths) {
    try {
      await rm(journalPath, { force: true });
    } catch {
      failures.push(journalPath);
    }
  }
  try {
    await fsyncParentDirectories(paths);
  } catch {
    failures.push(...new Set(paths.map((journalPath) => path.dirname(journalPath))));
  }
  return failures;
}

async function restoreEntry(
  entry: StagedManagedPath,
  operations: ManagedRestoreFileOperations,
): Promise<void> {
  if (path.dirname(entry.originalPath) !== path.dirname(entry.stagedPath)) {
    throw new ManagedPathDeletionError(
      `恢复隔离文件要求原路径与 tombstone 位于同一目录: ${entry.originalPath}`,
    );
  }
  const [original, staged] = await Promise.all([
    regularFileState(entry.originalPath),
    regularFileState(entry.stagedPath),
  ]);
  if (original.state === "present" && staged.state === "missing") return;
  if (original.state === "missing" && staged.state === "present") {
    // link(2) is an atomic NO-REPLACE operation: unlike rename(), it cannot
    // overwrite a path which appeared after the preflight. The following
    // unlink completes a same-directory restore while leaving a recoverable
    // two-hard-link state if the process crashes between the two syscalls.
    try {
      await operations.createHardLink(entry.stagedPath, entry.originalPath);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "EEXIST") {
        throw new ManagedPathDeletionError(
          `恢复隔离文件时原路径已被占用，拒绝覆盖: ${entry.originalPath}`,
          error,
        );
      }
      throw error;
    }
    // The new name must be durable before the old name can be removed. Without
    // this barrier, a crash may persist unlink(tombstone) while losing the
    // preceding link(original), leaving neither name after recovery.
    await operations.fsyncDirectory(path.dirname(entry.originalPath));
    await operations.unlink(entry.stagedPath);
    return;
  }
  if (original.state === "present" && staged.state === "present") {
    if (isSameFileIdentity(original.stats, staged.stats)) {
      // Crash recovery for the link-created-but-not-yet-unlinked state.
      await operations.unlink(entry.stagedPath);
      return;
    }
    throw new ManagedPathDeletionError(
      `恢复隔离文件时原路径已被不同文件占用，拒绝覆盖: ${entry.originalPath}`,
    );
  }
  throw new ManagedPathDeletionError(`恢复隔离文件时原路径和隔离路径均不存在: ${entry.originalPath}`);
}

export async function restoreStagedManagedPaths(
  entries: StagedManagedPath[],
  operations: ManagedRestoreFileOperations = DEFAULT_RESTORE_FILE_OPERATIONS,
): Promise<void> {
  const failures: unknown[] = [];
  for (const entry of [...entries].reverse()) {
    try {
      await restoreEntry(entry, operations);
    } catch (error) {
      failures.push(error);
    }
  }
  try {
    // Sync every parent, including entries already restored before a crash,
    // before unlinking the only durable recovery record.
    await fsyncParentDirectories(entries.map((entry) => entry.originalPath));
  } catch (error) {
    failures.push(error);
  }
  if (failures.length > 0) {
    throw new ManagedPathDeletionError("恢复已隔离文件失败", failures);
  }
  const journalFailures = await removeJournals(entries);
  if (journalFailures.length > 0) {
    throw new ManagedPathDeletionError("恢复文件后清理隔离日志失败", journalFailures);
  }
}

export async function stageManagedPathsForDeletion(
  candidatePaths: string[],
  rootPath: string,
  metadata: ManagedDeletionMetadata,
): Promise<StagedManagedPath[]> {
  if (!isValidManagedDeletionMetadata(metadata)) {
    throw new ManagedPathDeletionError("文件隔离日志元数据无效");
  }
  const uniqueLexicalPaths = [...new Set(
    candidatePaths.map((candidate) => assertLexicallyManagedPath(candidate, rootPath)),
  )];
  const existingPaths: string[] = [];
  for (const candidatePath of uniqueLexicalPaths) {
    if (await pathKind(candidatePath) === "missing") continue;
    const managedPath = await assertNoSymlinkAncestor(candidatePath, rootPath);
    const stats = await lstat(managedPath);
    if (!stats.isFile() || stats.isSymbolicLink()) {
      throw new ManagedPathDeletionError(`受管删除协议只允许普通文件: ${managedPath}`);
    }
    existingPaths.push(managedPath);
  }
  if (existingPaths.length === 0) return [];

  const operationId = crypto.randomUUID();
  const plannedEntries = existingPaths.map((originalPath, index) => ({
    originalPath,
    stagedPath: buildStagedPath(originalPath, operationId, index),
  }));
  const journal: ManagedDeletionJournal = {
    version: JOURNAL_VERSION,
    operationId,
    createdAt: new Date().toISOString(),
    metadata,
    entries: plannedEntries,
  };
  const journalPath = await writeJournalAtomically(rootPath, journal);
  const entries = plannedEntries.map((entry) => ({ ...entry, journalPath }));
  const staged: StagedManagedPath[] = [];

  try {
    for (const entry of entries) {
      await rename(entry.originalPath, entry.stagedPath);
      staged.push(entry);
    }
    await fsyncParentDirectories(entries.map((entry) => entry.originalPath));
    return entries;
  } catch (error) {
    try {
      // Include every planned entry: untouched paths are recognized as already
      // restored, while successfully renamed paths are moved back.
      await restoreStagedManagedPaths(entries);
    } catch (restoreError) {
      throw new ManagedPathDeletionError("隔离文件失败，且无法完整恢复已隔离文件", {
        error,
        restoreError,
        staged,
      });
    }
    throw new ManagedPathDeletionError("隔离待删除文件失败", error);
  }
}

export async function purgeStagedManagedPaths(entries: StagedManagedPath[]): Promise<string[]> {
  const failedPaths: string[] = [];
  for (const entry of entries) {
    try {
      const staged = await regularFileState(entry.stagedPath);
      if (staged.state === "present") await unlink(entry.stagedPath);
    } catch {
      failedPaths.push(entry.stagedPath);
    }
  }
  try {
    await fsyncParentDirectories(entries.map((entry) => entry.stagedPath));
  } catch {
    failedPaths.push(...new Set(entries.map((entry) => path.dirname(entry.stagedPath))));
  }
  if (failedPaths.length === 0) {
    failedPaths.push(...await removeJournals(entries));
  }
  return failedPaths;
}

function parseJournal(raw: string, expectedPath: string, rootPath: string): ManagedDeletionJournal {
  const parsed = JSON.parse(raw) as Partial<ManagedDeletionJournal>;
  if (
    parsed.version !== JOURNAL_VERSION
    || typeof parsed.operationId !== "string"
    || !UUID_V4_PATTERN.test(parsed.operationId)
    || typeof parsed.createdAt !== "string"
    || !isCanonicalIsoTimestamp(parsed.createdAt)
    || !isValidManagedDeletionMetadata(parsed.metadata)
    || !Array.isArray(parsed.entries)
    || parsed.entries.length === 0
  ) {
    throw new ManagedPathDeletionError(`文件隔离日志格式无效: ${expectedPath}`);
  }
  const expectedJournalPath = journalPathFor(rootPath, parsed.operationId);
  if (path.resolve(expectedPath) !== expectedJournalPath) {
    throw new ManagedPathDeletionError(`文件隔离日志名称与操作 ID 不一致: ${expectedPath}`);
  }
  const entries = parsed.entries.map((entry, index) => {
    if (!entry || typeof entry.originalPath !== "string" || typeof entry.stagedPath !== "string") {
      throw new ManagedPathDeletionError(`文件隔离日志条目无效: ${expectedPath}`);
    }
    const originalPath = assertLexicallyManagedPath(entry.originalPath, rootPath);
    const stagedPath = assertLexicallyManagedPath(entry.stagedPath, rootPath);
    if (
      path.dirname(originalPath) !== path.dirname(stagedPath)
      || path.basename(stagedPath) !== `${TOMBSTONE_PREFIX}${parsed.operationId}-${index}`
    ) {
      throw new ManagedPathDeletionError(`文件隔离日志路径不符合短 tombstone 协议: ${expectedPath}`);
    }
    return { originalPath, stagedPath };
  });
  return {
    version: JOURNAL_VERSION,
    operationId: parsed.operationId,
    createdAt: parsed.createdAt,
    metadata: parsed.metadata as ManagedDeletionMetadata,
    entries,
  };
}

async function purgeReconciledEntries(entries: StagedManagedPath[]): Promise<void> {
  const failures: unknown[] = [];
  // Preflight the complete operation before deleting any tombstone. If one
  // original has reappeared (an ABA/new-generation file), retaining every
  // tombstone is the only state from which a later reconciler can recover.
  for (const entry of entries) {
    try {
      const [original] = await Promise.all([
        regularFileState(entry.originalPath),
        regularFileState(entry.stagedPath),
      ]);
      if (original.state === "present") {
        throw new ManagedPathDeletionError(
          `已提交删除的原路径重新出现，拒绝清理可能属于新一代的文件: ${entry.originalPath}`,
        );
      }
    } catch (error) {
      failures.push(error);
    }
  }
  if (failures.length > 0) {
    throw new ManagedPathDeletionError("清理已提交删除操作的原路径预检失败", failures);
  }
  for (const entry of entries) {
    try {
      const staged = await regularFileState(entry.stagedPath);
      if (staged.state === "present") await unlink(entry.stagedPath);
    } catch (error) {
      failures.push(error);
    }
  }
  try {
    await fsyncParentDirectories(entries.map((entry) => entry.stagedPath));
  } catch (error) {
    failures.push(error);
  }
  if (failures.length > 0) {
    throw new ManagedPathDeletionError("清理已提交删除操作的残留文件失败", failures);
  }
  const journalFailures = await removeJournals(entries);
  if (journalFailures.length > 0) {
    throw new ManagedPathDeletionError("清理已提交删除操作的隔离日志失败", journalFailures);
  }
}

export async function reconcileStagedDeletionJournals(
  rootPath: string,
  resolveAction: (
    journal: ManagedDeletionJournal,
  ) => Promise<ManagedDeletionReconcileAction | ManagedDeletionReconcileResolution>,
): Promise<ManagedDeletionReconcileResult> {
  const result: ManagedDeletionReconcileResult = {
    scannedJournals: 0,
    restoredJournals: 0,
    purgedJournals: 0,
    skippedJournals: 0,
    failedJournals: 0,
    failedJournalNames: [],
  };
  const directory = journalDirectory(rootPath);
  let names: string[];
  try {
    await assertManagedJournalDirectory(rootPath);
    names = await readdir(directory);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return result;
    throw error;
  }

  // Never race another process that is still writing/fsyncing its temporary
  // journal. Only our exact UUID temp format and a conservative stale age are
  // eligible for cleanup.
  const removedTemporaryNames: string[] = [];
  for (const temporaryName of names.filter((entry) => JOURNAL_TEMP_PATTERN.test(entry))) {
    const temporaryPath = path.join(directory, temporaryName);
    const stats = await lstat(temporaryPath).catch(() => null);
    if (!stats || Date.now() - stats.mtimeMs < STALE_JOURNAL_TEMP_AGE_MS) continue;
    try {
      await rm(temporaryPath, { force: true });
      removedTemporaryNames.push(temporaryName);
    } catch {}
  }
  if (removedTemporaryNames.length > 0) {
    await fsyncDirectory(directory);
  }

  for (const name of names.filter((entry) => entry.endsWith(".json")).sort()) {
    result.scannedJournals += 1;
    const currentJournalPath = path.join(directory, name);
    try {
      const journalStats = await lstat(currentJournalPath);
      if (
        !journalStats.isFile()
        || journalStats.isSymbolicLink()
        || journalStats.size > MAX_JOURNAL_BYTES
      ) {
        throw new ManagedPathDeletionError(
          `文件隔离日志必须是大小不超过 ${MAX_JOURNAL_BYTES} 字节的普通文件: ${currentJournalPath}`,
        );
      }
      const journal = parseJournal(await readFile(currentJournalPath, "utf8"), currentJournalPath, rootPath);
      const entries = journal.entries.map((entry) => ({ ...entry, journalPath: currentJournalPath }));
      // Validate the real parents again before trusting a durable journal.
      for (const entry of entries) {
        await assertNoSymlinkAncestor(entry.originalPath, rootPath);
        await assertNoSymlinkAncestor(entry.stagedPath, rootPath);
        await regularFileState(entry.originalPath);
        await regularFileState(entry.stagedPath);
      }
      const requestedResolution = await resolveAction(journal);
      const resolution = typeof requestedResolution === "string"
        ? { action: requestedResolution }
        : requestedResolution;
      try {
        if (resolution.action === "skip") {
          result.skippedJournals += 1;
        } else if (resolution.action === "restore") {
          await restoreStagedManagedPaths(entries);
          result.restoredJournals += 1;
        } else if (resolution.action === "purge") {
          await purgeReconciledEntries(entries);
          result.purgedJournals += 1;
        } else {
          throw new ManagedPathDeletionError(`未知的文件隔离日志处理动作: ${String(resolution.action)}`);
        }
      } finally {
        await resolution.release?.();
      }
    } catch {
      result.failedJournals += 1;
      result.failedJournalNames.push(name);
    }
  }
  return result;
}
