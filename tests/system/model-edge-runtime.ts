import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";

export const MODEL_EDGE_CHECKOUT_ROOT = path.resolve(import.meta.dir, "..", "..");
export const MODEL_EDGE_ROOTS = {
  guided: path.join(MODEL_EDGE_CHECKOUT_ROOT, "py", "guided_training"),
  preparation: path.join(MODEL_EDGE_CHECKOUT_ROOT, "py", "production_preparation"),
  trained: path.join(MODEL_EDGE_CHECKOUT_ROOT, "py", "trained_models"),
} as const;
const MODEL_EDGE_LOCK_PATH = path.join(MODEL_EDGE_CHECKOUT_ROOT, ".model-edge-runner.lock");

type RootKind = keyof typeof MODEL_EDGE_ROOTS;

export interface ModelEdgeArtifactBaseline {
  guided: Set<string>;
  preparation: Set<string>;
  trained: Set<string>;
}

export interface ModelEdgeOwnedArtifacts {
  baseline: ModelEdgeArtifactBaseline;
  experimentIds: number[];
  studentId: number;
  guidedDirs: string[];
  preparationDirs: string[];
  modelPaths: string[];
}

export interface CleanupTask {
  id: string;
  run: () => void | Promise<void>;
}

export interface CleanupOperations {
  list: (root: string) => string[];
  lstat: (entryPath: string) => fs.Stats;
  unlink: (entryPath: string) => void;
  removeGuidedDir: (entryPath: string) => boolean;
  removePreparationDir: (entryPath: string) => boolean;
  removeModelGroup: (entryPath: string) => void;
}

const filesystemOperations = {
  list: (root: string) => {
    try {
      return fs.readdirSync(root);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw error;
    }
  },
  lstat: (entryPath: string) => fs.lstatSync(entryPath),
  unlink: (entryPath: string) => fs.unlinkSync(entryPath),
};

const assertKnownModelRoots = () => {
  const roots = Object.values(MODEL_EDGE_ROOTS);
  if (new Set(roots).size !== roots.length) {
    throw new Error("[model-edge] configured model roots must be distinct");
  }
  for (const root of roots) {
    const relative = path.relative(MODEL_EDGE_CHECKOUT_ROOT, root);
    if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
      throw new Error(`[model-edge] configured model root escapes checkout: ${root}`);
    }
    let current = MODEL_EDGE_CHECKOUT_ROOT;
    for (const component of relative.split(path.sep)) {
      current = path.join(current, component);
      try {
        if (fs.lstatSync(current).isSymbolicLink()) {
          throw new Error(`[model-edge] configured model root has symlink component: ${current}`);
        }
      } catch (error) {
        if (isMissing(error)) break;
        throw error;
      }
    }
  }
};

const immediateEntries = (root: string) => new Set(filesystemOperations.list(root));

export const captureModelEdgeArtifactBaseline = (): ModelEdgeArtifactBaseline => {
  assertKnownModelRoots();
  return {
    guided: immediateEntries(MODEL_EDGE_ROOTS.guided),
    preparation: immediateEntries(MODEL_EDGE_ROOTS.preparation),
    trained: immediateEntries(MODEL_EDGE_ROOTS.trained),
  };
};

const assertImmediateOwnedPath = (
  entryPath: string,
  kind: RootKind,
  prefixes: string[],
) => {
  const root = MODEL_EDGE_ROOTS[kind];
  const resolved = path.resolve(entryPath);
  const relative = path.relative(root, resolved);
  if (
    !relative
    || relative.startsWith("..")
    || path.isAbsolute(relative)
    || relative.includes(path.sep)
    || !prefixes.some((prefix) => relative.startsWith(prefix))
  ) {
    throw new Error(`[model-edge] refusing non-owned ${kind} artifact path: ${entryPath}`);
  }
  return { resolved, entry: relative };
};

const isMissing = (error: unknown) => (error as NodeJS.ErrnoException).code === "ENOENT";

const removeExactDirectory = (
  entryPath: string,
  remover: (ownedPath: string) => boolean,
  operations: CleanupOperations,
) => {
  let stats: fs.Stats;
  try {
    stats = operations.lstat(entryPath);
  } catch (error) {
    if (isMissing(error)) return;
    throw error;
  }
  if (!stats.isDirectory() || stats.isSymbolicLink()) {
    throw new Error(`[model-edge] owned session artifact is not a directory: ${entryPath}`);
  }
  if (!remover(entryPath)) {
    throw new Error(`[model-edge] owned session artifact failed containment validation: ${entryPath}`);
  }
};

const removeExactTrainedEntry = (
  entryPath: string,
  operations: CleanupOperations,
) => {
  let stats: fs.Stats;
  try {
    stats = operations.lstat(entryPath);
  } catch (error) {
    if (isMissing(error)) return;
    throw error;
  }
  if (stats.isDirectory()) {
    throw new Error(`[model-edge] refusing trained-model directory cleanup: ${entryPath}`);
  }
  if (entryPath.endsWith(".pkl")) {
    operations.removeModelGroup(entryPath);
    try {
      operations.lstat(entryPath);
    } catch (error) {
      if (isMissing(error)) return;
      throw error;
    }
    throw new Error(`[model-edge] trained-model artifact remained after cleanup: ${entryPath}`);
  }
  operations.unlink(entryPath);
};

export const buildModelEdgeCleanupTasks = (
  owned: ModelEdgeOwnedArtifacts,
  operations: CleanupOperations,
): CleanupTask[] => {
  assertKnownModelRoots();
  const prefixes = owned.experimentIds.map(
    (experimentId) => `exp${experimentId}-student${owned.studentId}-`,
  );
  if (prefixes.length === 0) return [];
  const tasks = new Map<string, CleanupTask>();

  const addDirectory = (rawPath: string, kind: "guided" | "preparation") => {
    const { resolved, entry } = assertImmediateOwnedPath(rawPath, kind, prefixes);
    if (owned.baseline[kind].has(entry)) return;
    const id = `${kind}:${resolved}`;
    tasks.set(id, {
      id,
      run: () => removeExactDirectory(
        resolved,
        kind === "guided" ? operations.removeGuidedDir : operations.removePreparationDir,
        operations,
      ),
    });
  };
  const addTrained = (rawPath: string) => {
    const { resolved, entry } = assertImmediateOwnedPath(rawPath, "trained", prefixes);
    if (owned.baseline.trained.has(entry)) return;
    const id = `trained:${resolved}`;
    tasks.set(id, { id, run: () => removeExactTrainedEntry(resolved, operations) });
  };

  const captureBuildFailure = (id: string, build: () => void) => {
    try {
      build();
    } catch (error) {
      tasks.set(`invalid:${id}`, {
        id: `invalid:${id}`,
        run: () => { throw error; },
      });
    }
  };
  for (const artifactDir of owned.guidedDirs) {
    captureBuildFailure(`guided:${artifactDir}`, () => addDirectory(artifactDir, "guided"));
  }
  for (const artifactDir of owned.preparationDirs) {
    captureBuildFailure(
      `preparation:${artifactDir}`,
      () => addDirectory(artifactDir, "preparation"),
    );
  }
  for (const modelPath of owned.modelPaths) {
    captureBuildFailure(`trained:${modelPath}`, () => addTrained(modelPath));
  }

  for (const [kind, root] of Object.entries(MODEL_EDGE_ROOTS) as Array<[RootKind, string]>) {
    let entries: string[] = [];
    captureBuildFailure(`list:${root}`, () => { entries = operations.list(root); });
    for (const entry of entries) {
      if (owned.baseline[kind].has(entry) || !prefixes.some((prefix) => entry.startsWith(prefix))) {
        continue;
      }
      const entryPath = path.join(root, entry);
      captureBuildFailure(`${kind}:${entryPath}`, () => {
        if (kind === "trained") addTrained(entryPath);
        else addDirectory(entryPath, kind);
      });
    }
  }
  return [...tasks.values()];
};

export const runCleanupTasks = async (tasks: CleanupTask[]) => {
  const errors: unknown[] = [];
  for (const task of tasks) {
    try {
      await task.run();
    } catch (error) {
      errors.push(new Error(`[model-edge] cleanup task failed: ${task.id}`, { cause: error }));
    }
  }
  if (errors.length === 1) throw errors[0];
  if (errors.length > 1) throw new AggregateError(errors, "model-edge artifact cleanup failed");
};

export const cleanupModelEdgeOwnedArtifacts = async (
  owned: ModelEdgeOwnedArtifacts,
  suppliedOperations?: CleanupOperations,
) => {
  let operations = suppliedOperations;
  if (!operations) {
    const [{
      removeGuidedTrainingArtifactDir,
      removeProductionPreparationArtifactDir,
    }, { removeModelArtifactGroup }] = await Promise.all([
      import("../../src/modules/models/model-cleanup.service"),
      import("../../src/modules/models/model-artifacts"),
    ]);
    operations = {
      ...filesystemOperations,
      removeGuidedDir: removeGuidedTrainingArtifactDir,
      removePreparationDir: removeProductionPreparationArtifactDir,
      removeModelGroup: removeModelArtifactGroup,
    };
  }
  return runCleanupTasks(buildModelEdgeCleanupTasks(owned, operations));
};

const aggregateErrors = (errors: unknown[], message: string) => {
  if (errors.length === 1) throw errors[0];
  if (errors.length > 1) throw new AggregateError(errors, message);
};

export const acquireOwnedFileLock = (lockPath: string) => {
  const token = randomUUID();
  let descriptor: number;
  try {
    descriptor = fs.openSync(lockPath, "wx", 0o600);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") {
      throw new Error(
        `[model-edge] exclusive runner lock is held: ${lockPath}. `
        + "If no runner is active, inspect and manually remove this stale lock file.",
      );
    }
    throw error;
  }
  const identity = fs.fstatSync(descriptor);
  try {
    fs.writeFileSync(descriptor, JSON.stringify({ token, pid: process.pid, started_at: new Date().toISOString() }));
    fs.fsyncSync(descriptor);
  } catch (error) {
    fs.closeSync(descriptor);
    try {
      const current = fs.lstatSync(lockPath);
      if (current.dev === identity.dev && current.ino === identity.ino) {
        fs.unlinkSync(lockPath);
      }
    } catch {
      // The original initialization error remains authoritative.
    }
    throw error;
  }
  fs.closeSync(descriptor);
  let released = false;
  return () => {
    if (released) return;
    released = true;
    let current: fs.Stats;
    try {
      current = fs.lstatSync(lockPath);
    } catch (error) {
      if (isMissing(error)) {
        throw new Error(`[model-edge] owned runner lock disappeared before release: ${lockPath}`);
      }
      throw error;
    }
    const contents = fs.readFileSync(lockPath, "utf8");
    if (
      !current.isFile()
      || current.dev !== identity.dev
      || current.ino !== identity.ino
      || !contents.includes(`"token":"${token}"`)
    ) {
      throw new Error(
        `[model-edge] runner lock ownership changed; replacement was not removed: ${lockPath}`,
      );
    }
    fs.unlinkSync(lockPath);
  };
};

export const acquireModelEdgeRunnerLock = () => acquireOwnedFileLock(MODEL_EDGE_LOCK_PATH);

export async function withOwnedFileLock<T>(
  lockPath: string,
  work: () => Promise<T>,
): Promise<T> {
  const release = acquireOwnedFileLock(lockPath);
  let result: T | undefined;
  const errors: unknown[] = [];
  try {
    result = await work();
  } catch (error) {
    errors.push(error);
  }
  try {
    release();
  } catch (error) {
    errors.push(error);
  }
  aggregateErrors(errors, "model-edge run and lock release failed");
  return result as T;
}

export const withModelEdgeRunnerLock = <T>(work: () => Promise<T>) =>
  withOwnedFileLock(MODEL_EDGE_LOCK_PATH, work);
