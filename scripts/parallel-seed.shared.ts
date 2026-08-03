import crypto from "node:crypto";
import path from "path";
import {
  lstat,
  mkdir,
  readdir,
  realpath,
  rm,
} from "node:fs/promises";
import mysql from "mysql2/promise";
import type { RowDataPacket } from "mysql2/promise";
import {
  ensureDatabaseManagedStorageRoot,
  getDatabaseManagedStorageRoot,
  type ManagedStorageEnvironment,
} from "../src/utils/managed-storage";

export const PARALLEL_USER_IDS = {
  admin: 900001,
  teacher: 900101,
  assistant: 900201,
} as const;

export const PARALLEL_ACCOUNTS = {
  adminUsername: "parallel_admin",
  teacherUsername: "parallel_teacher",
  assistantUsername: "parallel_assistant",
} as const;

export const PARALLEL_CLASS = {
  id: 8401,
  name: "并发实验测试班",
  code: "PARALLEL-SHIYAN-2026",
} as const;

export const PARALLEL_DATASET = {
  name: "Parallel-Shiyan-并发测试数据集",
  description: "用于 shiyan 多学生并行实验的干净数据集，包含 2 个产品与 48 个月连续月度样本。",
  fileLabel: "parallel-shiyan-concurrency.csv",
  manualDisplayName: "并发实验测试手册.pdf",
  manualFileLabel: "parallel-shiyan-concurrency-manual.pdf",
  manualDescription: "用于 shiyan 多学生并发实验验证的最小手册。",
  industry: "CONC_并发制造业",
  company: "CONC_样本企业A",
  unit: "件",
  startYear: 2021,
  monthCount: 48,
  trainStartIndex: 0,
  trainEndIndex: 35,
  evaluateStartIndex: 36,
  evaluateEndIndex: 41,
  products: [
    "CONC_标准件A",
    "CONC_波动件B",
  ] as const,
} as const;

export const PARALLEL_SCENARIOS = [
  "start-only",
  "training-burst",
  "mixed-load",
] as const;

export type ParallelScenarioName = (typeof PARALLEL_SCENARIOS)[number];

export const PARALLEL_MODEL_SELECTIONS = {
  base: [
    "moving_average",
    "exponential_smoothing",
    "arima",
    "lstm",
  ],
  ensemble: [
    "weighted_ensemble",
    "boosting_ensemble",
    "stacking_ensemble",
  ],
} as const;

const REQUIRED_ENV_KEYS = ["DB_HOST", "DB_USER", "DB_DATABASE"] as const;
const SYSTEM_DATABASES = new Set(["mysql", "information_schema", "performance_schema", "sys"]);
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);
const MODEL_TYPES = [
  "ma",
  "es",
  "arima",
  "lstm",
  "weighted_avg",
  "boosting",
  "stacking",
] as const;

export function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}

export function assertBaseEnv() {
  for (const key of REQUIRED_ENV_KEYS) {
    requireEnv(key);
  }
}

function isEnabled(...values: Array<string | undefined>) {
  return values.some((value) => value === "1");
}

function hasForceFlag() {
  return process.argv.includes("--force");
}

function warnDeprecatedEnv(oldName: string, newName: string) {
  if (process.env[oldName]) {
    console.warn(`⚠️  ${oldName} is deprecated, use ${newName}`);
  }
}

export function getCliPositionalArgs() {
  return process.argv
    .slice(2)
    .filter((value) => value !== "--" && !value.startsWith("--"));
}

export function assertParallelSeedSafety(input: {
  operationName: string;
  commandExample: string;
}) {
  assertBaseEnv();

  if (!hasForceFlag()) {
    throw new Error(
      `${input.operationName} requires --force. Re-run with \`${input.commandExample}\`.`,
    );
  }

  const dbHost = requireEnv("DB_HOST");
  const dbDatabase = requireEnv("DB_DATABASE");

  if (SYSTEM_DATABASES.has(dbDatabase.toLowerCase())) {
    throw new Error(`Refusing to run against system database "${dbDatabase}".`);
  }

  warnDeprecatedEnv("DB_SETUP_ALLOW_PROD", "DB_RESET_ALLOW_PROD");
  if (
    process.env.NODE_ENV === "production" &&
    !isEnabled(process.env.DB_RESET_ALLOW_PROD, process.env.DB_SETUP_ALLOW_PROD)
  ) {
    throw new Error(
      "Refusing to run in production. Set DB_RESET_ALLOW_PROD=1 to override.",
    );
  }

  warnDeprecatedEnv("DB_SETUP_ALLOW_REMOTE", "DB_RESET_ALLOW_REMOTE");
  if (
    !LOCAL_HOSTS.has(dbHost.toLowerCase()) &&
    !isEnabled(process.env.DB_RESET_ALLOW_REMOTE, process.env.DB_SETUP_ALLOW_REMOTE)
  ) {
    throw new Error(
      `Refusing to run against remote host "${dbHost}". Set DB_RESET_ALLOW_REMOTE=1 to override.`,
    );
  }
}

export async function createDbConnection() {
  assertBaseEnv();

  return await mysql.createConnection({
    host: requireEnv("DB_HOST"),
    user: requireEnv("DB_USER"),
    password: process.env.DB_PASSWORD ?? "",
    database: requireEnv("DB_DATABASE"),
    charset: "utf8mb4",
    multipleStatements: true,
  });
}

interface AdvisoryLockRow extends RowDataPacket {
  acquired?: number | null;
  released?: number | null;
}

export async function acquireParallelSeedLock(
  connection: Awaited<ReturnType<typeof createDbConnection>>,
) {
  const databaseHash = crypto
    .createHash("sha256")
    .update(requireEnv("DB_DATABASE"))
    .digest("hex")
    .slice(0, 32);
  const lockName = `fangzhen:parallel-seed:${databaseHash}`;
  const [rows] = await connection.query<AdvisoryLockRow[]>(
    "SELECT GET_LOCK(?, 30) AS acquired",
    [lockName],
  );
  if (Number(rows[0]?.acquired) !== 1) {
    throw new Error("Timed out waiting for the parallel seed lock");
  }

  let released = false;
  return async () => {
    if (released) return;
    released = true;
    const [releaseRows] = await connection.query<AdvisoryLockRow[]>(
      "SELECT RELEASE_LOCK(?) AS released",
      [lockName],
    );
    if (Number(releaseRows[0]?.released) !== 1) {
      throw new Error("Parallel seed lock ownership was lost before release");
    }
  };
}

export const PARALLEL_STUDENT_USERNAMES = Array.from(
  { length: 100 },
  (_, index) => String(20246001 + index),
);

export const PARALLEL_STUDENT_GROUPS = {
  all: PARALLEL_STUDENT_USERNAMES,
  trainingBurst: PARALLEL_STUDENT_USERNAMES,
  mixedStartOnly: [
    ...PARALLEL_STUDENT_USERNAMES.slice(0, 10),
    ...PARALLEL_STUDENT_USERNAMES.slice(50),
  ],
  mixedTraining: PARALLEL_STUDENT_USERNAMES.slice(10, 30),
  mixedProduction: PARALLEL_STUDENT_USERNAMES.slice(30, 50),
} as const;

export function safeFileName(value: string): string {
  return value.replace(/[\/\\?%*:|"<>]/g, "-");
}

export function getUploadsRoot() {
  return path.resolve(process.cwd(), "uploads");
}

function getCurrentStorageEnvironment(): ManagedStorageEnvironment | null {
  const databaseName = process.env.DB_DATABASE?.trim();
  const databaseHost = process.env.DB_HOST?.trim();
  if (!databaseName || !databaseHost) return null;

  const configuredPort = process.env.DB_PORT?.trim();
  const databasePort = configuredPort ? Number(configuredPort) : undefined;
  return {
    DB_DATABASE: databaseName,
    DB_HOST: databaseHost,
    DB_PORT: databasePort,
    E2E_ALLOW_DB_RESET: process.env.E2E_ALLOW_DB_RESET,
  };
}

function getCurrentDatabaseManagedRoot(managedRoot: string): string {
  const environment = getCurrentStorageEnvironment();
  return environment
    ? getDatabaseManagedStorageRoot(managedRoot, environment)
    : path.resolve(managedRoot);
}

async function ensureCurrentDatabaseManagedRoot(category: "manuals" | "datasets" | "reports") {
  const environment = getCurrentStorageEnvironment();
  if (!environment) throw new Error("Database storage identity is required before seeding files");
  const uploadsRoot = getUploadsRoot();
  // Creating only the trusted top-level uploads component first lets the
  // runtime helper reject an existing symlink before any category or .db leaf
  // can be created through it.
  await mkdir(uploadsRoot, { recursive: true });
  return ensureDatabaseManagedStorageRoot(path.join(uploadsRoot, category), environment);
}

export function getManualsRoot() {
  return getCurrentDatabaseManagedRoot(path.resolve(getUploadsRoot(), "manuals"));
}

export function getDatasetsRoot() {
  return getCurrentDatabaseManagedRoot(path.resolve(getUploadsRoot(), "datasets"));
}

export function getReportsRoot() {
  return getCurrentDatabaseManagedRoot(path.resolve(getUploadsRoot(), "reports"));
}

export const ensureManualsRoot = () => ensureCurrentDatabaseManagedRoot("manuals");
export const ensureDatasetsRoot = () => ensureCurrentDatabaseManagedRoot("datasets");
export const ensureReportsRoot = () => ensureCurrentDatabaseManagedRoot("reports");

export function getTrainedModelsRoot() {
  return path.resolve(process.cwd(), "py", "trained_models");
}

export function getGuidedTrainingRoot() {
  return path.resolve(process.cwd(), "py", "guided_training");
}

export function getProductionPreparationRoot() {
  return path.resolve(process.cwd(), "py", "production_preparation");
}

const isPathStrictlyInside = (candidatePath: string, rootPath: string) => {
  const relativePath = path.relative(rootPath, candidatePath);
  return relativePath.length > 0
    && !relativePath.startsWith("..")
    && !path.isAbsolute(relativePath);
};

const isPathInsideOrEqual = (candidatePath: string, rootPath: string) =>
  candidatePath === rootPath
  || isPathStrictlyInside(candidatePath, rootPath);

const getParallelManagedRoots = () => [
  getManualsRoot(),
  getDatasetsRoot(),
  getReportsRoot(),
  getTrainedModelsRoot(),
  getGuidedTrainingRoot(),
  getProductionPreparationRoot(),
];

export function assertParallelManagedPath(targetPath: string): string {
  if (!targetPath.trim()) {
    throw new Error("Refusing to remove an empty parallel fixture path");
  }

  const resolvedPath = path.resolve(targetPath);
  const managedRoots = getParallelManagedRoots();
  if (!managedRoots.some((rootPath) => isPathStrictlyInside(resolvedPath, rootPath))) {
    throw new Error(
      `Refusing to remove a path outside parallel fixture roots: ${resolvedPath}`,
    );
  }
  return resolvedPath;
}

export function assertParallelManagedPaths(
  paths: Array<string | null | undefined>,
): string[] {
  return [
    ...new Set(
      paths
        .filter((value): value is string => value !== null && value !== undefined)
        .map(assertParallelManagedPath),
    ),
  ];
}

const getExistingManagedRoot = async (rootPath: string) => {
  try {
    const rootStats = await lstat(rootPath);
    if (rootStats.isSymbolicLink() || !rootStats.isDirectory()) {
      throw new Error(
        `Refusing to use a non-directory or symbolic-link fixture root: ${rootPath}`,
      );
    }
    const [canonicalWorkspace, canonicalRoot] = await Promise.all([
      realpath(process.cwd()),
      realpath(rootPath),
    ]);
    if (!isPathStrictlyInside(canonicalRoot, canonicalWorkspace)) {
      throw new Error(
        `Refusing to use a fixture root outside the workspace: ${canonicalRoot}`,
      );
    }
    return canonicalRoot;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      // The database namespace leaf may not exist yet. Validate the nearest
      // existing ancestor before allowing a later recursive mkdir; otherwise
      // category/.db -> /outside plus a missing <hash> leaf would be mistaken
      // for a harmless missing root and the seed would write outside the repo.
      const workspacePath = path.resolve(process.cwd());
      const canonicalWorkspace = await realpath(workspacePath);
      let ancestor = path.dirname(path.resolve(rootPath));
      while (isPathInsideOrEqual(ancestor, workspacePath)) {
        try {
          const canonicalAncestor = await realpath(ancestor);
          if (!isPathInsideOrEqual(canonicalAncestor, canonicalWorkspace)) {
            throw new Error(
              `Refusing to create a fixture root through a symbolic link outside the workspace: ${rootPath}`,
            );
          }
          return null;
        } catch (ancestorError) {
          if ((ancestorError as NodeJS.ErrnoException).code !== "ENOENT") {
            throw ancestorError;
          }
          if (ancestor === workspacePath) break;
          ancestor = path.dirname(ancestor);
        }
      }
      throw new Error(`Unable to validate missing fixture root ancestry: ${rootPath}`);
    }
    throw error;
  }
};

const getNearestExistingCanonicalPath = async (
  targetPath: string,
  rootPath: string,
) => {
  let currentPath = targetPath;
  while (isPathInsideOrEqual(currentPath, rootPath)) {
    try {
      return await realpath(currentPath);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        throw error;
      }
      if (currentPath === rootPath) return null;
      currentPath = path.dirname(currentPath);
    }
  }
  return null;
};

const assertNoParallelSymlinkEscape = async (targetPath: string) => {
  const rootPath = getParallelManagedRoots().find((candidateRoot) =>
    isPathStrictlyInside(targetPath, candidateRoot)
  );
  if (!rootPath) {
    throw new Error(
      `Refusing to remove a path outside parallel fixture roots: ${targetPath}`,
    );
  }

  const canonicalRoot = await getExistingManagedRoot(rootPath);
  if (!canonicalRoot) {
    return;
  }
  const canonicalExistingPath = await getNearestExistingCanonicalPath(
    targetPath,
    rootPath,
  );
  if (
    !canonicalExistingPath
    || !isPathInsideOrEqual(canonicalExistingPath, canonicalRoot)
  ) {
    throw new Error(
      `Refusing to remove a path that escapes its fixture root through a symbolic link: ${targetPath}`,
    );
  }
};

export async function validateParallelManagedPaths(
  paths: Array<string | null | undefined>,
): Promise<string[]> {
  const validatedPaths = assertParallelManagedPaths(paths);
  for (const targetPath of validatedPaths) {
    await assertNoParallelSymlinkEscape(targetPath);
  }
  return validatedPaths;
}

async function expandModelArtifactGroups(paths: readonly string[]) {
  const expanded = new Set(paths);
  const trainedModelsRoot = getTrainedModelsRoot();

  for (const targetPath of paths) {
    if (
      path.extname(targetPath).toLowerCase() !== ".pkl"
      || !isPathStrictlyInside(targetPath, trainedModelsRoot)
    ) {
      continue;
    }

    const directoryPath = path.dirname(targetPath);
    const artifactPrefix = `${path.basename(targetPath, ".pkl")}_`;
    try {
      for (const siblingName of await readdir(directoryPath)) {
        if (siblingName.startsWith(artifactPrefix)) {
          expanded.add(assertParallelManagedPath(path.join(directoryPath, siblingName)));
        }
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        throw error;
      }
    }
  }

  return [...expanded];
}

export async function rmMany(paths: Array<string | null | undefined>) {
  const validatedPaths = await validateParallelManagedPaths(paths);
  const expandedPaths = await expandModelArtifactGroups(validatedPaths);
  await validateParallelManagedPaths(expandedPaths);
  for (const targetPath of expandedPaths) {
    await assertNoParallelSymlinkEscape(targetPath);
    await rm(targetPath, { recursive: true, force: true });
  }
}

export function buildModelArtifactPaths(experimentId: number, studentId: number) {
  const trainedModelsRoot = getTrainedModelsRoot();
  return MODEL_TYPES.flatMap((modelType) => [
    path.resolve(
      trainedModelsRoot,
      `exp${experimentId}-student${studentId}-${modelType}.pkl`,
    ),
    path.resolve(
      trainedModelsRoot,
      `exp${experimentId}-student${studentId}-${modelType}-production.pkl`,
    ),
  ]);
}

export async function cleanupLegacyStudentModelFiles(studentId: number) {
  const trainedModelsRoot = getTrainedModelsRoot();
  try {
    const files = await readdir(trainedModelsRoot);
    const matchingFiles = files
      .filter((fileName) => fileName.startsWith(`${studentId}_`))
      .map((fileName) => path.resolve(trainedModelsRoot, fileName));
    await rmMany(matchingFiles);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
      throw error;
    }
  }
}

export function formatSqlTimestamp(date: Date): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  return [
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`,
    `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`,
  ].join(" ");
}
