import path from "path";
import { readdir, rm } from "node:fs/promises";
import mysql from "mysql2/promise";

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

export const PARALLEL_STUDENT_USERNAMES = Array.from(
  { length: 50 },
  (_, index) => String(20246001 + index),
);

export const PARALLEL_STUDENT_GROUPS = {
  all: PARALLEL_STUDENT_USERNAMES,
  trainingBurst: PARALLEL_STUDENT_USERNAMES,
  mixedStartOnly: PARALLEL_STUDENT_USERNAMES.slice(0, 10),
  mixedTraining: PARALLEL_STUDENT_USERNAMES.slice(10, 30),
  mixedProduction: PARALLEL_STUDENT_USERNAMES.slice(30, 50),
} as const;

export function safeFileName(value: string): string {
  return value.replace(/[\/\\?%*:|"<>]/g, "-");
}

export async function rmMany(paths: Array<string | null | undefined>) {
  await Promise.all(
    paths
      .filter((value): value is string => Boolean(value))
      .map((targetPath) => rm(targetPath, { recursive: true, force: true })),
  );
}

export function getUploadsRoot() {
  return path.resolve(process.cwd(), "uploads");
}

export function getManualsRoot() {
  return path.resolve(getUploadsRoot(), "manuals");
}

export function getDatasetsRoot() {
  return path.resolve(getUploadsRoot(), "datasets");
}

export function buildModelArtifactPaths(experimentId: number, studentId: number) {
  const trainedModelsRoot = path.resolve(process.cwd(), "py", "trained_models");
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
  const trainedModelsRoot = path.resolve(process.cwd(), "py", "trained_models");
  try {
    const files = await readdir(trainedModelsRoot);
    const matchingFiles = files
      .filter((fileName) => fileName.startsWith(`${studentId}_`))
      .map((fileName) => path.resolve(trainedModelsRoot, fileName));
    await rmMany(matchingFiles);
  } catch {
    // Ignore missing directories in partial environments.
  }
}

export function formatSqlTimestamp(date: Date): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  return [
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`,
    `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`,
  ].join(" ");
}
