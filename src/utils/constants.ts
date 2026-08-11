/**
 * 全局常量配置
 * 统一管理文件大小限制、超时时间、行数限制等
 */

import { availableParallelism } from "node:os";

function parsePositiveIntegerEnv(name: string): number | null {
  const rawValue = process.env[name]?.trim();
  if (!rawValue) return null;

  if (!/^[1-9]\d*$/.test(rawValue)) {
    throw new Error(`${name} must be a positive integer, got: ${rawValue}`);
  }

  const parsed = Number(rawValue);
  if (!Number.isSafeInteger(parsed)) {
    throw new Error(`${name} must be a positive integer, got: ${rawValue}`);
  }
  return parsed;
}

/** 模型执行的不可配置硬上限：10 分钟。 */
export const MODEL_EXECUTION_HARD_CAP_MS = 10 * 60 * 1000;
/** A guided/production orchestration process may give one nested model the full cap. */
export const MODEL_ORCHESTRATION_HARD_CAP_MS = 630_000;

function parseModelTimeoutEnv(name: string, fallbackMs: number): number {
  const value = parsePositiveIntegerEnv(name) ?? fallbackMs;
  if (value > MODEL_EXECUTION_HARD_CAP_MS) {
    throw new Error(
      `${name} must be <= ${MODEL_EXECUTION_HARD_CAP_MS}, got: ${value}`,
    );
  }
  return value;
}

/**
 * 模型 HTTP 超时必须保持：operation + 最坏进程终止 + 响应余量 <= nginx < browser。
 * nginx 和前端当前是静态部署配置，因此后端只允许缩短 operation deadline，
 * 不允许通过单个环境变量把它提高到传输层预算之外。
 */
export const MODEL_HTTP_TIMEOUT_BUDGETS = {
  OPERATION_DEADLINE_HARD_CAP: 660_000,
  NGINX_READ: 675_000,
  BROWSER_REQUEST: 690_000,
  /** Includes the bounded failure-status write plus response serialization/network time. */
  RESPONSE_RESERVE: 6_000,
} as const;

const ADF_EXECUTION_TIMEOUT_MS = 30_000;

/** The production nginx exact ADF route uses this independent short budget. */
export const ADF_HTTP_TIMEOUT_BUDGETS = {
  NGINX_READ: 45_000,
  RESPONSE_RESERVE: MODEL_HTTP_TIMEOUT_BUDGETS.RESPONSE_RESERVE,
} as const;

/** Best-effort DB failure recording must fit inside the response reserve. */
const MODEL_FAILURE_RECORD_TIMEOUT_MS = 2_000;

// Business DB work runs while a process-local admission permit is held. mysql2
// has no default per-query or acquire timeout, so a hung query or a
// starved pool would keep the permit occupied far past the operation deadline.
// These bound both so the permit is always released promptly. Generous relative
// to the healthy sub-second query time, but far below the operation deadline.
const MODEL_DB_QUERY_TIMEOUT_MS =
  parsePositiveIntegerEnv("MODEL_DB_QUERY_TIMEOUT_MS") ?? 30_000;
const MODEL_DB_CONNECTION_ACQUIRE_TIMEOUT_MS =
  parsePositiveIntegerEnv("MODEL_DB_CONNECTION_ACQUIRE_TIMEOUT_MS") ?? 10_000;
const MODEL_PROCESS_TERMINATION_GRACE_MS =
  parsePositiveIntegerEnv("MODEL_PROCESS_TERMINATION_GRACE_MS") ?? 5_000;
const ADF_OPERATION_CLEANUP_BUDGET_MS =
  MODEL_PROCESS_TERMINATION_GRACE_MS
  + ADF_HTTP_TIMEOUT_BUDGETS.RESPONSE_RESERVE;
const ADF_SINGLE_OPERATION_MAX_MS =
  ADF_HTTP_TIMEOUT_BUDGETS.NGINX_READ - ADF_OPERATION_CLEANUP_BUDGET_MS;

if (ADF_SINGLE_OPERATION_MAX_MS < ADF_EXECUTION_TIMEOUT_MS) {
  throw new Error(
    "MODEL_PROCESS_TERMINATION_GRACE_MS leaves less than "
    + `${ADF_EXECUTION_TIMEOUT_MS}ms for ADF inside the `
    + `${ADF_HTTP_TIMEOUT_BUDGETS.NGINX_READ}ms nginx ADF-route budget`,
  );
}

const MODEL_OPERATION_CLEANUP_BUDGET_MS =
  MODEL_PROCESS_TERMINATION_GRACE_MS
  + MODEL_HTTP_TIMEOUT_BUDGETS.RESPONSE_RESERVE;
const MODEL_SINGLE_OPERATION_MAX_MS = Math.min(
  MODEL_HTTP_TIMEOUT_BUDGETS.OPERATION_DEADLINE_HARD_CAP,
  MODEL_HTTP_TIMEOUT_BUDGETS.NGINX_READ - MODEL_OPERATION_CLEANUP_BUDGET_MS,
);

if (MODEL_SINGLE_OPERATION_MAX_MS <= 0) {
  throw new Error(
    "MODEL_PROCESS_TERMINATION_GRACE_MS exceeds the nginx model-route cleanup budget",
  );
}

const MODEL_SINGLE_OPERATION_TIMEOUT_MS =
  parsePositiveIntegerEnv("MODEL_OPERATION_DEADLINE_MS")
  ?? MODEL_HTTP_TIMEOUT_BUDGETS.OPERATION_DEADLINE_HARD_CAP;

if (MODEL_SINGLE_OPERATION_TIMEOUT_MS > MODEL_SINGLE_OPERATION_MAX_MS) {
  throw new Error(
    `MODEL_OPERATION_DEADLINE_MS must be <= ${MODEL_SINGLE_OPERATION_MAX_MS} `
    + `for the ${MODEL_HTTP_TIMEOUT_BUDGETS.NGINX_READ}ms nginx model-route budget, `
    + `got: ${MODEL_SINGLE_OPERATION_TIMEOUT_MS}`,
  );
}

const SYSTEM_PARALLELISM = Math.max(1, availableParallelism());
const DEFAULT_MODEL_JOB_CAP = 50;
const DEFAULT_PDF_JOB_CAP = 12;
const DEFAULT_MAX_CONCURRENT_MODEL_JOBS = Math.min(SYSTEM_PARALLELISM, DEFAULT_MODEL_JOB_CAP);
const DEFAULT_MAX_CONCURRENT_PDF_JOBS = Math.min(
  DEFAULT_PDF_JOB_CAP,
  Math.max(1, Math.ceil(SYSTEM_PARALLELISM / 4)),
);

// ============================================
// 文件大小限制 (bytes)
// ============================================
export const FILE_SIZE_LIMITS = {
  /** CSV 文件最大大小: 5MB */
  CSV_MAX_SIZE: 5 * 1024 * 1024,
  /** 数据集 CSV 文件最大大小: 50MB */
  DATASET_CSV_MAX_SIZE: 50 * 1024 * 1024,
  /** PDF 文件最大大小: 50MB */
  PDF_MAX_SIZE: 50 * 1024 * 1024,
} as const;

// ============================================
// CSV 解析限制
// ============================================
export const CSV_LIMITS = {
  /** 单个产品分段允许包含的最大数据行数 */
  DATASET_SEGMENT_MAX_ROWS: 10_000,
  /** 一个管理员数据集上传文件允许包含的总数据行数 */
  DATASET_UPLOAD_MAX_ROWS: 100_000,
  /** 单个数据集允许拆分出的最大行业/公司/产品组合数 */
  DATASET_UPLOAD_MAX_SEGMENTS: 1_000,
  /** 批量导入学生最大行数 */
  BATCH_IMPORT_MAX_ROWS: 1_000,
} as const;

// ============================================
// 内容长度限制
// ============================================
export const CONTENT_LIMITS = {
  /** Markdown 报告最大长度 (字符) */
  MARKDOWN_MAX_LENGTH: 500000,
  /** 反馈内容最大长度 */
  FEEDBACK_MAX_LENGTH: 10000,
  /** 描述字段最大长度 */
  DESCRIPTION_MAX_LENGTH: 5000,
} as const;

// ============================================
// 超时时间 (毫秒)
// ============================================
export const TIMEOUTS = {
  /** 所有 Python 模型任务都不可突破的硬上限 */
  MODEL_EXECUTION_HARD_CAP: MODEL_EXECUTION_HARD_CAP_MS,
  /** 只允许编排一个嵌套模型任务的 Python 父进程硬上限。 */
  MODEL_ORCHESTRATION_HARD_CAP: MODEL_ORCHESTRATION_HARD_CAP_MS,
  /** 单次后端模型操作的总时限；每个 guided step / prepare / predict 独立计时。 */
  MODEL_SINGLE_OPERATION: MODEL_SINGLE_OPERATION_TIMEOUT_MS,
  /** 当前终止、解锁和 nginx 预算下允许的单次操作最大值。 */
  MODEL_SINGLE_OPERATION_MAX: MODEL_SINGLE_OPERATION_MAX_MS,
  /** deadline 后为 Python 终止、进程内资源释放和响应传输预留的总预算。 */
  MODEL_OPERATION_CLEANUP_BUDGET: MODEL_OPERATION_CLEANUP_BUDGET_MS,
  /** lease 释放后记录模型失败状态的 best-effort 总时限。 */
  MODEL_FAILURE_RECORD: MODEL_FAILURE_RECORD_TIMEOUT_MS,
  /** 持有并发额度期间业务查询的最长执行时间（超时即释放 lease）。 */
  MODEL_DB_QUERY: MODEL_DB_QUERY_TIMEOUT_MS,
  /** 持有并发额度期间从连接池获取业务连接的最长等待时间。 */
  MODEL_DB_CONNECTION_ACQUIRE: MODEL_DB_CONNECTION_ACQUIRE_TIMEOUT_MS,
  /** Python 收到 SIGTERM 后允许优雅退出的时间。 */
  MODEL_PROCESS_TERMINATION_GRACE: MODEL_PROCESS_TERMINATION_GRACE_MS,
  /** ADF 从接收请求到进入清理前的总时限。 */
  ADF_OPERATION: Math.min(ADF_SINGLE_OPERATION_MAX_MS, MODEL_SINGLE_OPERATION_TIMEOUT_MS),
  /** ADF deadline 后为 Python 终止、admission 释放和响应传输预留的预算。 */
  ADF_OPERATION_CLEANUP_BUDGET: ADF_OPERATION_CLEANUP_BUDGET_MS,
  /** Python 模型训练超时 */
  MODEL_TRAINING: parseModelTimeoutEnv("MODEL_TRAINING_TIMEOUT_MS", 300000),
  /** Python 模型预测超时 */
  MODEL_PREDICTION: parseModelTimeoutEnv("MODEL_PREDICTION_TIMEOUT_MS", 60000),
  /** ADF 平稳性检测超时 */
  ADF: ADF_EXECUTION_TIMEOUT_MS,
  /** 单次 PDF 的 Puppeteer 启动、渲染、写入和关闭总预算 */
  PDF_GENERATION: parsePositiveIntegerEnv("PDF_GENERATION_TIMEOUT_MS") ?? 30000,
  /** PDF 生成排队等待超时 */
  PDF_QUEUE: parsePositiveIntegerEnv("PDF_QUEUE_TIMEOUT_MS") ?? 60000,
} as const;

// ============================================
// 并发限制
// ============================================
const MAX_CONCURRENT_MODEL_JOBS =
  parsePositiveIntegerEnv("MAX_CONCURRENT_MODEL_JOBS") ??
  parsePositiveIntegerEnv("MODEL_CONCURRENCY") ??
  DEFAULT_MAX_CONCURRENT_MODEL_JOBS;
const MAX_CONCURRENT_PDF_JOBS =
  parsePositiveIntegerEnv("MAX_CONCURRENT_PDF_JOBS") ??
  parsePositiveIntegerEnv("PDF_CONCURRENCY") ??
  DEFAULT_MAX_CONCURRENT_PDF_JOBS;

export const CONCURRENCY_LIMITS = {
  /** 当前系统可用并行度（通常等于可用 CPU 核/线程数） */
  SYSTEM_PARALLELISM,
  /** 默认模型任务上限；64 核 / 100 人课堂默认允许 50 个活跃任务，其余请求快速返回 429。 */
  DEFAULT_MODEL_JOB_CAP,
  /** 默认 PDF 任务上限；PDF/Chromium 更重，自动值按 CPU/4 估算并封顶 */
  DEFAULT_PDF_JOB_CAP,
  /** 最大并发模型训练/生产准备/预测任务 */
  MAX_CONCURRENT_MODEL_JOBS,
  /** 最大并发 PDF 生成（Puppeteer）任务 */
  MAX_CONCURRENT_PDF_JOBS,
} as const;

// ============================================
// Python 模型运行时
// ============================================
export const PYTHON_RUNTIME_CONFIG = {
  /** 每个 Python 模型子进程允许使用的原生计算线程数。默认 1，避免高并发时 BLAS/TF 线程爆炸。 */
  MODEL_PROCESS_THREADS: parsePositiveIntegerEnv("PYTHON_MODEL_THREADS") ?? 1,
  /** TensorFlow inter-op 线程数通常保持 1，更适合多进程并发。 */
  TF_INTER_OP_THREADS: parsePositiveIntegerEnv("TF_NUM_INTEROP_THREADS") ?? 1,
} as const;

// ============================================
// 磁盘清理保留期
// ============================================
export const CLEANUP_RETENTION = {
  /** 导出的 ZIP/CSV 下载文件默认保留 7 天 */
  EXPORT_FILES_MS: (parsePositiveIntegerEnv("EXPORT_FILE_RETENTION_DAYS") ?? 7) * 24 * 60 * 60 * 1000,
  /** 上传临时目录中的残留文件默认保留 24 小时 */
  TEMP_FILES_MS: (parsePositiveIntegerEnv("TEMP_FILE_RETENTION_HOURS") ?? 24) * 60 * 60 * 1000,
  /** guided training 无 DB 记录的孤儿目录默认保留 24 小时，避开创建会话的短暂窗口 */
  GUIDED_TRAINING_ORPHAN_MS: (parsePositiveIntegerEnv("GUIDED_TRAINING_ORPHAN_RETENTION_HOURS") ?? 24) * 60 * 60 * 1000,
  /** production preparation 无 DB 记录的孤儿目录默认保留 24 小时 */
  PRODUCTION_PREPARATION_ORPHAN_MS: (parsePositiveIntegerEnv("PRODUCTION_PREPARATION_ORPHAN_RETENTION_HOURS") ?? 24) * 60 * 60 * 1000,
  /** 普通模型产物默认保留 30 天；仅清理已完成、无记录或长期无活动的实验产物 */
  MODEL_ARTIFACTS_MS: (parsePositiveIntegerEnv("MODEL_ARTIFACT_RETENTION_DAYS") ?? 30) * 24 * 60 * 60 * 1000,
} as const;

// ============================================
// 数据库连接池
// ============================================
/**
 * A single Bun instance never pins database connections for model mutexes.
 * The pool therefore serves only short queries/transactions. Keep a floor of
 * 40 connections for classroom bursts and scale the default modestly with the
 * process-local model capacity; explicit DB_CONNECTION_LIMIT may raise it.
 */
const MIN_CONNECTION_LIMIT = Math.max(
  40,
  Math.ceil(CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS / 2),
);
const CONNECTION_LIMIT =
  parsePositiveIntegerEnv("DB_CONNECTION_LIMIT") ?? MIN_CONNECTION_LIMIT;

export const DB_POOL_CONFIG = {
  MIN_CONNECTION_LIMIT,
  CONNECTION_LIMIT,
} as const;

// ============================================
// 认证限流
// ============================================
export const RATE_LIMITS = {
  /** 登录限流窗口。默认 15 分钟。 */
  AUTH_LOGIN_WINDOW_MS: parsePositiveIntegerEnv("AUTH_LOGIN_RATE_LIMIT_WINDOW_MS") ?? 15 * 60 * 1000,
  /** 单 IP 登录尝试上限。默认偏高，避免同一教室 NAT 误伤。 */
  AUTH_LOGIN_MAX_PER_IP: parsePositiveIntegerEnv("AUTH_LOGIN_RATE_LIMIT_MAX_PER_IP") ?? 100,
  /** 单账号登录尝试上限。 */
  AUTH_LOGIN_MAX_PER_ACCOUNT: parsePositiveIntegerEnv("AUTH_LOGIN_RATE_LIMIT_MAX_PER_ACCOUNT") ?? 10,
  /** 自助注册限流窗口。默认 1 小时。 */
  AUTH_REGISTER_WINDOW_MS: parsePositiveIntegerEnv("AUTH_REGISTER_RATE_LIMIT_WINDOW_MS") ?? 60 * 60 * 1000,
  /** 单 IP 自助注册上限。 */
  AUTH_REGISTER_MAX_PER_IP: parsePositiveIntegerEnv("AUTH_REGISTER_RATE_LIMIT_MAX_PER_IP") ?? 50,
  /** 单用户名自助注册上限。 */
  AUTH_REGISTER_MAX_PER_ACCOUNT: parsePositiveIntegerEnv("AUTH_REGISTER_RATE_LIMIT_MAX_PER_ACCOUNT") ?? 3,
} as const;

// ============================================
// 用户相关常量
// ============================================
export const USER_CONSTANTS = {
  /** 教师邮箱域名 */
  TEACHER_EMAIL_DOMAIN: "teacher.edu",
  /** 助教邮箱域名 */
  ASSISTANT_EMAIL_DOMAIN: "assistant.edu",
  /** 学生邮箱域名 */
  STUDENT_EMAIL_DOMAIN: "student.edu",
  /** 手机号正则 */
  PHONE_REGEX: /^\d{6,15}$/,
  /** 学号正则 (8-32位数字) */
  STUDENT_ID_REGEX: /^\d{8,32}$/,
  /** 最小姓名长度 */
  MIN_NAME_LENGTH: 2,
  /** 最大姓名长度 */
  MAX_NAME_LENGTH: 64,
  /** 新密码最小长度；既有账号登录不受此策略变更影响。 */
  MIN_PASSWORD_LENGTH: 8,
  /** bcrypt cost */
  BCRYPT_COST: 10,
} as const;

// ============================================
// 认证相关常量
// ============================================
export const AUTH_CONSTANTS = {
  /** JWT 签名与验证算法。签发和中间件必须保持一致。 */
  JWT_ALGORITHM: "HS256",
  /** JWT 有效期（秒） */
  JWT_EXPIRATION_SECONDS: 60 * 60 * 2,
} as const;

// ============================================
// 路径配置 (可被环境变量覆盖)
// ============================================
export const PATHS = {
  /** Chromium 可执行文件路径 */
  CHROMIUM_PATH: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
} as const;
