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

const SYSTEM_PARALLELISM = Math.max(1, availableParallelism());
const DEFAULT_TRAIN_JOB_CAP = 50;
const DEFAULT_PREDICT_JOB_CAP = 50;
const DEFAULT_PREPARE_JOB_CAP = 25;
const DEFAULT_PDF_JOB_CAP = 12;
// 训练/预测按满 CPU 封顶到 50；生产准备较少见，按 CPU 的一半封顶到 25。
const DEFAULT_MAX_CONCURRENT_TRAIN_JOBS = Math.min(SYSTEM_PARALLELISM, DEFAULT_TRAIN_JOB_CAP);
const DEFAULT_MAX_CONCURRENT_PREDICT_JOBS = Math.min(SYSTEM_PARALLELISM, DEFAULT_PREDICT_JOB_CAP);
const DEFAULT_MAX_CONCURRENT_PREPARE_JOBS = Math.min(
  DEFAULT_PREPARE_JOB_CAP,
  Math.max(1, Math.ceil(SYSTEM_PARALLELISM / 2)),
);
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
  /** CSV 最大行数 */
  MAX_ROWS: 10000,
  /** 批量导入学生最大行数 */
  BATCH_IMPORT_MAX_ROWS: 1000,
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
  /** Python 模型训练超时 */
  MODEL_TRAINING: 300000,
  /** Python 模型预测超时 */
  MODEL_PREDICTION: 60000,
  /** ADF 平稳性检测超时 */
  ADF: 30000,
  /** PDF 生成超时（Puppeteer 执行） */
  PDF_GENERATION: parsePositiveIntegerEnv("PDF_GENERATION_TIMEOUT_MS") ?? 30000,
  /** PDF 生成排队等待超时 */
  PDF_QUEUE: parsePositiveIntegerEnv("PDF_QUEUE_TIMEOUT_MS") ?? 60000,
} as const;

// ============================================
// 并发限制
// ============================================
// 训练、生产准备、预测、PDF 四种行为各自独立限流，互不挤占。
// 每个行为只有一个对应的环境变量；未设置时使用该行为的自适应默认值。
const MAX_CONCURRENT_TRAIN_JOBS =
  parsePositiveIntegerEnv("MAX_CONCURRENT_TRAIN_JOBS") ??
  DEFAULT_MAX_CONCURRENT_TRAIN_JOBS;
const MAX_CONCURRENT_PREPARE_JOBS =
  parsePositiveIntegerEnv("MAX_CONCURRENT_PREPARE_JOBS") ??
  DEFAULT_MAX_CONCURRENT_PREPARE_JOBS;
const MAX_CONCURRENT_PREDICT_JOBS =
  parsePositiveIntegerEnv("MAX_CONCURRENT_PREDICT_JOBS") ??
  DEFAULT_MAX_CONCURRENT_PREDICT_JOBS;
const MAX_CONCURRENT_PDF_JOBS =
  parsePositiveIntegerEnv("MAX_CONCURRENT_PDF_JOBS") ??
  DEFAULT_MAX_CONCURRENT_PDF_JOBS;

export const CONCURRENCY_LIMITS = {
  /** 当前系统可用并行度（通常等于可用 CPU 核/线程数） */
  SYSTEM_PARALLELISM,
  /** 默认训练任务上限（含引导训练）；按 64 核 / 50 人课堂配置，封顶 50 */
  DEFAULT_TRAIN_JOB_CAP,
  /** 默认生产准备任务上限；较少见，按 CPU/2 估算并封顶 25 */
  DEFAULT_PREPARE_JOB_CAP,
  /** 默认预测任务上限；较轻量，封顶 50 */
  DEFAULT_PREDICT_JOB_CAP,
  /** 默认 PDF 任务上限；PDF/Chromium 更重，自动值按 CPU/4 估算并封顶 */
  DEFAULT_PDF_JOB_CAP,
  /** 最大并发训练任务（引导训练共用此池） */
  MAX_CONCURRENT_TRAIN_JOBS,
  /** 最大并发生产准备任务 */
  MAX_CONCURRENT_PREPARE_JOBS,
  /** 最大并发预测任务 */
  MAX_CONCURRENT_PREDICT_JOBS,
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
  /** 普通模型产物默认保留 30 天；仅清理已完成、无记录或长期无活动的实验产物 */
  MODEL_ARTIFACTS_MS: (parsePositiveIntegerEnv("MODEL_ARTIFACT_RETENTION_DAYS") ?? 30) * 24 * 60 * 60 * 1000,
} as const;

// ============================================
// 数据库连接池
// ============================================
/**
 * 并发限流改为进程内信号量后，模型任务不再持有任何 advisory lock 连接，
 * 连接池不再需要满足旧的 "> 2 × 模型并发" 死锁约束。
 *
 * 现在连接池只需覆盖同时在途的数据库查询：最坏情况下四个池全部满载，
 * 每个在跑的任务偶发一条查询，再加上其他模块（auth、experiments、reports 等）
 * 的常规流量。因此按 "四池容量之和 + 查询余量" 保守估算。
 *
 * QUERY_HEADROOM：留给常规接口/事务的余量，默认 max(40, 池容量之和 / 4)。
 */
const ACTIVE_JOB_CEILING =
  CONCURRENCY_LIMITS.MAX_CONCURRENT_TRAIN_JOBS +
  CONCURRENCY_LIMITS.MAX_CONCURRENT_PREPARE_JOBS +
  CONCURRENCY_LIMITS.MAX_CONCURRENT_PREDICT_JOBS +
  CONCURRENCY_LIMITS.MAX_CONCURRENT_PDF_JOBS;
const QUERY_HEADROOM =
  parsePositiveIntegerEnv("DB_QUERY_HEADROOM") ??
  Math.max(40, Math.ceil(ACTIVE_JOB_CEILING / 4));
const CONNECTION_LIMIT =
  parsePositiveIntegerEnv("DB_CONNECTION_LIMIT") ??
  ACTIVE_JOB_CEILING + QUERY_HEADROOM;

export const DB_POOL_CONFIG = {
  ACTIVE_JOB_CEILING,
  QUERY_HEADROOM,
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
  /** 最小密码长度 */
  MIN_PASSWORD_LENGTH: 6,
  /** bcrypt cost */
  BCRYPT_COST: 10,
} as const;

// ============================================
// 认证相关常量
// ============================================
export const AUTH_CONSTANTS = {
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
