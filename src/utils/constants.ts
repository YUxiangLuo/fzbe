/**
 * 全局常量配置
 * 统一管理文件大小限制、超时时间、行数限制等
 */

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
  PDF_GENERATION: 30000,
  /** PDF 生成排队等待超时 */
  PDF_QUEUE: 30000,
} as const;

// ============================================
// 并发限制
// ============================================
export const CONCURRENCY_LIMITS = {
  /** 最大并发模型训练任务 */
  MAX_CONCURRENT_MODEL_JOBS: 16,
  /** 最大并发 PDF 生成（Puppeteer）任务 */
  MAX_CONCURRENT_PDF_JOBS: 3,
} as const;

// ============================================
// 数据库连接池
// ============================================
/**
 * 连接池大小必须 > 2 × MAX_CONCURRENT_MODEL_JOBS，否则会死锁：
 * 每个模型任务持有 2 个连接（执行锁 + 槽位锁），训练/预测逻辑还需要
 * 额外连接执行数据库查询。若连接池耗尽，查询等待连接释放，但连接被锁
 * 占用、锁又要等训练完成——形成循环等待。
 *
 * QUERY_HEADROOM：除了锁连接，其他模块（auth、experiments、reports 等）
 * 也共享同一连接池。留出足够余量让普通查询、事务与锁连接并行。
 */
const LOCK_CONNECTIONS = 2 * CONCURRENCY_LIMITS.MAX_CONCURRENT_MODEL_JOBS;
const QUERY_HEADROOM = 18;
export const DB_POOL_CONFIG = {
  LOCK_CONNECTIONS,
  QUERY_HEADROOM,
  CONNECTION_LIMIT: LOCK_CONNECTIONS + QUERY_HEADROOM,
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
// 路径配置 (可被环境变量覆盖)
// ============================================
export const PATHS = {
  /** Chromium 可执行文件路径 */
  CHROMIUM_PATH: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
} as const;
