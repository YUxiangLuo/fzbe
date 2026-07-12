# 并发控制说明

本文档说明后端当前的并发控制逻辑、默认容量、部署参数和测试方式。相关实现主要位于：

- `src/utils/constants.ts`
- `src/utils/semaphore.ts`
- `src/utils/keyed-lock.ts`
- `src/modules/models/model-locks.ts`
- `src/modules/models/models.controller.ts`
- `src/modules/models/models.service.ts`
- `src/modules/reports/reports.controller.ts`
- `src/modules/reports/reports.service.ts`
- `src/config/database.ts`

## 目标场景

默认配置面向单台后端实例，按 `availableParallelism()` 看到的 CPU 可用并行度自动生成并发策略。

核心设计：**训练、生产准备、预测、PDF 生成四种行为各自拥有独立的并发池**，互不挤占。任一池被占满只会限流该行为，其余三种行为不受影响。四个池都是**进程内实现**（`Semaphore` / `KeyedLock`），不再使用 MySQL advisory lock，因此模型任务不再占用数据库连接，连接池大小与并发上限解耦。

核心目标：

- 训练、生产准备、预测各自不能无限启动 Python 子进程，且互相隔离。
- 同一学生的同一模型不能被重复训练/准备/预测。
- PDF 生成要限流，超过队列等待时间后让前端可重试。
- 数据库连接池不再被并发上限绑架（旧的“锁占连接导致死锁”问题已消除）。
- 不配置并发环境变量时即可自动生成合理策略；显式环境变量仅用于人工覆盖。

## 配置入口

所有主要并发参数都在 `src/utils/constants.ts` 统一计算。

| 配置项 | 默认值 | 说明 |
| --- | ---: | --- |
| `MAX_CONCURRENT_TRAIN_JOBS` | `min(availableParallelism(), 50)` | 训练池上限（引导训练共用此池）。 |
| `MAX_CONCURRENT_PREDICT_JOBS` | `min(availableParallelism(), 50)` | 预测池上限。 |
| `MAX_CONCURRENT_PREPARE_JOBS` | `min(ceil(availableParallelism() / 2), 25)` | 生产准备池上限。 |
| `MAX_CONCURRENT_PDF_JOBS` | `min(ceil(availableParallelism() / 4), 12)` | PDF/Puppeteer 并发生成数。 |
| `PDF_QUEUE_TIMEOUT_MS` | `60000` | PDF 队列等待超时。超时后接口返回 `503`。 |
| `PDF_GENERATION_TIMEOUT_MS` | `30000` | 单个 `page.pdf()` 的生成超时。 |
| `DB_QUERY_HEADROOM` | `max(40, ceil(poolSum / 4))` | 除四个池的活动任务外，给普通查询、事务和其他接口保留的连接余量。`poolSum = train + prepare + predict + pdf`。 |
| `DB_CONNECTION_LIMIT` | `poolSum + DB_QUERY_HEADROOM` | MySQL pool 连接数。显式配置后不再自动计算。 |
| `PYTHON_MODEL_THREADS` | `1` | 每个 Python 模型子进程的 BLAS/NumExpr/TensorFlow intra-op 线程默认值。 |
| `TF_NUM_INTEROP_THREADS` | `1` | TensorFlow inter-op 线程默认值。 |

所有这些环境变量都必须是正整数；非法值会在模块加载时直接报错。每个行为只有一个对应的环境变量，互不影响；未设置时使用该行为的自适应默认值。

## 64 核 / 50 学生默认容量

如果服务进程能通过 `availableParallelism()` 看到 64 核，默认计算为：

```text
MAX_CONCURRENT_TRAIN_JOBS   = min(64, 50) = 50
MAX_CONCURRENT_PREDICT_JOBS = min(64, 50) = 50
MAX_CONCURRENT_PREPARE_JOBS = min(ceil(64 / 2), 25) = 25
MAX_CONCURRENT_PDF_JOBS     = min(ceil(64 / 4), 12) = 12
PDF_QUEUE_TIMEOUT_MS = 60000
PDF_GENERATION_TIMEOUT_MS = 30000
PYTHON_MODEL_THREADS = 1
TF_NUM_INTEROP_THREADS = 1

poolSum             = 50 + 25 + 50 + 12 = 137
DB_QUERY_HEADROOM   = max(40, ceil(137 / 4)) = 40
DB_CONNECTION_LIMIT = 137 + 40 = 177
```

单实例生产部署建议优先保持这些并发环境变量未设置，让后端自动计算。数据库侧的 `max_connections` 必须大于等于自动计算出的 `DB_CONNECTION_LIMIT`；64 核课堂场景为 `177`，建议数据库设置为 `250+`，给迁移、管理连接和临时排查留余量。

> 注意：四个池的上限之和（默认 137 个模型/PDF 任务）在极端峰值下可能超过物理 CPU 数。这是隔离性的代价，靠 `PYTHON_MODEL_THREADS=1`（每进程单原生线程）保持可承受。如果确实观察到 CPU 被打满，优先下调单个池的上限，而不是把线程数调大。

## 模型任务并发

三种模型行为各有一个独立的进程内信号量池：

- 训练池：`MAX_CONCURRENT_TRAIN_JOBS`（引导训练也从此池取槽位）
- 生产准备池：`MAX_CONCURRENT_PREPARE_JOBS`
- 预测池：`MAX_CONCURRENT_PREDICT_JOBS`

每个请求在 controller 中的流程为：先取执行锁，再取对应池的槽位，`finally` 中先放槽位再放执行锁。

### 执行锁（409）

`acquireModelExecutionLock(experimentId, modelType)` 使用进程内 `KeyedLock`，键为 `fangzhen-model:exp:{experimentId}:{modelType}`。用于避免同一个学生的同一个模型被重复训练、准备或预测。拿不到该锁时接口返回：

```http
409 Conflict
{ "error": "同一模型正在训练或预测，请稍后再试" }
```

### 行为池槽位（429）

`acquireModelSlot(kind)`（`kind` 为 `"train" | "prepare" | "predict"`）对相应的 `Semaphore` 执行**非阻塞** `tryAcquire()`。该池已满时立即返回 `null`，接口返回：

```http
429 Too Many Requests
{ "error": "模型服务繁忙，请稍后再试" }
```

因为三个池相互独立，训练池打满**不会**让预测或生产准备接口返回 429，反之亦然。这是本设计的核心保证。

### cached production artifact 的例外

基础模型 `ma`、`es`、`arima`、`lstm` 的 `prepare-production` 在拿到执行锁后，会先检查是否已经存在可复用的 production artifact。

如果 artifact 可复用，接口直接返回，不占用生产准备池，也不启动 Python 子进程：

```json
{
  "status": "success",
  "results": {
    "saved_model": "...",
    "artifact_phase": "production",
    "reused_existing": true
  }
}
```

ensemble 模型（`weighted_avg`、`boosting`、`stacking`）不走这个前置绕过逻辑。即使已有 cached artifact，其 `prepare-production` 仍先进入生产准备池，再由 service 层判断是否可复用，避免绕过生产准备并发上限。

## Python 子进程线程控制

每个模型任务最终可能启动 Python 子进程。为了避免大量 Python 进程各自再启动多个 BLAS 或 TensorFlow 原生线程，后端在启动 Python 子进程时设置以下环境变量：

```text
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
MKL_NUM_THREADS
VECLIB_MAXIMUM_THREADS
NUMEXPR_NUM_THREADS
TF_NUM_INTRAOP_THREADS
TF_NUM_INTEROP_THREADS
TF_CPP_MIN_LOG_LEVEL
```

默认 `PYTHON_MODEL_THREADS=1`，`TF_NUM_INTEROP_THREADS=1`。如果进程环境中已经显式设置了更具体的变量，例如 `OPENBLAS_NUM_THREADS=2`，则保留该显式值。

## 数据库连接池与启动前校验

并发限流改为进程内信号量后，模型任务**不再持有任何数据库 advisory lock 连接**，旧的“连接池必须 > 2 × 模型并发，否则锁占连接导致死锁”的约束已不再适用。

连接池现在只需覆盖同时在途的数据库查询：四个池全部满载时的活动任务数，加上其他模块（auth、experiments、reports 等）的常规流量。默认公式：

```text
poolSum             = MAX_CONCURRENT_TRAIN_JOBS + MAX_CONCURRENT_PREPARE_JOBS
                    + MAX_CONCURRENT_PREDICT_JOBS + MAX_CONCURRENT_PDF_JOBS
DB_QUERY_HEADROOM   = max(40, ceil(poolSum / 4))
DB_CONNECTION_LIMIT = poolSum + DB_QUERY_HEADROOM
```

`src/config/database.ts` 使用 `DB_POOL_CONFIG.CONNECTION_LIMIT` 创建 mysql2 pool，`queueLimit` 由 `DB_QUEUE_LIMIT`（默认 100）控制。

启动时 `runServerStartupPreflight()` 做两项校验：

- `connectionLimit >= DB_QUERY_HEADROOM`（一个基本下限，避免连接池被配置得连常规查询余量都不够）。
- 数据库 `max_connections >= connectionLimit`。

任一条件不满足，后端会记录 fatal 日志并退出，在启动阶段暴露部署配置错误。

## PDF 生成并发

PDF 生成使用进程内 `Semaphore`，并发上限为 `MAX_CONCURRENT_PDF_JOBS`。与模型池不同，PDF 池是**阻塞排队**的：

```ts
await pdfSemaphore.acquire(TIMEOUTS.PDF_QUEUE);
```

如果还有可用名额，立即生成 PDF。如果名额已满，请求进入队列。等待超过 `PDF_QUEUE_TIMEOUT_MS` 后，`Semaphore` 抛出 `SemaphoreTimeoutError`，controller 返回：

```http
503 Service Unavailable
{ "error": "PDF 生成服务繁忙，排队等待超过 N 秒，请稍后再试" }
```

这个失败发生在报告事务写入前；已生成的临时 PDF 会清理，因此学生稍后重试是安全的。进入 PDF 生成槽位后，`page.pdf()` 本身还受 `PDF_GENERATION_TIMEOUT_MS` 控制。

## 运行时观测

`GET /api/v1/runtime-info` 会返回当前后端实际生效的配置，以及四个池的瞬时状态：

```json
{
  "data": {
    "app": "fangzhen-be",
    "database": "...",
    "port": 3001,
    "concurrency": {
      "systemParallelism": 64,
      "maxConcurrentTrainJobs": 50,
      "maxConcurrentPrepareJobs": 25,
      "maxConcurrentPredictJobs": 50,
      "maxConcurrentPdfJobs": 12,
      "dbConnectionLimit": 177,
      "dbQueryHeadroom": 40,
      "pythonModelProcessThreads": 1,
      "pythonTensorflowInterOpThreads": 1
    },
    "timeouts": {
      "pdfQueueMs": 60000,
      "pdfGenerationMs": 30000
    },
    "modelPools": {
      "train":   { "activeJobs": 0, "pendingJobs": 0, "maxConcurrentJobs": 50 },
      "prepare": { "activeJobs": 0, "pendingJobs": 0, "maxConcurrentJobs": 25 },
      "predict": { "activeJobs": 0, "pendingJobs": 0, "maxConcurrentJobs": 50 }
    },
    "pdfQueue": {
      "activeJobs": 0,
      "pendingJobs": 0,
      "maxConcurrentJobs": 12,
      "queueTimeoutMs": 60000,
      "generationTimeoutMs": 30000
    }
  }
}
```

前端报告提交会读取 `runtime-info.timeouts`，按 `pdfQueueMs + pdfGenerationMs + 30s` 计算请求超时，并保底 120s。建议把这个接口作为发布后人工核对项之一。

## 本地和服务器测试

配置单测：

```bash
bun test tests/unit/config
```

其中：

- `concurrency.test.ts` 覆盖四个池的自适应默认值、每个 per-behavior 变量独立生效、以及解耦后的 `DB_CONNECTION_LIMIT` 公式。
- `semaphore.test.ts` 覆盖 `Semaphore`（含非阻塞 `tryAcquire`）。
- `keyed-lock.test.ts` 覆盖执行锁去重语义。

四个池的行为语义如下（限流点均由上述单测与线程内实现覆盖）：

- 各行为池容量内请求成功；池耗尽时对应接口返回 `429`（PDF 为排队超时后 `503`）。
- **池隔离**：任一池被占满不会让其他行为返回 `429`。
- 同一实验同一模型被执行锁占用时返回 `409`。
- 基础模型已有可复用 production artifact 时不占用生产准备池；融合模型 `prepare-production` 仍受生产准备池限制。

## 调参建议

单实例部署默认不需要配置并发参数。发布后先查看 `GET /api/v1/runtime-info`，确认自动计算结果符合服务器 CPU 数。

- 如果某个行为出现较多 `429`，只需针对性调高该行为的池上限（如 `MAX_CONCURRENT_PREDICT_JOBS`），不会影响其他行为。调大任一模型池后，确认自动或显式的 `DB_CONNECTION_LIMIT` 和数据库 `max_connections` 足够。
- 如果 PDF 提交出现较多 `503`，优先提高 `PDF_QUEUE_TIMEOUT_MS`（如 `120000`）。只有在确认 CPU、Chromium 进程数和响应时间都健康时，再提高 `MAX_CONCURRENT_PDF_JOBS`。
- 如果服务器 CPU 被 Python 子进程打满，优先保持或降低 `PYTHON_MODEL_THREADS`，或下调某个模型池上限，不要只把线程数调大。当前设计是多进程并发、每进程少线程，比较适合课堂并发。
