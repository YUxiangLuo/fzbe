# 并发控制说明

本文档说明后端当前的并发控制逻辑、默认容量、部署参数和测试方式。相关实现主要位于：

- `src/utils/constants.ts`
- `src/config/database.ts`
- `src/modules/models/models.controller.ts`
- `src/modules/models/models.service.ts`
- `src/modules/reports/reports.controller.ts`
- `src/modules/reports/reports.service.ts`
- `src/utils/semaphore.ts`

## 目标场景

默认配置面向单台后端实例，按 `availableParallelism()` 看到的 CPU 可用并行度自动生成并发策略；64 核服务器会自动得到 50 个模型槽位和 12 个 PDF 槽位。100 名学生同时点击模型步骤时，50 个请求进入执行，另外 50 个请求收到带 `Retry-After` 的容量响应，不能先占用实验锁连接。

核心目标：

- 模型训练、生产准备、预测不能无限启动 Python 子进程。
- 数据库 advisory lock 不能耗尽连接池并造成死锁。
- PDF 生成要限流，超过队列等待时间后让前端可重试。
- 不配置并发环境变量时即可自动生成合理策略；显式环境变量仅用于确有需要的人工覆盖。

## 配置入口

所有主要并发参数都在 `src/utils/constants.ts` 统一计算。

| 配置项 | 默认值 | 说明 |
| --- | ---: | --- |
| `MAX_CONCURRENT_MODEL_JOBS` | `min(availableParallelism(), 50)` | 全局模型任务槽位数。覆盖 `MODEL_CONCURRENCY`。 |
| `MODEL_CAPACITY_RETRY_AFTER_SECONDS` | `3` | 模型容量耗尽时返回的 `Retry-After` 秒数。 |
| `MODEL_CONCURRENCY` | 无 | 旧环境变量名，仍兼容。仅在 `MAX_CONCURRENT_MODEL_JOBS` 未设置时生效。 |
| `MAX_CONCURRENT_PDF_JOBS` | `min(ceil(availableParallelism() / 4), 12)` | PDF/Puppeteer 并发生成数。覆盖 `PDF_CONCURRENCY`。 |
| `PDF_CONCURRENCY` | 无 | 旧环境变量名，仍兼容。仅在 `MAX_CONCURRENT_PDF_JOBS` 未设置时生效。 |
| `PDF_QUEUE_TIMEOUT_MS` | `60000` | PDF 队列等待超时。超时后接口返回 `503`。 |
| `PDF_GENERATION_TIMEOUT_MS` | `30000` | 单个 `page.pdf()` 的生成超时。 |
| `DB_QUERY_HEADROOM` | `max(40, ceil(modelJobs / 2))` | 除模型锁连接外，给普通查询、事务和其他接口保留的连接余量。 |
| `DB_CONNECTION_LIMIT` | `2 * modelJobs + queryHeadroom` | MySQL pool 连接数。显式配置后不再自动计算。 |
| `PYTHON_MODEL_THREADS` | `1` | 每个 Python 模型子进程的 BLAS/NumExpr/TensorFlow intra-op 线程默认值。 |
| `TF_NUM_INTEROP_THREADS` | `1` | TensorFlow inter-op 线程默认值。 |

所有这些环境变量都必须是正整数；非法值会在模块加载时直接报错。

## 64 核 / 100 学生默认容量（50 个模型槽位）

如果服务进程能通过 `availableParallelism()` 看到 64 核，默认计算为：

```text
MAX_CONCURRENT_MODEL_JOBS = min(64, 50) = 50
MAX_CONCURRENT_PDF_JOBS = min(ceil(64 / 4), 12) = 12
PDF_QUEUE_TIMEOUT_MS = 60000
PDF_GENERATION_TIMEOUT_MS = 30000
PYTHON_MODEL_THREADS = 1
TF_NUM_INTEROP_THREADS = 1
MODEL_CAPACITY_RETRY_AFTER_SECONDS = 3
DB_QUERY_HEADROOM = max(40, ceil(50 / 2)) = 40
DB_CONNECTION_LIMIT = 2 * 50 + 40 = 140
```

单实例生产部署建议优先保持这些并发环境变量未设置，让后端自动计算。只有当容器、systemd 或 cgroup 让 `availableParallelism()` 看到的 CPU 数不符合预期，或需要临时限流/扩容时，才显式覆盖。

数据库侧的 `max_connections` 必须大于等于自动计算出的 `DB_CONNECTION_LIMIT`。64 核课堂场景自动连接池为 `140`，建议数据库设置为 `200+`，给迁移、管理连接和临时排查留余量。

## 模型任务并发

模型计算接口使用两层数据库锁和一层本实例连接保护：

1. 全局模型槽位锁：`fangzhen-model:slot:{slotNumber}`
2. 单实验执行锁：`fangzhen-model:exp:{experimentId}`
3. 本实例实验锁 admission 上限：`MAX_CONCURRENT_MODEL_JOBS`

训练、guided 步骤、生产准备和预测必须先取得全局槽位，再取得实验锁。容量外请求因此不会长期占用实验锁连接。单实验锁覆盖该实验的全部七种模型以及激活、完成和清理操作，避免基础模型变化与融合模型操作交叉执行。拿不到该锁时，接口返回：

```http
409 Conflict
{ "error": "同一模型正在训练或预测，请稍后再试" }
```

全局模型槽位锁用于限制整个后端同时运行的模型任务数量。`acquireModelSlot()` 会从 `1` 到 `MAX_CONCURRENT_MODEL_JOBS` 尝试 MariaDB/MySQL `GET_LOCK(name, 0)`。所有槽位都被占用时，模型接口返回：

```http
429 Too Many Requests
Retry-After: 3

{ "error": "模型服务繁忙，请稍后再试" }
```

前端读取 `Retry-After` 后最多自动重试三次，使用指数退避和 0–25% 随机抖动。`409/429` 不消耗模型业务失败次数；自动尝试用尽后仍保留人工重试。

课堂网络通常让全部学生共享一个 NAT 出口。默认登录限流因此为单 IP 每 15 分钟 300 次，自助注册为单 IP 每小时 150 次；单账号限制仍分别保持 10 次和 3 次，避免为了容纳 100 人而取消账号级防护。

训练、生产准备、预测都会经过同一个全局模型槽位池。这里没有独立的“生产槽位”；生产准备只是因为可能启动 Python 子进程生成 production artifact，所以也需要占用全局模型槽位。模型任务结束后，无论成功、失败还是超时，controller 的 `finally` 都会释放槽位锁和执行锁。

### cached production artifact 的例外

基础模型 `ma`、`es`、`arima`、`lstm` 的 `prepare-production` 在拿到单实验单模型执行锁后，会先检查是否已经存在可复用的 production artifact。

基础模型会先在实验锁内做一次快速缓存检查。如果 artifact 可复用，接口直接返回：

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

这种路径不会占用全局模型槽位，也不会启动 Python 子进程。缓存未命中时会先释放快速检查锁，再按“槽位 → 实验锁”的统一顺序重新准入，并在 service 层重新检查缓存。

ensemble 模型不走这个槽位前置绕过逻辑。即使已有 cached artifact，`weighted_avg`、`boosting`、`stacking` 的 `prepare-production` 仍然先进入全局模型槽位，再由 service 层判断是否可复用。这样可以避免 ensemble 的 artifact 检查或派生逻辑绕过模型并发上限。

## Python 子进程线程控制

每个模型任务最终可能启动 Python 子进程。为了避免 50 个 Python 进程各自再启动多个 BLAS 或 TensorFlow 原生线程，后端在启动 Python 子进程时设置以下环境变量：

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

默认 `PYTHON_MODEL_THREADS=1`，`TF_NUM_INTEROP_THREADS=1`。这两个项目配置是权威值，会覆盖 systemd、容器镜像或宿主机继承的 `OPENBLAS_NUM_THREADS` 等变量，避免隐藏的线程数把 50 个 Python 进程再次成倍放大。需要调线程时只修改这两个受校验的配置入口。

## 数据库连接池与启动前校验

模型任务使用 MariaDB/MySQL advisory lock。每个并发模型任务最多会长期持有两类连接：

- 执行锁连接
- 槽位锁连接

模型任务内部和其他接口还需要普通查询连接。如果连接池只够锁使用，可能出现锁占着连接、业务查询又等待连接释放的循环等待。所有计算路径都先申请槽位，并且本实例同时持有的实验锁数被限制为模型槽位数，所以容量外请求不能侵占 `queryHeadroom`。

因此连接池默认按下面公式计算：

```text
lockConnections = 2 * MAX_CONCURRENT_MODEL_JOBS
queryHeadroom = max(40, ceil(MAX_CONCURRENT_MODEL_JOBS / 2))
DB_CONNECTION_LIMIT = lockConnections + queryHeadroom
```

`src/config/database.ts` 使用 `DB_POOL_CONFIG.CONNECTION_LIMIT` 创建 mysql2 pool，`queueLimit` 为 `100`。

启动时 `runServerStartupPreflight()` 会做两项校验：

- `connectionLimit >= 2 * MAX_CONCURRENT_MODEL_JOBS + DB_QUERY_HEADROOM`
- 数据库 `max_connections >= connectionLimit`

任一条件不满足，后端会记录 fatal 日志并退出。这样可以在启动阶段暴露部署配置错误，而不是上课时才出现死锁或大面积请求挂起。

## PDF 生成并发

PDF 生成使用进程内 `Semaphore`，并发上限为 `MAX_CONCURRENT_PDF_JOBS`。

报告提交进入 `submitReport()` 后，会先执行：

```ts
await pdfSemaphore.acquire(TIMEOUTS.PDF_QUEUE);
```

如果还有可用名额，立即生成 PDF。如果名额已满，请求进入队列。等待超过 `PDF_QUEUE_TIMEOUT_MS` 后，`Semaphore` 抛出 `SemaphoreTimeoutError`，controller 返回：

```http
503 Service Unavailable
{ "error": "PDF 生成服务繁忙，排队等待超过 N 秒，请稍后再试" }
```

这个失败发生在报告事务写入前；已生成的临时 PDF 会清理。因此学生稍后重试是安全的。

进入 PDF 生成槽位后，`page.pdf()` 本身还受 `PDF_GENERATION_TIMEOUT_MS` 控制。如果 PDF 生成失败，service 会清理本次生成的文件，并释放信号量。

默认 `MAX_CONCURRENT_PDF_JOBS=min(ceil(availableParallelism() / 4), 12)`，是为了让较小服务器自动降低 Puppeteer/Chromium 压力，而 64 核课堂服务器自动达到 12 个 PDF 槽位。100 人同时提交会形成 9 批；最后一批约在 `8 × 单批耗时` 后开始。默认 60 秒队列要求单批尾延迟低于约 7.5 秒，否则部分请求会返回 `503`。应在目标服务器实测 PDF 尾延迟后再提高 `PDF_QUEUE_TIMEOUT_MS`，不直接增加 Chromium 并发。

## 运行时观测

`GET /api/v1/runtime-info` 会返回当前后端实际生效的配置和 PDF 队列瞬时状态，便于部署后确认和课堂监控：

```json
{
  "data": {
    "app": "fangzhen-be",
    "database": "...",
    "port": 3001,
    "concurrency": {
      "systemParallelism": 64,
      "maxConcurrentModelJobs": 50,
      "maxConcurrentPdfJobs": 12,
      "modelCapacityRetryAfterSeconds": 3,
      "dbConnectionLimit": 140,
      "dbLockConnections": 100,
      "dbQueryHeadroom": 40,
      "pythonModelProcessThreads": 1,
      "pythonTensorflowInterOpThreads": 1
    },
    "timeouts": {
      "pdfQueueMs": 60000,
      "pdfGenerationMs": 30000
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

前端报告提交会读取 `runtime-info.timeouts`，按 `pdfQueueMs + pdfGenerationMs + 30s` 计算请求超时，并保底 120s。这样当后端提高 `PDF_QUEUE_TIMEOUT_MS` 时，前端不会提前中断仍在排队或生成中的报告请求。

如果使用反向代理或容器部署，建议把这个接口作为发布后人工核对项之一。

## 本地和服务器测试

配置单测：

```bash
bun test tests/unit/config/concurrency.test.ts
```

完整后端单测：

```bash
bun test tests/unit
```

前端仓库提供 API 级并发 e2e，默认会用小并发参数启动受控后端，适配本机和 64 核服务器：

```bash
cd ../fangzhen-bun
bun run e2e:shiyan:concurrency
```

可单独测试模型或 PDF：

```bash
bun run e2e:shiyan:concurrency:model
bun run e2e:shiyan:concurrency:pdf
```

真实浏览器 UI 并发回归：

```bash
bun run e2e:shiyan:concurrency:ui
```

不启动 Python、直接验证 100 个不同实验同时争抢 50 个槽位及数据库查询余量：

```bash
MAX_CONCURRENT_MODEL_JOBS=50 \
DB_CONNECTION_LIMIT=140 \
MODEL_LOCK_BURST_REQUESTS=100 \
bun run verify:model-lock-burst
```

预期结果必须是 `50 acquired / 50 capacity / 0 conflicts`，并且持锁期间的 `SELECT 1` 能立即完成。该检查已接入后端 CI。

64 核服务器 50 学生压力档：

```bash
bun run e2e:shiyan:concurrency:server
```

API/UI 压力档使用并发测试学生 `20246001` 到 `20246050` 验证容量内真实训练和 PDF；上述 `verify:model-lock-burst` 另行覆盖 100 个不同实验的容量外突发。压力档显式设置：

```env
E2E_CONCURRENCY_MODEL_JOBS=50
E2E_CONCURRENCY_MODEL_CAPACITY_REQUESTS=50
E2E_CONCURRENCY_PDF_JOBS=12
E2E_CONCURRENCY_PDF_STUDENTS=50
E2E_CONCURRENCY_PDF_QUEUE_TIMEOUT_MS=60000
E2E_CONCURRENCY_EXPECT_PDF_503=0
DB_CONNECTION_LIMIT=140
```

其中 `E2E_CONCURRENCY_EXPECT_PDF_503=0` 表示服务器压力档允许 50 个 PDF 请求全部在队列内成功，不强制要求出现 `503`。运行前确认数据库 `max_connections >= 140`，建议 `200+`。

该 e2e 覆盖：

- 模型容量内请求成功。
- 模型槽位耗尽时训练接口返回 `429`。
- 全局模型槽位耗尽时生产准备接口返回 `429`。
- 模型槽位耗尽时生产预测接口返回 `429`。
- 同一实验同一模型被执行锁占用时返回 `409`。
- 基础模型已有可复用 production artifact 时不占用全局模型槽位。
- 融合模型 production prepare 仍受全局模型槽位限制。
- PDF 队列超时时报告提交返回 `503`。
- PDF `503` 后顺序重试可以成功生成报告。

## 调参建议

单实例部署默认不需要配置并发参数。发布后先查看 `GET /api/v1/runtime-info`，确认自动计算结果符合服务器 CPU 数：

```text
MAX_CONCURRENT_MODEL_JOBS = min(availableParallelism(), 50)
MAX_CONCURRENT_PDF_JOBS = min(ceil(availableParallelism() / 4), 12)
DB_CONNECTION_LIMIT = 2 * MAX_CONCURRENT_MODEL_JOBS + max(40, ceil(MAX_CONCURRENT_MODEL_JOBS / 2))
```

如果模型请求出现较多 `429`，先确认是否真的有超过自动槽位数的模型任务同时运行，再考虑显式增加 `MAX_CONCURRENT_MODEL_JOBS`。增加模型并发时，必须同步确认自动或显式的 `DB_CONNECTION_LIMIT` 和数据库 `max_connections` 足够。

如果 PDF 提交出现较多 `503`，优先提高 `PDF_QUEUE_TIMEOUT_MS`，例如 `120000`。只有在确认 CPU、Chromium 进程数和响应时间都健康时，再显式提高 `MAX_CONCURRENT_PDF_JOBS`。

如果服务器 CPU 被 Python 子进程打满，优先保持或降低 `PYTHON_MODEL_THREADS`，不要只降低 `MAX_CONCURRENT_MODEL_JOBS`。当前设计是多进程并发，每个进程少线程，比较适合课堂并发。
