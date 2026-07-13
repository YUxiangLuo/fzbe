# 并发控制说明

本文说明后端当前的模型与 PDF 并发控制、默认容量、超时、断线语义和验证方法。训练 / prepare / predict 的完整设计记录见仓库根目录的 `TRAIN_PREPARE_PREDICT_CONCURRENCY_REFACTOR_PLAN.md`。

## 1. 目标场景

默认配置面向单台 64 核服务器、单个 Bun 后端实例和 100 名学生课堂同步发起请求：

```text
MAX_CONCURRENT_MODEL_JOBS = min(64, 50) = 50
MAX_CONCURRENT_PDF_JOBS = min(ceil(64 / 4), 12) = 12
DB_QUERY_HEADROOM = max(40, ceil(50 / 2)) = 40
DB_CONNECTION_LIMIT = 2 * 50 + 40 = 140
PYTHON_MODEL_THREADS = 1
TF_NUM_INTEROP_THREADS = 1
```

100 个模型请求突发时，最多 50 个进入模型资源区，另外 50 个立即返回 429。模型准入不建立进程内等待队列。

## 2. 配置入口

主要配置由 `src/utils/constants.ts` 统一计算。

| 配置项 | 默认值 | 说明 |
| --- | ---: | --- |
| `MAX_CONCURRENT_MODEL_JOBS` | `min(availableParallelism(), 50)` | 训练、prepare、predict、ADF 共用的模型容量 |
| `MODEL_CONCURRENCY` | 无 | 旧名称；只在新名称未设置时生效 |
| `MODEL_OPERATION_DEADLINE_MS` | `660000` | 每次调用独立的总 deadline；660 秒也是硬上限，只允许缩短 |
| `MODEL_LOCK_CONNECTION_ACQUIRE_TIMEOUT_MS` | `2000` | 获取 advisory-lock 专用连接的上限 |
| `MODEL_LOCK_QUERY_TIMEOUT_MS` | `2000` | 获取、校验、释放锁 SQL 的上限 |
| `MODEL_PROCESS_TERMINATION_GRACE_MS` | `5000` | Python `SIGTERM` 后的退出宽限期 |
| `MODEL_FAILURE_RECORD`（内部常量） | `2000` | lease 释放后 best-effort 失败回写总上限；计入 6 秒响应余量 |
| `MODEL_TRAINING_TIMEOUT_MS` | `300000` | 普通 Python 训练预算 |
| `MODEL_PREDICTION_TIMEOUT_MS` | `60000` | Python 预测预算 |
| `MAX_CONCURRENT_PDF_JOBS` | `min(ceil(availableParallelism()/4), 12)` | PDF/Puppeteer 并发数 |
| `PDF_QUEUE_TIMEOUT_MS` | `60000` | PDF 排队上限 |
| `PDF_GENERATION_TIMEOUT_MS` | `30000` | 单次 PDF 生成上限 |
| `DB_QUERY_HEADROOM` | `max(40, ceil(modelJobs/2))` | 普通查询、事务和其他模块的连接余量 |
| `DB_CONNECTION_LIMIT` | `2 * modelJobs + queryHeadroom` | mysql2 pool 连接数 |
| `PYTHON_MODEL_THREADS` | `1` | 单个 Python 进程的原生计算线程数 |
| `TF_NUM_INTEROP_THREADS` | `1` | TensorFlow inter-op 线程数 |

所有显式配置必须为正整数。Python 模型进程预算还有不可配置的 600,000 ms 硬上限。

## 3. 模型资源获取协议

`src/modules/models/model-operation-lease.ts` 是模型资源的唯一编排层。完整模型操作固定按以下顺序获取资源：

```text
请求 schema 校验
  -> service operation context（开始单次 deadline）
  -> 权限和必要数据预检
  -> process-local admission.tryAcquire()（不排队）
  -> MariaDB 全局 slot GET_LOCK(..., 0)（不排队）
  -> 单实验单模型 execution GET_LOCK(..., 0)
  -> DB / Python / 工件 / commit
```

释放顺序固定相反：

```text
execution -> slot -> admission
```

释放由嵌套 `finally` 保证。即使某次锁释放查询失败，无法确认安全的连接会被销毁，后续资源仍继续释放。锁连接获取和 slot/execution 获取查询都监听 operation deadline signal：连接池等待会立即返回，迟到且未执行 SQL 的健康连接归还 pool；在途获取查询则销毁专用连接。heartbeat 与主动 ownership verification 共用同一个在途查询；释放开始时若该查询仍未结束，会直接销毁专用连接，不再把 `RELEASE_LOCK` 排到 mysql2 命令队列后面。查询超时时同样立即销毁 handle 连接。

### 3.1 各操作使用的 lease

| 操作 | 模式 |
| --- | --- |
| guided training 正常 step | slot + execution |
| prepare-production | slot + execution |
| predict | slot + execution |
| ADF | slot only |
| completed guided session 恢复/物化 | execution only |
| guided session discard | execution only |

可复用 production artifact 也必须先通过 admission、slot 和 execution lock，不存在 cached prepare 的容量绕过。

### 3.2 过载和互斥响应

模型容量满返回：

```http
429 Too Many Requests
Retry-After: 2

{"error":"模型服务繁忙，请稍后再试","code":"MODEL_CAPACITY_BUSY","retryable":true}
```

同一实验、同一模型已经运行返回 409，错误码 `MODEL_BUSY`。锁连接/查询不可用或锁所有权丢失返回结构化 503，包括 `MODEL_LOCK_CONNECTION_TIMEOUT`、`MODEL_LOCK_QUERY_TIMEOUT`、`MODEL_LOCK_ACQUISITION_FAILED`、`MODEL_LOCK_LOST` 和 `MODEL_LOCK_VERIFY_FAILED`。

## 4. 单次操作 deadline

`MODEL_OPERATION_DEADLINE_MS=660000` 限制一次后端调用，不限制整个实验：

- 每个 guided step 重新计时；
- 每次 prepare 重新计时；
- 每次 predict 重新计时；
- ADF 也建立独立 context，但 ADF Python 自身仍为 30 秒。

660 秒也是当前静态 nginx/浏览器部署契约下的硬上限。启动时使用以下最坏预算校验配置组合：

```text
operation deadline
+ process termination grace
+ 2 × lock query timeout
+ response reserve
<= nginx read timeout
< browser timeout

默认：660s + 5s + 2×2s + 6s = 675s < 690s
```

因此 deadline 只能缩短；如果提高终止宽限期或锁查询 timeout 导致默认 deadline 不再满足公式，后端会拒绝启动。

deadline 覆盖 service 内权限/数据预检、准入、锁、Python、工件处理和提交。锁相关数据库操作另有 2 秒上限。普通业务查询没有通用的 mysql2 取消机制，因此在可控异步边界检查 deadline；Python 进程可被主动终止。

HTTP 超时层级为：

```text
Python <= 600s
backend operation = 660s
nginx model route = 675s
browser model request = 690s
```

Bun 的普通 `idleTimeout` 最大为 255 秒。长模型路由由 `src/index.ts` 精确匹配并执行 `server.timeout(request, 0)`，因此不会被 Bun idle timeout 提前截断。

## 5. 页面关闭与客户端断线

已接受的后端模型任务不使用 HTTP `Request.signal`。模型 lease 只响应：

1. 服务端单次 operation deadline；
2. advisory lock ownership loss。

页面关闭、跳转、浏览器离线或客户端主动 abort 不会终止 Python，也不会提前释放 admission、slot 或 execution lock。任务继续执行，直到成功、内部错误或服务端 deadline。

nginx 的 `/api/v1/models/` 配置必须保持：

```nginx
proxy_read_timeout 675s;
proxy_ignore_client_abort on;
```

ADF 的精确路由 `/api/v1/tools/adf` 也配置 `proxy_ignore_client_abort on`，并使用 45 秒 upstream 预算。后端从该预算反推 ADF operation deadline（默认 32 秒；若全局 operation deadline 被主动缩短则取两者较小值），并在启动时校验 30 秒 Python 上限、进程终止、一次 slot 解锁及响应余量能够全部容纳；不满足时拒绝启动。Bun 对该精确路由同样执行 `server.timeout(request, 0)`。

前端可以中止自己的 fetch 并避免卸载后更新 UI；这不再构成后端取消信号。

## 6. Python 进程与锁所有权

Linux 下主训练、guided、模型产物校验/复制和 ADF Python 都通过 `setsid` 建立进程组。deadline、锁丢失或 Python 自身超时时：

1. 对进程组发送 `SIGTERM`；
2. 等待默认 5 秒；
3. 未退出则发送 `SIGKILL`；
4. 即使 setsid leader 已退出，仍通过 PGID 检查整个进程组；必要时向残留成员发送 `SIGKILL`；
5. 确认整个进程组消失后，才进入资源清理和释放。

advisory lock 每 10 秒执行 ownership heartbeat。guided commit、prepare artifact 晋升和 predict 返回前还会主动执行 `IS_USED_LOCK`。mysql2 的查询 timeout 只拒绝 callback、不会取消底层命令，所以 heartbeat/主动校验一旦超时就使 handle 失效并销毁连接。

guided failure 回写带有 attempt/status fencing：只更新仍处于 `running/python_completed` 的同一 attempt，以及仍由该 attempt 占有且状态为 `running` 的 session。晚到的旧失败不能把 `committed/completed` 改回 `failed`。本版本仍没有引入通用 fencing token；多实例或外部 worker 场景需要另行设计数据库 schema 和迁移。

ADF 使用独立的 30 秒 Python 上限，同时接收 lease termination signal。ADF 自有超时返回结构化 504；operation deadline 或 slot ownership loss 会终止并等待整个 ADF 进程组退出，然后释放 slot/admission。

## 7. 数据库连接池

每个完整模型任务在容量区内最多长期持有两个锁连接：一个全局 slot、一个 execution lock。连接池默认公式为：

```text
lockConnections = 2 * MAX_CONCURRENT_MODEL_JOBS
queryHeadroom = max(40, ceil(MAX_CONCURRENT_MODEL_JOBS / 2))
DB_CONNECTION_LIMIT = lockConnections + queryHeadroom
```

启动预检要求：

- `connectionLimit >= lockConnections + queryHeadroom`；
- MariaDB/MySQL `max_connections >= connectionLimit`。

64 核默认 pool 为 140。生产建议数据库 `max_connections` 设置为 200 以上，为迁移、管理连接和排查保留余量。

## 8. Python 线程控制

模型子进程默认设置以下环境变量为单线程或显式配置值：

```text
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
MKL_NUM_THREADS
VECLIB_MAXIMUM_THREADS
NUMEXPR_NUM_THREADS
TF_NUM_INTRAOP_THREADS
TF_NUM_INTEROP_THREADS
```

课堂并发采用“多进程、每进程少线程”。增加 `MAX_CONCURRENT_MODEL_JOBS` 前，必须同时观察 CPU、内存、Python RSS、DB 连接和尾延迟。

## 9. PDF 并发

PDF 使用独立的进程内 `Semaphore`，允许有界排队：

- 最大活跃任务数：`MAX_CONCURRENT_PDF_JOBS`；
- 队列等待超过 `PDF_QUEUE_TIMEOUT_MS` 返回 503；
- `page.pdf()` 超过 `PDF_GENERATION_TIMEOUT_MS` 失败；
- 事务前失败及临时文件都会清理，随后释放 semaphore。

模型 admission 是 fail-fast、无等待；PDF semaphore 的排队语义与模型不同，不应混淆。

## 10. 运行时观测

`GET /api/v1/runtime-info` 返回：

- 实际 CPU 并行度、模型/PDF 容量、DB pool 与 Python 线程配置；
- operation、Python、锁和 PDF 超时；
- operation 动态最大值、deadline 后清理预算及 nginx/browser 预算；
- `modelOperations.admission.capacity/active/pending`；
- operation 的开始、完成、失败、deadline、429、409 计数；
- 当前各类 operation 的活跃 lease；
- advisory lock 的活跃、获取失败、所有权丢失和释放失败计数；
- PDF 活跃任务及队列长度。

模型 admission 的 `pending` 应始终为 0。课堂突发时 `capacityRejected` 增长是预期过载保护；锁连接超时、ownership loss 或 release failure 持续增长则需要告警和排查。

## 11. 测试

后端完整检查：

```bash
bun run typecheck
bun test --isolate
```

关键回归：

```bash
bun test tests/unit/config/ # 这些配置/并发测试也必须在非 isolate 模式自洽
bun test tests/unit/config/bounded-pool-transaction.test.ts --isolate
bun test tests/unit/config/model-operation-lease.test.ts --isolate
bun test tests/unit/config/model-operation-classroom.test.ts --isolate
bun test tests/unit/config/model-locks.test.ts --isolate
bun test tests/unit/config/adf-cancellation.test.ts --isolate
bun test tests/unit/config/process-tree.test.ts --isolate
```

前端 API 级并发套件位于 `fzfe/scripts/shiyan-concurrency-suite.ts`。其 reusable prepare 用例现在要求缓存命中仍受全局 admission/slot 限制。

64 核服务器上的自动化入口：

```bash
cd ../fzfe
bun run e2e:shiyan:concurrency:server
```

并发基线现包含 100 个学生账号。套件先让 100 个已认证学生同时进入测试专用的完整 `slot + execution` lease，其中 50 个持有、50 个快速 429；全部响应后断言 `admission.active=0`、`admission.pending=0`、`activeLeases=0` 和 `modelLocks.active=0`。server profile 将登录 IP 限额提高到 1000，套件内部还会跨场景复用同一学生 token，避免容量阶段之后产生第 101 次登录。测试专用 hold 路由只在 `E2E_SERVER=1` 时注册；复用已有后端前会从 `runtime-info` 明确验证该标志，普通后端会被立即拒绝而不是运行到 404。真实发布验收还应覆盖浏览器断线后后端继续，以及错误、deadline 和锁丢失后所有资源计数归零。
