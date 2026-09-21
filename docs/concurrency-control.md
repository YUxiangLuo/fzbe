# 单实例并发控制说明

本文说明后端在**单台服务器、单个 Bun 后端进程**部署约束下的模型、PDF、文件删除和数据库并发控制。

## 1. 不可变部署约束

生产环境只允许运行一个 `fzbe` 进程：

- nginx 只能配置一个后端 upstream；
- `.env` 必须设置 `NODE_ENV=production`（或 `BUN_ENV=production`），生产环境固定绑定 `127.0.0.1:3001`，偏离该地址会启动失败；`update.sh` 会在停服前再次校验模式和地址；
- systemd 只能维护一个后端 service 实例，并使用 `KillMode=control-group`，保证重启时 Bun 与 Python/Chromium 子进程一起退出；`update.sh` 会校验该属性并 fail-closed；
- 发布必须原地重启，不得使用会让新旧进程重叠的 rolling/surge 策略；
- 不得另外启动会直接执行模型任务或受管文件删除的 worker；
- 管理员 `GET /api/v1/runtime-info` 固定返回
  `deploymentMode=single_instance` 和 `coordinationScope=process`。

这里的“单实例”不代表“没有并发”。一个 Bun 进程仍会同时处理多个学生、多个浏览器标签页、Python 子进程、PDF 任务和数据库事务，因此进程内限流、互斥、deadline、事务锁和乐观版本控制都必须保留。

服务进程会先占用监听端口，再执行数据库恢复和文件清理。第二个按生产约束启动的进程因此会在接触这些共享状态之前因端口占用而失败；启动完成前监听器只返回 503。迁移和并发测试 seed 是独立运维进程，不属于在线服务实例。它们仍可使用短期数据库 advisory lock，避免两个运维命令同时修改 schema/测试基线。

## 2. 默认目标场景

默认配置面向单台 64 核服务器、单个 Bun 后端进程和 100 名学生课堂突发：

```text
MAX_CONCURRENT_MODEL_JOBS = min(64, 50) = 50
MAX_CONCURRENT_PDF_JOBS = min(ceil(64 / 4), 12) = 12
DB_MIN_CONNECTION_LIMIT = max(40, ceil(50 / 2)) = 40
DB_CONNECTION_LIMIT = 40
PYTHON_MODEL_THREADS = 1
TF_NUM_INTEROP_THREADS = 1
```

100 个模型请求同时到达时，最多 50 个进入模型资源区，其余请求立即返回 429。模型准入不建立等待队列。

## 3. 配置入口

跨后端、nginx 和浏览器的静态模型/ADF 预算以 `contracts/model-http-timeouts.json` 为唯一数据源，生成和发布规则见 [model-http-timeouts.md](./model-http-timeouts.md)。`src/utils/constants.ts` 校验该契约，并在其上应用只能缩短有效 deadline 的环境覆盖。

| 配置项 | 默认值 | 说明 |
| --- | ---: | --- |
| `MAX_CONCURRENT_MODEL_JOBS` | `min(availableParallelism(), 50)` | 训练、prepare、predict、ADF 共用的进程内容量 |
| `MODEL_CONCURRENCY` | 无 | 旧名称；只在新名称未设置时生效 |
| `MODEL_OPERATION_DEADLINE_MS` | `660000` | 每次调用独立的总 deadline；660 秒为硬上限 |
| `MODEL_PROCESS_TERMINATION_GRACE_MS` | `5000` | Python `SIGTERM` 后的退出宽限期 |
| `MODEL_DB_CONNECTION_ACQUIRE_TIMEOUT_MS` | `10000` | 模型事务获取连接的上限 |
| `MODEL_DB_QUERY_TIMEOUT_MS` | `30000` | 模型事务单次查询上限 |
| `MODEL_TRAINING_TIMEOUT_MS` | `300000` | 普通 Python 训练预算 |
| `MODEL_PREDICTION_TIMEOUT_MS` | `60000` | Python 预测预算 |
| `MAX_CONCURRENT_PDF_JOBS` | `min(ceil(availableParallelism()/4), 12)` | PDF/Puppeteer 并发数 |
| `PDF_QUEUE_TIMEOUT_MS` | `60000` | PDF 排队上限 |
| `PDF_GENERATION_TIMEOUT_MS` | `30000` | 单次 PDF 生成上限 |
| `DB_CONNECTION_LIMIT` | `max(40, ceil(modelJobs/2))` | 短查询和事务共享的 mysql2 pool |
| `PYTHON_MODEL_THREADS` | `1` | 模型、产物探测、ADF 的默认原生计算线程数；显式原生环境变量优先 |
| `TF_NUM_INTEROP_THREADS` | `1` | TensorFlow inter-op 线程数 |

所有显式整数配置必须为正整数。Python 模型进程还有不可配置的 600,000 ms 硬上限。旧 advisory-lock 时代的 `MODEL_LOCK_CONNECTION_ACQUIRE_TIMEOUT_MS`、`MODEL_LOCK_QUERY_TIMEOUT_MS` 和 `DB_QUERY_HEADROOM` 已移除；生产启动会拒绝仍包含这些键的 `.env`，避免静默忽略旧容量配置。

## 4. 模型资源协议

`src/modules/models/model-operation-lease.ts` 是唯一编排层：

```text
请求 schema 校验
  -> service operation context（启动单次 deadline）
  -> 权限和必要数据预检
  -> process-local admission.tryAcquire()（不排队）
  -> 可选的 experiment/model 进程内 execution mutex
  -> DB / Python / 工件 / commit
```

释放顺序固定为：

```text
execution mutex -> admission
```

JavaScript 在同一个事件循环中同步完成 execution mutex 的 Map check-and-set，因此同一实验、同一模型的互斥不存在异步抢占窗口。token 校验保证迟到调用不能释放另一个调用的新锁。过期 session 和模型产物清理复用同一组 execution mutex：单 session 清理获取对应模型锁，整实验产物清理同步获取该实验全部模型锁；锁繁忙时跳过并留待下一轮，绝不删除正在被训练、准备或预测使用的文件。

### 4.1 各操作使用的资源

| 操作 | admission | execution mutex |
| --- | :---: | :---: |
| guided training 创建/恢复 | 是 | 是 |
| guided training step | 是 | 是 |
| completed session 物化、discard | 是 | 是 |
| production preparation step | 是 | 是 |
| predict | 是 | 是 |
| ADF | 是 | 否 |
| session 元数据读取 | 否 | 否 |

所有真正进入 lease 的调用都占用 admission。初始化步骤即使可能复用产物，也必须先进入容量保护；已经完成且版本仍匹配的 session 查询只返回持久化元数据，不执行模型工作，因此可以不进入 lease。

### 4.2 过载和互斥响应

容量满时返回：

```http
429 Too Many Requests
Retry-After: 2

{"error":"模型服务繁忙，请稍后再试","code":"MODEL_CAPACITY_BUSY","retryable":true}
```

同一实验、同一模型正在执行时返回 409，错误码 `MODEL_BUSY`。进程内 execution mutex 不依赖数据库连接，不再存在数据库模型锁连接、查询或心跳错误；如果内部 token 一致性校验异常，仍会 fail-closed 返回 `MODEL_EXECUTION_LOCK_LOST` 503。

## 5. 单次 operation deadline

`MODEL_OPERATION_DEADLINE_MS=660000` 限制一次后端调用，而不是整个实验：

- 每个 guided step 独立计时；
- 每个 prepare step 独立计时；
- 每次 predict 独立计时；
- ADF 使用较短的独立预算。

当前契约生成的静态传输层预算为：

```text
Python <= 600s
backend operation = 660s
nginx model route = 675s
browser request = 690s
```

启动配置必须满足：

```text
operation deadline + process termination grace + response reserve
<= nginx read timeout < browser timeout
```

默认清理预算为 `5s + 6s = 11s`。ADF 为 `30s` Python 上限、`34s` operation 上限和 `45s` nginx 上限。

普通 mysql2 查询无法原生通过 AbortSignal 取消，因此模型 operation signal 通过 AsyncLocalStorage 传播到数据库 helper；helper 显式借用连接，并对连接获取、查询、`BEGIN` 和 `ROLLBACK` 设置上限。operation 截止或查询/事务控制超时时会销毁对应连接，`COMMIT` 中止或超时则按结果未知 fail-closed。数据集查找也使用同一有界查询 helper，文件读取和 CSV 解析会响应 operation signal。

销毁客户端连接只保证调用停止等待、连接不再被复用，不能保证 MariaDB/MySQL 上的查询和锁立即消失。服务端可能继续执行当前语句，直到发现连接关闭才清理事务；必须结合真实数据库会话和锁等待观测判断资源是否回收。普通 CRUD 尚未统一使用模型 deadline；浏览器超时后的写入仍可能提交，客户端应重新读取状态和版本确认结果。

## 6. 客户端断线与 Python 进程

已接受的模型任务不使用 HTTP `Request.signal`。页面关闭、路由跳转或浏览器离线不会提前终止后端任务，也不会提前释放 admission/execution mutex。任务只在以下条件结束：

1. 成功；
2. 内部失败；
3. 服务端 operation deadline。

nginx 模型路由必须继续使用 `proxy_ignore_client_abort on`。

Linux 下模型和 ADF 通过 `setsid` 建立进程组。超时后依次执行：

1. 向进程组发送 `SIGTERM`；
2. 等待宽限期；
3. 必要时发送 `SIGKILL`；
4. 确认整个进程组退出后释放 execution mutex 和 admission。

普通、guided 和 auxiliary runner 的异常分支均等待进程树终止；主进程提前退出或关闭输出流不能抢先结束 lease。回归使用真实的拒绝 `SIGTERM` 的后代进程，同时检查进程组存活和同一执行键重新准入。

模型、辅助脚本及 ADF 共用 `src/utils/python-process-env.ts` 的环境构造函数。`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 等显式配置优先于 `PYTHON_MODEL_THREADS`；管理员运行时信息的 `concurrency.pythonNativeProcessEnv` 显示实际传入子进程的值，不能把产品默认值视为所有部署的有效值。

模型 run、guided attempt 和 production preparation 仍使用数据库状态、attempt ID、revision、checksum 和提交后复核防止迟到结果覆盖新状态。这些机制处理的是异步结果一致性，不能因为单实例而删除。

## 7. 数据库连接池和事务并发

模型任务不再为容量或互斥长期占用数据库连接。连接池只服务短查询和事务：

```text
minimum = max(40, ceil(MAX_CONCURRENT_MODEL_JOBS / 2))
DB_CONNECTION_LIMIT = explicit override or minimum
```

64 核默认从旧设计的 140 降为 40。启动预检要求：

- `DB_CONNECTION_LIMIT >= minimum`；
- MariaDB/MySQL `max_connections >= DB_CONNECTION_LIMIT`。

数据库 `FOR UPDATE`、统一的 users → experiment 锁顺序、deadlock retry、提交 ACK 丢失复核和 `state_version` 乐观锁与实例数量无关，全部保留。

首次创建、驳回重做和模型 run 激活事务使用单次 `READ COMMITTED`，避免不同学生在空索引范围上形成 RR gap lock 死锁。它们继续锁定学生及实验实体，并在事务内复核权限、最新尝试、配置 revision 和活跃 run；不修改连接池的会话默认隔离级别。

上述路径对明确回滚的死锁/锁等待超时最多重试三次，间隔 50ms、150ms。模型重试仅重新执行落库事务，复用已完成的 Python 结果、run ID 和 attempt。回滚失败或 COMMIT ACK 丢失不得进入此重试；创建/重做的未知提交返回 `EXPERIMENT_COMMIT_OUTCOME_UNKNOWN` 503，需读取当前实验确认，模型继续使用原有 receipt 恢复协议。

模型 run 落库的查询/BEGIN 超时会销毁连接，随后对已关闭连接的回滚失败不得覆盖原有 `MODEL_DB_QUERY_TIMEOUT` / `MODEL_DB_TRANSACTION_TIMEOUT` 503。只有已确认回滚的 SQL 死锁/锁等待超时才能重放；回滚失败仍然禁止事务重试。

默认 DB 等待队列为 100。已知的连接池队列耗尽或服务器连接数耗尽统一返回 `DATABASE_BUSY` 503 和 `Retry-After: 2`。这表示容量不足，不应被解读为所有写入都可以自动重放。

报告预览的两项测验证据和模型观测数据按顺序读取，每名学生最多保留一个证据读取请求；50 人同步预览不会再主动扇出 150 个请求。两项测验必须完整，只有模型观测可降级。其他后台流量仍共享连接池，所以该约束不等于任意混合负载的容量保证。

学期创建/更新在同一已借连接内读取返回数据后提交；班级成员重复记录分支也复用当前连接。持有连接的服务不得再次向同一池借连接来完成自身响应。

## 8. 受管文件删除

数据集、手册和用户删除共享一个 process-wide deletion permit，容量为 1：

```text
deletion permit -> ordered DB row locks -> stage files -> DB commit -> purge/restore
```

该 permit 在提交结果复核及文件 restore/purge 完成前不会释放，因此单实例中不再需要每个删除操作额外持有数据库 named lock。durable deletion journal 和数据库 owner probe 继续负责宕机/ACK 丢失恢复。

## 9. PDF 并发

PDF 使用独立的进程内 `Semaphore`，允许等待时间有上限的 FIFO 排队，当前没有队列条目数上限：

- 活跃上限：`MAX_CONCURRENT_PDF_JOBS`；
- 等待超过 `PDF_QUEUE_TIMEOUT_MS` 返回 503；
- Chromium 启动、页面渲染、`page.pdf()` 和关闭共用 `PDF_GENERATION_TIMEOUT_MS` 墙钟预算；
- 超时会终止 Chromium，并给关闭流程最多 2 秒清理时间；
- 所有失败路径清理临时 PDF 并释放 permit；生成超时返回 504。

模型 admission 是 fail-fast，PDF 按等待时间限制排队，两者语义不同。同实验重复提交尚未在 PDF 生成前合并。

## 10. 认证限流和其他进程内控制

登录/注册固定窗口限流、latest-request gate、运行时计数和清理 timer 都是进程内状态。在单实例约束下它们就是全局状态，无需 Redis、leader election 或共享队列。

nginx 必须覆盖客户端提交的代理 IP 头，后端只信任 nginx 设置的 `X-Real-IP`；这与单实例无关，是限流身份可信度要求。

登录默认每 IP 15 分钟 100 次；50 名学生共享 NAT 时，两轮登录即可耗尽配额。部署验收需包含教师流量、误输密码和重登，按校园实际出口拓扑配置预算。

手册、数据集、报告和导出文件的单范围 `206` 响应使用 64 KiB 有界流读取。即使客户端请求 `bytes=0-`，后端也不得按完整文件或完整范围分配内存；客户端取消时必须关闭文件句柄。

## 11. 运行时观测

管理员调用 `GET /api/v1/runtime-info` 可查看：

- `deploymentMode=single_instance`、`coordinationScope=process`；
- 进程生命周期内的 HTTP 活跃数、总量、方法/状态码类别和聚合延迟；每个请求由 nginx 生成并覆盖 `X-Request-ID`，后端在响应与结构化日志中回显同一标识；
- CPU、模型/PDF 容量、DB pool、Python 线程数；
- operation/Python/PDF 超时；
- admission active/pending 和各 operation 活跃数；
- process-local execution mutex 的 active/acquired/conflicts；
- PDF 活跃任务和队列长度。

`admission.pending` 应始终为 0；`capacityRejected` 和 execution conflicts 是预期过载信号。所有请求结束后，admission、execution mutex 和 PDF active 数必须归零。

## 12. 测试

```bash
bun run test:check
bun test tests/unit/config/model-execution-lock.test.ts --isolate
bun test tests/unit/config/model-operation-lease.test.ts --isolate
bun test tests/unit/config/model-operation-classroom.test.ts --isolate
bun test tests/unit/config/python-lease-process-tree.test.ts --isolate
# 需要本地、空的专用测试库，以及 E2E_ALLOW_DB_RESET=1 和完整 DB_* 环境。
bun run test:system:concurrency
```

真实服务矩阵：

```bash
cd ../fzfe
bun run test:e2e:concurrency
bun run test:e2e:concurrency:classroom
bun run test:e2e:concurrency:mixed
bun run test:e2e:concurrency:ui
```

并发 E2E 必须通过真实 HTTP hold 请求占用同一 Bun 进程的 admission/execution mutex，不能再通过外部数据库 named lock 模拟容量。

`test:system:concurrency` 在真实 MySQL/MariaDB 上测试三轮 50 人创建、50 人重做、同学生竞争、三轮 12 人 run 提交、50 个学期写入和连接隔离恢复；`bun run ci` 将其作为本地 SQL gate。它不运行 Python。

前端默认并发命令依次执行 `protocols`（容量拒绝及恢复）与 `classroom`（50 人真实创建、提交两项测验、复用页面函数预览）。原来的 `all` 已更名为 `protocols`，避免将小规模协议测试误当整班负载验收。`mixed` 仍为独立的 MA/LSTM、预测和 PDF 混合测量。`E2E_CONCURRENCY_EXPECT_PDF_503=0` 严格要求所有 PDF 首轮成功，出现 503 即失败，不能靠串行补发变绿。
