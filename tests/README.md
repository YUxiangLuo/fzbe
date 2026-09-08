# 后端测试结构

后端测试按执行层级和失败含义分成三类。新增测试应进入对应层级，不再向一个笼统目录或 CI YAML 里追加零散命令。

| 层级 | 位置 | 作用 | 常用入口 |
| --- | --- | --- | --- |
| Bun 单元/契约 | `tests/unit/<domain>/` | TypeScript 模块、服务、策略、schema 与请求边界 | `bun run test:unit` |
| Python 工作流 | `py/test/` | 真实 CLI、CSV、保存/加载、guided parity 和生产准备链路 | `bun run test:workflow` |
| Python deep | `py/deep_tests/` | 七类算法的公式、数据情形、泄漏、持久化、稳定性和资源矩阵 | `bun run test:deep:fast` |
| 系统边界矩阵 | `tests/system/` | 七类模型的 Python 与真实 API/数据库边界 | `bun run test:system:model-edge` |
| 真实 SQL 并发 | `tests/system/concurrency-classroom.ts` | 50 人创建/重做、run 激活、连接池阻塞及隔离恢复 | `bun run test:system:concurrency` |

## 日常命令

```bash
# 查看所有测试文件/脚本的分类和数量；存在未分类测试时会失败
bun run test:list

# TypeScript
bun run test:unit
bun run test:unit:models
bun run test:unit:services
bun run test:check              # typecheck + OpenAPI + inventory + 全部单测

# Python 端到端工作流
bun run test:workflow:quick       # contract
bun run test:workflow            # CI: contract + smoke + parity + regression
bun run test:workflow:full        # 再加入 diagnostic + stress

# 独立算法深测
bun run test:deep:fast
bun run test:deep:full
bun run test:deep:nightly
bun run test:deep:resource

# 系统级模型边界
bun run test:system:model-edge:python

# API/all 档会重置数据库，必须显式使用一次性测试库
RUN_ID="$(date -u +%Y%m%d%H%M%S)"
export E2E_ALLOW_DB_RESET=1
export DB_DATABASE="fz_pr_${RUN_ID}"
bun run test:system:model-edge:api
bun run test:system:model-edge

# 本地完整回归（耗时）
bun run test:full
```

`test:workflow:*` 由 `py/test/run.py` 的清单统一编排。可用
`py/venv/bin/python py/test/run.py --list` 查看 CI 精确包含哪些脚本，或用
`--suite contract|smoke|parity|regression|diagnostic|stress` 单独运行一层。
CI 只调用 runner，不再逐项复制脚本路径。

`test:system:model-edge:python` 不访问数据库，不需要重置授权。API/all
档只有在 `E2E_ALLOW_DB_RESET=1` 且数据库名为 `fangzhen_e2e`、
`fz_e2e_*`、`fz_pr_*` 或 `fz_test_*` 时才允许重置；默认开发库和其他
名称会在执行任何重置命令前被拒绝。
此外数据库 host 默认必须是 `localhost`、`127.0.0.1` 或 `::1`，且
`NODE_ENV=production` 或 `BUN_ENV=production` 默认被拒绝。确需使用远程一次性测试库或 production-mode
进程时，分别显式设置 `DB_RESET_ALLOW_REMOTE=1`、`DB_RESET_ALLOW_PROD=1`；
旧的 `DB_SETUP_ALLOW_REMOTE`/`DB_SETUP_ALLOW_PROD` 只作为兼容别名保留。

`test:system:concurrency` 额外要求本地且初始为空的专用测试库；只创建并清理自己拥有的表，不替换已有数据。它固定池大小 40、等待队列 100，将会话基线设为 RR 后验证局部 RC 不泄漏。CI 在 canonical-phone SQL gate 清空测试表后执行它。Python、HTTP 预览和混合负载另由相应回归覆盖，不能用 SQL gate 替代。

## Bun 单测领域

```text
tests/unit/
├── config/       # 并发、数据库、进程与运行时配置
├── controllers/  # HTTP controller
├── models/       # 模型运行、产物、预测、训练与结果转换
├── policies/     # 更新、提交等业务策略
├── routing/      # 应用路由装配
├── schemas/      # Zod 与 OpenAPI 请求/响应契约
├── security/     # 认证、权限、限流和启动安全
├── services/     # 非模型领域服务
└── utils/        # 纯工具
```

测试文件与被测域同名，使用 `*.test.ts`；跨语言或 API 边界测试额外使用
`*.contract.test.ts`。共享 harness 放在所属域的 `support/` 中，不作为独立测试发现。

## 新增测试的边界

- 单函数、服务分支和请求校验进入 `tests/unit/<domain>/`。
- 需要启动真实 Python CLI 或验证 artifact 生命周期的场景进入 `py/test/`，并登记到 `WORKFLOWS`。
- 算法性质、更多数据形状、随机稳定性和资源清理进入 `py/deep_tests/`，并标注对应 profile。
- 需要真实应用路由和数据库的跨层矩阵进入 `tests/system/`；这类破坏性测试不并入默认 `test:full`。
- 诊断脚本不能混入默认 CI；只有能明确判定成功/失败的脚本才能登记为 workflow。
- 提交前至少运行 `bun run test:list` 和受影响的最小套件。
