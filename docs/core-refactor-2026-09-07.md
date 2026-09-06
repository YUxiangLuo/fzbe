本批完成核心模块职责拆分，保留现有 API、数据库结构、模型算法和生产计划计算口径。跨学期改造已按用户决定标记为不实施。

| 原入口 | 保留职责 | 移出的职责 |
| --- | --- | --- |
| production-preparation.service.ts | 准备会话编排、身份/产物检查、Python 调用、结果提交 | production-preparation-state.ts 管理步骤协议、进度校验、响应投影；production-preparation-attempt-store.ts 管理 attempt 安装、receipt 重放认领、中间结果提交及失败记录事务 |
| model-run-store.ts | 运行记录、依赖关系和实验结果持久化 | model-training-observability.ts 整理和限制模型诊断输出 |
| 前端 ProductionPlanContextV2.tsx | React 生命周期、用户操作、保存编排 | productionPlanTypes.ts 定义状态；productionPlanState.ts 处理初始化/快照转换；productionPlanCalculation.ts 计算完整 MPS |

共享 JSON 列解析与对象判断移入 model-json.ts；model-context.ts 保留原导出。原服务和 Context 也保留既有公开导出，调用方无需切换接口。新的步骤协议与诊断模块可以在未配置数据库的进程中单独导入，不再通过纯数据处理路径初始化连接池。

事务模块原样保留锁顺序、CAS 条件、租约判断、提交结果未知时的连接处理及失败记录规则；没有抽象成通用“自动重试事务”而改变这些差异。MPS 计算保留首两期教学数据、固定一个月提前期、产能约束、缺货补偿、舍入规则和期数校验。诊断信息通过调用方传入的回调报告，计算模块不依赖 React 或直接操作全局状态。

| 入口文件 | 拆分前行数 | 拆分后行数 |
| --- | ---: | ---: |
| production-preparation.service.ts | 2060 | 1456 |
| model-run-store.ts | 1832 | 1502 |
| ProductionPlanContextV2.tsx | 906 | 545 |

行数变化只表示职责移动，不代表删除了相应业务能力。剩余编排与持久化代码仍有继续拆分空间；本批先建立独立可测的边界。

验证结果：

- 后端全量 Bun 单测：1101 通过，0 失败。
- 前端全量 Bun 单测：819 通过，0 失败。
- 两端 TypeScript、测试清单、模型 HTTP 超时契约、生成 API 类型一致性及正式前端构建均通过。
- 对比拆分前源码，37 个移出的函数/常量初始化实现文本完全一致，包含事务 SQL 与 CAS 条件。
- 用拆分前 MPS 实现作为对照，覆盖 2/3/6/12 期、零/有限/充足产能、整数/小数初始库存与需求的 192 组确定性组合，输出逐项一致。
- 新增独立测试覆盖步骤进度前缀、完成状态、损坏列数据、无提升 Boosting 直接完成、恢复响应，以及库存/产能/缺货递推、不可变输入、期数错误和诊断回调。

本批没有修改 Python 源码、界面结构或数据库，因此未执行 Python 全量工作流、浏览器全角色 E2E 和数据库迁移。课堂混合负载性能尚未在本批测量，不能据此次拆分宣称吞吐或内存改善。
