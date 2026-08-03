# Python workflow tests

这里验证真实 Python 入口和跨进程生命周期；它与 `py/deep_tests` 的独立
`unittest` 算法矩阵分工明确。所有可执行脚本都登记在 `run.py`，CI 不直接维护脚本列表。

| suite | 覆盖范围 | 默认 CI |
| --- | --- | :---: |
| `contract` | 指标、数值数据契约、模型元数据 | 是 |
| `smoke` | 四个基础模型、三类融合模型、LSTM 组合、artifact 复用、生产准备 | 是 |
| `parity` | guided/standalone 与 guided/regular ensemble 一致性 | 是 |
| `regression` | 已修复问题的端到端复现 | 是 |
| `diagnostic` | 设计数据与 LSTM 特征诊断 | 否 |
| `stress` | focused edge-case；36 期数据下四基础模型全部 2/3/4 成员融合组合 | 否 |

```bash
py/venv/bin/python py/test/run.py --list
py/venv/bin/python py/test/run.py --profile quick
py/venv/bin/python py/test/run.py --profile ci
py/venv/bin/python py/test/run.py --profile full
py/venv/bin/python py/test/run.py --suite smoke
py/venv/bin/python py/test/run.py --suite contract --suite parity
```

runner 为每个脚本创建独立进程，固定哈希、BLAS 和 TensorFlow 线程环境，并在最后给出统一失败清单。新增脚本后必须把脚本、suite 和一句可验证的目的登记到 `WORKFLOWS`。

`stress` 中的 `model_edge_cases.py` 使用 `focused` 档，只执行它独有的输入与失败
边界，不会再次嵌套回放已由 contract/smoke/parity 覆盖的脚本。需要显式聚合
全部 Python 生命周期时使用 `bun run test:system:model-edge:python`。
