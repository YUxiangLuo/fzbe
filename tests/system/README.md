# 系统测试

这里放需要同时跨越 TypeScript 应用、Python 模型进程和数据库的测试入口。

`model-edge-cases.ts` 覆盖 MA、ES、ARIMA、LSTM、加权平均、Stacking 和 Boosting
的 Python/API 边界矩阵。Python 层可独立运行；API 层会执行数据库重置与 E2E
播种，因此没有并入默认 `test:full`。

```bash
bun run test:system:model-edge:python

# API/all 会清空并重建数据库，只能使用一次性测试库
RUN_ID="$(date -u +%Y%m%d%H%M%S)"
export E2E_ALLOW_DB_RESET=1
export DB_DATABASE="fz_pr_${RUN_ID}"
bun run test:system:model-edge:api
bun run test:system:model-edge
```

API/all 会在运行任何 Python 矩阵或数据库命令前检查授权和数据库名；只有
`fangzhen_e2e`、`fz_e2e_*`、`fz_pr_*`、`fz_test_*` 被允许。其余数据库
会直接拒绝，`--force` 不能绕过该检查。
