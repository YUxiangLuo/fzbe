# 学生端 Dogfood 数据库与运行手册

本套夹具用于真实前后端下的学生端探索式测试。种子脚本只允许写入名称以
`fz_student_dogfood_` 开头的本地数据库，并且要求显式传入测试密码和
`--force`，避免误操作开发或生产数据。

## 准备隔离环境

从后端仓库根目录执行。每一轮使用新的运行标识、数据库和仓库外输出目录；
密码只保存在当前 shell 环境，不写入命令、文档或 manifest。

```bash
cd fzbe

export DOGFOOD_RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
export DB_HOST="127.0.0.1"
export DB_USER="<local-database-user>"
export DB_DATABASE="fz_student_dogfood_${DOGFOOD_RUN_ID//-/_}"
export E2E_ALLOW_DB_RESET="1"
export DOGFOOD_OUTPUT_ROOT="$(pwd)/../dogfood-output/student-${DOGFOOD_RUN_ID}"
# 种子脚本会用这个密钥签发“已完成但未提交报告”的恢复样本；本轮启动后端时
# 必须复用完全相同的值。
export JWT_SECRET="$(openssl rand -hex 32)"

read -rsp "MariaDB password: " DB_PASSWORD
export DB_PASSWORD
printf "\n"
read -rsp "Temporary dogfood student password: " DOGFOOD_STUDENT_PASSWORD
export DOGFOOD_STUDENT_PASSWORD
printf "\n"

bun run db:reset --force
bun run dogfood:student:seed --force
bun run dogfood:student:verify
```

验证器会检查：

- 9 个账号的密码与最新实验/报告状态；
- 3 个数据集、5 个产品分段及所有 CSV 文件；
- 4 个完成态生产计划是否通过应用的正式完成态契约；
- 3 个报告 PDF 是否存在于真实下载目录，并且具有有效的页面、Helvetica 字体、
  文本流、stream 长度和 xref 偏移；可见文本还必须匹配账号、报告状态及
  待评/驳回重做/评分场景。

机器可读明细写入
`$DOGFOOD_OUTPUT_ROOT/database-manifest.json`。manifest 不包含数据库密码、
学生密码或 JWT；运行结束后可将整个 `$DOGFOOD_OUTPUT_ROOT` 作为 CI/PR
证据归档。

## 账号矩阵

所有账号使用 `$DOGFOOD_STUDENT_PASSWORD` 指定的本轮临时密码。

| 账号 | 场景 | 主要用途 |
| --- | --- | --- |
| `20990001` | 全新账号 | 从登录到报告提交的真实主路径 |
| `20990002` | 步骤 3 恢复 | 继续/新建选择、产品页、路由守卫 |
| `20990003` | 步骤 5 模型阶段 | 四种基础模型、三种融合模型和训练失败恢复 |
| `20990004` | 报告待评分 | 状态页、PDF、返回实验首页 |
| `20990005` | 报告已驳回 | 驳回原因、重新实验、重新提交入口 |
| `20990006` | 遗留损坏生产状态 | 不完整旧预测的恢复、错误说明和重试 |
| `20990007` | 超长 Unicode 资料 | 顶栏、侧栏、个人中心、窄屏和编辑资料 |
| `20990008` | 完成但无报告 | 首页决策与历史异常边界 |
| `20990009` | 报告已评分 | 分数、教师反馈和报告 PDF |

## 数据集矩阵

| 数据集 | 分段 | 用途 |
| --- | ---: | --- |
| Dogfood 标准季节趋势数据集 | 2 × 48 个月 | Happy path、全部模型、生产计划 |
| Dogfood 零销量与尖峰边界数据集 | 24/36 个月 | 零值、极端值、图表、指标和失败文案 |
| Dogfood 长名称与国际化多变量数据集 | 60 个月 | Unicode、溢出、多变量 LSTM、响应式 |

## 启动

后端继续复用上面的数据库环境和种子阶段已经生成的 `JWT_SECRET`。不要在种子
完成后轮换这个密钥，否则 `20990008` 的生产预测凭证会失效。这里也必须保留
`E2E_ALLOW_DB_RESET=1`：`e2e-server` 启动时会执行过期数据清理，因此即使数据库
已经存在，也只允许连接上面创建的 disposable dogfood 数据库：

```bash
export PORT="54230"
export AUTH_LOGIN_RATE_LIMIT_MAX_PER_IP="1000"
bun run src/e2e-server.ts
```

另开终端，从前端仓库根目录启动：

```bash
cd fzfe
export VITE_API_URL="http://127.0.0.1:54230/api/v1"
bun run dev -- --host 127.0.0.1 --port 55230
```

浏览器验收统一使用 `agent-browser`：

```bash
agent-browser \
  --session "fz-student-dogfood-${DOGFOOD_RUN_ID}" \
  --allowed-domains 127.0.0.1 \
  open http://127.0.0.1:55230/login.html
```

## 深测覆盖

1. 登录、错误凭据、角色切换、退出和会话隔离。
2. 介绍、开始/继续实验、直接深链和全部七步导航守卫。
3. 行业、公司、产品、历史数据、三类数据集和窗口边界。
4. MA、ES、ARIMA、LSTM 与 Weighted、Boosting、Stacking 的介绍、参数、训练、结果和失败恢复。
5. 外层模型评价、基线、区间证据、模型测验。
6. 生产预测、产能场景、安全库存、MPS、计划测验。
7. 报告预览、提交、待评、驳回、重提、已评分和 PDF。
8. 个人资料、长文本、桌面/平板/手机、键盘与 WCAG 快速审计。
9. 刷新、前进后退、双标签页版本冲突、离线/接口错误和控制台。

测试完成后关闭前后端进程，并按保留策略归档或删除本轮独立数据库与
`$DOGFOOD_OUTPUT_ROOT`；不要把运行时模型、PDF、HAR、截图或凭据提交到仓库。
