# MariaDB 与 MySQL 8 双库兼容规范与机房运维清单

本项目后端必须同时兼容 **MariaDB（开发与现有环境）** 与 **MySQL 8（8.0 / 8.4，机房生产环境）**。所有业务 SQL、迁移脚本、重置/种子脚本与系统测试均需在两套数据库引擎上零错误、零弃用告警通过。

---

## 一、双库 SQL 编码规范与禁忌

| 场景 | 禁止写法 | 原因 | 正确写法 |
| --- | --- | --- | --- |
| `JSON` 列等值比较（CAS） | `col = ?`、`col = CAST(? AS JSON)`、`col = JSON_EXTRACT(?, '$')` | MySQL 8 下 `col = ?` 对 JSON 列与字符串恒假；MariaDB 下 `CAST(? AS JSON)` 报语法错误 `1064`；单侧 `JSON_EXTRACT` 在 MariaDB 下因格式化空格与紧凑 `LONGTEXT` 不相等 | 统一使用 `jsonColumnEqualsSql()` 生成双侧 `JSON_EXTRACT(col, '$') = JSON_EXTRACT(?, '$')` |
| 应用层比对数据库读回的 `JSON` 值 | `JSON.stringify(a) === JSON.stringify(b)` | MySQL 8 原生 JSON 按「键长度升序 + 字典序」重排对象键，且 `mysql2` 返回已解析对象；直接 `JSON.stringify` 会把未改动的 JSON 误判为变更 | 统一使用 `src/utils/stable-json.ts` 的 `stableJsonStringify()` 按规范键序序列化后再比较 |
| 修改外键 `DELETE_RULE` | 单条 `ALTER TABLE t DROP FOREIGN KEY fk, ADD CONSTRAINT fk ...` | MySQL 8 拒绝在同一条 `ALTER TABLE` 中删除并重建同名外键，直接报 `ER_FK_DUP_NAME (1826)` | 拆分为两条独立 `ALTER TABLE` 语句：先 `DROP FOREIGN KEY`，再 `ADD CONSTRAINT` |
| 派生表对 `TIMESTAMP` 条件聚合 | `MIN/MAX(CASE WHEN ... THEN ts_col END)` | MySQL 8 严格模式（`NO_ZERO_DATE`）下省略 `ELSE NULL` 与类型转换可能物化出 `'0000-00-00 00:00:00'` 导致 `ERROR 1292` | 显式写成 `CAST(ts_col AS DATETIME) ELSE NULL END` |
| 查询 `information_schema` | `SELECT table_name, column_name FROM information_schema...` 后按 `row.table_name` 取值 | MySQL 8 默认返回大写列名 `TABLE_NAME` / `COLUMN_NAME`，小写属性访问得到 `undefined` | 始终显式别名：`SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name` |
| `CREATE DATABASE` | 省略字符集与排序规则 | MySQL 8 默认创建 `utf8mb4_0900_ai_ci`，与建表基线 `utf8mb4_unicode_ci` 混用会导致联表报 `ER_CANT_AGGREGATE_2COLLATIONS (1267)` | 始终显式指定 `CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci` |
| `ON DUPLICATE KEY UPDATE` | `col = VALUES(col)` 或 `VALUES (...) AS new_row` | `VALUES(col)` 在 MySQL 8.0 / 8.4 触发弃用告警 `Warning 1287`；`AS new_row` 别名语法在 MariaDB 不支持 | 直接绑定参数 `col = ?` |
| 判断字符串前后空白 | `BINARY col <> BINARY TRIM(col)` | `BINARY expr` 在 MySQL 8.0 / 8.4 触发弃用告警 `Warning 1287` | 使用 `CHAR_LENGTH(col) <> CHAR_LENGTH(TRIM(col))` |
| InnoDB `FULLTEXT` 布尔查询 | 允许 `@`、`-` 进入 `MATCH ... AGAINST (... IN BOOLEAN MODE)` | `@` 在 MySQL 8 与 MariaDB 均触发 `ERROR 1064` 语法错误；`-` 会被解释为排除运算符 | 仅允许 `[\w\s.]` 走全文索引，邮箱等含 `@` / `-` 词条安全回落到 `LIKE ... ESCAPE '='` |
| `ER_DUP_ENTRY` 冲突解析 | 仅匹配 `users.username` 带表前缀格式 | MariaDB 报错键名不带表前缀（如 `'username'`），MySQL 8 带表前缀（如 `'users.username'`） | 用 `/for key '([^']+)'/i` 提取键名后取 `.split(".").pop()` 归一化判断 |
| 批量 `INSERT` 后推算主键 | `firstInsertId + i` | 隐式假定 `auto_increment_increment = 1`，主主/集群步长 > 1 时关联错位 | 同事务内按唯一业务键（如 `WHERE username IN (...)`）回查 `user_id` |
| SQL 字符串字面量 | `table_type = "BASE TABLE"` | 开启 `ANSI_QUOTES` 时双引号被解析为标识符 | SQL 字符串一律使用单引号 `'BASE TABLE'` |

---

## 二、双库自动化验证门禁

- **本地当前库兼容测试**（已接入 `bun run ci` / `./scripts/ci-local.sh full`）：
  ```bash
  E2E_ALLOW_DB_RESET=1 DB_DATABASE=fz_test_db_compat bun run test:system:db-compat
  ```
- **一键 Docker MySQL 8 完整门禁**（自动拉起临时容器跑通 `db-compat`、`user-phone-canonical`、`concurrency` 后自动销毁）：
  ```bash
  bun run test:system:mysql8
  # 或通过 ci-local.sh 入口：
  ./scripts/ci-local.sh mysql8
  # 验证 MySQL 8.4 LTS：
  MYSQL8_DOCKER_IMAGE=mysql:8.4 bun run test:system:mysql8
  ```

---

## 三、学校机房 MySQL 8 升级与热修回退核对清单

> 若机房曾手动应用过 `CAST(? AS JSON)` 本地热修，`./update.sh` 会在 `git status` 检查处中止发布，必须先备份并清理本地修改。

### 1. 发布前只读核对

在机房 MySQL 8 上执行以下只读查询并留档：

```sql
SELECT VERSION(), @@sql_mode, @@max_connections, @@auto_increment_increment, @@lower_case_table_names;
SELECT DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME
  FROM information_schema.SCHEMATA
 WHERE SCHEMA_NAME = DATABASE();
```

核对要点：
- `@@max_connections` 应大于后端连接池上限（建议 `>= 200`）；后端启动时也会在日志输出数据库版本及非阻断告警。
- 若库默认排序规则为 `utf8mb4_0900_ai_ci`，可在维护窗口执行：
  ```sql
  ALTER DATABASE `<your_db_name>` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
  ```
  （仅修改库默认排序规则，不重建存量表数据，可防止未来建表或临时表与主表联表报 `1267`）。

### 2. 清理本地热修并执行正式发布

```bash
cd /path/to/fzbe
# 1) 备份本地热修 diff
git diff > /tmp/fzbe-mysql8-hotfix-backup.patch
git status --short

# 2) 确认仅含该热修后回退本地改动
git checkout -- \
  src/modules/models/production-preparation.service.ts \
  src/modules/models/production-preparation-attempt-store.ts

# 3) 执行标准发布（自动完成 staging 校验、迁移与健康探活）
./update.sh
```

### 3. 上线后验收路径

1. 查看服务启动日志，确认出现 `Database runtime: version=8.x...` 且无异常告警。
2. 用测试学生账号完整走通：
   - 第 5 步训练任意模型；
   - 第 7 步触发生产模型准备与需求预测（验证进度推进正常、无静默零进度）；
   - 修改期初库存或产能后**再次保存生产计划**（验证不误报 `PRODUCTION_FORECAST_EVIDENCE_REQUIRED`）；
   - 提交并再次覆写实验报告（验证报告提交与 PDF 生成正常）。
3. 用管理员账号搜索含 `@` 的完整邮箱地址（验证不再报错 `1064`）。
