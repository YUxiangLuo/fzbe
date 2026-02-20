# Admin 端前后端联调用文档（后端接口参考）

基于当前后端代码整理，适用于管理端（Admin）前端联调与实现。

## 1. 鉴权与通用约定

### 1.1 登录

- `POST /api/v1/sessions`
- 请求体：

```json
{
  "username": "admin",
  "password": "123456",
  "role": "Admin"
}
```

- 成功响应：

```json
{
  "token": "<JWT>",
  "user": {
    "role": "Admin"
  }
}
```

### 1.2 鉴权头

- 除公开接口外，均需：
- `Authorization: Bearer <JWT>`

### 1.3 错误格式

- 常见格式：`{ "error": "..." }`
- 部分接口会附带额外字段：
- `details`
- `invalid_rows`
- `conflicts`
- `current_class`

前端建议统一按 `HTTP status + error` 处理，不要只靠中文文案精确匹配。

## 2. 文件路径与下载/预览规则（重要）

后端返回的文件路径是**相对公共路径**，例如：

- `manuals/xxx.pdf`
- `datasets/xxx.csv`
- `reports/xxx.pdf`
- `exports/xxx.zip`

前端拼接规则：

- `finalUrl = ${API_BASE}/${file_path}`
- 不要再手动加 `uploads/` 前缀
- 访问 `reports/*`、`exports/*` 时必须携带 `Authorization: Bearer <JWT>`，后端会做对象级权限校验

静态资源路由：

- `/manuals/*`
- `/datasets/*`
- `/reports/*`
- `/exports/*`

## 3. 管理端核心接口

## 3.1 用户管理

### 管理员常用

- `GET /api/v1/users`
- 用途：分页获取全部用户
- 权限：`Admin`

- `GET /api/v1/users/search?q=...&page=1&limit=10`
- 用途：搜索用户
- 权限：`Admin`

- `POST /api/v1/users/teachers`
- 用途：创建教师（单个）
- 权限：`Admin`

- `POST /api/v1/users/assistants`
- 用途：创建助教（单个）
- 权限：`Admin`

- `POST /api/v1/assistants`
- 用途：创建助教并直接分配到班级（`class_ids`）
- 权限：`Admin` / `Teacher`（教师只能分配到自己班级）

- `POST /api/v1/users/teachers/batch`（`multipart/form-data`）
- 字段：`file`(CSV), `password`(可选)
- 用途：批量创建教师
- 权限：`Admin`

- `POST /api/v1/users/assistants/batch`（`multipart/form-data`）
- 字段：`file`(CSV), `password`(可选)
- 用途：批量创建助教
- 权限：`Admin`

- `GET /api/v1/users/:userId`
- 用途：查询用户详情
- 权限：`Admin` 或本人

- `GET /api/v1/assistants`
- 用途：助教列表
- 权限：`Admin` / `Teacher`

- `GET /api/v1/teachers/:teacherId/assistants`
- 用途：查询某教师名下助教
- 权限：`Admin` 或该教师本人

- `GET /api/v1/students/unenrolled`
- 用途：查询未入班学生
- 权限：`Admin` / `Teacher` / `Assistant`

- `PUT /api/v1/users/:userId/password`
- 用途：管理员强制改密码
- 权限：`Admin`

- `POST /api/v1/users/:userId/reset-password`
- 用途：重置学生密码（默认或指定新密码）
- 权限：`Admin` / `Teacher` / `Assistant`（后两者受班级归属限制）

- `DELETE /api/v1/users/:userId`
- 用途：删除用户
- 权限：`Admin`

### 个人信息

- `GET /api/v1/users/me`
- `PUT /api/v1/users/me`
- `PUT /api/v1/users/me/password`

## 3.2 班级管理

- `POST /api/v1/classes`（`multipart/form-data`）
- 字段：`class_name`，`class_code`(可选)，`student_list`(CSV，可选)
- 用途：创建班级，并可批量导入学生
- 权限：`Teacher` / `Admin`

- `GET /api/v1/classes`
- 用途：获取全部班级
- 权限：`Admin`

- `GET /api/v1/classes/:classId`
- 用途：班级详情（含学生和助教列表）
- 权限：`Admin` / 班级教师 / 班级助教 / 班级学生

- `PUT /api/v1/classes/:classId`
- 用途：更新班级名称/班级码
- 权限：`Admin` / 班级教师

- `DELETE /api/v1/classes/:classId`
- 用途：删除班级
- 权限：`Admin` / 班级教师

- `GET /api/v1/teachers/:teacherId/classes`
- 权限：`Admin` 或该教师本人

- `GET /api/v1/assistants/:assistantId/classes`
- 权限：`Admin` 或该助教本人

## 3.3 班级成员管理

- `POST /api/v1/classes/:classId/students`
- 用途：创建新学生并加入班级
- 权限：`Admin` / `Teacher` / `Assistant`

- `POST /api/v1/classes/:classId/enroll`
- 请求体：`{ "student_id": 123 }`
- 用途：把已有学生加入班级
- 权限：`Admin` / `Teacher` / `Assistant`

- `GET /api/v1/classes/:classId/students`
- 用途：获取班级学生列表
- 权限：`Admin` / 班级教师 / 班级助教

- `DELETE /api/v1/classes/:classId/students/:studentId`
- 用途：移出班级学生
- 权限：`Admin` / 班级教师 / 班级助教

- `POST /api/v1/classes/:classId/assistants`
- 请求体：`{ "assistant_id": 123 }`
- 用途：给班级添加助教
- 权限：`Admin` / 班级教师

- `GET /api/v1/classes/:classId/assistants`
- 用途：获取班级助教列表
- 权限：`Admin` / 班级教师 / 班级助教

- `DELETE /api/v1/classes/:classId/assistants/:assistantId`
- 用途：移除班级助教
- 权限：`Admin` / 班级教师

## 3.4 成绩、报告与导出

- `GET /api/v1/classes/:classId/reports`
- 用途：班级报告列表（含 `pdf_file_path`）

- `PUT /api/v1/reports/:reportId`
- 用途：批改报告/更新状态/反馈

- `GET /api/v1/classes/:classId/grading-policy`
- `PUT /api/v1/classes/:classId/grading-policy`
- 用途：获取/更新成绩权重

- `GET /api/v1/classes/:classId/grade-summaries`
- 用途：学生成绩汇总

- `GET /api/v1/classes/:classId/experiment-runs`
- 用途：实验进度概览

- `GET /api/v1/classes/:classId/experiment-events`
- 用途：实验步骤事件时间线

- `GET /api/v1/classes/:classId/report-archive`
- 返回：`{ "file_path": "exports/xxx.zip" }`

- `GET /api/v1/classes/:classId/grade-export.csv`
- 返回：`{ "file_path": "exports/xxx.csv" }`

- `GET /api/v1/classes/:classId/report-export.csv`
- 返回：`{ "file_path": "exports/xxx.csv" }`

以上接口权限由班级归属校验控制，`Admin` 全量可访问。

## 3.5 手册管理（Admin）

- `POST /api/v1/manuals`（`multipart/form-data`）
- 字段：`file`(PDF), `file_name`, `description`(可选)
- 权限：`Admin`

- `GET /api/v1/manuals`
- 权限：`Admin`

- `PUT /api/v1/manuals/:manualId`
- 权限：`Admin`

- `DELETE /api/v1/manuals/:manualId`
- 权限：`Admin`

- `GET /api/v1/manuals/active`
- 公开接口（无需 JWT）

## 3.6 数据集管理

- `POST /api/v1/datasets`（`multipart/form-data`）
- 字段：`file`(CSV), `data_name`, `description`(可选)
- 权限：`Admin`

- `PUT /api/v1/datasets/:datasetId`
- `DELETE /api/v1/datasets/:datasetId`
- 权限：`Admin`

- `GET /api/v1/datasets`
- `GET /api/v1/datasets/:datasetId`
- `GET /api/v1/datasets/industries`
- `GET /api/v1/datasets/industries/:industry/companies`
- `GET /api/v1/datasets/industries/:industry/companies/:company/products`
- `GET /api/v1/datasets/industries/:industry/companies/:company/products/:product/fields`
- `GET /api/v1/datasets/industries/:industry/companies/:company/products/:product/sales`
- 权限：已登录用户（JWT）

## 4. 前端落地建议

- 把角色控制放在前端路由守卫，后端继续做最终鉴权。
- 所有下载/预览都走统一函数处理 `file_path`。
- 对 `409` 的冲突错误做专门 UI（如重复用户、重复分配、重复导入）。
- 对批量导入接口展示 `invalid_rows`/`conflicts`，支持“失败明细”弹窗。

可复用示例：

```ts
export function resolveFileUrl(apiBase: string, filePath: string) {
  return `${apiBase.replace(/\/$/, "")}/${filePath.replace(/^\//, "")}`;
}
```
