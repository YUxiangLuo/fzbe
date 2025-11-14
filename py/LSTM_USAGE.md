# LSTM 模型使用指南

## 输入参数说明

### 训练模式

#### 必需参数

1. **`--data`** (必需)
   - 格式：JSON字符串，包含时间序列数据
   - 要求：必须包含目标字段（`sales` 或 `y`）
   - 示例：
   ```json
   [
     {"month": "2024-01", "sales": 100.5},
     {"month": "2024-02", "sales": 105.2},
     {"month": "2024-03", "sales": 110.8}
   ]
   ```

#### 可选参数

2. **`--lstm_target_feature`** (可选，默认: `"sales"`)
   - 目标特征的字段名
   - 如果数据中有 `y` 字段，会自动检测
   - 示例：`--lstm_target_feature="y"` 或 `--lstm_target_feature="sales"`

3. **`--lstm_features`** (可选，默认: `[]`)
   - 额外的特征字段列表（多变量预测）
   - 示例：`--lstm_features='["price", "promotion"]'`
   - 数据格式：
   ```json
   [
     {"month": "2024-01", "sales": 100, "price": 50, "promotion": 1},
     {"month": "2024-02", "sales": 105, "price": 48, "promotion": 0}
   ]
   ```

4. **`--lookback`** (可选，默认: `10`)
   - 回看窗口大小（使用过去多少个时间步预测未来）
   - 必须 > 0 且 < 数据长度
   - 示例：`--lookback=5`

5. **`--epochs`** (可选，默认: `20`)
   - 训练轮数（客户规格：20轮）
   - 示例：`--epochs=20`

6. **`--batch_size`** (可选，默认: `6`)
   - 批量大小（客户规格：6）
   - 影响训练稳定性和速度
   - 示例：`--batch_size=6`

7. **`--lr`** (可选，默认: `0.01`)
   - 初始学习率（客户规格：0.01）
   - 示例：`--lr=0.01`

8. **`--lstm_normalization`** (可选，默认: `"zscore"`)
   - 归一化方法
   - 选项：`"zscore"` 或 `"minmax"`
   - 示例：`--lstm_normalization="zscore"`

9. **`--train_test_split`** (可选，默认: `0.8`)
   - 训练/测试数据分割比例
   - 范围：0.0 < 值 < 1.0
   - 0.8 表示 80% 训练，20% 测试
   - 示例：`--train_test_split=0.8`

10. **`--save_model`** (可选，默认: `"/tmp/lstm_model.pkl"`)
    - 保存模型的路径
    - 会自动生成对应的 `.pth` 文件保存PyTorch权重
    - 示例：`--save_model="/path/to/model.pkl"`

11. **`--pretty`** (可选，默认: `False`)
    - 是否美化JSON输出
    - 示例：`--pretty`

---

### 预测模式

#### 必需参数

1. **`--load_model`** (必需)
   - 已保存的模型文件路径
   - 示例：`--load_model="/path/to/model.pkl"`

2. **`--forecast_steps`** (必需)
   - 预测未来多少步
   - 必须 > 0
   - 示例：`--forecast_steps=5`

#### 可选参数

3. **`--data`** (可选)
   - 提供最新的观测数据（JSON格式）
   - 如果不提供，使用训练时保存的最后 `lookback` 个数据点
   - 示例：与训练模式的 `--data` 格式相同

4. **`--pretty`** (可选，默认: `False`)
   - 是否美化JSON输出
   - 示例：`--pretty`

---

## 完整示例

### 示例 1: 基本训练（单变量）

```python
from argparse import Namespace
import json

# 准备数据
data = [
    {"month": "2024-01", "sales": 100},
    {"month": "2024-02", "sales": 105},
    {"month": "2024-03", "sales": 110},
    # ... 更多数据
]

args = Namespace(
    data=json.dumps(data),
    lstm_target_feature="sales",
    lstm_features=[],
    lookback=10,
    epochs=20,
    batch_size=6,
    lr=0.01,
    lstm_normalization="zscore",
    train_test_split=0.8,
    save_model="/path/to/my_model.pkl",
    pretty=True
)

from models.lstm import run
run(args)
```

### 示例 2: 多变量训练

```python
from argparse import Namespace
import json

# 准备多变量数据
data = [
    {"month": "2024-01", "sales": 100, "price": 50, "promotion": 1},
    {"month": "2024-02", "sales": 105, "price": 48, "promotion": 0},
    {"month": "2024-03", "sales": 110, "price": 52, "promotion": 1},
    # ... 更多数据
]

args = Namespace(
    data=json.dumps(data),
    lstm_target_feature="sales",
    lstm_features=["price", "promotion"],  # 额外特征
    lookback=10,
    epochs=20,
    batch_size=6,
    lr=0.01,
    save_model="/path/to/multivar_model.pkl",
    pretty=True
)

from models.lstm import run
run(args)
```

### 示例 3: 加载模型进行预测

```python
from argparse import Namespace

args = Namespace(
    load_model="/path/to/my_model.pkl",
    forecast_steps=5,  # 预测未来5步
    pretty=True
)

from models.lstm import run
run(args)
```

### 示例 4: 使用新数据进行预测

```python
from argparse import Namespace
import json

# 提供最新数据
recent_data = [
    {"month": "2024-10", "sales": 150},
    {"month": "2024-11", "sales": 155},
    # ... 至少提供 lookback 个数据点
]

args = Namespace(
    load_model="/path/to/my_model.pkl",
    forecast_steps=3,
    data=json.dumps(recent_data),  # 使用最新数据
    pretty=True
)

from models.lstm import run
run(args)
```

---

## 输出格式

### 训练模式输出

```json
{
  "mode": "lstm_trained",
  "normalization": "zscore",
  "target_feature": "sales",
  "features": [],
  "lookback": 10,
  "epochs": 20,
  "metrics": {
    "rmse": 2.34,
    "mae": 1.89,
    "mape": 3.45,
    "r2": 0.95
  },
  "predictions": [
    {"prediction": 115.5, "std_dev": 2.1},
    {"prediction": 118.2, "std_dev": 2.1}
  ],
  "saved_model": "/path/to/my_model.pkl",
  "saved_pytorch_model": "/path/to/my_model.pth"
}
```

### 预测模式输出

```json
{
  "mode": "lstm_loaded_predict",
  "forecast_steps": 5,
  "predictions": [
    {"prediction": 120.5, "std_dev": 2.1},
    {"prediction": 122.3, "std_dev": 2.1},
    {"prediction": 124.1, "std_dev": 2.1},
    {"prediction": 125.8, "std_dev": 2.1},
    {"prediction": 127.5, "std_dev": 2.1}
  ]
}
```

---

## 数据要求

### 最小数据量
- 数据长度必须 > `lookback`
- 建议：至少 `lookback * 10` 个数据点以获得良好效果
- 训练需要足够的数据来创建训练和测试序列

### 数据格式
- **JSON数组**：必须是列表格式
- **必需字段**：目标字段（`sales` 或 `y`）
- **可选字段**：
  - `month`：时间标签（可选）
  - 额外特征：如 `price`, `promotion` 等

### 数据质量
- **不允许 NaN/inf 值**：数据必须是有限数值
- **数值类型**：所有特征必须可转换为浮点数
- **一致性**：所有记录必须包含相同的字段

---

## 常见错误

### 错误 1: 数据太短
```
[VALIDATION] Data length (5) must be > lookback (10)
```
**解决**：增加数据量或减小 `lookback`

### 错误 2: 缺少目标字段
```
[DATA] Cannot determine target key. Expecting 'y' or 'sales'.
```
**解决**：确保数据包含 `sales` 或 `y` 字段，或使用 `--lstm_target_feature` 指定

### 错误 3: 包含 NaN 值
```
[DATA] Data contains NaN or inf values
```
**解决**：清理数据，移除或填充 NaN 值

### 错误 4: forecast_steps 无效
```
[VALIDATION] forecast_steps must be > 0
```
**解决**：确保 `--forecast_steps` 是正整数

---

## 模型规格（基于客户需求）

- **LSTM层数**: 3层（增加模型深度，学习复杂时间序列模式）
- **每层隐藏单元数**: 288个（提供足够容量捕捉复杂性）
- **输出特征数量**: 1个（预测目标值）
- **损失函数**: MSE（均方误差）
- **优化器**: Adam（自适应学习率优化）
- **最大训练轮数**: 20轮
- **批量大小**: 6
- **初始学习率**: 0.01
- **学习率衰减率**: 0.99（逐渐减小学习率以细化参数）

---

## Python 直接调用示例

```python
#!/usr/bin/env python3
import json
from argparse import Namespace
from models.lstm import run

# 创建测试数据
data = [{"month": f"2024-{i:02d}", "sales": 100 + i * 2} for i in range(1, 51)]

# 训练
args_train = Namespace(
    data=json.dumps(data),
    lstm_target_feature="sales",
    lstm_features=[],
    lookback=10,
    epochs=20,
    batch_size=6,
    lr=0.01,
    save_model="my_lstm.pkl",
    pretty=True
)

print("Training...")
run(args_train)

# 预测
args_predict = Namespace(
    load_model="my_lstm.pkl",
    forecast_steps=5,
    pretty=True
)

print("\nPredicting...")
run(args_predict)
```
