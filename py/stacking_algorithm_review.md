# Stacking Algorithm Review Report

## 📚 教科书式Stacking定义

**标准Stacking (Stacked Generalization)** 由Wolpert (1992)提出：

### 正确流程：

```
训练阶段：
├── Level 0 (Base Models)
│   ├── 将训练集分为K折或train/validation
│   ├── 基础模型在训练部分训练
│   ├── 在验证部分生成out-of-fold预测
│   └── 用全部训练集重新训练（用于最终预测）
│
└── Level 1 (Meta-Model)
    ├── 输入：基础模型的out-of-fold预测
    ├── 目标：原始训练集标签
    └── 训练元学习器

预测阶段：
├── 基础模型（在全部训练集上训练过）预测新数据
└── 元学习器基于基础预测给出最终预测

评估阶段：
└── 在独立的测试集上评估最终预测
```

---

## ⚠️ 当前实现的问题

### 问题1：严重的数据泄露 (Data Leakage)

**当前代码 (lines 189-234):**

```python
# Level 0: Train base models
for model_name in models_to_use:
    result = run_base_model_subprocess(
        train_start=train_range.start,  # 训练集：[0, 14]
        train_end=train_range.stop - 1,
        eval_start=eval_range.start,    # 评估集：[15, 19] ❌
        eval_end=eval_range.stop - 1,
    )
    y_pred = extract_predictions_from_result(result)
    base_model_predictions.append(y_pred)  # 来自评估集的预测

# Level 1: Train meta-model
X_meta_train = np.array(base_model_predictions).T  # ❌ 评估集的预测
y_meta_train = parsed_data.y[eval_range]           # ❌ 评估集的标签

meta_model.fit(X_meta_train, y_meta_train)  # ❌ 在评估集上训练！
y_pred_final = meta_model.predict(X_meta_train)  # ❌ 在评估集上预测！
```

**问题分析：**

1. 基础模型在 `train_range` 上训练
2. 基础模型在 `eval_range` 上生成预测
3. 元模型用 `eval_range` 的预测和标签训练
4. 元模型在同一个 `eval_range` 上评估

**这等同于：**
```
测试集 = 训练集
```

**后果：**
- 严重过拟合
- 性能指标过于乐观（misleading）
- 无法反映真实泛化能力
- 不符合机器学习基本原则

---

### 问题2：不符合Stacking定义

**标准Stacking要求：**
- 元模型的训练数据必须是"未见过"的预测（out-of-fold或hold-out）
- 避免元模型"记住"基础模型在训练集上的模式

**当前实现：**
- 元模型直接在评估集上训练和评估
- 没有out-of-fold预测
- 没有独立的验证集

---

## ✅ 正确的实现方案

### 方案：Hold-Out Stacking（适合小数据集教学）

```python
def train_and_evaluate_correct(
    parsed_data: ParsedData,
    train_range: range,
    eval_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Any],
):
    """
    正确的Stacking实现：
    1. 将训练集分为level0_train和level1_train
    2. 基础模型在level0_train上训练，在level1_train上预测
    3. 元模型在level1_train的预测上训练
    4. 基础模型在完整train_range上重新训练
    5. 在独立的eval_range上评估
    """

    # Step 1: 分割训练集
    # 训练集划分：前80%用于level0，后20%用于level1
    train_len = len(train_range)
    level0_split = int(train_len * 0.8)

    level0_train_range = range(train_range.start, train_range.start + level0_split)
    level1_train_range = range(train_range.start + level0_split, train_range.stop)

    print(f"Level 0 training: {level0_train_range}")
    print(f"Level 1 training: {level1_train_range}")
    print(f"Final evaluation: {eval_range}")

    # Step 2: 训练基础模型并生成level1预测
    level1_predictions = []
    base_models_temp = {}

    for model_name in models_to_use:
        # 在level0上训练，在level1上预测（用于元模型训练）
        result = run_base_model_subprocess(
            train_start=level0_train_range.start,
            train_end=level0_train_range.stop - 1,
            eval_start=level1_train_range.start,  # ✅ 在level1上预测
            eval_end=level1_train_range.stop - 1,
            ...
        )
        y_pred_level1 = extract_predictions_from_result(result)
        level1_predictions.append(y_pred_level1)

    # Step 3: 训练元模型（使用level1的预测）
    X_meta_train = np.array(level1_predictions).T  # ✅ level1的预测
    y_meta_train = parsed_data.y[level1_train_range]  # ✅ level1的标签

    meta_model = LinearRegression()
    meta_model.fit(X_meta_train, y_meta_train)  # ✅ 在独立的level1集上训练

    # Step 4: 在完整训练集上重新训练基础模型（用于最终预测）
    eval_predictions = []
    base_model_states = {}

    for model_name in models_to_use:
        # 在完整train_range上训练，在eval_range上预测
        result = run_base_model_subprocess(
            train_start=train_range.start,  # ✅ 完整训练集
            train_end=train_range.stop - 1,
            eval_start=eval_range.start,    # ✅ 独立评估集
            eval_end=eval_range.stop - 1,
            ...
        )
        y_pred_eval = extract_predictions_from_result(result)
        eval_predictions.append(y_pred_eval)
        base_model_states[model_name] = {...}

    # Step 5: 最终预测和评估
    X_meta_eval = np.array(eval_predictions).T  # ✅ eval_range的预测
    y_pred_final = meta_model.predict(X_meta_eval)  # ✅ 元模型预测
    y_true_final = parsed_data.y[eval_range]  # ✅ 真实标签

    # ✅ 在独立的eval_range上计算指标
    metrics = compute_metrics(y_true_final, y_pred_final)

    return metrics, base_model_states, meta_model, ...
```

---

## 📊 对比分析

| 维度 | 当前实现 | 正确实现 |
|------|---------|---------|
| 数据划分 | train_range + eval_range | level0_train + level1_train + eval_range |
| 元模型训练集 | eval_range（测试集） ❌ | level1_train（验证集） ✅ |
| 元模型评估集 | eval_range（同训练集） ❌ | eval_range（独立集） ✅ |
| 数据泄露 | 有严重泄露 ❌ | 无泄露 ✅ |
| 符合Stacking定义 | 否 ❌ | 是 ✅ |
| 性能估计 | 过于乐观 ❌ | 真实可靠 ✅ |

---

## 🎯 修复优先级

**严重性：高 🔴**

**原因：**
1. 违反机器学习基本原则（测试集=训练集）
2. 性能指标不可信
3. 不符合教科书定义
4. 可能误导学生

**建议：**
立即修复train_and_evaluate函数，实施正确的hold-out stacking流程。

---

## 📝 参考文献

- Wolpert, D. H. (1992). "Stacked generalization". Neural Networks, 5(2), 241-259.
- Breiman, L. (1996). "Stacked regressions". Machine Learning, 24(1), 49-64.
- 周志华《机器学习》第8章：集成学习
