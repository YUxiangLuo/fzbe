# 集成模型算法Review总结

## 🔴 严重问题：所有三个集成模型都存在数据泄露

---

## 1️⃣ Stacking (stacking.py) ✅ 已修复

### ❌ 原始问题：
```python
# 基础模型在train上训练，在eval上预测
base_predictions = [模型在eval上的预测]

# 元模型用eval的预测和标签训练 ← ❌ 数据泄露
meta_model.fit(base_predictions, y[eval_range])

# 在同一个eval上评估 ← ❌ 测试集=训练集
```

### ✅ 修复方案（已实施）：
- 将train_range分为level0(80%)和level1(20%)
- 基础模型在level0上训练，在level1上预测
- 元模型在level1的预测上训练（out-of-fold）
- 基础模型在完整train上重新训练
- 在独立的eval上最终评估

### 验证结果：
```
✅ PASS: Stacking Data Split
✅ PASS: Leakage Prevention
Total: 2/2 tests passed (100%)
```

---

## 2️⃣ Weighted Average (weighted_avg.py) ❌ 需要修复

### 算法原理：
使用**Inverse Variance Weighting**，权重与预测误差的方差成反比：
```python
weight_i = 1 / variance_i
```

### ❌ 当前问题 (lines 168-199)：
```python
# 基础模型在train上训练，在eval上预测
result = run_base_model_subprocess(
    train_start=train_range.start,
    train_end=train_range.stop - 1,
    eval_start=eval_range.start,  # ❌ 在eval上预测
    eval_end=eval_range.stop - 1,
)

# 用eval集的预测计算权重 ← ❌ 数据泄露
for name in models_to_use:
    residuals = y_true - np.array(base_model_predictions[name])
    variance = np.var(residuals)  # ❌ eval集的方差
    weights[name] = 1 / (variance + 1e-9)  # ❌ 权重基于eval集

# 用同样的权重在eval集上评估 ← ❌ 测试集=训练集
```

### ✅ 正确实现方案：

```python
def train_and_evaluate_correct(
    parsed_data: ParsedData,
    train_range: range,
    eval_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Any],
):
    # Step 1: 分割train_range为weight_train(80%)和weight_val(20%)
    train_len = len(train_range)
    weight_split = int(train_len * 0.8)
    weight_train_range = range(train_range.start, train_range.start + weight_split)
    weight_val_range = range(train_range.start + weight_split, train_range.stop)

    # Step 2: Phase 1 - 在weight_train上训练，在weight_val上预测
    weight_val_predictions = {}
    for model in models:
        pred = model.fit(weight_train).predict(weight_val)
        weight_val_predictions[model] = pred

    # Step 3: 计算权重（基于weight_val的方差，避免泄露）
    y_true_val = y[weight_val_range]
    weights = {}
    for model in models:
        residuals = y_true_val - weight_val_predictions[model]
        variance = np.var(residuals)  # ✅ 在独立的weight_val上计算
        weights[model] = 1 / (variance + 1e-9)

    # 归一化权重
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}

    # Step 4: Phase 2 - 在完整train上重新训练，在eval上预测
    eval_predictions = {}
    for model in models:
        pred = model.fit(train_range).predict(eval_range)
        eval_predictions[model] = pred

    # Step 5: 最终加权预测和评估（在独立的eval集上）
    y_pred = sum(weights[m] * eval_predictions[m] for m in models)
    y_true = y[eval_range]  # ✅ 独立测试集
    metrics = evaluate(y_true, y_pred)  # ✅ 真实性能
```

---

## 3️⃣ Boosting (boosting.py) ❌ 需要修复

### 算法原理：
使用**Greedy Forward Selection + AdaBoost-style Weighting**：
1. 迭代选择最佳模型
2. 更新样本权重（错误样本权重增加）
3. 计算模型权重（基于加权误差）
4. 贪心选择：只添加能改善整体RMSE的模型

### ❌ 当前问题 (lines 130-208)：
```python
for stage in range(len(models_to_use)):
    # 所有候选模型在train上训练，在eval上预测
    for model_name in available_models:
        result = run_base_model_subprocess(
            train_start=train_range.start,
            train_end=train_range.stop - 1,
            eval_start=eval_range.start,  # ❌ 在eval上预测
            eval_end=eval_range.stop - 1,
        )

        # 用eval集的误差选择模型 ← ❌ 数据泄露
        y_pred = extract_predictions_from_result(result)
        errors = np.abs(np.array(y_pred) - y_true_eval)  # ❌ eval集的误差
        weighted_error = np.sum(sample_weights * errors)  # ❌ 基于eval

    # 选择最佳模型 ← ❌ 基于eval集的表现
    best_model = min(stage_results, key=lambda x: x["weighted_error"])

    # 计算权重 ← ❌ 基于eval集的误差
    beta = weighted_error / (1 - weighted_error)
    model_weight = np.log(1 / beta)

    # 贪心选择 ← ❌ 基于eval集的RMSE
    if temp_rmse < best_overall_rmse:
        add_model()

    # 更新样本权重 ← ❌ 基于eval集的误差
    sample_weights *= np.power(beta, 1 - normalized_errors)
```

**所有决策都基于eval集的表现 → 严重的数据泄露！**

### ✅ 正确实现方案：

```python
def train_and_evaluate_correct(
    parsed_data: ParsedData,
    train_range: range,
    eval_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Any],
):
    # Step 1: 分割train_range为boost_train(80%)和boost_val(20%)
    train_len = len(train_range)
    boost_split = int(train_len * 0.8)
    boost_train_range = range(train_range.start, train_range.start + boost_split)
    boost_val_range = range(train_range.start + boost_split, train_range.stop)

    # Step 2: Greedy Forward Selection (在boost_val上进行)
    best_chain = []
    best_weights = {}
    sample_weights = np.ones(len(boost_val_range)) / len(boost_val_range)
    y_true_val = y[boost_val_range]

    for stage in range(len(models)):
        stage_results = []

        # 在boost_train上训练，在boost_val上预测
        for model in available_models:
            pred_val = model.fit(boost_train).predict(boost_val)

            # ✅ 使用boost_val的误差（不是eval）
            errors = np.abs(pred_val - y_true_val)
            weighted_error = np.sum(sample_weights * errors)
            stage_results.append((model, weighted_error, pred_val))

        # ✅ 基于boost_val选择最佳模型
        best_model, best_error, best_pred = min(stage_results, key=lambda x: x[1])

        # 计算模型权重
        beta = best_error / (1 - best_error)
        model_weight = np.log(1 / beta)

        # ✅ 贪心选择（基于boost_val的RMSE）
        # 检查是否改善了整体表现
        if improves_performance:
            best_chain.append(best_model)
            best_weights[best_model] = model_weight

            # 更新样本权重（for next iteration）
            errors = np.abs(best_pred - y_true_val)
            sample_weights *= np.power(beta, 1 - normalized_errors)
            sample_weights /= np.sum(sample_weights)

    # Step 3: 在完整train上重新训练选中的模型链
    final_models = {}
    for model in best_chain:
        final_models[model] = model.fit(train_range)

    # Step 4: 在独立的eval集上评估
    eval_predictions = {}
    for model in best_chain:
        eval_predictions[model] = final_models[model].predict(eval_range)

    # 加权组合
    y_pred = sum(best_weights[m] * eval_predictions[m] for m in best_chain)
    y_true = y[eval_range]  # ✅ 独立测试集
    metrics = evaluate(y_true, y_pred)  # ✅ 真实性能
```

---

## 📊 总结对比

| 模型 | 状态 | 数据泄露类型 | 影响 |
|------|------|------------|------|
| **Stacking** | ✅ 已修复 | 元模型在测试集上训练和评估 | 性能指标过于乐观 |
| **Weighted Avg** | ❌ 需修复 | 权重基于测试集方差计算 | 权重不可靠，性能指标不可信 |
| **Boosting** | ❌ 需修复 | 所有选择和权重基于测试集 | 模型链选择有偏，性能严重过估 |

---

## 🎯 修复优先级

**严重性：高 🔴🔴🔴**

### 影响范围：
1. **教学正确性** - 学生学到错误的集成学习方法
2. **性能可信度** - 所有评估指标都过于乐观
3. **理论一致性** - 不符合机器学习教科书定义

### 建议action plan：
1. ✅ **Stacking** - 已完成修复和测试
2. ⏭️ **Weighted Average** - 立即修复（相对简单）
3. ⏭️ **Boosting** - 立即修复（较复杂，需仔细处理贪心选择）

### 修复后验证：
- [ ] 更新weighted_avg.py的train_and_evaluate函数
- [ ] 更新boosting.py的train_and_evaluate函数
- [ ] 创建robustness测试验证修复
- [ ] 确保所有模型符合教科书定义

---

## 📝 参考文献

**Stacking:**
- Wolpert, D. H. (1992). "Stacked generalization". Neural Networks.

**Weighted Average:**
- Ensemble methods教科书中的Inverse Variance Weighting
- 交叉验证用于权重估计

**Boosting:**
- Freund, Y., & Schapire, R. E. (1997). "A decision-theoretic generalization of on-line learning and an application to boosting". Journal of Computer and System Sciences.
- 标准AdaBoost算法流程
