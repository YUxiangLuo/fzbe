-- --------------------------------------------------------
-- Virtual Simulation System Seed Data File
-- Test data for development and testing
-- All passwords: $2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG
-- --------------------------------------------------------

-- Clear existing data (in reverse order of dependencies)
SET FOREIGN_KEY_CHECKS = 0;

TRUNCATE TABLE `experiment_step_events`;
TRUNCATE TABLE `quiz_answers`;
TRUNCATE TABLE `experiment_reports`;
TRUNCATE TABLE `experiment_grade`;
TRUNCATE TABLE `experiment_status`;
TRUNCATE TABLE `question_bank`;
TRUNCATE TABLE `grade_weights`;
TRUNCATE TABLE `dataset_segments`;
TRUNCATE TABLE `datasets`;
TRUNCATE TABLE `experiment_manuals`;
TRUNCATE TABLE `class_assistants`;
TRUNCATE TABLE `enrollments`;
TRUNCATE TABLE `classes`;
TRUNCATE TABLE `users`;

SET FOREIGN_KEY_CHECKS = 1;

-- ========================================================
-- 1. Users
-- ========================================================

INSERT INTO `users` (`user_id`, `username`, `password_hash`, `full_name`, `email`, `phone_number`, `role`, `created_at`) VALUES
(1, 'admin', '$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG', 'System Administrator', 'admin@university.edu', '010-1234-5678', 'Admin', '2024-01-01 09:00:00');

-- ========================================================
-- 2. Question Bank
-- ========================================================

INSERT INTO `question_bank` (`question_id`, `knowledge_point`, `question_type`, `question_text`, `options`, `correct_answers`, `creator_id`) VALUES
-- Forecasting Model Questions (20)
(1, '预测模型-时间序列基础', 'Single Choice', '以下哪项不是时间序列的经典组成部分？', '{"A":"趋势 (Trend)","B":"季节性 (Seasonality)","C":"周期性 (Cyclicality)","D":"回归 (Regression)"}', '["D"]', 1),
(2, '预测模型-时间序列基础', 'True/False', '平稳时间序列的均值和方差不随时间变化。', '["正确","错误"]', '["正确"]', 1),
(3, '预测模型-时间序列基础', 'Single Choice', 'ADF检验（Augmented Dickey-Fuller test）的主要用途是什么？', '{"A":"检验序列的自相关性","B":"检验序列的正态性","C":"检验序列的平稳性","D":"检验模型的拟合优度"}', '["C"]', 1),
(4, '预测模型-模型评估指标', 'Multiple Choice', '以下哪些是评估时间序列预测准确性的常用指标？', '{"A":"RMSE (均方根误差)","B":"MAE (平均绝对误差)","C":"R-squared (决定系数)","D":"AUC (曲线下面积)"}', '["A","B","C"]', 1),
(5, '预测模型-模型评估指标', 'Single Choice', 'RMSE和MAE这两个指标，哪个对异常值（outliers）更敏感？', '{"A":"RMSE","B":"MAE","C":"两者相同","D":"无法比较"}', '["A"]', 1),
(6, '预测模型-移动平均法(MA)', 'Single Choice', '简单移动平均模型的主要缺点是什么？', '{"A":"计算复杂","B":"不适用于季节性数据","C":"对历史数据赋予相同权重，滞后于趋势变化","D":"需要大量历史数据"}', '["C"]', 1),
(7, '预测模型-指数平滑法(ES)', 'Single Choice', '在简单指数平滑中，较高的alpha（平滑常数）值意味着什么？', '{"A":"对旧观测值赋予更多权重","B":"对近期观测值赋予更多权重","C":"对变化响应较慢","D":"预测结果更平滑"}', '["B"]', 1),
(8, '预测模型-ARIMA模型', 'Multiple Choice', 'ARIMA(p,d,q)模型通常涉及哪些参数？', '{"A":"自回归阶数 (p)","B":"差分阶数 (d)","C":"移动平均阶数 (q)","D":"季节周期 (s)"}', '["A","B","C"]', 1),
(9, '预测模型-ARIMA模型', 'Single Choice', '在ARIMA模型中，参数“d”的作用是什么？', '{"A":"确定模型的自回归部分","B":"通过差分使序列平稳","C":"确定模型的移动平均部分","D":"调整季节性影响"}', '["B"]', 1),
(10, '预测模型-ARIMA模型', 'True/False', 'ACF图（自相关函数图）和PACF图（偏自相关函数图）可用于帮助确定ARIMA模型的p和q值。', '["正确","错误"]', '["正确"]', 1),
(11, '预测模型-LSTM深度学习', 'True/False', 'LSTM网络是一种特殊的循环神经网络（RNN），能有效解决长期依赖问题。', '["正确","错误"]', '["正确"]', 1),
(12, '预测模型-LSTM深度学习', 'Single Choice', '以下哪个不是LSTM单元中的“门”结构？', '{"A":"输入门 (Input Gate)","B":"遗忘门 (Forget Gate)","C":"输出门 (Output Gate)","D":"处理门 (Process Gate)"}', '["D"]', 1),
(13, '预测模型-LSTM深度学习', 'True/False', '在训练LSTM模型之前，对输入数据进行归一化（Normalization）通常是一个好习惯，可以加速模型收敛。', '["正确","错误"]', '["正确"]', 1),
(14, '预测模型-模型融合技术', 'Single Choice', '将多个不同模型的预测结果进行加权平均，属于哪种集成学习方法？', '{"A":"Boosting","B":"Stacking","C":"Bagging","D":"Ensemble Weighted Average"}', '["D"]', 1),
(15, '预测模型-模型融合技术', 'True/False', 'Stacking集成模型的思想是训练一个“元模型”（meta-model）来学习如何最好地结合基模型的预测结果。', '["正确","错误"]', '["正确"]', 1),
(16, '预测模型-时间序列基础', 'Single Choice', '“白噪声”序列具有以下哪个特征？', '{"A":"强烈的趋势性","B":"固定的季节模式","C":"序列值之间没有相关性","D":"方差随时间增大"}', '["C"]', 1),
(17, '预测模型-模型评估指标', 'Single Choice', '如果一个模型的预测值总是系统性地高于或低于实际值，这表明模型存在什么问题？', '{"A":"高方差 (High Variance)","B":"高偏差 (High Bias)","C":"过拟合 (Overfitting)","D":"欠拟合 (Underfitting)"}', '["B"]', 1),
(18, '预测模型-ARIMA模型', 'Single Choice', 'ADF检验的原假设（Null Hypothesis）是什么？', '{"A":"序列是平稳的","B":"序列存在单位根（即不平稳）","C":"序列服从正态分布","D":"序列没有自相关性"}', '["B"]', 1),
(19, '预测模型-指数平滑法(ES)', 'Multiple Choice', 'Holt-Winters指数平滑方法可以处理时间序列中的哪些成分？', '{"A":"水平 (Level)","B":"趋势 (Trend)","C":"季节性 (Seasonality)","D":"周期性 (Cyclicality)"}', '["A","B","C"]', 1),
(20, '预测模型-模型评估指标', 'True/False', '交叉验证（Cross-validation）是一种有效评估模型在独立数据集上表现的方法。', '["正确","错误"]', '["正确"]', 1),

-- Production Planning Questions (20)
(21, '生产计划-MPS基本原理', 'Single Choice', '主生产计划（MPS）的主要目的是什么？', '{"A":"详细说明车间的日常生产活动","B":"规划企业长期的产能需求","C":"明确在特定时间段内要生产的最终产品的数量和时间","D":"管理原材料的采购"}', '["C"]', 1),
(22, '生产计划-MPS基本原理', 'Single Choice', '物料需求计划（MRP）的核心输入不包括以下哪项？', '{"A":"主生产计划 (MPS)","B":"物料清单 (BOM)","C":"库存状态文件 (Inventory Records)","D":"销售人员绩效报告"}', '["D"]', 1),
(23, '生产计划-库存与缺货', 'Multiple Choice', '以下哪些是与库存相关的常见成本？', '{"A":"持有成本 (Holding Costs)","B":"订购成本 (Ordering Costs)","C":"缺货成本 (Shortage Costs)","D":"营销成本 (Marketing Costs)"}', '["A","B","C"]', 1),
(24, '生产计划-服务水平评估', 'Single Choice', '在库存管理中，95%的客户服务水平通常意味着什么？', '{"A":"95%的客户对产品质量满意","B":"客户95%的需求可以从现有库存中得到满足，无需等待","C":"库存年周转率为95%","D":"生产线的效率为95%"}', '["B"]', 1),
(25, '生产计划-安全库存策略', 'True/False', '安全库存（Safety Stock）是为了应对需求或供应提前期的不确定性而额外持有的库存。', '["正确","错误"]', '["正确"]', 1),
(26, '生产计划-库存与缺货', 'Single Choice', '经济订货批量（EOQ）模型旨在找到一个最佳订货量，使得什么达到最小？', '{"A":"年订购成本","B":"年持有成本","C":"年总库存成本（订购成本+持有成本）","D":"缺货风险"}', '["C"]', 1),
(27, '生产计划-MPS基本原理', 'Single Choice', '“提前期”（Lead Time）指的是什么？', '{"A":"产品在仓库中存储的时间","B":"从下达订单到收到货物的总时间","C":"生产一个产品所需的总工时","D":"客户等待产品送达的时间"}', '["B"]', 1),
(28, '生产计划-产能规划', 'Single Choice', '粗能力计划（RCCP）是用来验证什么的可行性的？', '{"A":"物料需求计划 (MRP)","B":"主生产计划 (MPS)","C":"车间作业计划","D":"采购计划"}', '["B"]', 1),
(29, '生产计划-MPS基本原理', 'True/False', '物料清单（BOM）描述了一个最终产品所有零部件、原材料的结构关系和数量。', '["正确","错误"]', '["正确"]', 1),
(30, '生产计划-库存与缺货', 'Single Choice', 'ABC库存分类法是根据什么原则对物料进行分类管理的？', '{"A":"物料的重量和体积","B":"物料的采购难度","C":"帕累托原则（80/20法则），即根据物料的年度消耗金额","D":"物料的保质期"}', '["C"]', 1),
(31, '生产计划-MPS基本原理', 'Single Choice', '需求预测是制定生产计划的起点，它通常分为长期、中期和短期预测。主生产计划（MPS）主要依赖于哪种预测？', '{"A":"长期预测","B":"中期预测","C":"短期预测","D":"以上所有"}', '["B"]', 1),
(32, '生产计划-产能规划', 'True/False', '产能（Capacity）是指一个生产系统在特定时间内的最大产出能力。', '["正确","错误"]', '["正确"]', 1),
(33, '生产计划-安全库存策略', 'Multiple Choice', '影响安全库存水平的因素主要有哪些？', '{"A":"需求的波动性","B":"供应提前期的稳定性","C":"期望的客户服务水平","D":"产品的销售价格"}', '["A","B","C"]', 1),
(34, '生产计划-MPS基本原理', 'Single Choice', '在MRP系统中，“净需求”是如何计算的？', '{"A":"毛需求 - 计划接收量","B":"毛需求 - 现有库存","C":"毛需求 - (现有库存 + 计划接收量)","D":"毛需求 + 安全库存"}', '["C"]', 1),
(35, '生产计划-MPS基本原理', 'Single Choice', '以下哪项是用来平衡需求与供给的短期计划工具？', '{"A":"销售与运营计划 (S&OP)","B":"主生产计划 (MPS)","C":"物料需求计划 (MRP)","D":"每日派工单"}', '["B"]', 1),
(36, '生产计划-库存与缺货', 'True/False', '“在制品”（Work-in-Process, WIP）库存指的是已经进入生产过程但尚未完成的产品。', '["正确","错误"]', '["正确"]', 1),
(37, '生产计划-MPS基本原理', 'Single Choice', '“需求管理”在生产计划体系中的主要作用是什么？', '{"A":"仅仅是进行销售预测","B":"控制生产成本","C":"平衡客户需求与企业供应能力，是计划的源头","D":"管理供应商"}', '["C"]', 1),
(38, '生产计划-产能规划', 'Multiple Choice', '提高生产线产能的常见方法有哪些？', '{"A":"增加设备或人力","B":"改进生产流程，消除瓶颈","C":"实行加班或多班制","D":"降低产品质量标准"}', '["A","B","C"]', 1),
(39, '生产计划-安全库存策略', 'Single Choice', '“再订货点”（Reorder Point）指的是库存水平下降到哪个点时需要下达新的订单？', '{"A":"零库存","B":"足以覆盖提前期内需求的库存量","C":"足以覆盖提前期内需求并加上安全库存的量","D":"最大库存水平"}', '["C"]', 1),
(40, '生产计划-MPS基本原理', 'True/False', '一个有效的生产计划系统可以帮助企业降低库存、提高客户满意度并优化资源利用。', '["正确","错误"]', '["正确"]', 1);
