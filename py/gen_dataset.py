#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成更合理的销售预测数据集

特点：
1. 明确的趋势：逐年增长 ~5%
2. 强季节性：夏季（6-8月）和年末（11-12月）销量高
3. 价格敏感性：价格降低 → 销量增加
4. 促销效果：促销期间销量提升 15-25%
5. 节假日效应：节假日销量提升 10-20%
6. 温度影响：温度适中时销量较好（假设是冷饮类产品）
"""

import json
import math
from datetime import datetime, timedelta

# 基础参数
BASE_SALES = 50000  # 基础销量
TREND_RATE = 0.004  # 月增长率 (~5% 年增长)
NOISE_STD = 0.08    # 噪声标准差

# 季节性系数（1-12月）
SEASONALITY = [
    0.85,  # 1月：冬季低迷
    0.80,  # 2月：最低
    0.90,  # 3月：回升
    1.05,  # 4月：春季
    1.10,  # 5月：旺季开始
    1.25,  # 6月：夏季高峰
    1.35,  # 7月：最高峰
    1.30,  # 8月：持续高位
    1.10,  # 9月：开始回落
    1.05,  # 10月：秋季
    1.20,  # 11月：年末促销
    1.25,  # 12月：圣诞/新年
]

def generate_dataset(start_year=2021, n_years=3):
    """生成 n 年的月度销售数据"""
    import random
    random.seed(42)  # 固定随机种子以保证可重复
    
    data = []
    
    for year_offset in range(n_years):
        year = start_year + year_offset
        
        for month in range(1, 13):
            month_index = len(data)
            
            # 1. 基础销量 + 趋势
            base = BASE_SALES * (1 + TREND_RATE * month_index)
            
            # 2. 季节性
            seasonal_factor = SEASONALITY[month - 1]
            
            # 3. 价格（逐年下降，反映市场竞争）
            base_price = 30.0
            price = base_price - 0.08 * month_index
            price_effect = 1.0 + (base_price - price) * 0.12  # 价格每降1元，销量增12%
            
            # 4. 促销（夏季和年末有促销）
            is_promo = 1 if month in [4, 6, 7, 11, 12] else 0
            promo_effect = 1.20 if is_promo else 1.0
            discount_rate = 0.15 if is_promo else 0.05
            
            # 5. 节假日
            is_holiday = 1 if month in [1, 4, 10, 12] else 0
            holiday_effect = 1.15 if is_holiday else 1.0
            
            # 6. 温度影响（假设产品在温暖天气卖得好）
            temp_map = [2, 4, 9, 15, 20, 25, 30, 29, 23, 17, 10, 5]
            temperature = temp_map[month - 1]
            # 最佳温度 25度，偏离则销量下降
            temp_effect = 1.0 - abs(temperature - 25) * 0.01
            
            # 7. 竞争对手价格（比我们略高）
            competitor_price = price + 0.5
            
            # 8. 综合计算销量
            sales = base * seasonal_factor * price_effect * promo_effect * holiday_effect * temp_effect
            
            # 9. 添加随机噪声
            noise = random.gauss(0, NOISE_STD)
            sales = sales * (1 + noise)
            
            # 10. 四舍五入
            sales = round(sales)
            
            # 构建数据记录
            record = {
                "ts": f"{year}-{month:02d}",
                "y": sales,
                "x": {
                    "price": round(price, 2),
                    "discount_rate": round(discount_rate, 2),
                    "promo": is_promo,
                    "is_holiday": is_holiday,
                    "temperature": round(temperature, 1),
                    "competitor_price": round(competitor_price, 2)
                }
            }
            
            data.append(record)
    
    return data


def generate_simple_format(data):
    """生成简单格式（只有 month 和 sales，供 ARIMA 使用）"""
    return [{"month": d["ts"], "sales": d["y"]} for d in data]


def print_statistics(data):
    """打印数据集统计信息"""
    sales = [d["y"] for d in data]
    print(f"数据点数: {len(data)}")
    print(f"销量范围: {min(sales):,} - {max(sales):,}")
    print(f"平均销量: {sum(sales)/len(sales):,.0f}")
    print(f"总增长率: {(sales[-1]/sales[0] - 1)*100:.1f}%")
    
    # 季节性验证
    print("\n月度平均销量（验证季节性）:")
    for m in range(1, 13):
        month_sales = [d["y"] for d in data if d["ts"].endswith(f"-{m:02d}")]
        avg = sum(month_sales) / len(month_sales)
        print(f"  {m:2d}月: {avg:>8,.0f}")


if __name__ == "__main__":
    # 生成 3 年数据（36 个月）
    data = generate_dataset(start_year=2021, n_years=3)
    
    print("=" * 60)
    print("数据集统计信息")
    print("=" * 60)
    print_statistics(data)
    
    # 输出 LSTM 格式（完整特征）
    print("\n" + "=" * 60)
    print("LSTM 格式数据（含特征）")
    print("=" * 60)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    # 输出 ARIMA 格式（仅时间序列）
    print("\n" + "=" * 60)
    print("ARIMA 格式数据（仅销量）")
    print("=" * 60)
    simple_data = generate_simple_format(data)
    print(json.dumps(simple_data, indent=2, ensure_ascii=False))
    
    # 建议的训练/验证/测试划分
    print("\n" + "=" * 60)
    print("建议的数据划分")
    print("=" * 60)
    n = len(data)
    train_end = int(n * 0.7)      # 70% 训练
    val_end = int(n * 0.85)       # 15% 验证
    print(f"总数据点: {n}")
    print(f"训练集: [0, {train_end-1}]  ({train_end} 个月)")
    print(f"验证集: [{train_end}, {val_end-1}]  ({val_end - train_end} 个月)")
    print(f"测试集: [{val_end}, {n-1}]  ({n - val_end} 个月)")
    print(f"\n训练集月份: {data[0]['ts']} 至 {data[train_end-1]['ts']}")
    print(f"验证集月份: {data[train_end]['ts']} 至 {data[val_end-1]['ts']}")
    print(f"测试集月份: {data[val_end]['ts']} 至 {data[n-1]['ts']}")