#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates a moderately complex time series dataset for testing.
"""
import json
import numpy as np
import pandas as pd
import os

def generate_complex_series(n=120):
    """
    Generates a time series with trend, seasonality, cycles, and noise.
    """
    # 1. Trend: Linear upward trend
    trend = np.linspace(100, 200, n)
    
    # 2. Seasonality: Strong yearly (12-month) seasonality
    time = np.arange(n)
    seasonality = 25 * np.sin(2 * np.pi * time / 12) + 15 * np.cos(2 * np.pi * time / 24)
    
    # 3. Cycle: A longer-term business cycle (e.g., 40 months)
    cycle = 15 * np.sin(2 * np.pi * time / 40)
    
    # 4. Noise: Random noise
    noise = np.random.normal(0, 5, n)
    
    # --- Combine to create the target variable 'sales' ---
    sales = trend + seasonality + cycle + noise
    
    # --- Create Features ---
    
    # 1. Numeric Feature: 'marketing_spend' - correlated with sales but with a lag and noise
    marketing_spend = np.roll(sales, 2) * 0.2 + np.random.normal(0, 10, n) + 20
    marketing_spend[:2] = marketing_spend[2] # Fill initial NaNs
    
    # 2. Numeric Feature: 'competitor_price' - independent pattern
    competitor_price = 99 - 10 * np.sin(2 * np.pi * time / 30) + np.random.normal(0, 2, n)
    
    # 3. Categorical Feature: 'promotion_type'
    promo_types = ['None', 'Discount', 'BOGO', 'Holiday Special']
    # Holiday special in December, Discount/BOGO more frequent in other months
    promo_cycle = [promo_types[i % 4] for i in range(n)]
    for i in range(n):
        if (i % 12) == 11: # December
            promo_cycle[i] = 'Holiday Special'
        elif np.random.rand() < 0.3:
             promo_cycle[i] = 'Discount'
        elif np.random.rand() < 0.15:
             promo_cycle[i] = 'BOGO'
        else:
             promo_cycle[i] = 'None'


    # --- Create DataFrame ---
    df = pd.DataFrame({
        'month': pd.to_datetime(pd.date_range(start='2020-01-01', periods=n, freq='MS')),
        'sales': np.round(sales, 2),
        'marketing_spend': np.round(marketing_spend, 2),
        'competitor_price': np.round(competitor_price, 2),
        'promotion_type': promo_cycle
    })
    
    # Convert month to string for JSON compatibility
    df['month'] = df['month'].dt.strftime('%Y-%m')
    
    return df

if __name__ == "__main__":
    dataset = generate_complex_series(120)
    # Save to a JSON file in the parent directory
    file_path = os.path.join(os.path.dirname(__file__), '..', 'complex_data.json')
    dataset.to_json(file_path, orient='records', indent=2)
    print(f"Successfully generated and saved complex data to {file_path}")
    print("\nSample of first 5 rows:")
    print(dataset.head().to_string())
