# py/src/data_utils.py
# -*- coding: utf-8 -*-
"""
Centralized data parsing and preparation utilities for all time series models.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from logger import log


class ParsedData:
    """
    A container for parsed and prepared time series data, using a pandas DataFrame
    as the underlying data structure to handle mixed data types.
    """


    def __init__(
        self,
        df: pd.DataFrame,
        target_key: str,
        time_key: Optional[str] = None,
        feature_keys: Optional[List[str]] = None,
    ):
        self.df = df
        self.target_key = target_key
        self.time_key = time_key
        self.feature_keys = feature_keys or []

    @property
    def n_obs(self) -> int:
        return len(self.df)

    @property
    def y(self) -> np.ndarray:
        """Returns the target variable as a NumPy array."""
        return self.df[self.target_key].values

    @property
    def X(self) -> Optional[np.ndarray]:
        """Returns the feature variables as a NumPy array, if they exist."""
        if not self.feature_keys:
            return None
        return self.df[self.feature_keys].values

    @property
    def y_featured(self) -> np.ndarray:
        """Returns y and X combined, with y as the first column."""
        if not self.feature_keys:
            return self.y.reshape(-1, 1)
        
        cols = [self.target_key] + self.feature_keys
        return self.df[cols].values

    @property
    def timestamps(self) -> Optional[List[str]]:
        if self.time_key:
            return self.df[self.time_key].astype(str).tolist()
        return None


def _parse_records(
    records: List[Dict[str, Any]],
    target_key: str,
    time_key: Optional[str] = None,
    feature_keys: Optional[List[str]] = None,
) -> ParsedData:
    """
    Internal helper to process a list of records into a ParsedData object.
    """
    df = pd.DataFrame(records)

    # Verify all necessary keys exist
    required_keys = [target_key] + (feature_keys or [])
    for key in required_keys:
        if key not in df.columns:
            raise ValueError(f"Error processing records: missing key '{key}'")

    # Attempt to convert numeric columns, coercing errors
    numeric_cols = [target_key] + (feature_keys or [])
    for col in numeric_cols:
        # This check is to avoid converting categorical features that are passed in feature_keys
        if pd.api.types.is_numeric_dtype(df[col]):
             df[col] = pd.to_numeric(df[col], errors='coerce')

    if df[required_keys].isnull().values.any():
        raise ValueError("Error processing records: contains null or non-numeric values in required columns.")

    return ParsedData(
        df=df,
        target_key=target_key,
        time_key=time_key,
        feature_keys=feature_keys,
    )


def load_data_from_json(
    json_string: str,
    target_key: Optional[str] = None,
    time_key: Optional[str] = None,
    feature_keys: Optional[List[str]] = None,
) -> ParsedData:
    """
    High-level function to parse a JSON string, automatically detect keys if not provided,
    and return a structured ParsedData object.
    """
    log.info(f"load_data_from_json received target_key: {target_key}, feature_keys: {feature_keys}")
    
    try:
        records = json.loads(json_string)
    except json.JSONDecodeError as e:
        log.error(f"JSONDecodeError: {e}")
        raise ValueError(f"Invalid JSON format: {e}")

    if not isinstance(records, list) or not records:
        log.error("JSON data is not a non-empty array of objects.")
        raise ValueError("JSON data must be a non-empty array of objects.")

    df = pd.DataFrame(records)
    log.debug(f"Initial DataFrame columns: {df.columns.tolist()}")

    # Try to find a default target key if not provided
    if not target_key:
        default_keys = ['y', 'sales', '销售数量']
        for key in default_keys:
            if key in df.columns:
                target_key = key
                log.info(f"Auto-detected target_key: '{target_key}'")
                break
    
    if not target_key or target_key not in df.columns:
        log.error(f"Target key '{target_key}' not found. Available columns: {df.columns.tolist()}")
        raise ValueError(f"Cannot determine target key. Please specify one. Available columns: {df.columns.tolist()}")

    # Auto-detect time_key and create composite timestamp if possible
    if time_key is None:
        if "年份" in df.columns and "月份" in df.columns:
            log.info("Found '年份' and '月份' columns, attempting to create composite timestamp.")
            try:
                # Ensure year and month are clean strings before combining
                year_str = df["年份"].astype(str)
                month_str = pd.to_numeric(df["月份"], errors='coerce').apply(lambda x: f"{int(x):02d}")
                
                if month_str.isnull().any():
                    log.warning("Could not create composite timestamp due to non-numeric '月份' values. Falling back.")
                    raise ValueError("Non-numeric month values")

                df["timestamp"] = year_str + "-" + month_str
                time_key = "timestamp"
                log.info(f"Created composite time_key: '{time_key}'")
            except Exception as e:
                log.warning(f"Could not create composite timestamp: {e}. Falling back to simpler time key.")
                if "月份" in df.columns: time_key = "月份"
                elif "年份" in df.columns: time_key = "年份"

        elif "month" in df.columns:
            time_key = "month"
        elif "date" in df.columns:
            time_key = "date"
        
        if time_key and time_key not in ["timestamp"]:
             log.info(f"Auto-detected time_key: '{time_key}'")

    # Auto-detect feature keys if they are not provided
    if feature_keys is None:
        all_keys = df.columns.tolist()
        keys_to_exclude = [target_key, time_key, "年份", "月份"]
        detected_features = [k for k in all_keys if k not in keys_to_exclude]
        feature_keys = detected_features if detected_features else []
        log.info(f"Auto-detected feature_keys: {feature_keys}")
        
    # Explicitly convert all feature and target columns to numeric, coercing errors
    cols_to_convert = [target_key] + [fk for fk in feature_keys if fk != target_key]
    log.debug(f"Attempting to convert columns to numeric: {cols_to_convert}")
    for col in cols_to_convert:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            original_non_nulls = df[col].notnull().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().sum() > original_non_nulls:
                log.error(f"Column '{col}' contains non-numeric values that were coerced to NaN.")
                raise ValueError(f"Column '{col}' contains non-numeric values that cannot be converted.")

    # --- CRITICAL: Filter the DataFrame to only the used columns ---
    columns_to_keep = [target_key]
    if time_key:
        columns_to_keep.append(time_key)
    if feature_keys:
        columns_to_keep.extend([fk for fk in feature_keys if fk not in columns_to_keep])
    
    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        log.error(f"Specified or inferred columns not found in dataset: {missing_cols}")
        raise ValueError(f"Specified or inferred columns not found in dataset: {missing_cols}")
        
    df_filtered = df[columns_to_keep]
    log.info(f"Final columns after filtering: {df_filtered.columns.tolist()}")
    
    if time_key == "timestamp" and "年份" in df_filtered.columns and "月份" in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=["年份", "月份"])
        log.info("Dropped original '年份' and '月份' columns.")

    return ParsedData(
        df=df_filtered,
        target_key=target_key,
        time_key=time_key,
        feature_keys=feature_keys,
    )
