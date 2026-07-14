# py/src/data_utils.py
# -*- coding: utf-8 -*-
"""
Centralized data parsing and preparation utilities for all time series models.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

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


def build_numeric_subprocess_records(parsed_data: "ParsedData") -> Tuple[List[Dict[str, float]], List[str]]:
    """
    Build row dicts for subprocess-based models and keep only numeric features.

    Some datasets include categorical metadata columns (e.g. 行业/公司/产品名称) that
    cannot be converted to float directly. This helper keeps numeric columns and skips
    categorical columns to avoid runtime float-conversion failures in ensemble models.
    """
    numeric_feature_keys: List[str] = []

    for feat_key in parsed_data.feature_keys:
        if feat_key not in parsed_data.df.columns:
            continue

        series = parsed_data.df[feat_key]
        if pd.api.types.is_numeric_dtype(series):
            numeric_feature_keys.append(feat_key)
            continue

        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().all():
            parsed_data.df[feat_key] = converted
            numeric_feature_keys.append(feat_key)
            continue

        log.warning("Skipping non-numeric feature for subprocess model: %s", feat_key)

    data_dicts: List[Dict[str, float]] = []
    for i in range(parsed_data.n_obs):
        row: Dict[str, float] = {parsed_data.target_key: float(parsed_data.y[i])}
        for feat_key in numeric_feature_keys:
            row[feat_key] = float(parsed_data.df.iloc[i][feat_key])
        data_dicts.append(row)

    return data_dicts, numeric_feature_keys


def _to_json_scalar(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-serializable Python scalars."""
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None

    # Keep native JSON scalar types unchanged.
    if isinstance(value, (str, int, float, bool)):
        return value

    # Fallback for pandas NA / other scalar-like values.
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


def build_subprocess_records(
    parsed_data: "ParsedData",
    feature_keys: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build subprocess row dicts while preserving categorical feature values.

    This is primarily for models (e.g. LSTM) that can consume categorical
    features via their own preprocessing pipeline.
    """
    source_keys = feature_keys if feature_keys is not None else parsed_data.feature_keys
    selected_feature_keys: List[str] = []

    for feat_key in source_keys:
        if not feat_key or feat_key == parsed_data.target_key:
            continue
        if feat_key not in parsed_data.df.columns:
            log.warning("Feature not found in dataframe, skipping: %s", feat_key)
            continue
        if feat_key in selected_feature_keys:
            continue
        selected_feature_keys.append(feat_key)

    # Convert numeric-like string columns to numeric, keep true categorical columns.
    for feat_key in selected_feature_keys:
        series = parsed_data.df[feat_key]
        if pd.api.types.is_numeric_dtype(series):
            continue
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().all():
            parsed_data.df[feat_key] = converted

    data_dicts: List[Dict[str, Any]] = []
    for i in range(parsed_data.n_obs):
        row: Dict[str, Any] = {parsed_data.target_key: float(parsed_data.y[i])}
        for feat_key in selected_feature_keys:
            row[feat_key] = _to_json_scalar(parsed_data.df.iloc[i][feat_key])
        data_dicts.append(row)

    return data_dicts, selected_feature_keys


_CATEGORICAL_NAME_HINTS = (
    "区域",
    "代码",
    "分类",
    "分组",
    "品类",
    "地区",
    "天气",
    "属性",
    "序号",
    "形态",
    "渠道",
    "类型",
    "编号",
    "编码",
    "类别",
    "等级",
    "组别",
)

_CATEGORICAL_TOKEN_HINTS = (
    "category",
    "class",
    "code",
    "flag",
    "group",
    "id",
    "label",
    "region",
    "segment",
    "channel",
    "type",
    "weather",
)

_CALENDAR_CATEGORICAL_NAME_HINTS = (
    "星期",
    "月份",
    "季度",
    "周次",
)

_CALENDAR_CATEGORICAL_TOKEN_HINTS = (
    "dayofweek",
    "dow",
    "month",
    "quarter",
    "weekday",
    "weekofyear",
)

_NUMERIC_NAME_HINTS = (
    "价格",
    "促销",
    "利用率",
    "搜索",
    "指数",
    "年份",
    "库存",
    "强度",
    "投入",
    "数量",
    "销量",
    "销售",
)

_NUMERIC_TOKEN_HINTS = (
    "year",
    "price",
    "sales",
    "inventory",
    "stock",
    "trend",
    "utilization",
    "value",
    "volume",
    "yearly",
)


def _normalize_feature_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _tokenize_feature_name(name: str) -> List[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", normalized.lower())
    return [token for token in tokens if token]


def _is_integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return False
    return bool(np.all(np.isclose(finite, np.round(finite), atol=1e-9)))


def _infer_single_feature_type(feature_name: str, series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "categorical"

    normalized_name = _normalize_feature_name(feature_name)
    name_tokens = set(_tokenize_feature_name(feature_name))
    numeric_series = series
    if not pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce")
        if not numeric_series.notna().all():
            return "categorical"

    non_null = pd.Series(numeric_series).dropna()
    if non_null.empty:
        return "numeric"

    unique_count = int(non_null.nunique(dropna=True))
    unique_ratio = unique_count / len(non_null)
    integer_like = _is_integer_like(non_null)

    if any(hint in normalized_name for hint in _NUMERIC_NAME_HINTS) or any(
        hint in name_tokens for hint in _NUMERIC_TOKEN_HINTS
    ):
        return "numeric"

    if any(hint in normalized_name for hint in _CALENDAR_CATEGORICAL_NAME_HINTS) or any(
        hint in name_tokens for hint in _CALENDAR_CATEGORICAL_TOKEN_HINTS
    ):
        return "categorical" if integer_like and unique_count <= 31 else "numeric"

    if any(hint in normalized_name for hint in _CATEGORICAL_NAME_HINTS) or any(
        hint in name_tokens for hint in _CATEGORICAL_TOKEN_HINTS
    ):
        return "categorical"

    low_cardinality_threshold = max(4, min(12, int(np.sqrt(len(non_null))) + 1))
    if integer_like and unique_count <= low_cardinality_threshold and unique_ratio <= 0.3:
        return "categorical"

    return "numeric"


def infer_feature_types(
    parsed_data: "ParsedData",
    target_key: Optional[str] = None,
    feature_keys: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Infer feature types from both observed values and feature-name heuristics.

    Numeric-like string columns are treated as numeric after coercion.
    """
    resolved_target_key = target_key or parsed_data.target_key
    source_keys = feature_keys if feature_keys is not None else parsed_data.feature_keys

    inferred: Dict[str, str] = {resolved_target_key: "numeric"}
    seen: set[str] = set()

    for feat_key in source_keys:
        if not feat_key or feat_key == resolved_target_key or feat_key in seen:
            continue
        seen.add(feat_key)

        if feat_key not in parsed_data.df.columns:
            log.warning("Feature not found in dataframe when inferring feature_types: %s", feat_key)
            continue

        series = parsed_data.df[feat_key]
        inferred_type = _infer_single_feature_type(feat_key, series)
        if inferred_type == "numeric" and not pd.api.types.is_numeric_dtype(series):
            converted = pd.to_numeric(series, errors="coerce")
            parsed_data.df[feat_key] = converted
        inferred[feat_key] = inferred_type

    return inferred


def _data_error(message: str) -> ValueError:
    if message.startswith("[DATA]"):
        return ValueError(message)
    return ValueError(f"[DATA] {message}")


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
        raise _data_error(f"Invalid JSON format: {e}")

    if not isinstance(records, list) or not records:
        log.error("JSON data is not a non-empty array of objects.")
        raise _data_error("JSON data must be a non-empty array of objects.")

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
        raise _data_error(f"Cannot determine target key. Please specify one. Available columns: {df.columns.tolist()}")

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

                # Convert 年份 and 月份 to numeric for use as features
                df["年份"] = pd.to_numeric(df["年份"], errors='coerce')
                df["月份"] = pd.to_numeric(df["月份"], errors='coerce')
                log.info("Converted '年份' and '月份' to numeric types for use as features")
            except Exception as e:
                log.warning(f"Could not create composite timestamp: {e}. Falling back to simpler time key.")
                if "月份" in df.columns:
                    time_key = "月份"
                elif "年份" in df.columns:
                    time_key = "年份"

        elif "month" in df.columns:
            time_key = "month"
        elif "date" in df.columns:
            time_key = "date"

        if time_key and time_key not in ["timestamp"]:
             log.info(f"Auto-detected time_key: '{time_key}'")

    # Auto-detect feature keys if they are not provided
    if feature_keys is None:
        all_keys = df.columns.tolist()
        # Only exclude target_key and time_key, keep 年份 and 月份 as numeric features
        keys_to_exclude = [target_key, time_key]
        detected_features = [k for k in all_keys if k not in keys_to_exclude]
        feature_keys = detected_features if detected_features else []
        log.info(f"Auto-detected feature_keys: {feature_keys}")
        
    # Target column must be numeric
    if target_key in df.columns and not pd.api.types.is_numeric_dtype(df[target_key]):
        df[target_key] = pd.to_numeric(df[target_key], errors='coerce')
        if df[target_key].isnull().any():
            raise _data_error(f"Target column '{target_key}' contains non-numeric values.")
        log.debug(f"Converted target column '{target_key}' to numeric")

    # Feature columns: convert if numeric-like, keep as-is if categorical
    for col in [fk for fk in (feature_keys or []) if fk != target_key and fk in df.columns]:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.isnull().sum() <= df[col].isnull().sum():
            df[col] = converted
            log.debug(f"Converted feature column '{col}' to numeric")
        else:
            log.debug(f"Keeping feature column '{col}' as categorical (dtype: {df[col].dtype})")

    # --- CRITICAL: Filter the DataFrame to only the used columns ---
    columns_to_keep = [target_key]
    if time_key:
        columns_to_keep.append(time_key)
    if feature_keys:
        columns_to_keep.extend([fk for fk in feature_keys if fk not in columns_to_keep])
    
    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        log.error(f"Specified or inferred columns not found in dataset: {missing_cols}")
        raise _data_error(f"Specified or inferred columns not found in dataset: {missing_cols}")
        
    df_filtered = df[columns_to_keep]
    log.info(f"Final columns after filtering: {df_filtered.columns.tolist()}")

    # Keep 年份 and 月份 as numeric features (no longer dropping them)
    # This allows models like LSTM to use month for seasonality

    return ParsedData(
        df=df_filtered,
        target_key=target_key,
        time_key=time_key,
        feature_keys=feature_keys,
    )
