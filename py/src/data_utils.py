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


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Parse numeric-like values while accepting presentation-only commas."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    normalized = series.map(
        lambda value: value.replace(",", "").strip() if isinstance(value, str) else value
    )
    return pd.to_numeric(normalized, errors="coerce")


def _non_finite_mask(series: pd.Series) -> np.ndarray:
    numeric_values = series.to_numpy(dtype=float, na_value=np.nan)
    return ~np.isfinite(numeric_values)


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
        time_frequency: Optional[str] = None,
    ):
        self.df = df
        self.target_key = target_key
        self.time_key = time_key
        self.feature_keys = feature_keys or []
        self.time_frequency = time_frequency

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
            if _non_finite_mask(series).any():
                raise _data_error(f"Feature column '{feat_key}' contains non-finite values.")
            numeric_feature_keys.append(feat_key)
            continue

        converted = _coerce_numeric_series(series)
        if converted.notna().all():
            if _non_finite_mask(converted).any():
                raise _data_error(f"Feature column '{feat_key}' contains non-finite values.")
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
            if _non_finite_mask(series).any():
                raise _data_error(f"Feature column '{feat_key}' contains non-finite values.")
            continue
        converted = _coerce_numeric_series(series)
        if converted.notna().all():
            if _non_finite_mask(converted).any():
                raise _data_error(f"Feature column '{feat_key}' contains non-finite values.")
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
    numeric = _coerce_numeric_series(series)
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
        numeric_series = _coerce_numeric_series(series)
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
            converted = _coerce_numeric_series(series)
            if converted.notna().any() and _non_finite_mask(converted.dropna()).any():
                raise _data_error(f"Feature column '{feat_key}' contains non-finite values.")
            parsed_data.df[feat_key] = converted
        inferred[feat_key] = inferred_type

    return inferred


def _data_error(message: str) -> ValueError:
    if message.startswith("[DATA]"):
        return ValueError(message)
    return ValueError(f"[DATA] {message}")


_YEAR_MONTH_SHAPE_PATTERN = re.compile(r"^\s*(\d{4})[-/](0?[1-9]|1[0-2])\s*$")
_YEAR_MONTH_PATTERN = re.compile(r"^\s*([1-9]\d{3})[-/](0?[1-9]|1[0-2])\s*$")


def _raise_time_axis_error(time_key: str, message: str) -> None:
    raise _data_error(f"Time column '{time_key}' {message}")


def _validate_order_and_spacing(
    values: np.ndarray,
    *,
    time_key: str,
    expected_step: Optional[float] = None,
) -> float:
    """Validate a numeric representation of a time axis without reordering rows."""
    if values.size <= 1:
        return float(expected_step if expected_step is not None else 0.0)

    differences = np.diff(values.astype(float))
    if np.any(np.isclose(differences, 0.0, rtol=0.0, atol=1e-12)):
        _raise_time_axis_error(time_key, "contains duplicate time points.")
    if np.any(differences < 0):
        _raise_time_axis_error(time_key, "must be strictly increasing in input row order.")

    step = float(expected_step if expected_step is not None else differences[0])
    if step <= 0 or not np.allclose(differences, step, rtol=1e-9, atol=1e-12):
        _raise_time_axis_error(
            time_key,
            "must use one fixed frequency with no missing periods.",
        )
    return step


def _validate_time_axis(df: pd.DataFrame, time_key: str) -> str:
    """
    Validate the canonical row order used by every model.

    Calendar months must be consecutive. Numeric indices and timestamp series
    must use one positive fixed step. The function intentionally never sorts:
    silently sorting would change the train/evaluation ranges selected by users.
    """
    if time_key not in df.columns:
        _raise_time_axis_error(time_key, "does not exist.")

    series = df[time_key]
    if series.isnull().any() or series.map(
        lambda value: isinstance(value, str) and not value.strip()
    ).any():
        _raise_time_axis_error(time_key, "contains empty values.")

    string_values = series.astype(str)
    month_matches = string_values.map(_YEAR_MONTH_PATTERN.fullmatch)
    month_shape_matches = string_values.map(_YEAR_MONTH_SHAPE_PATTERN.fullmatch)
    if (month_shape_matches.notna() & month_matches.isna()).any():
        _raise_time_axis_error(
            time_key,
            "contains YYYY-MM values with years outside 1000-9999.",
        )
    if month_matches.notna().all():
        month_ordinals = np.asarray(
            [
                int(match.group(1)) * 12 + int(match.group(2)) - 1
                for match in month_matches
            ],
            dtype=np.int64,
        )
        _validate_order_and_spacing(
            month_ordinals,
            time_key=time_key,
            expected_step=1.0,
        )
        return "monthly"

    numeric = _coerce_numeric_series(series)
    if numeric.notna().all():
        if _non_finite_mask(numeric).any():
            _raise_time_axis_error(time_key, "contains non-finite values.")
        step = _validate_order_and_spacing(
            numeric.to_numpy(dtype=float),
            time_key=time_key,
        )
        return "numeric" if len(numeric) <= 1 else f"numeric_step:{step:g}"

    # Only accept date-like strings if every value parses.  Partial coercion is
    # dangerous because it makes an invalid time label look like missing data.
    try:
        timestamps = pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:  # pandas < 2.0 compatibility
        timestamps = pd.to_datetime(series, errors="coerce")
    if timestamps.isnull().any():
        _raise_time_axis_error(
            time_key,
            "must contain numeric indices, YYYY-MM values, or parseable dates.",
        )

    timestamp_ns = timestamps.astype("int64").to_numpy(dtype=np.int64)
    if timestamp_ns.size > 1:
        differences = np.diff(timestamp_ns)
        if np.any(differences == 0):
            _raise_time_axis_error(time_key, "contains duplicate time points.")
        if np.any(differences < 0):
            _raise_time_axis_error(time_key, "must be strictly increasing in input row order.")

        # Consecutive calendar months are fixed-frequency even though their
        # durations vary between 28 and 31 days.
        month_ordinals = np.asarray(
            [stamp.year * 12 + stamp.month - 1 for stamp in timestamps],
            dtype=np.int64,
        )
        if np.all(np.diff(month_ordinals) == 1):
            return "monthly"

        _validate_order_and_spacing(timestamp_ns, time_key=time_key)

    return "datetime_fixed"


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

    # Auto-detect time_key and create a strict composite timestamp when possible.
    if time_key is None:
        if "年份" in df.columns and "月份" in df.columns:
            combined_month_matches = df["月份"].astype(str).map(_YEAR_MONTH_PATTERN.fullmatch)
            if combined_month_matches.notna().all():
                # Legacy database rows may retain a redundant 年份 header while
                # 月份 already carries the authoritative YYYY-MM value.
                time_key = "月份"
                log.info("Using combined legacy '月份' values as the monthly time axis.")
            else:
                log.info("Found separate '年份' and '月份' columns, creating a composite monthly timestamp.")
                years = _coerce_numeric_series(df["年份"])
                months = _coerce_numeric_series(df["月份"])
                valid_years = (
                    years.notna()
                    & np.isfinite(years)
                    & years.eq(years.round())
                    & (years >= 1000)
                    & (years <= 9999)
                )
                valid_months = (
                    months.notna()
                    & np.isfinite(months)
                    & months.eq(months.round())
                    & (months >= 1)
                    & (months <= 12)
                )
                if not bool(valid_years.all() and valid_months.all()):
                    raise _data_error(
                        "Columns '年份' and '月份' must contain a valid four-digit year and month 1-12."
                    )

                year_values = years.astype(int)
                month_values = months.astype(int)
                df["timestamp"] = [
                    f"{year:04d}-{month:02d}"
                    for year, month in zip(year_values, month_values)
                ]
                time_key = "timestamp"
                df["年份"] = year_values
                df["月份"] = month_values
                log.info(f"Created composite time_key: '{time_key}'")

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
        
    # Target column must be numeric and finite. Run this for numeric dtypes too,
    # because pandas accepts values such as Infinity without producing nulls.
    target_values = _coerce_numeric_series(df[target_key])
    if target_values.isnull().any():
        raise _data_error(f"Target column '{target_key}' contains non-numeric values.")
    if _non_finite_mask(target_values).any():
        raise _data_error(f"Target column '{target_key}' contains non-finite values.")
    df[target_key] = target_values
    log.debug(f"Converted target column '{target_key}' to finite numeric values")

    # Feature columns: convert if numeric-like, keep as-is if categorical
    for col in [fk for fk in (feature_keys or []) if fk != target_key and fk in df.columns]:
        if pd.api.types.is_numeric_dtype(df[col]):
            if _non_finite_mask(df[col]).any():
                raise _data_error(f"Feature column '{col}' contains non-finite values.")
            continue
        converted = _coerce_numeric_series(df[col])
        if converted.isnull().sum() <= df[col].isnull().sum():
            if _non_finite_mask(converted).any():
                raise _data_error(f"Feature column '{col}' contains non-finite values.")
            df[col] = converted
            log.debug(f"Converted feature column '{col}' to numeric")
        else:
            log.debug(f"Keeping feature column '{col}' as categorical (dtype: {df[col].dtype})")

    time_frequency = _validate_time_axis(df, time_key) if time_key else None
    if time_key:
        log.info("Validated time axis '%s' with frequency '%s'", time_key, time_frequency)

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
        time_frequency=time_frequency,
    )
