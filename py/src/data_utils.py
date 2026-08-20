# py/src/data_utils.py
# -*- coding: utf-8 -*-
"""
Centralized data parsing and preparation utilities for all time series models.
"""

from __future__ import annotations

import hashlib
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
        source_records: Optional[List[Dict[str, Any]]] = None,
    ):
        self.df = df
        self.target_key = target_key
        self.time_key = time_key
        self.feature_keys = feature_keys or []
        self.source_records = source_records

    def source_value(self, row_index: int, column: str) -> Any:
        """Return the per-row JSON scalar before DataFrame dtype promotion.

        Internally constructed ParsedData instances have no source snapshot and
        safely fall back to the DataFrame value.
        """
        if self.source_records is not None and row_index < len(self.source_records):
            row = self.source_records[row_index]
            if column in row:
                return row[column]
        return self.df.iloc[row_index][column]

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


def _validate_monthly_calendar_cadence(df: pd.DataFrame, time_key: Optional[str]) -> None:
    """Reject invalid, sparse, duplicate, and reversed calendar timestamps."""
    if not time_key or time_key not in df.columns:
        return

    raw_values = df[time_key].astype(str).str.strip()
    calendar_shape_mask = raw_values.str.match(r"^\d{4}-\d+(?:$|[-T ].*)")
    if not calendar_shape_mask.any():
        return

    valid_month_mask = raw_values.str.match(
        r"^\d{4}-(?:0?[1-9]|1[0-2])(?:$|[-T ].*)"
    )
    if not calendar_shape_mask.all() or not valid_month_mask.all():
        raise _data_error(f"Calendar timestamp column '{time_key}' contains invalid values.")

    try:
        periods = pd.PeriodIndex(raw_values, freq="M")
    except (TypeError, ValueError) as exc:
        raise _data_error(f"Calendar timestamp column '{time_key}' contains invalid values.") from exc

    ordinals = periods.asi8
    invalid_positions = np.flatnonzero(np.diff(ordinals) != 1)
    if invalid_positions.size:
        index = int(invalid_positions[0])
        raise _data_error(
            "Monthly calendar timestamps must be contiguous: "
            f"{raw_values.iloc[index]} -> {raw_values.iloc[index + 1]}."
        )


def _normalize_calendar_parts(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Validate separate year/month fields before composing calendar timestamps."""
    year_values = pd.to_numeric(df["年份"], errors="coerce")
    month_values = pd.to_numeric(df["月份"], errors="coerce")
    valid_years = (
        year_values.notna()
        & np.isfinite(year_values)
        & np.isclose(year_values, np.round(year_values), atol=1e-9)
        & year_values.between(1000, 9999)
    )
    valid_months = (
        month_values.notna()
        & np.isfinite(month_values)
        & np.isclose(month_values, np.round(month_values), atol=1e-9)
        & month_values.between(1, 12)
    )
    if not bool(valid_years.all()) or not bool(valid_months.all()):
        raise _data_error("Separate year/month columns contain invalid calendar values.")
    return year_values.astype(int), month_values.astype(int)


TRAINING_PROVENANCE_SCHEMA = "fz.training-prefix.v1"


def _canonical_provenance_scalar(value: Any) -> Dict[str, Any]:
    if isinstance(value, (np.bool_, bool)):
        return {"type": "boolean", "value": bool(value)}
    if value is None:
        return {"type": "null"}
    if isinstance(value, (np.integer, int)):
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise _data_error("Training provenance cannot encode non-finite numeric values.")
        if numeric == 0.0:
            numeric = 0.0
        return {"type": "float", "value": numeric.hex()}
    if not isinstance(value, str) and bool(pd.isna(value)):
        return {"type": "null"}
    if isinstance(value, str):
        return {"type": "string", "value": value}
    raise _data_error(
        f"Training provenance cannot encode scalar type {type(value).__name__}."
    )


def _canonical_numeric_consumed_scalar(value: Any) -> Dict[str, Any]:
    if isinstance(value, (np.bool_, bool)) or value is None:
        raise _data_error("Training provenance numeric values must be finite numbers.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise _data_error("Training provenance numeric values must be finite numbers.") from exc
    if not np.isfinite(numeric):
        raise _data_error("Training provenance cannot encode non-finite numeric values.")
    if numeric == 0.0:
        numeric = 0.0
    return {"type": "float", "value": numeric.hex()}


def build_training_provenance(
    parsed_data: "ParsedData",
    train_range: range,
    *,
    feature_keys: Optional[List[str]] = None,
    feature_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if train_range.step != 1 or train_range.start < 0 or train_range.stop > parsed_data.n_obs:
        raise _data_error("Training provenance range is invalid.")
    if train_range.start >= train_range.stop:
        raise _data_error("Training provenance range must contain at least one row.")

    consumed_features = list(feature_keys or [])
    if any(not isinstance(key, str) or not key for key in consumed_features):
        raise _data_error("Training provenance feature names must be non-empty strings.")
    if len(set(consumed_features)) != len(consumed_features):
        raise _data_error("Training provenance feature names must not contain duplicates.")
    role_keys = {parsed_data.target_key}
    if parsed_data.time_key:
        role_keys.add(parsed_data.time_key)
    collisions = [key for key in consumed_features if key in role_keys]
    if collisions:
        raise _data_error(
            "Training provenance features collide with target/time roles: "
            + ", ".join(collisions)
        )
    columns = [parsed_data.target_key]
    if parsed_data.time_key:
        columns.append(parsed_data.time_key)
    columns.extend(consumed_features)
    missing_columns = [column for column in columns if column not in parsed_data.df.columns]
    if missing_columns:
        raise _data_error(
            "Training provenance columns are missing from the dataset: "
            + ", ".join(missing_columns)
        )

    feature_semantics = [
        "categorical"
        if feature_types and feature_types.get(column) == "categorical"
        else "numeric"
        if (
            feature_types and feature_types.get(column) == "numeric"
        ) or pd.api.types.is_numeric_dtype(parsed_data.df[column])
        else "categorical"
        for column in consumed_features
    ]

    def canonical_row(index: int) -> List[Dict[str, Any]]:
        values = [
            _canonical_numeric_consumed_scalar(
                parsed_data.df.iloc[index][parsed_data.target_key]
            )
        ]
        if parsed_data.time_key:
            values.append(
                _canonical_provenance_scalar(
                    parsed_data.source_value(index, parsed_data.time_key)
                )
            )
        for column, semantics in zip(consumed_features, feature_semantics):
            value = (
                parsed_data.df.iloc[index][column]
                if semantics == "numeric"
                else parsed_data.source_value(index, column)
            )
            values.append(
                _canonical_numeric_consumed_scalar(value)
                if semantics == "numeric"
                else _canonical_provenance_scalar(value)
            )
        return values

    canonical = {
        "schema": TRAINING_PROVENANCE_SCHEMA,
        "train_start_index": train_range.start,
        "train_end_index": train_range.stop - 1,
        "target_key": parsed_data.target_key,
        "time_key": parsed_data.time_key,
        "feature_keys": consumed_features,
        "feature_semantics": feature_semantics,
        "columns": columns,
        "rows": [canonical_row(index) for index in train_range],
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": TRAINING_PROVENANCE_SCHEMA,
        "train_start_index": train_range.start,
        "train_end_index": train_range.stop - 1,
        "target_key": parsed_data.target_key,
        "time_key": parsed_data.time_key,
        "feature_keys": consumed_features,
        "feature_semantics": feature_semantics,
        "columns": columns,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def attach_training_provenance(
    state: Dict[str, Any],
    parsed_data: "ParsedData",
    train_range: range,
    *,
    feature_keys: Optional[List[str]] = None,
    feature_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    state["training_provenance"] = build_training_provenance(
        parsed_data,
        train_range,
        feature_keys=feature_keys,
        feature_types=feature_types,
    )
    return state


def validate_training_provenance(
    state: Dict[str, Any],
    parsed_data: "ParsedData",
    expected_train_range: range,
    *,
    feature_keys: Optional[List[str]] = None,
    feature_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    actual = state.get("training_provenance")
    if not isinstance(actual, dict):
        raise ValueError("[VALIDATION] Base artifact is missing training provenance.")
    expected = build_training_provenance(
        parsed_data,
        expected_train_range,
        feature_keys=feature_keys,
        feature_types=feature_types,
    )
    if actual != expected:
        raise ValueError("[VALIDATION] Base artifact training provenance does not match the requested training range and dataset.")
    return expected


def build_numeric_subprocess_records(parsed_data: "ParsedData") -> Tuple[List[Dict[str, Any]], List[str]]:
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

    data_dicts: List[Dict[str, Any]] = []
    for i in range(parsed_data.n_obs):
        row: Dict[str, Any] = {parsed_data.target_key: float(parsed_data.y[i])}
        if parsed_data.time_key:
            row[parsed_data.time_key] = _to_json_scalar(
                parsed_data.source_value(i, parsed_data.time_key)
            )
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
    feature_types: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build subprocess row dicts while preserving categorical feature values.

    This is primarily for models (e.g. LSTM) that can consume categorical
    features via their own preprocessing pipeline.
    """
    source_keys = feature_keys if feature_keys is not None else parsed_data.feature_keys
    if len(set(source_keys)) != len(source_keys):
        raise _data_error("Feature list must not contain duplicate names.")
    role_keys = {parsed_data.target_key}
    if parsed_data.time_key:
        role_keys.add(parsed_data.time_key)
    collisions = [key for key in source_keys if key in role_keys]
    if collisions:
        raise _data_error(
            "Feature list collides with target/time roles: " + ", ".join(collisions)
        )
    selected_feature_keys: List[str] = []

    for feat_key in source_keys:
        if not feat_key:
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
        if parsed_data.time_key:
            row[parsed_data.time_key] = _to_json_scalar(
                parsed_data.source_value(i, parsed_data.time_key)
            )
        for feat_key in selected_feature_keys:
            semantics = feature_types.get(feat_key) if feature_types else None
            value = (
                parsed_data.source_value(i, feat_key)
                if semantics == "categorical"
                else parsed_data.df.iloc[i][feat_key]
            )
            row[feat_key] = _to_json_scalar(value)
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
    inference_range: Optional[range] = None,
) -> Dict[str, str]:
    """
    Infer feature types from observed values in the declared training range.

    Numeric-like string columns are treated as numeric after coercion. The
    resolved schema is then applied to the full dataframe without re-inferring.
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
        inference_series = series if inference_range is None else series.iloc[inference_range]
        inferred_type = _infer_single_feature_type(feat_key, inference_series)
        if inferred_type == "numeric" and not pd.api.types.is_numeric_dtype(series):
            converted_inference = _coerce_numeric_series(inference_series)
            if converted_inference.notna().any() and _non_finite_mask(converted_inference.dropna()).any():
                raise _data_error(f"Feature column '{feat_key}' contains non-finite values.")
            parsed_data.df[feat_key] = _coerce_numeric_series(series)
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

    if (
        not isinstance(records, list)
        or not records
        or not all(isinstance(record, dict) for record in records)
    ):
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

    # Separate year/month fields are an explicit calendar contract even when a
    # different time key was requested. Validate them before any construction.
    calendar_parts: Optional[Tuple[pd.Series, pd.Series]] = None
    if "年份" in df.columns and "月份" in df.columns:
        calendar_parts = _normalize_calendar_parts(df)

    # Auto-detect time_key and create a composite timestamp when possible.
    if time_key is None:
        if calendar_parts is not None:
            log.info("Found valid '年份' and '月份' columns, creating composite timestamp.")
            year_values, month_values = calendar_parts
            df["timestamp"] = (
                year_values.map(lambda value: f"{value:04d}")
                + "-"
                + month_values.map(lambda value: f"{value:02d}")
            )
            time_key = "timestamp"
            df["年份"] = year_values
            df["月份"] = month_values
            log.info("Created composite time_key and normalized year/month features")
        elif (
            "月份" in df.columns
            and df["月份"].astype(str).str.strip().str.match(r"^\d{4}-\d+").any()
        ):
            time_key = "月份"
        elif "month" in df.columns:
            time_key = "month"
        elif "date" in df.columns:
            time_key = "date"

        if time_key and time_key != "timestamp":
            log.info(f"Auto-detected time_key: '{time_key}'")

    _validate_monthly_calendar_cadence(df, time_key)

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

    source_records: List[Dict[str, Any]] = []
    for index, original in enumerate(records):
        source_records.append({
            column: (
                original[column]
                if column in original
                else _to_json_scalar(df_filtered.iloc[index][column])
            )
            for column in columns_to_keep
        })

    return ParsedData(
        df=df_filtered,
        target_key=target_key,
        time_key=time_key,
        feature_keys=feature_keys,
        source_records=source_records,
    )
