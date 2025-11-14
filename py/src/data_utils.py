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
    try:
        records = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    if not isinstance(records, list) or not records:
        raise ValueError("JSON data must be a non-empty array of objects.")

    first_record = records[0]
    if not isinstance(first_record, dict):
        raise ValueError("Each record must be a JSON object/dict.")

    # Auto-detect keys if not provided
    if target_key is None:
        if "y" in first_record:
            target_key = "y"
        elif "sales" in first_record:
            target_key = "sales"
        else:
            raise ValueError("Cannot determine target key. Expecting 'y' or 'sales'.")

    # Validate that the determined or provided target_key exists
    if target_key not in first_record:
        raise ValueError(f"Provided or auto-detected target key '{target_key}' not in data.")

    if time_key is None and "month" in first_record:
        time_key = "month"

    # Auto-detect feature keys if they are not provided
    if feature_keys is None:
        # Detect all other columns as potential features
        all_keys = list(first_record.keys())
        detected_features = [k for k in all_keys if k not in [target_key, time_key]]
        feature_keys = detected_features if detected_features else None

    # We can't use the old _parse_records directly as it was designed for numpy
    # The new approach is to create the DataFrame and then the ParsedData object
    
    # Rigorous check for missing keys, as DataFrame creation can hide them with NaNs
    if records:
        first_record_keys = set(records[0].keys())
        for i, record in enumerate(records[1:], 1):
            record_keys = set(record.keys())
            if record_keys != first_record_keys:
                missing = first_record_keys - record_keys
                extra = record_keys - first_record_keys
                raise ValueError(f"Error processing record {i}: Key mismatch. Missing: {missing or 'None'}, Extra: {extra or 'None'}")

    df = pd.DataFrame(records)

    return ParsedData(
        df=df,
        target_key=target_key,
        time_key=time_key,
        feature_keys=feature_keys,
    )
