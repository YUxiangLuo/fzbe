# py/src/data_utils.py
# -*- coding: utf-8 -*-
"""
Centralized data parsing and preparation utilities for all time series models.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np


class ParsedData:
    """
    A container for parsed and prepared time series data, ensuring a consistent
    data contract across different models.
    """

    def __init__(
        self,
        y: np.ndarray,
        target_key: str,
        timestamps: Optional[List[str]] = None,
        time_key: Optional[str] = None,
        X: Optional[np.ndarray] = None,
        feature_keys: Optional[List[str]] = None,
    ):
        self.y = y
        self.target_key = target_key
        self.timestamps = timestamps
        self.time_key = time_key
        self.X = X
        self.feature_keys = feature_keys

    @property
    def n_obs(self) -> int:
        return len(self.y)

    @property
    def y_featured(self) -> np.ndarray:
        """Returns y and X combined, with y as the first column."""
        if self.X is None:
            return self.y.reshape(-1, 1)
        return np.hstack([self.y.reshape(-1, 1), self.X])


def _parse_records(
    records: List[Dict[str, Any]],
    target_key: str,
    time_key: Optional[str] = None,
    feature_keys: Optional[List[str]] = None,
) -> ParsedData:
    """
    Internal helper to process a list of records into a ParsedData object.
    """
    y_values: List[float] = []
    timestamps: List[str] = []
    feature_values: List[List[float]] = [] if feature_keys else None

    for i, record in enumerate(records):
        try:
            y_values.append(float(record[target_key]))
            if feature_keys:
                feature_values.append([float(record[key]) for key in feature_keys])
            if time_key:
                timestamps.append(str(record[time_key]))
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Error processing record {i}: {e}")

    return ParsedData(
        y=np.asarray(y_values, dtype=np.float64),
        target_key=target_key,
        timestamps=timestamps if time_key else None,
        time_key=time_key,
        X=np.asarray(feature_values, dtype=np.float64) if feature_keys else None,
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

    return _parse_records(records, target_key, time_key, feature_keys)
