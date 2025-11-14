# py/src/data_utils.py
# -*- coding: utf-8 -*-
"""
Centralized data parsing and preparation utilities for all time series models.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


class TimeSeriesData:
    """
    A container for parsed and prepared time series data, ensuring a consistent
    data contract across different models.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        target_key: str,
        time_key: Optional[str] = None,
        feature_types: Optional[Dict[str, str]] = None,
    ):
        if not isinstance(records, list) or not records:
            raise ValueError("Input 'records' must be a non-empty list of dicts.")

        self.target_key = target_key
        self.time_key = time_key
        self.feature_types = feature_types if feature_types is not None else {}

        self.y, self.timestamps = self._extract_y_and_timestamps(records)
        self.feature_names, self.X = self._extract_features(records)

    def _extract_y_and_timestamps(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Extracts the target variable (y) and optional timestamps."""
        y_values: List[float] = []
        timestamps: List[str] = []
        has_timestamps = self.time_key is not None

        for i, record in enumerate(records):
            if self.target_key not in record:
                raise ValueError(f"[DATA] Record {i} is missing target key '{self.target_key}'")
            try:
                y_val = float(record[self.target_key])
                if not np.isfinite(y_val):
                    raise ValueError(f"must be finite, but got {y_val}")
                y_values.append(y_val)
            except (ValueError, TypeError) as e:
                raise ValueError(f"[DATA] Target '{self.target_key}' in record {i} is invalid: {e}")

            if has_timestamps:
                if self.time_key not in record:
                    raise ValueError(f"[DATA] Record {i} is missing time key '{self.time_key}'")
                timestamps.append(str(record[self.time_key]))

        return np.asarray(y_values, dtype=np.float64), timestamps if has_timestamps else None

    def _extract_features(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        Extracts features from the 'x' dictionary in each record, if present.
        Currently handles numeric features only. Categorical support can be added later.
        """
        # Check if any record has the 'x' key for features
        if not any("x" in r and isinstance(r.get("x"), dict) for r in records):
            return [], None

        # Determine the full, ordered set of feature names from the first record with 'x'
        feature_names: List[str] = []
        for r in records:
            if "x" in r and isinstance(r.get("x"), dict):
                feature_names = sorted(r["x"].keys())
                break
        
        if not feature_names:
            return [], None

        X_values: List[List[float]] = []
        for i, record in enumerate(records):
            features = record.get("x", {})
            if sorted(features.keys()) != feature_names:
                raise ValueError(f"[DATA] Feature keys in record {i} do not match the initial set.")

            row: List[float] = []
            for key in feature_names:
                try:
                    val = float(features[key])
                    if not np.isfinite(val):
                        raise ValueError("must be finite")
                    row.append(val)
                except (ValueError, TypeError):
                    raise ValueError(f"[DATA] Feature '{key}' in record {i} must be a valid number.")
            X_values.append(row)

        return feature_names, np.asarray(X_values, dtype=np.float64)

    @property
    def n_obs(self) -> int:
        return len(self.y)


def parse_data_from_json(
    json_str: str,
    target_key: str,
    time_key: Optional[str] = None,
    feature_types: Optional[Dict[str, str]] = None,
) -> TimeSeriesData:
    """
    Parses a JSON string into a structured TimeSeriesData object.

    This is the main entry point for all models to standardize data ingestion.

    Args:
        json_str: The raw JSON data string.
        target_key: The dictionary key for the target variable (e.g., "sales").
        time_key: The optional dictionary key for the timestamp (e.g., "month").
        feature_types: Optional dict specifying feature types (e.g., {"promo": "binary"}).
                       (Currently placeholder for future expansion).

    Returns:
        A TimeSeriesData object containing y, X, timestamps, etc.
    """
    try:
        records = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"[DATA] Invalid JSON format: {e}")

    if not isinstance(records, list):
        raise ValueError("[DATA] JSON data must be an array of objects.")

    return TimeSeriesData(
        records=records,
        target_key=target_key,
        time_key=time_key,
        feature_types=feature_types,
    )

# --- Legacy single-series parser for compatibility ---
# This can be phased out as models are updated.
def parse_sales_data(json_str):
    """Legacy function to parse simple {"month": ..., "sales": ...} data."""
    arr = json.loads(json_str)
    months = [item["month"] for item in arr]
    sales = [float(item["sales"]) for item in arr]
    return months, sales

def slice_series(series, start, end):
    """Legacy slicing function."""
    return series[start:end]