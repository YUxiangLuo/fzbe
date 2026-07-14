#!/usr/bin/env python3
"""Smoke checks for the CSV/JSON numeric contract shared by all base models."""

import json
import math
import sys
from pathlib import Path


PY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PY_ROOT / "src"))

from data_utils import load_data_from_json  # noqa: E402


def assert_grouped_numeric_values() -> None:
    records = [
        {"年份": "2024", "月份": "1", "销售数量": "1,234.50", "价格指数": "2,500"},
        {"年份": "2024", "月份": "2", "销售数量": "1,250", "价格指数": "2,510.25"},
    ]
    parsed = load_data_from_json(
        json.dumps(records, ensure_ascii=False),
        target_key="销售数量",
        feature_keys=["价格指数"],
    )
    assert parsed.y.tolist() == [1234.5, 1250.0]
    assert parsed.df["价格指数"].tolist() == [2500.0, 2510.25]
    assert all(math.isfinite(float(value)) for value in parsed.y)


def assert_invalid_targets_rejected() -> None:
    for value, expected in (
        ("not-a-number", "non-numeric"),
        ("0x10", "non-numeric"),
        ("Infinity", "non-finite"),
        ("1e309", "non-finite"),
    ):
        records = [{"年份": "2024", "月份": "1", "销售数量": value}]
        try:
            load_data_from_json(json.dumps(records, ensure_ascii=False))
        except ValueError as exc:
            message = str(exc)
            assert message.startswith("[DATA]")
            assert expected in message
        else:
            raise AssertionError(f"invalid target was accepted: {value}")


def assert_invalid_numeric_features_rejected() -> None:
    records = [
        {"年份": "2024", "月份": "1", "销售数量": 10, "价格指数": float("inf")},
    ]
    try:
        load_data_from_json(
            json.dumps(records, ensure_ascii=False),
            target_key="销售数量",
            feature_keys=["价格指数"],
        )
    except ValueError as exc:
        message = str(exc)
        assert message.startswith("[DATA]")
        assert "non-finite" in message
    else:
        raise AssertionError("numeric-dtype non-finite feature was accepted")


def main() -> None:
    assert_grouped_numeric_values()
    assert_invalid_targets_rejected()
    assert_invalid_numeric_features_rejected()
    print(json.dumps({"status": "ok", "contract": "finite_grouped_numeric"}, indent=2))


if __name__ == "__main__":
    main()
