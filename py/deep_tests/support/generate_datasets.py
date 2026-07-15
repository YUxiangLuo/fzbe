#!/usr/bin/env python3
"""Create and verify deterministic, strict-JSON deep-test datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path
from typing import Any

GENERATOR_REVISION = 2
DEFAULT_SEED = 20250308
DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"


def _month(index: int) -> str:
    return f"{2020 + index // 12:04d}-{index % 12 + 1:02d}"


def generate_level() -> list[dict[str, Any]]:
    return [{"month": _month(i), "y": 120} for i in range(36)]


def generate_trend() -> list[dict[str, Any]]:
    return [{"month": _month(i), "y": 80 + 2 * i} for i in range(48)]


def generate_recursive_ma() -> list[dict[str, Any]]:
    values = [30 + (i % 5) for i in range(21)] + [10, 20, 40]
    return [{"month": _month(i), "y": value} for i, value in enumerate(values)]


def generate_mixed_features() -> list[dict[str, Any]]:
    regions = ["华北", "华南", "西部"]
    return [
        {
            "month": _month(i),
            "y": 70 + i + (i % 4) * 3,
            "price": round(12.5 + (i % 7) * 0.25, 2),
            "visits": f"{1000 + 17 * i:,}",
            "region": regions[i % len(regions)],
            "年份": 2020 + i // 12,
            "月份": i % 12 + 1,
        }
        for i in range(48)
    ]


def generate_seasonal_pulse() -> list[dict[str, Any]]:
    return [
        {"month": _month(i), "y": 90 + i // 3 + (24 if i % 12 in (0, 1) else 0)}
        for i in range(60)
    ]


def generate_gap_probe() -> list[dict[str, Any]]:
    return [
        {"month": _month(i), "y": 50 + 2 * i + (i % 3), "promo": i % 2, "region": "A" if i % 2 else "B"}
        for i in range(40)
    ]


def generate_signed_residual() -> list[dict[str, Any]]:
    return [{"month": _month(i), "y": (-1 if i % 2 else 1) * (5 + i % 7)} for i in range(36)]


def generate_seeded_arx(seed: int) -> list[dict[str, Any]]:
    """Seeded ARX process with numeric and categorical exogenous features."""
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    previous = 100.0
    for i in range(96):
        driver = 40 + 2 * (i % 12)
        promo = 1 if i % 9 in (0, 1) else 0
        innovation = (int(rng.random() * 6001) - 3000) / 1000
        value = round(100 + 0.65 * (previous - 100) + 0.35 * (driver - 51) + 4 * promo + innovation, 6)
        records.append({
            "month": _month(i),
            "y": value,
            "driver": driver,
            "promo": promo,
            "region": ("north", "south", "west")[i % 3],
        })
        previous = value
    return records


def generate_invalid_cases() -> list[dict[str, Any]]:
    return [
        {"name": "empty", "expected_tag": "[DATA]"},
        {"name": "not_list", "expected_tag": "[DATA]"},
        {"name": "missing_target", "expected_tag": "[DATA]"},
        {"name": "non_numeric_target", "expected_tag": "[DATA]"},
        {"name": "nan", "expected_tag": "[DATA]"},
        {"name": "infinity", "expected_tag": "[DATA]"},
        {"name": "missing_feature", "expected_tag": "[DATA]"},
    ]


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def generate_all(seed: int = DEFAULT_SEED) -> dict[str, list[dict[str, Any]]]:
    return {
        "trend_48.json": generate_trend(),
        "level_36.json": generate_level(),
        "recursive_ma_24.json": generate_recursive_ma(),
        "mixed_features_48.json": generate_mixed_features(),
        "seasonal_pulse_60.json": generate_seasonal_pulse(),
        "gap_probe_40.json": generate_gap_probe(),
        "signed_residual_36.json": generate_signed_residual(),
        "seeded_arx_96.json": generate_seeded_arx(seed),
        "invalid_contract_cases.json": generate_invalid_cases(),
    }


SPECS: dict[str, dict[str, Any]] = {
    "trend_48.json": {"splits": {"train": [0, 35], "eval": [40, 43], "gap": [36, 39]}, "formula": "month=2020-01+t months; y=80+2*t"},
    "level_36.json": {"splits": {"train": [0, 27], "eval": [30, 33], "gap": [28, 29]}, "formula": "month=2020-01+t months; y=120"},
    "recursive_ma_24.json": {"splits": {"train": [0, 23]}, "formula": "month=2020-01+t months; y=30+(t mod 5) for t=0..20; y[21:24]=[10,20,40]"},
    "mixed_features_48.json": {"splits": {"train": [0, 35], "eval": [40, 43], "gap": [36, 39]}, "formula": "month=2020-01+t months; y=70+t+3*(t mod 4); price=round(12.5+0.25*(t mod 7),2); visits=comma_format(1000+17*t); region=[华北,华南,西部][t mod 3]; 年份=2020+floor(t/12); 月份=1+(t mod 12)"},
    "seasonal_pulse_60.json": {"splits": {"train": [0, 43], "eval": [48, 51], "gap": [44, 47]}, "formula": "month=2020-01+t months; y=90+floor(t/3)+24*I(t mod 12 in {0,1})"},
    "gap_probe_40.json": {"splits": {"train": [0, 27], "eval": [32, 35], "gap": [28, 31]}, "formula": "month=2020-01+t months; y=50+2*t+(t mod 3); promo=t mod 2; region='A' if t odd else 'B'"},
    "signed_residual_36.json": {"splits": {"train": [0, 27], "eval": [30, 33], "gap": [28, 29]}, "formula": "month=2020-01+t months; y=(-1)^t*(5+(t mod 7)) with positive sign at even t"},
    "seeded_arx_96.json": {"splits": {"train": [0, 63], "eval": [68, 75], "gap": [64, 67]}, "formula": "month=2020-01+t months; driver=40+2*(t mod 12); promo=I(t mod 9 in {0,1}); region=[north,south,west][t mod 3]; e=(floor(Random(seed).random()*6001)-3000)/1000; y[t]=round(100+0.65*(y[t-1]-100)+0.35*(driver-51)+4*promo+e,6), y[-1]=100"},
    "invalid_contract_cases.json": {"splits": {}, "formula": "ordered names=[empty,not_list,missing_target,non_numeric_target,nan,infinity,missing_feature]; expected_tag='[DATA]' for every row"},
}


def build_manifest(fixtures: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    entries = []
    for name, records in fixtures.items():
        columns = sorted({key for record in records for key in record})
        entries.append({
            "file": name,
            "sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
            "rows": len(records),
            "columns": columns,
            "target_key": "y" if "y" in columns else None,
            "time_key": "month" if "month" in columns else None,
            "splits": SPECS[name]["splits"],
            "formula": SPECS[name]["formula"],
        })
    return {"generator_revision": GENERATOR_REVISION, "seed": seed, "fixtures": entries}


def write_to(directory: Path, seed: int = DEFAULT_SEED) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fixtures = generate_all(seed)
    for name, records in fixtures.items():
        (directory / name).write_bytes(canonical_json_bytes(records))
    (directory / "manifest.json").write_bytes(canonical_json_bytes(build_manifest(fixtures, seed)))


def check(directory: Path = DATASET_DIR, seed: int = DEFAULT_SEED) -> list[str]:
    mismatches: list[str] = []
    with tempfile.TemporaryDirectory(prefix="fz-deep-generator-") as temp:
        generated = Path(temp)
        write_to(generated, seed)
        expected_names = sorted(path.name for path in generated.iterdir())
        actual_names = sorted(path.name for path in directory.iterdir()) if directory.exists() else []
        if actual_names != expected_names:
            mismatches.append(f"file set differs: expected={expected_names}, actual={actual_names}")
        for name in expected_names:
            actual = directory / name
            if not actual.exists() or actual.read_bytes() != (generated / name).read_bytes():
                mismatches.append(f"byte mismatch: {name}")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.write:
        write_to(DATASET_DIR, args.seed)
        print(f"wrote {len(generate_all(args.seed))} fixtures and manifest to {DATASET_DIR}")
        return 0
    mismatches = check(DATASET_DIR, args.seed)
    if mismatches:
        for mismatch in mismatches:
            print(mismatch)
        return 1
    print(f"verified {len(generate_all(args.seed))} fixtures and manifest (revision={GENERATOR_REVISION}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
