from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

from .core import adf_fixed_d


def read_series(input_path: Path, value_key: str) -> List[float]:
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
        if value_key not in df.columns:
            raise KeyError(f"CSV missing column '{value_key}'")
        series = df[value_key].astype(float).tolist()
        return series
    else:
        # assume JSON array
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError("JSON must be an array (list) of objects")
        series = [float(item[value_key]) for item in data]
        return series


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ADF test with fixed differencing orders (d=0,1,2)")
    p.add_argument("input", type=str, help="JSON array or file path (.json/.csv)")
    p.add_argument("--value-key", default="sales")
    p.add_argument("--diff-orders", default="0,1,2")
    p.add_argument("--regression", default="c", choices=["n", "c", "ct", "ctt"])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--round", dest="ndigits", type=int, default=3)
    p.add_argument("--pretty", action="store_true")
    args = p.parse_args(argv)

    try:
        # ✅ 尝试先解析为 JSON 字符串
        data = json.loads(args.input)
        if not isinstance(data, list):
            raise ValueError("JSON must be a list")
        series = [float(item[args.value_key]) for item in data]
    except json.JSONDecodeError:
        # 否则再当文件读取
        path = Path(args.input)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1
        series = read_series(path, args.value_key)

    diff_orders = [int(x) for x in args.diff_orders.split(",") if x.strip()]
    results = adf_fixed_d(series, diff_orders, args.regression, args.alpha, args.ndigits)

    print(json.dumps(results, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0

if __name__ == "__main__":
    sys.exit(main())
