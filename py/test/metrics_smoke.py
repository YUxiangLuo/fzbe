#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path


PY_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PY_SRC))

from metrics import mae, mape, r2, rmse


def assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"expected {expected}, got {actual}")


def main() -> None:
    assert_close(rmse([1, 2, 3], [1, 2, 5]), math.sqrt(4 / 3))
    assert_close(mae([1, 2, 3], [1, 2, 5]), 2 / 3)
    assert_close(r2([1, 2, 3], [1, 2, 3]), 1.0)

    assert_close(mape([100], [90]), 10.0)
    assert_close(mape([-100], [-90]), 10.0)

    zero_actual_mape = mape([0], [1])
    if not math.isfinite(zero_actual_mape) or zero_actual_mape <= 0:
        raise AssertionError(f"expected finite positive zero-actual MAPE, got {zero_actual_mape}")

    print(json.dumps({"model": "metrics", "status": "ok"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
