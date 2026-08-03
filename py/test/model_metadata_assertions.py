EXPECTED_METHOD_METADATA = {
    "ma": {
        "method_name": "递推移动平均",
        "forecast_strategy": "recursive_roll_forward",
        "note_fragments": ("移动平均", "递推", "不声明 95% 覆盖保证"),
    },
    "es": {
        "method_name": "一次指数平滑",
        "forecast_strategy": "flat_simple_exponential_smoothing",
        "note_fragments": ("一次指数平滑", "多步", "不声明 95% 覆盖保证"),
    },
    "arima": {
        "method_name": "ARIMA 逐步搜索定阶",
        "forecast_strategy": "arima_multi_step_forecast",
        "note_fragments": ("ARIMA", "stepwise"),
    },
    "lstm_direct": {
        "method_name": "LSTM 历史多特征销量预测",
        "forecast_strategy": "direct_multi_horizon",
        "note_fragments": ("直接多步", "递归", "不声明 95% 覆盖保证"),
    },
    "lstm_autoregressive": {
        "method_name": "LSTM 历史多特征销量预测",
        "forecast_strategy": "autoregressive",
        "note_fragments": ("递归", "回退", "不声明 95% 覆盖保证"),
    },
    "weighted_avg": {
        "method_name": "选择残差 MSE 倒数加权平均融合",
        "forecast_strategy": "validation_residual_mse_weighted_average",
        "note_fragments": ("残差", "权重", "不复用权重选择残差"),
    },
    "stacking": {
        "method_name": "滚动起点折外预测 Stacking 融合",
        "forecast_strategy": "rolling_origin_oof_nonnegative_least_squares",
        "note_fragments": ("滚动起点", "无截距", "选择后校准"),
    },
    "boosting": {
        "method_name": "残差提升融合",
        "forecast_strategy": "greedy_residual_additive_ensemble",
        "note_fragments": ("残差", "贪心", "第 0 阶段", "shrinkage"),
    },
}


def assert_method_metadata(payload: dict, model_key: str) -> None:
    expected = EXPECTED_METHOD_METADATA[model_key]
    for field in ("method_name", "forecast_strategy"):
        if payload.get(field) != expected[field]:
            raise AssertionError(
                f"{model_key} {field} mismatch: "
                f"actual={payload.get(field)!r}, expected={expected[field]!r}"
            )

    notes = payload.get("implementation_notes")
    if not isinstance(notes, list) or not notes or not all(isinstance(note, str) and note.strip() for note in notes):
        raise AssertionError(f"{model_key} implementation_notes must be a non-empty string list, got {notes!r}")

    for fragment in expected["note_fragments"]:
        if not any(fragment in note for note in notes):
            raise AssertionError(f"{model_key} implementation_notes missing fragment {fragment!r}: {notes!r}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    source_dir = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source_dir))

    from models import arima, boosting, es, lstm, ma, stacking, weighted_avg

    modules_by_key = {
        "ma": ma,
        "es": es,
        "arima": arima,
        "lstm_direct": lstm,
        "weighted_avg": weighted_avg,
        "stacking": stacking,
        "boosting": boosting,
    }
    for key, module in modules_by_key.items():
        assert_method_metadata(
            {
                "method_name": module.METHOD_NAME,
                "forecast_strategy": (
                    module.DIRECT_FORECAST_STRATEGY
                    if key == "lstm_direct"
                    else module.FORECAST_STRATEGY
                ),
                "implementation_notes": (
                    module.DIRECT_IMPLEMENTATION_NOTES
                    if key == "lstm_direct"
                    else module.IMPLEMENTATION_NOTES
                ),
            },
            key,
        )
    print("model_metadata_assertions: PASS")
