EXPECTED_METHOD_METADATA = {
    "ma": {
        "method_name": "递推移动平均",
        "forecast_strategy": "recursive_roll_forward",
        "note_fragments": ("移动平均", "递推"),
    },
    "es": {
        "method_name": "一次指数平滑",
        "forecast_strategy": "flat_simple_exponential_smoothing",
        "note_fragments": ("一次指数平滑", "多步"),
    },
    "arima": {
        "method_name": "ARIMA 自动定阶",
        "forecast_strategy": "arima_multi_step_forecast",
        "note_fragments": ("ARIMA", "stepwise"),
    },
    "lstm_direct": {
        "method_name": "历史协变量直接多步 LSTM 销量预测",
        "forecast_strategy": "direct_multi_horizon",
        "note_fragments": ("直接多步", "递归"),
    },
    "weighted_avg": {
        "method_name": "验证残差MSE倒数加权平均融合",
        "forecast_strategy": "validation_residual_mse_weighted_average",
        "note_fragments": ("残差", "权重"),
    },
    "stacking": {
        "method_name": "时间顺序留出 Stacking 融合",
        "forecast_strategy": "holdout_nonnegative_least_squares",
        "note_fragments": ("时间顺序", "无截距"),
    },
    "boosting": {
        "method_name": "残差提升融合",
        "forecast_strategy": "greedy_residual_additive_ensemble",
        "note_fragments": ("残差", "贪心"),
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
