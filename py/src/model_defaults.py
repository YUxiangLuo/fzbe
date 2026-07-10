"""Shared defaults for forecasting model entry points."""

MIN_LSTM_LOOK_BACK = 3
MAX_LSTM_LOOK_BACK = 6


def resolve_lstm_look_back(training_size: int, explicit_value: int | None = None) -> int:
    """Keep an explicit history window or select the shared automatic default."""
    if explicit_value is not None:
        return int(explicit_value)
    return max(
        MIN_LSTM_LOOK_BACK,
        min(MAX_LSTM_LOOK_BACK, int(training_size) // 4),
    )
