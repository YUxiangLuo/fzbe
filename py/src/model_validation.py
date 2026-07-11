MIN_EVALUATION_POINTS = 2


def validate_minimum_evaluation_points(eval_start: int, eval_end: int) -> None:
    evaluation_size = eval_end - eval_start + 1
    if evaluation_size < MIN_EVALUATION_POINTS:
        raise ValueError(
            f"[VALIDATION] Evaluation range must contain at least "
            f"{MIN_EVALUATION_POINTS} points, got {evaluation_size}."
        )
