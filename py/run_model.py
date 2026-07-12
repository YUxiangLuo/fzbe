import argparse
import sys
import os

# Add src/models to path to import model modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import all model modules
from models import arima, es, lstm, ma, stacking, weighted_avg, boosting

MODEL_MODULES = {
    "arima": arima,
    "es": es,
    "ma": ma,
    "lstm": lstm,
    "stacking": stacking,
    "weighted_avg": weighted_avg,
    "boosting": boosting,
}


def _format_known_error(exc: Exception):
    message = str(exc).strip()
    if message.startswith("[VALIDATION]"):
        return f"[VALIDATION] {message[13:].strip()}".strip()
    if message.startswith("[DATA]"):
        return f"[DATA] {message[7:].strip()}".strip()
    return None

def main():
    parser = argparse.ArgumentParser(description="Run a specified time series model.")
    parser.add_argument("model_type", type=str, help="Type of model to run (e.g., arima, lstm)")
    
    # Add common arguments that all models might use
    parser.add_argument("--data", type=str, help="JSON string of input data")
    parser.add_argument("--data-stdin", action="store_true", help="Read the JSON input data from stdin")
    parser.add_argument("--train-start", type=int, help="Training data start index")
    parser.add_argument("--train-end", type=int, help="Training data end index")
    parser.add_argument("--eval-start", type=int, help="Evaluation data start index")
    parser.add_argument("--eval-end", type=int, help="Evaluation data end index")
    parser.add_argument("--save-model", type=str, help="Path to save the trained model")
    parser.add_argument("--load-model", type=str, help="Path to load a trained model")
    parser.add_argument("--forecast-steps", type=int, help="Number of steps to forecast")
    parser.add_argument("--fit-only", action="store_true", help="Fit a production model without evaluation")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    # Add model-specific arguments (these will be parsed by the individual model's run function)
    # MA
    parser.add_argument("--window", type=int, help="MA window size")
    # ES
    parser.add_argument("--alpha", type=float, help="Exponential smoothing alpha value")
    # ARIMA
    parser.add_argument("--d", type=int, help="ARIMA differencing order")
    parser.add_argument("--max-p", type=int, help="ARIMA max p order for auto_arima")
    parser.add_argument("--max-q", type=int, help="ARIMA max q order for auto_arima")
    # LSTM
    parser.add_argument("--look-back", type=int, help="LSTM look back window")
    parser.add_argument("--epochs", type=int, help="LSTM training epochs")
    parser.add_argument("--feature-types", type=str, help="LSTM feature types as JSON string")
    parser.add_argument("--lstm-features", type=str, help="LSTM feature list as JSON array or legacy comma-separated string")
    parser.add_argument("--lstm-features-json", type=str, help="LSTM feature list as JSON array")
    parser.add_argument("--lstm-target-feature", type=str, help="LSTM target feature column name")
    parser.add_argument("--normalization", type=str, choices=['minmax', 'zscore'], help="LSTM normalization method")
    # Ensemble Models
    parser.add_argument("--models", type=str, help="Comma-separated list of base models for stacking/weighted_avg/boosting")
    parser.add_argument("--base-model-files", type=str, help="JSON object mapping base model names to saved model artifact paths")
    # Boosting-specific
    parser.add_argument("--learning-rate", type=float, help="Boosting learning rate (shrinkage parameter)")
    parser.add_argument("--max-rounds", type=int, help="Boosting maximum number of rounds")

    # Parse args. We need to parse unknown args because each model's run function
    # expects a Namespace object with all its specific args, even if not defined here.
    args, unknown = parser.parse_known_args()

    if args.data_stdin:
        if args.data is not None:
            raise ValueError("[VALIDATION] Use either --data or --data-stdin, not both.")
        args.data = sys.stdin.read()
        if not args.data:
            raise ValueError("[DATA] stdin did not contain model input data.")

    if args.model_type not in MODEL_MODULES:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Pass all parsed args (including unknown ones) to the specific model's run function
    # The individual model's run function will then parse its specific args from this Namespace
    model_module = MODEL_MODULES[args.model_type]
    model_module.run(args)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        formatted_error = _format_known_error(exc)
        if formatted_error is None:
            raise
        print(formatted_error, file=sys.stderr, flush=True)
        sys.exit(1)
