# src/main.py
import argparse, sys
from models import ma, es, arima


def main():
    parser = argparse.ArgumentParser(description="Forecasting model runner")
    # --- Common arguments ---
    parser.add_argument("--method", choices=["ma", "es", "arima", "lstm", "weighted_average", "boosting", "stacking", "stackingv2"], default="ma")
    parser.add_argument("--data", required=False)
    parser.add_argument("--train_start", type=int, required=False)
    parser.add_argument("--train_end", type=int, required=False)
    parser.add_argument("--eval_start", type=int, required=False)
    parser.add_argument("--eval_end", type=int, required=False)
    parser.add_argument("--pretty", action="store_true")

    # --- Model-specific arguments ---
    # MA
    parser.add_argument("--window", type=int, required=False)
    # ES
    parser.add_argument("--alpha", type=float, required=False)
    # ARIMA
    parser.add_argument("--d", type=int, required=False)
    parser.add_argument("--p_max", type=int, required=False, default=3)
    parser.add_argument("--q_max", type=int, required=False, default=3)
    # LSTM
    parser.add_argument("--feature_types", type=str, required=False)
    parser.add_argument("--lookback", type=int, required=False, default=4)
    parser.add_argument("--horizon", type=int, required=False, default=1)
    parser.add_argument("--norm", type=str, required=False, default="zscore")
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--epochs", type=int, required=False, default=50)
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--hidden", type=int, required=False, default=64)
    parser.add_argument("--layers", type=int, required=False, default=1)
    parser.add_argument("--dropout", type=float, required=False, default=0.0)

    # --- Common for trainable models (ARIMA, LSTM) ---
    parser.add_argument("--save_model", type=str, required=False, help="Path to save trained model.")
    parser.add_argument("--load_model", type=str, required=False, help="Path to load model for prediction.")
    parser.add_argument("--forecast_steps", type=int, required=False, help="Steps to forecast when loading a model.")

    # --- Ensemble / Boosting arguments ---
    parser.add_argument("--models", nargs="+", required=False, help="Paths to base models for ensemble/boosting.")
    parser.add_argument("--save_ensemble", type=str, required=False, help="Path to save ensemble/boosting config.")
    parser.add_argument("--load_ensemble", type=str, required=False, help="Path to load ensemble/boosting config.")
    
    # Weighted Average Ensemble specific
    parser.add_argument(
        "--weight_strategy",
        type=str,
        required=False,
        default="inverse_variance",
        choices=["equal", "inverse_variance", "inverse_rmse", "grid_search", "linear_regression"],
        help="Weight strategy for weighted average ensemble."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=1.0,
        help="Boosting shrinkage factor applied at each stage (>0)."
    )

    # Stacking specific
    parser.add_argument(
        "--meta_l2",
        type=float,
        required=False,
        default=0.1,
        help="L2 regularisation strength for stacking meta-model (>=0)."
    )
    parser.add_argument("--base_models", nargs="+", required=False, help="Base models for Stacking v2.")

    args = parser.parse_args()
    try:
        if args.method == "ma":
            if not args.load_model and (args.window is None or args.data is None or args.train_start is None):
                raise ValueError("MA training requires --window, --data, and train/eval ranges.")
            ma.run(args)

        elif args.method == "es":
            if not args.load_model and (args.alpha is None or args.data is None or args.train_start is None):
                raise ValueError("ES training requires --alpha, --data, and train/eval ranges.")
            es.run(args)

        elif args.method == "arima":
            if not args.load_model and (args.d is None or args.data is None or args.train_start is None):
                raise ValueError("ARIMA training requires --d, --data, and train/eval ranges.")
            arima.run(args)

        elif args.method == "lstm":
            from models import lstm
            if not args.load_model and (args.data is None or args.train_start is None):
                raise ValueError("LSTM training requires --data and train/eval ranges.")
            lstm.run(args)

        elif args.method == "weighted_average":
            from models import weighted_average
            if not args.load_ensemble and (args.models is None or len(args.models) < 2 or args.data is None or args.eval_start is None):
                raise ValueError("Weighted average training requires at least 2 --models, --data, and --eval_start/--eval_end.")
            weighted_average.run(args)
            
        elif args.method == "boosting":
            from models import boosting
            if not args.load_ensemble and (args.models is None or args.data is None or args.train_start is None or args.eval_start is None):
                 raise ValueError("Boosting training requires --models, --data, and train/eval ranges.")
            boosting.run(args)

        elif args.method == "stacking":
            from models import stacking
            if not args.load_ensemble and (args.models is None or args.data is None or args.eval_start is None):
                raise ValueError("Stacking training requires --models, --data, and --eval_start/--eval_end.")
            stacking.run(args)

        elif args.method == "stackingv2":
            from models import stackingv2
            if args.data is None or args.eval_start is None:
                raise ValueError("Stacking v2 training requires --data and --eval_start/--eval_end.")
            stackingv2.run(args)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
