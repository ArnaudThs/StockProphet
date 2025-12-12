"""
Main entry point for the StockProphet trading system.

This script orchestrates the full pipeline:
1. Build feature dataset (OHLCV + technicals + RNN predictions + sentiment)
2. Train RecurrentPPO agent on the enriched dataset
3. Evaluate agent performance

Usage:
    python -m project_refactored.main [--mode train|evaluate|full] [--simple-rnn]
"""
import argparse
import os

from .config import (
    PPO_TIMESTEPS, VEC_NORMALIZE_PATH, PPO_WINDOW_SIZE, PPO_TRAIN_RATIO,
    RECURRENT_PPO_MODEL_PATH, ENV_TYPE, INITIAL_CAPITAL
)
from .pipeline import build_feature_dataset, prepare_for_ppo
from .envs.trading_env import (
    create_train_test_envs, create_eval_callback_env, load_test_env
)
from .models.ppo import (
    create_model, create_continuous_model, create_callbacks,
    train_model, save_model, load_model
)
from .evaluate import evaluate_agent_auto


def train_mode(
    timesteps: int = PPO_TIMESTEPS,
    include_rnn: bool = True,
    include_sentiment: bool = True,
    probabilistic_rnn: bool = True,
    tensorboard_log: str = "./ppo_logs/"
):
    """
    Run the full training pipeline.

    Args:
        timesteps: Total training timesteps
        include_rnn: Include RNN predictions in features
        include_sentiment: Include sentiment in features
        probabilistic_rnn: Use Probabilistic Multi-Horizon LSTM (vs simple LSTM)
        tensorboard_log: Path for tensorboard logs

    Returns:
        Tuple of (model, train_env, test_env)
    """
    rnn_type = "Probabilistic Multi-Horizon" if probabilistic_rnn else "Simple"

    print("=" * 60)
    print("STOCKPROPHET - TRAINING MODE (RecurrentPPO)")
    print("=" * 60)
    print(f"RNN Type: {rnn_type if include_rnn else 'Disabled'}")

    # Step 1: Build feature dataset
    print("\n[1/5] Building feature dataset...")
    df = build_feature_dataset(
        include_rnn=include_rnn,
        include_sentiment=include_sentiment,
        probabilistic_rnn=probabilistic_rnn,
        verbose=True
    )

    # Step 2: Prepare data for PPO
    print("\n[2/5] Preparing data for PPO...")
    prices, signal_features, feature_cols = prepare_for_ppo(df)
    print(f"Prices shape: {prices.shape}")
    print(f"Signal features shape: {signal_features.shape}")
    print(f"Number of features: {len(feature_cols)}")

    # Step 3: Create environments
    print("\n[3/5] Creating environments...")
    train_env, test_env, train_bound, test_bound = create_train_test_envs(
        df, prices, signal_features,
        window_size=PPO_WINDOW_SIZE,
        train_ratio=PPO_TRAIN_RATIO
    )

    eval_env = create_eval_callback_env(
        df, prices, signal_features,
        frame_bound=test_bound,
        window_size=PPO_WINDOW_SIZE
    )

    # Step 4: Create and train model
    print(f"\n[4/5] Training RecurrentPPO model...")
    print(f"Environment type: {ENV_TYPE}")
    if ENV_TYPE == 'continuous':
        print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")

    # Select appropriate model based on env type
    if ENV_TYPE == 'continuous':
        model = create_continuous_model(
            train_env,
            tensorboard_log=tensorboard_log,
            verbose=1
        )
    else:
        model = create_model(
            train_env,
            tensorboard_log=tensorboard_log,
            verbose=1
        )

    callbacks = create_callbacks(
        train_env, eval_env,
        eval_freq=5000,
        checkpoint_freq=10000
    )

    model = train_model(model, total_timesteps=timesteps, callbacks=callbacks)

    # Step 5: Save model
    print("\n[5/5] Saving model...")
    save_model(model, train_env)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return model, train_env, test_env


def evaluate_mode(
    include_rnn: bool = True,
    include_sentiment: bool = True,
    probabilistic_rnn: bool = True,
    model_path: str = None,
    vec_normalize_path: str = None
):
    """
    Run evaluation on a trained RecurrentPPO model.

    Args:
        include_rnn: Include RNN predictions in features
        include_sentiment: Include sentiment in features
        probabilistic_rnn: Use Probabilistic Multi-Horizon LSTM
        model_path: Path to saved model
        vec_normalize_path: Path to saved VecNormalize stats

    Returns:
        Evaluation results dictionary
    """
    if model_path is None:
        model_path = RECURRENT_PPO_MODEL_PATH
    if vec_normalize_path is None:
        vec_normalize_path = VEC_NORMALIZE_PATH

    print("=" * 60)
    print("STOCKPROPHET - EVALUATION MODE (RecurrentPPO)")
    print("=" * 60)

    # Check if model exists
    model_path = str(model_path)  # Convert Path to string if needed
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using: python -m project_refactored.main --mode train")
        return None

    # Step 1: Build feature dataset
    print("\n[1/3] Building feature dataset...")
    df = build_feature_dataset(
        include_rnn=include_rnn,
        include_sentiment=include_sentiment,
        probabilistic_rnn=probabilistic_rnn,
        verbose=True
    )

    # Step 2: Prepare data and create test environment
    print("\n[2/3] Preparing test environment...")
    prices, signal_features, feature_cols = prepare_for_ppo(df)

    total_len = len(df)
    train_end = int(total_len * PPO_TRAIN_RATIO)
    test_bound = (train_end, total_len)

    test_env = load_test_env(
        df, prices, signal_features,
        frame_bound=test_bound,
        window_size=PPO_WINDOW_SIZE,
        vec_normalize_path=vec_normalize_path
    )

    # Step 3: Load model and evaluate
    print(f"\n[3/3] Loading RecurrentPPO model and evaluating...")
    model = load_model(model_path)

    results = evaluate_agent_auto(model, test_env, episodes=1)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return results


def full_mode(
    timesteps: int = PPO_TIMESTEPS,
    include_rnn: bool = True,
    include_sentiment: bool = True,
    probabilistic_rnn: bool = True,
    tensorboard_log: str = "./ppo_logs/"
):
    """
    Run full pipeline: train and then evaluate.

    Args:
        timesteps: Total training timesteps
        include_rnn: Include RNN predictions in features
        include_sentiment: Include sentiment in features
        probabilistic_rnn: Use Probabilistic Multi-Horizon LSTM
        tensorboard_log: Path for tensorboard logs

    Returns:
        Tuple of (model, results)
    """
    print("=" * 60)
    print("STOCKPROPHET - FULL PIPELINE (RecurrentPPO)")
    print("=" * 60)

    # Train
    model, train_env, test_env = train_mode(
        timesteps=timesteps,
        include_rnn=include_rnn,
        include_sentiment=include_sentiment,
        probabilistic_rnn=probabilistic_rnn,
        tensorboard_log=tensorboard_log
    )

    if model is None:
        return None, None

    # Sync normalization stats to test env
    test_env.obs_rms = train_env.obs_rms
    test_env.ret_rms = train_env.ret_rms

    # Evaluate
    print("\n" + "=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)

    results = evaluate_agent_auto(model, test_env, episodes=1)

    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETE")
    print("=" * 60)

    return model, results


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="StockProphet Trading System (RecurrentPPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train RecurrentPPO with probabilistic RNN:
    python -m project_refactored.main --mode train --timesteps 100000

  Train with simple RNN (point predictions):
    python -m project_refactored.main --mode train --simple-rnn

  Evaluate an existing model:
    python -m project_refactored.main --mode evaluate

  Full pipeline (train + evaluate):
    python -m project_refactored.main --mode full

  Train without RNN predictions:
    python -m project_refactored.main --mode train --no-rnn
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "full"],
        default="full",
        help="Mode to run: train, evaluate, or full (default: full)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=PPO_TIMESTEPS,
        help=f"Total training timesteps (default: {PPO_TIMESTEPS})"
    )
    parser.add_argument(
        "--no-rnn",
        action="store_true",
        help="Exclude RNN predictions from features"
    )
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Exclude sentiment from features"
    )
    parser.add_argument(
        "--simple-rnn",
        action="store_true",
        help="Use simple LSTM (point predictions) instead of Probabilistic Multi-Horizon LSTM"
    )
    parser.add_argument(
        "--tensorboard",
        type=str,
        default="./ppo_logs/",
        help="Tensorboard log directory"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model for evaluation (default: from config)"
    )

    args = parser.parse_args()

    include_rnn = not args.no_rnn
    include_sentiment = not args.no_sentiment
    probabilistic_rnn = not args.simple_rnn

    if args.mode == "train":
        train_mode(
            timesteps=args.timesteps,
            include_rnn=include_rnn,
            include_sentiment=include_sentiment,
            probabilistic_rnn=probabilistic_rnn,
            tensorboard_log=args.tensorboard
        )
    elif args.mode == "evaluate":
        evaluate_mode(
            include_rnn=include_rnn,
            include_sentiment=include_sentiment,
            probabilistic_rnn=probabilistic_rnn,
            model_path=args.model_path
        )
    else:  # full
        full_mode(
            timesteps=args.timesteps,
            include_rnn=include_rnn,
            include_sentiment=include_sentiment,
            probabilistic_rnn=probabilistic_rnn,
            tensorboard_log=args.tensorboard
        )


if __name__ == "__main__":
    main()
