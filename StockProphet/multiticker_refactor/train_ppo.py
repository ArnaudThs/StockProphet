"""
Script to train PPO agent on the unified feature dataset.
Source: Reinforcement.ipynb training cells
"""
import argparse

from multiticker_refactor.config import PPO_TIMESTEPS, PPO_WINDOW_SIZE, PPO_TRAIN_RATIO
from multiticker_refactor.pipeline import build_feature_dataset, prepare_for_ppo
from multiticker_refactor.envs.trading_env import (
    create_train_test_envs, create_eval_callback_env
)
from multiticker_refactor.models.ppo import (
    create_ppo_model, create_callbacks, train_ppo, save_ppo_model
)


def main(
    timesteps: int = PPO_TIMESTEPS,
    include_rnn: bool = True,
    include_sentiment: bool = True,
    tensorboard_log: str = "./ppo_logs/"
):
    """
    Main training function.

    Args:
        timesteps: Total training timesteps
        include_rnn: Include RNN predictions in features
        include_sentiment: Include sentiment in features
        tensorboard_log: Path for tensorboard logs
    """
    print("=" * 60)
    print("PPO TRADING AGENT TRAINING")
    print("=" * 60)

    # Step 1: Build feature dataset
    print("\n[Step 1] Building feature dataset...")
    df = build_feature_dataset(
        include_rnn=include_rnn,
        include_sentiment=include_sentiment,
        verbose=True
    )

    # Step 2: Prepare data for PPO
    print("\n[Step 2] Preparing data for PPO...")
    prices, signal_features, feature_cols = prepare_for_ppo(df)
    print(f"Prices shape: {prices.shape}")
    print(f"Signal features shape: {signal_features.shape}")
    print(f"Number of features: {len(feature_cols)}")

    # Step 3: Create environments
    print("\n[Step 3] Creating environments...")
    train_env, test_env, train_bound, test_bound = create_train_test_envs(
        df, prices, signal_features,
        window_size=PPO_WINDOW_SIZE,
        train_ratio=PPO_TRAIN_RATIO
    )

    # Create eval callback env (uses test bounds)
    eval_env = create_eval_callback_env(
        df, prices, signal_features,
        frame_bound=test_bound,
        window_size=PPO_WINDOW_SIZE
    )

    # Step 4: Create PPO model
    print("\n[Step 4] Creating PPO model...")
    model = create_ppo_model(
        train_env,
        tensorboard_log=tensorboard_log,
        verbose=1
    )

    # Step 5: Create callbacks
    print("\n[Step 5] Setting up callbacks...")
    callbacks = create_callbacks(
        train_env, eval_env,
        eval_freq=5000,
        checkpoint_freq=10000
    )

    # Step 6: Train
    print(f"\n[Step 6] Training for {timesteps:,} timesteps...")
    model = train_ppo(model, total_timesteps=timesteps, callbacks=callbacks)

    # Step 7: Save model
    print("\n[Step 7] Saving model...")
    save_ppo_model(model, train_env)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return model, train_env, test_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO trading agent")
    parser.add_argument("--timesteps", type=int, default=PPO_TIMESTEPS,
                        help=f"Total training timesteps (default: {PPO_TIMESTEPS})")
    parser.add_argument("--no-rnn", action="store_true",
                        help="Exclude RNN predictions from features")
    parser.add_argument("--no-sentiment", action="store_true",
                        help="Exclude sentiment from features")
    parser.add_argument("--tensorboard", type=str, default="./ppo_logs/",
                        help="Tensorboard log directory")

    args = parser.parse_args()

    main(
        timesteps=args.timesteps,
        include_rnn=not args.no_rnn,
        include_sentiment=not args.no_sentiment,
        tensorboard_log=args.tensorboard
    )
