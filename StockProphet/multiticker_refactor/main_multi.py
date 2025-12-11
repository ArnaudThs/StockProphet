"""
Multi-Ticker Trading System - Main Entry Point

Train and evaluate RecurrentPPO agent on multiple tickers simultaneously.

Usage:
    # Train with 3 tickers (default AAPL, MSFT, GOOGL)
    python -m multiticker_refactor.main_multi --mode train --timesteps 200000

    # Evaluate existing model
    python -m multiticker_refactor.main_multi --mode evaluate

    # Full pipeline (train + evaluate)
    python -m multiticker_refactor.main_multi --mode full --timesteps 200000

    # Custom tickers
    python -m multiticker_refactor.main_multi --mode train --tickers AAPL TSLA NVDA
"""

import argparse
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn

from .config import (
    # Tickers and dates
    TICKERS,
    START_DATE,
    END_DATE,
    # Environment parameters
    INITIAL_CAPITAL,
    TRANSACTION_FEE_PCT,
    SHORT_BORROW_RATE,
    REWARD_VOLATILITY_WINDOW,
    # PPO training parameters
    PPO_TIMESTEPS,
    PPO_TRAIN_RATIO,
    PPO_VAL_RATIO,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_BATCH_SIZE,
    PPO_N_EPOCHS,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF_START,
    PPO_ENT_COEF_END,
    PPO_EVAL_FREQ,
    PPO_CHECKPOINT_FREQ,
    # VecNormalize parameters
    VECNORMALIZE_NORM_OBS,
    VECNORMALIZE_NORM_REWARD,
    VECNORMALIZE_CLIP_OBS,
    VECNORMALIZE_CLIP_REWARD,
)
from .pipeline import build_multi_ticker_dataset, prepare_multi_ticker_for_ppo
from .envs import create_train_val_test_envs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Ticker Trading System")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "full"],
        default="full",
        help="Operating mode: train, evaluate, or full pipeline"
    )

    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=TICKERS,
        help="List of ticker symbols (max 3)"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=PPO_TIMESTEPS,
        help="Number of PPO training timesteps"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=START_DATE,
        help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=END_DATE,
        help="End date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--no-rnn",
        action="store_true",
        help="Skip RNN training"
    )

    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Skip sentiment data"
    )

    parser.add_argument(
        "--simple-rnn",
        action="store_true",
        help="Use simple RNN (not probabilistic)"
    )

    return parser.parse_args()


def train_multi_ticker(
    tickers: list,
    start_date: str,
    end_date: str,
    timesteps: int,
    include_rnn: bool = True,
    include_sentiment: bool = True,
    probabilistic_rnn: bool = True
):
    """
    Train RecurrentPPO agent on multi-ticker environment.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timesteps: Number of training timesteps
        include_rnn: Whether to train RNNs
        include_sentiment: Whether to include sentiment
        probabilistic_rnn: Use probabilistic LSTM

    Returns:
        Trained model, VecNormalize wrapper, metadata
    """
    print("=" * 70)
    print("MULTI-TICKER TRADING SYSTEM - TRAINING")
    print("=" * 70)
    print(f"\nTickers: {tickers}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Training timesteps: {timesteps:,}")
    print(f"RNN: {include_rnn} (probabilistic: {probabilistic_rnn})")
    print(f"Sentiment: {include_sentiment}")
    print(f"Policy: RecurrentPPO (MlpLstmPolicy)\n")

    # =========================================================================
    # Step 1: Build multi-ticker dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: BUILDING MULTI-TICKER DATASET")
    print("=" * 70)

    df, metadata = build_multi_ticker_dataset(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        include_rnn=include_rnn,
        include_sentiment=include_sentiment,
        probabilistic_rnn=probabilistic_rnn,
        verbose=True
    )

    # =========================================================================
    # Step 2: Prepare data for PPO
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: PREPARING DATA FOR PPO")
    print("=" * 70)

    ppo_data = prepare_multi_ticker_for_ppo(df, tickers, validate=True)
    print(f"\n✅ Data prepared:")
    print(f"   Prices array: {ppo_data['prices'].shape}")
    print(f"   Signal features: {ppo_data['signal_features'].shape}")
    print(f"   Ticker map: {ppo_data['ticker_map']}")

    # =========================================================================
    # Step 3: Create environments
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: CREATING TRAIN/VAL/TEST ENVIRONMENTS")
    print("=" * 70)

    train_env, val_env, test_env = create_train_val_test_envs(
        prices=ppo_data['prices'],
        signal_features=ppo_data['signal_features'],
        ticker_map=ppo_data['ticker_map'],
        train_ratio=PPO_TRAIN_RATIO,
        val_ratio=PPO_VAL_RATIO,
        initial_capital=INITIAL_CAPITAL,
        transaction_fee_pct=TRANSACTION_FEE_PCT,
        short_borrow_rate=SHORT_BORROW_RATE,
        reward_volatility_window=REWARD_VOLATILITY_WINDOW
    )

    print(f"\n✅ Environments created:")
    print(f"   Train: {train_env.frame_bound}")
    print(f"   Val:   {val_env.frame_bound}")
    print(f"   Test:  {test_env.frame_bound}")

    # Wrap in DummyVecEnv
    train_env_vec = DummyVecEnv([lambda: train_env])
    val_env_vec = DummyVecEnv([lambda: val_env])

    # VecNormalize (normalize observations, NOT rewards)
    train_env_vec = VecNormalize(
        train_env_vec,
        norm_obs=VECNORMALIZE_NORM_OBS,
        norm_reward=VECNORMALIZE_NORM_REWARD,
        clip_obs=VECNORMALIZE_CLIP_OBS,
        clip_reward=VECNORMALIZE_CLIP_REWARD
    )

    val_env_vec = VecNormalize(
        val_env_vec,
        norm_obs=VECNORMALIZE_NORM_OBS,
        norm_reward=VECNORMALIZE_NORM_REWARD,
        clip_obs=VECNORMALIZE_CLIP_OBS,
        clip_reward=VECNORMALIZE_CLIP_REWARD,
        training=False  # Don't update stats on validation
    )

    # =========================================================================
    # Step 4: Create RecurrentPPO model with constrained policy
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: CREATING RECURRENT PPO MODEL")
    print("=" * 70)

    try:
        from sb3_contrib import RecurrentPPO
        from .models.constrained_policy import (
            ConstrainedRecurrentPolicy,
            get_constrained_policy_kwargs
        )
        print("\n✅ Using RecurrentPPO with ConstrainedRecurrentPolicy")
        print("   - Squashed Gaussian (tanh) instead of clipping")
        print("   - log_std clamped to [-2.0, 0.5] (std ∈ [0.14, 1.65])")
    except ImportError:
        raise ImportError(
            "sb3-contrib is required for RecurrentPPO. Install with: pip install sb3-contrib"
        )

    # Constrained policy kwargs - prevents std explosion
    policy_kwargs = get_constrained_policy_kwargs(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        lstm_hidden_size=64,
        n_lstm_layers=1,
        log_std_init=-0.5,   # std ≈ 0.6 (reasonable exploration)
        log_std_min=-2.0,    # std >= 0.14 (prevents collapse)
        log_std_max=0.5,     # std <= 1.65 (prevents explosion)
    )

    model = RecurrentPPO(
        ConstrainedRecurrentPolicy,
        train_env_vec,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF_START,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./ppo_multi_logs/"
    )

    print(f"   Policy: ConstrainedRecurrentPolicy")
    print(f"   Learning rate: {PPO_LEARNING_RATE}")
    print(f"   n_steps: {PPO_N_STEPS}")
    print(f"   Batch size: {PPO_BATCH_SIZE}")
    print(f"   Entropy coef: {PPO_ENT_COEF_START}")

    # =========================================================================
    # Step 5: Train model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: TRAINING PPO AGENT")
    print("=" * 70)

    # Callbacks
    eval_callback = EvalCallback(
        val_env_vec,
        best_model_save_path="./ppo_multi_best_model/",
        log_path="./ppo_multi_eval_logs/",
        eval_freq=PPO_EVAL_FREQ,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=PPO_CHECKPOINT_FREQ,
        save_path="./ppo_multi_checkpoints/",
        name_prefix="ppo_multi_trading_model"
    )

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback]
    )

    print("\n✅ Training complete!")

    # =========================================================================
    # Step 6: Save model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: SAVING MODEL")
    print("=" * 70)

    model_path = "./saved_models_multi/ppo_multi_trading.zip"
    vecnorm_path = "./saved_models_multi/vec_normalize_multi.pkl"

    model.save(model_path)
    train_env_vec.save(vecnorm_path)

    print(f"\n✅ Model saved:")
    print(f"   Model: {model_path}")
    print(f"   VecNormalize: {vecnorm_path}")

    # Also save metadata
    metadata['ppo_data'] = ppo_data
    np.save("./saved_models_multi/metadata_multi.npy", metadata)

    return model, train_env_vec, metadata


def save_episode_data_for_demo(episode_data: dict, metadata: dict, test_env):
    """
    Save episode data for Streamlit dashboard.

    Args:
        episode_data: Dict with portfolio_values, actions, rewards, etc.
        metadata: Metadata dict with ppo_data, tickers, etc.
        test_env: Test environment instance
    """
    import json
    from pathlib import Path

    demo_dir = Path("./episode_data/latest")
    demo_dir.mkdir(parents=True, exist_ok=True)

    ppo_data = metadata['ppo_data']
    tickers = ppo_data['tickers']
    prices_array = ppo_data['prices']

    # Get dates from ppo_data if available
    dates_array = ppo_data.get('dates', None)

    # Get test set indices
    total_len = len(prices_array)
    test_start = int(total_len * (PPO_TRAIN_RATIO + PPO_VAL_RATIO))

    # Convert lists to arrays
    portfolio_history = np.array(episode_data['portfolio_values'])
    actions = np.array(episode_data['actions'])
    rewards = np.array(episode_data['rewards'])

    # Get test set prices (shape: n_days, n_tickers)
    test_prices = prices_array[test_start:test_start + len(portfolio_history)]

    # Create date mapping if dates are available
    date_mapping = {}
    if dates_array is not None:
        test_dates = dates_array[test_start:test_start + len(portfolio_history)]
        # Create dict: {0: "2024-01-01", 1: "2024-01-02", ...}
        date_mapping = {i: str(date) for i, date in enumerate(test_dates)}

    # Save arrays
    np.save(demo_dir / "portfolio_history.npy", portfolio_history)
    np.save(demo_dir / "actions.npy", actions)
    np.save(demo_dir / "rewards.npy", rewards)
    np.save(demo_dir / "prices.npy", test_prices)

    # Save metadata JSON
    demo_metadata = {
        'tickers': tickers,
        'initial_capital': float(INITIAL_CAPITAL),
        'n_steps': len(portfolio_history),
        'final_value': float(portfolio_history[-1]),
        'total_profit': float(portfolio_history[-1] - INITIAL_CAPITAL),
        'return_pct': float((portfolio_history[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100),
        'dates': date_mapping  # Add date mapping
    }

    with open(demo_dir / "metadata.json", 'w') as f:
        json.dump(demo_metadata, f, indent=2)

    print(f"\n✅ Demo data saved to {demo_dir}")


def evaluate_multi_ticker(
    model_path: str = "./saved_models_multi/ppo_multi_trading.zip",
    vecnorm_path: str = "./saved_models_multi/vec_normalize_multi.pkl",
    metadata_path: str = "./saved_models_multi/metadata_multi.npy"
):
    """
    Evaluate trained multi-ticker PPO agent.

    Args:
        model_path: Path to saved model
        vecnorm_path: Path to VecNormalize stats
        metadata_path: Path to metadata

    Returns:
        Evaluation results dict
    """
    print("=" * 70)
    print("MULTI-TICKER TRADING SYSTEM - EVALUATION")
    print("=" * 70)

    # Load metadata
    metadata = np.load(metadata_path, allow_pickle=True).item()
    ppo_data = metadata['ppo_data']
    tickers = ppo_data['tickers']

    print(f"\n✅ Loaded metadata:")
    print(f"   Tickers: {tickers}")
    print(f"   Data shape: {ppo_data['prices'].shape}")

    # Create test environment
    _, _, test_env = create_train_val_test_envs(
        prices=ppo_data['prices'],
        signal_features=ppo_data['signal_features'],
        ticker_map=ppo_data['ticker_map'],
        train_ratio=PPO_TRAIN_RATIO,
        val_ratio=PPO_VAL_RATIO,
        initial_capital=INITIAL_CAPITAL,
        transaction_fee_pct=TRANSACTION_FEE_PCT,
        short_borrow_rate=SHORT_BORROW_RATE,
        reward_volatility_window=REWARD_VOLATILITY_WINDOW
    )

    test_env_vec = DummyVecEnv([lambda: test_env])

    # Load VecNormalize
    test_env_vec = VecNormalize.load(vecnorm_path, test_env_vec)
    test_env_vec.training = False
    test_env_vec.norm_reward = False

    # Load model (RecurrentPPO)
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path)
    except ImportError:
        raise ImportError(
            "sb3-contrib is required for RecurrentPPO. Install with: pip install sb3-contrib"
        )

    print("\n✅ Model and environment loaded")

    # Run evaluation
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION ON TEST SET")
    print("=" * 70)

    obs = test_env_vec.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    # Track episode data for Streamlit demo
    episode_data = {
        'portfolio_values': [],
        'positions': [],  # Shape: (n_steps, n_tickers)
        'actions': [],
        'rewards': [],
        'prices': [],
        'dates': []
    }

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env_vec.step(action)
        total_reward += reward[0]
        step_count += 1

        # Save episode data
        episode_data['portfolio_values'].append(info[0]['portfolio_value'])
        episode_data['rewards'].append(reward[0])
        episode_data['actions'].append(action[0])

        if step_count % 50 == 0:
            print(f"Step {step_count}: Portfolio value = ${info[0]['portfolio_value']:,.2f}")

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTotal steps: {step_count}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final portfolio value: ${info[0]['portfolio_value']:,.2f}")
    print(f"Total profit: ${info[0]['total_profit']:,.2f}")
    print(f"Return: {(info[0]['total_profit'] / INITIAL_CAPITAL) * 100:.2f}%")

    # Save episode data for Streamlit demo
    save_episode_data_for_demo(episode_data, metadata, test_env)

    return {
        'total_reward': total_reward,
        'final_value': info[0]['portfolio_value'],
        'total_profit': info[0]['total_profit'],
        'return_pct': (info[0]['total_profit'] / INITIAL_CAPITAL) * 100
    }


def main():
    """Main entry point."""
    args = parse_args()

    if args.mode in ["train", "full"]:
        # Determine include_sentiment: CLI --no-sentiment overrides config default
        from .config import INCLUDE_SENTIMENT
        include_sentiment = False if args.no_sentiment else INCLUDE_SENTIMENT

        train_multi_ticker(
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            timesteps=args.timesteps,
            include_rnn=not args.no_rnn,
            include_sentiment=include_sentiment,
            probabilistic_rnn=not args.simple_rnn
        )

    if args.mode in ["evaluate", "full"]:
        evaluate_multi_ticker()


if __name__ == "__main__":
    main()
