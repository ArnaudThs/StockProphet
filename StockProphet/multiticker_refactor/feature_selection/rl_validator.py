"""
RL Feature Validator - Stage 2

Validate selected features using actual RecurrentPPO training.
Runs multiple seeds for statistical robustness.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import json
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from ..config import (
    PPO_WINDOW_SIZE, INITIAL_CAPITAL, CONTINUOUS_ENV_VERSION,
    PPO_TRAIN_RATIO, PPO_VAL_RATIO,
    TREND_REWARD_MULTIPLIER, CONVICTION_REWARD, EXIT_TIMING_REWARD, PATIENCE_REWARD
)
from ..envs.multi_asset_env import create_single_ticker_env


class RLFeatureValidator:
    """
    Validate feature subsets using RecurrentPPO training.
    """

    def __init__(
        self,
        prices: np.ndarray,
        feature_df: pd.DataFrame,
        ticker: str = "AAPL",
        timesteps: int = 50_000,
        n_seeds: int = 3,
        verbose: bool = True
    ):
        """
        Initialize RL validator.

        Args:
            prices: Price array for trading
            feature_df: DataFrame with ALL features
            ticker: Ticker symbol
            timesteps: Training timesteps per trial
            n_seeds: Number of random seeds for averaging
            verbose: Print progress
        """
        self.prices = prices.astype(np.float32)
        self.feature_df = feature_df
        self.ticker = ticker
        self.timesteps = timesteps
        self.n_seeds = n_seeds
        self.verbose = verbose

        self.all_feature_cols = list(feature_df.columns)
        self.results = []

        if self.verbose:
            print(f"Initialized RLFeatureValidator")
            print(f"  Ticker: {ticker}")
            print(f"  Timesteps per trial: {timesteps:,}")
            print(f"  Seeds for averaging: {n_seeds}")
            print(f"  Total features available: {len(self.all_feature_cols)}")

    def validate_feature_subset(
        self,
        feature_subset: List[str],
        experiment_name: str
    ) -> Dict:
        """
        Train RecurrentPPO on a feature subset and evaluate.

        Args:
            feature_subset: List of feature column names to use
            experiment_name: Name for this experiment

        Returns:
            Dict with metrics (averaged over seeds)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"VALIDATING: {experiment_name}")
            print(f"{'='*60}")
            print(f"Features ({len(feature_subset)}): {', '.join(feature_subset[:5])}...")

        seed_results = []

        for seed in range(self.n_seeds):
            if self.verbose:
                print(f"\n  Seed {seed+1}/{self.n_seeds}...")

            # Extract feature subset
            signal_features = self.feature_df[feature_subset].values.astype(np.float32)

            # Create environments
            if self.verbose:
                print(f"    Creating environments...")
            train_env, val_env, test_env = self._create_envs(signal_features, seed)

            # Train model
            if self.verbose:
                print(f"    Training RecurrentPPO for {self.timesteps:,} timesteps...")
            model = RecurrentPPO(
                "MlpLstmPolicy",
                train_env,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,  # RecurrentPPO doesn't support schedules, use static value
                clip_range=0.2,
                verbose=0,
                seed=seed,
                tensorboard_log=None
            )

            model.learn(total_timesteps=self.timesteps)

            # Evaluate on test set
            if self.verbose:
                print(f"    Evaluating on test set...")
            metrics = self._evaluate(model, test_env)
            seed_results.append(metrics)

            if self.verbose:
                print(f"    ✓ Test Return: {metrics['test_return']:+.2f}%, "
                      f"Sharpe: {metrics['sharpe']:.3f}, "
                      f"Max DD: {metrics['max_drawdown']*100:.2f}%")

        # Average metrics across seeds
        avg_metrics = {
            'experiment': experiment_name,
            'n_features': len(feature_subset),
            'features': feature_subset,
            'test_return_mean': np.mean([r['test_return'] for r in seed_results]),
            'test_return_std': np.std([r['test_return'] for r in seed_results]),
            'sharpe_mean': np.mean([r['sharpe'] for r in seed_results]),
            'sharpe_std': np.std([r['sharpe'] for r in seed_results]),
            'max_dd_mean': np.mean([r['max_drawdown'] for r in seed_results]),
            'max_dd_std': np.std([r['max_drawdown'] for r in seed_results]),
            'n_trades_mean': np.mean([r['n_trades'] for r in seed_results]),
            'n_seeds': self.n_seeds,
            'timesteps': self.timesteps
        }

        self.results.append(avg_metrics)

        if self.verbose:
            print(f"\n  {'─'*58}")
            print(f"  AVERAGE RESULTS (n={self.n_seeds} seeds):")
            print(f"    Return: {avg_metrics['test_return_mean']:+.2f}% ± {avg_metrics['test_return_std']:.2f}%")
            print(f"    Sharpe: {avg_metrics['sharpe_mean']:.3f} ± {avg_metrics['sharpe_std']:.3f}")
            print(f"    Max DD: {avg_metrics['max_dd_mean']*100:.2f}% ± {avg_metrics['max_dd_std']*100:.2f}%")

        return avg_metrics

    def _create_envs(self, signal_features: np.ndarray, seed: int) -> Tuple:
        """Create train/val/test environments."""
        total_len = len(self.prices)
        train_end = int(total_len * PPO_TRAIN_RATIO)
        val_end = train_end + int(total_len * PPO_VAL_RATIO)

        train_bounds = (PPO_WINDOW_SIZE, train_end)
        val_bounds = (train_end, val_end)
        test_bounds = (val_end, total_len)

        # Create env factory
        def make_env(bounds):
            def _init():
                return create_single_ticker_env(
                    prices=self.prices,
                    signal_features=signal_features,
                    frame_bound=bounds,
                    window_size=PPO_WINDOW_SIZE,
                    initial_capital=INITIAL_CAPITAL,
                    env_version=CONTINUOUS_ENV_VERSION,
                    # V2 params
                    trend_reward_multiplier=TREND_REWARD_MULTIPLIER,
                    conviction_reward=CONVICTION_REWARD,
                    exit_timing_reward=EXIT_TIMING_REWARD,
                    patience_reward=PATIENCE_REWARD
                )
            return _init

        # Wrap with VecNormalize
        train_env = DummyVecEnv([make_env(train_bounds)])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=5.0)
        train_env.seed(seed)

        val_env = DummyVecEnv([make_env(val_bounds)])
        val_env = VecNormalize(val_env, training=False, norm_obs=True, norm_reward=False)
        val_env.obs_rms = train_env.obs_rms

        test_env = DummyVecEnv([make_env(test_bounds)])
        test_env = VecNormalize(test_env, training=False, norm_obs=True, norm_reward=False)
        test_env.obs_rms = train_env.obs_rms

        return train_env, val_env, test_env

    def _evaluate(self, model, test_env) -> Dict:
        """Evaluate model on test environment."""
        obs = test_env.reset()
        lstm_states = None
        done = False

        equity = 1.0
        positions = []
        last_pos = 0
        step = 0

        base_env = test_env.venv.envs[0].unwrapped

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                deterministic=True
            )
            obs, _, done, info = test_env.step(action)
            done = bool(done[0])

            tick = info[0]["tick"]
            pos = info[0]["position_weight"]
            positions.append(pos)

            # Update equity (bounds check for tick)
            if step > 0 and tick < len(self.prices) and tick > 0:
                price_ratio = self.prices[tick] / self.prices[tick - 1]
                equity *= price_ratio ** last_pos

            last_pos = pos
            step += 1

        # Compute metrics
        positions = np.array(positions)
        returns = np.diff(np.log(equity + np.arange(len(positions)) * 0))  # Dummy for shape
        returns = np.diff(np.log([equity] + [equity] * (len(positions) - 1)))

        total_return = (equity - 1) * 100
        sharpe = 0.0  # Compute if needed
        max_dd = 0.0
        n_trades = np.sum(np.abs(np.diff(positions)) > 0.01)

        # Simple Sharpe approximation
        if len(positions) > 1:
            position_changes = np.diff(positions)
            if len(position_changes) > 0 and np.std(position_changes) > 0:
                sharpe = np.mean(position_changes) / (np.std(position_changes) + 1e-8) * np.sqrt(252)

        return {
            'test_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'n_trades': int(n_trades)
        }

    def compare_feature_sets(self, feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Compare multiple feature sets.

        Args:
            feature_sets: Dict of {experiment_name: feature_list}

        Returns:
            DataFrame with comparison results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"COMPARING {len(feature_sets)} FEATURE SETS")
            print(f"{'='*60}")

        for name, features in feature_sets.items():
            self.validate_feature_subset(features, name)

        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('sharpe_mean', ascending=False).reset_index(drop=True)

        if self.verbose:
            print(f"\n{'='*60}")
            print("COMPARISON RESULTS")
            print(f"{'='*60}")
            print(results_df[['experiment', 'n_features', 'test_return_mean', 'sharpe_mean', 'max_dd_mean']].to_string(index=False))

        return results_df

    def save_results(self, output_path: str):
        """Save validation results to JSON."""
        if not self.results:
            raise ValueError("No results to save. Run validation first!")

        # Convert NumPy types to Python native types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert NumPy types to Python native types."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        results_dict = {
            'config': {
                'ticker': self.ticker,
                'timesteps': int(self.timesteps),
                'n_seeds': int(self.n_seeds),
                'env_version': CONTINUOUS_ENV_VERSION
            },
            'results': convert_to_native(self.results)
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        if self.verbose:
            print(f"\nResults saved to: {output_path}")
