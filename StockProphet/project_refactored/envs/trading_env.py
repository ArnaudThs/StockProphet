"""
Trading environment wrappers for PPO training.
Source: Reinforcement.ipynb environment setup cells

Supports both:
- Discrete environment (FlexibleTradingEnv): action in {0=short, 1=long}
- Continuous environment (ContinuousTradingEnv): action in [-1, 1] position weight
"""
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import environments from gym-anytrading
import sys
sys.path.append("/Users/pnl1f276/code/ArnaudThs/StockProphet/gym-anytrading")
from gym_anytrading.envs.flexible_env import FlexibleTradingEnv
from gym_anytrading.envs.continuous_env import ContinuousTradingEnv

from project_refactored.config import (
    PPO_WINDOW_SIZE, PPO_TRAIN_RATIO, REWARD_CONFIG, VEC_NORMALIZE_PATH,
    ENV_TYPE, INITIAL_CAPITAL, CONTINUOUS_ENV_CONFIG
)


def make_env(
    df: pd.DataFrame,
    prices: np.ndarray,
    signal_features: np.ndarray,
    frame_bound: tuple,
    window_size: int = PPO_WINDOW_SIZE,
    reward_config: dict = None,
    env_type: str = None
):
    """
    Factory function to create a trading environment.

    Args:
        df: DataFrame with features
        prices: Array of close prices
        signal_features: Array of features (n_samples, n_features)
        frame_bound: Tuple of (start_tick, end_tick)
        window_size: Observation window size
        reward_config: Reward configuration dict
        env_type: 'discrete' or 'continuous' (default from config)

    Returns:
        Callable that creates the environment
    """
    if env_type is None:
        env_type = ENV_TYPE

    if env_type == 'continuous':
        return make_continuous_env(
            df, prices, signal_features, frame_bound, window_size
        )
    else:
        return make_discrete_env(
            df, prices, signal_features, frame_bound, window_size, reward_config
        )


def make_discrete_env(
    df: pd.DataFrame,
    prices: np.ndarray,
    signal_features: np.ndarray,
    frame_bound: tuple,
    window_size: int = PPO_WINDOW_SIZE,
    reward_config: dict = None
):
    """
    Factory function to create a discrete FlexibleTradingEnv.

    Args:
        df: DataFrame with features
        prices: Array of close prices
        signal_features: Array of features (n_samples, n_features)
        frame_bound: Tuple of (start_tick, end_tick)
        window_size: Observation window size
        reward_config: Reward configuration dict

    Returns:
        Callable that creates the environment
    """
    if reward_config is None:
        reward_config = REWARD_CONFIG

    def _init():
        env = FlexibleTradingEnv(
            df=df,
            prices=prices,
            signal_features=signal_features,
            window_size=window_size,
            frame_bound=frame_bound,
            include_position_in_obs=True,
            fee=reward_config.get('fee', 0.0005),
            holding_cost=reward_config.get('holding_cost', 0.0),
            short_borrow_cost=reward_config.get('short_borrow_cost', 0.0),
        )
        # Apply extra shaping attributes only if the env supports them
        for k, v in reward_config.items():
            if hasattr(env, k):
                setattr(env, k, v)
        return env

    return _init


def make_continuous_env(
    df: pd.DataFrame,
    prices: np.ndarray,
    signal_features: np.ndarray,
    frame_bound: tuple,
    window_size: int = PPO_WINDOW_SIZE,
    initial_capital: float = None,
    env_config: dict = None
):
    """
    Factory function to create a continuous ContinuousTradingEnv.

    Args:
        df: DataFrame with features
        prices: Array of close prices
        signal_features: Array of features (n_samples, n_features)
        frame_bound: Tuple of (start_tick, end_tick)
        window_size: Observation window size
        initial_capital: Starting capital in dollars
        env_config: Environment configuration dict

    Returns:
        Callable that creates the environment
    """
    if initial_capital is None:
        initial_capital = INITIAL_CAPITAL
    if env_config is None:
        env_config = CONTINUOUS_ENV_CONFIG

    def _init():
        env = ContinuousTradingEnv(
            df=df,
            prices=prices,
            signal_features=signal_features,
            window_size=window_size,
            frame_bound=frame_bound,
            initial_capital=initial_capital,
            fee=env_config.get('fee', 0.001),
            short_borrow_rate=env_config.get('short_borrow_rate', 0.0001),
            include_position_in_obs=True,
        )
        return env

    return _init


def create_train_test_envs(
    df: pd.DataFrame,
    prices: np.ndarray,
    signal_features: np.ndarray,
    window_size: int = PPO_WINDOW_SIZE,
    train_ratio: float = PPO_TRAIN_RATIO,
    reward_config: dict = None,
    env_type: str = None
) -> tuple:
    """
    Create train and test environments with VecNormalize wrappers.

    Args:
        df: DataFrame with features
        prices: Array of close prices
        signal_features: Array of features
        window_size: Observation window size
        train_ratio: Fraction of data for training
        reward_config: Reward configuration dict
        env_type: 'discrete' or 'continuous' (default from config)

    Returns:
        Tuple of (train_env, test_env, train_frame_bound, test_frame_bound)
    """
    if reward_config is None:
        reward_config = REWARD_CONFIG
    if env_type is None:
        env_type = ENV_TYPE

    total_len = len(df)
    train_end = int(total_len * train_ratio)

    train_frame_bound = (window_size, train_end)
    test_frame_bound = (train_end, total_len)

    print(f"Environment type: {env_type}")
    print(f"Total samples: {total_len}")
    print(f"Train: {train_frame_bound} ({train_end - window_size} steps)")
    print(f"Test:  {test_frame_bound} ({total_len - train_end} steps)")
    if env_type == 'continuous':
        print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")

    # Create training environment
    train_env = DummyVecEnv([
        make_env(df, prices, signal_features, train_frame_bound, window_size, reward_config, env_type)
    ])
    # Optimized VecNormalize: disable reward normalization to preserve learning signal
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,  # Critical: don't mask sparse rewards
        clip_obs=5.0,       # Tighter clipping
        clip_reward=None    # No reward clipping
    )

    # Create test environment (separate, not for training)
    test_env = DummyVecEnv([
        make_env(df, prices, signal_features, test_frame_bound, window_size, reward_config, env_type)
    ])
    test_env = VecNormalize(test_env, training=False, norm_obs=True, norm_reward=False)

    return train_env, test_env, train_frame_bound, test_frame_bound


def create_eval_callback_env(
    df: pd.DataFrame,
    prices: np.ndarray,
    signal_features: np.ndarray,
    frame_bound: tuple,
    window_size: int = PPO_WINDOW_SIZE,
    reward_config: dict = None,
    env_type: str = None
) -> VecNormalize:
    """
    Create evaluation environment for callbacks during training.

    Args:
        df: DataFrame with features
        prices: Array of close prices
        signal_features: Array of features
        frame_bound: Frame bound for evaluation
        window_size: Observation window size
        reward_config: Reward configuration dict
        env_type: 'discrete' or 'continuous' (default from config)

    Returns:
        VecNormalize wrapped environment
    """
    if reward_config is None:
        reward_config = REWARD_CONFIG
    if env_type is None:
        env_type = ENV_TYPE

    eval_env = DummyVecEnv([
        make_env(df, prices, signal_features, frame_bound, window_size, reward_config, env_type)
    ])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    return eval_env


def load_test_env(
    df: pd.DataFrame,
    prices: np.ndarray,
    signal_features: np.ndarray,
    frame_bound: tuple,
    window_size: int = PPO_WINDOW_SIZE,
    reward_config: dict = None,
    vec_normalize_path: str = None,
    env_type: str = None
) -> VecNormalize:
    """
    Load test environment with saved VecNormalize stats.

    Args:
        df: DataFrame with features
        prices: Array of close prices
        signal_features: Array of features
        frame_bound: Frame bound for testing
        window_size: Observation window size
        reward_config: Reward configuration dict
        vec_normalize_path: Path to saved VecNormalize stats
        env_type: 'discrete' or 'continuous' (default from config)

    Returns:
        VecNormalize wrapped environment with loaded stats
    """
    if reward_config is None:
        reward_config = REWARD_CONFIG
    if vec_normalize_path is None:
        vec_normalize_path = VEC_NORMALIZE_PATH
    if env_type is None:
        env_type = ENV_TYPE

    test_env = DummyVecEnv([
        make_env(df, prices, signal_features, frame_bound, window_size, reward_config, env_type)
    ])
    test_env = VecNormalize.load(vec_normalize_path, test_env)
    test_env.training = False
    test_env.norm_reward = False

    return test_env


def sync_normalize_stats(train_env: VecNormalize, eval_env: VecNormalize):
    """Sync normalization statistics from train to eval environment."""
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
