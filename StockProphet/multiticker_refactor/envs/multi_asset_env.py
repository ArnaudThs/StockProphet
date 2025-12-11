"""
Multi-Asset Trading Environment - Wrapper for UnifiedTradingEnv

This module provides backward-compatible factory functions that use UnifiedTradingEnv.
The actual environment implementation is in trading_env_unified.py.
"""

import numpy as np
from typing import Tuple, Dict
from .trading_env_unified import UnifiedTradingEnv


# Re-export UnifiedTradingEnv as MultiAssetContinuousEnv for backward compatibility
MultiAssetContinuousEnv = UnifiedTradingEnv


def create_multi_asset_env(
    prices: np.ndarray,
    signal_features: np.ndarray,
    ticker_map: Dict[int, str],
    initial_capital: float = 10_000.0,
    frame_bound: Tuple[int, int] = None,
    **kwargs
) -> UnifiedTradingEnv:
    """
    Factory function to create multi-asset continuous trading environment.

    Args:
        prices: Close prices, shape (n_timesteps, n_tickers)
        signal_features: All features, shape (n_timesteps, n_features)
        ticker_map: Dict mapping array index to ticker symbol
        initial_capital: Starting capital
        frame_bound: (start_idx, end_idx) for train/val/test split
        **kwargs: Additional environment parameters

    Returns:
        UnifiedTradingEnv instance
    """
    return UnifiedTradingEnv(
        prices=prices,
        signal_features=signal_features,
        ticker_map=ticker_map,
        initial_capital=initial_capital,
        frame_bound=frame_bound,
        **kwargs
    )


def create_train_val_test_envs(
    prices: np.ndarray,
    signal_features: np.ndarray,
    ticker_map: Dict[int, str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    initial_capital: float = 10_000.0,
    **kwargs
) -> Tuple[UnifiedTradingEnv, UnifiedTradingEnv, UnifiedTradingEnv]:
    """
    Create train, validation, and test environments with 60/20/20 split.

    Args:
        prices: Close prices, shape (n_timesteps, n_tickers)
        signal_features: All features, shape (n_timesteps, n_features)
        ticker_map: Dict mapping array index to ticker symbol
        train_ratio: Fraction for training (default 0.6)
        val_ratio: Fraction for validation (default 0.2)
        initial_capital: Starting capital
        **kwargs: Additional environment parameters

    Returns:
        Tuple of (train_env, val_env, test_env)
    """
    total_len = len(prices)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    train_frame_bound = (0, train_end)
    val_frame_bound = (train_end, val_end)
    test_frame_bound = (val_end, total_len)

    train_env = create_multi_asset_env(
        prices, signal_features, ticker_map,
        initial_capital=initial_capital,
        frame_bound=train_frame_bound,
        **kwargs
    )

    val_env = create_multi_asset_env(
        prices, signal_features, ticker_map,
        initial_capital=initial_capital,
        frame_bound=val_frame_bound,
        **kwargs
    )

    test_env = create_multi_asset_env(
        prices, signal_features, ticker_map,
        initial_capital=initial_capital,
        frame_bound=test_frame_bound,
        **kwargs
    )

    return train_env, val_env, test_env
