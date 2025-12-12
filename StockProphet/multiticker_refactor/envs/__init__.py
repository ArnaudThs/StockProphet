"""
Environments module for StockProphet.

This module provides a unified Gymnasium trading environment:
- trading_env.py: Unified environment with factory functions and backward compatibility
"""

from .trading_env import (
    UnifiedTradingEnv,
    create_multi_asset_env,
    create_train_val_test_envs,
    create_single_ticker_env,
    make_env,
    make_continuous_env,
    create_train_test_envs,
    create_eval_callback_env,
    load_test_env,
    sync_normalize_stats,
)

__all__ = [
    # Core environment class
    'UnifiedTradingEnv',

    # Factory functions (multi-asset)
    'create_multi_asset_env',
    'create_train_val_test_envs',
    'create_single_ticker_env',

    # Wrapper functions (backward compatibility)
    'make_env',
    'make_continuous_env',
    'create_train_test_envs',
    'create_eval_callback_env',
    'load_test_env',
    'sync_normalize_stats',
]
