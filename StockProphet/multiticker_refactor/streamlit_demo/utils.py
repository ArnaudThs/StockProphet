"""
Utility functions for Streamlit dashboard.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple


def load_episode_data(data_dir: str = None) -> Dict:
    """
    Load episode data from saved files.

    Args:
        data_dir: Directory containing episode data (if None, auto-detect)

    Returns:
        Dict with all episode data
    """
    if data_dir is None:
        # Auto-detect: try multiple possible locations
        possible_paths = [
            Path("./episode_data/latest"),  # From StockProphet/
            Path("../episode_data/latest"),  # From streamlit_demo/
            Path("./StockProphet/episode_data/latest"),  # From repo root
            Path(__file__).parent.parent.parent / "episode_data" / "latest"  # Absolute from script
        ]

        for path in possible_paths:
            if path.exists() and (path / "metadata.json").exists():
                data_dir = path
                break
        else:
            raise FileNotFoundError(
                "Episode data not found. Tried locations:\n" +
                "\n".join(f"  - {p}" for p in possible_paths) +
                "\n\nRun evaluation first:\n  cd StockProphet\n  python -m multiticker_refactor.main_multi --mode evaluate"
            )

    data_dir = Path(data_dir)

    # Load metadata
    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    # Load arrays
    portfolio_history = np.load(data_dir / "portfolio_history.npy")
    actions = np.load(data_dir / "actions.npy")
    rewards = np.load(data_dir / "rewards.npy")
    prices = np.load(data_dir / "prices.npy")

    return {
        'metadata': metadata,
        'portfolio_history': portfolio_history,
        'actions': actions,
        'rewards': rewards,
        'prices': prices
    }


def calculate_metrics(portfolio_history: np.ndarray, initial_capital: float) -> Dict:
    """
    Calculate portfolio performance metrics.

    Args:
        portfolio_history: Array of portfolio values over time
        initial_capital: Starting capital

    Returns:
        Dict with performance metrics
    """
    returns = np.diff(portfolio_history) / portfolio_history[:-1]

    final_value = portfolio_history[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    # Sharpe ratio (annualized, assuming daily data)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(portfolio_history)
    drawdown = (portfolio_history - cummax) / cummax
    max_drawdown = np.min(drawdown) * 100

    # Win rate
    positive_days = np.sum(returns > 0)
    win_rate = positive_days / len(returns) * 100 if len(returns) > 0 else 0

    return {
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_days': len(portfolio_history)
    }


def get_current_allocation(actions: np.ndarray, step: int, tickers: list) -> Dict:
    """
    Get current portfolio allocation at given step.

    Args:
        actions: Array of actions (shape: n_steps, n_tickers+1)
        step: Current step index
        tickers: List of ticker symbols

    Returns:
        Dict mapping ticker to allocation percentage
    """
    if step >= len(actions):
        step = len(actions) - 1

    action = actions[step]

    # Normalize action (sum of abs = 1.0)
    normalized = action / (np.abs(action).sum() + 1e-8)

    allocation = {}
    for i, ticker in enumerate(tickers):
        allocation[ticker] = normalized[i] * 100  # Convert to percentage

    allocation['Cash'] = normalized[-1] * 100

    return allocation


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:+.2f}%"
