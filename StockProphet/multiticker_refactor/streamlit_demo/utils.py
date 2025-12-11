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

    Prepends Day 0 (initial state with 100% cash) so that:
    - Day 0 = Initial state, 100% cash, $10,000, no action yet
    - Day 1 = First action taken, portfolio still $10,000 (result not known yet)
    - Day 2 = Portfolio shows result of Day 1's action, Day 2's action taken
    - etc.

    Timeline:
    - portfolio[N] = value at START of day N (before day N's action resolves)
    - actions[N] = action taken on day N (result shows in portfolio[N+1])

    Args:
        data_dir: Directory containing episode data (if None, auto-detect)

    Returns:
        Dict with all episode data (with Day 0 prepended)
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

    # Load arrays (as saved: portfolio[i] = value AFTER action[i] executed)
    raw_portfolio = np.load(data_dir / "portfolio_history.npy")
    raw_actions = np.load(data_dir / "actions.npy")
    rewards = np.load(data_dir / "rewards.npy")
    prices = np.load(data_dir / "prices.npy")

    initial_capital = metadata['initial_capital']
    n_tickers = len(metadata['tickers'])

    # Reframe: portfolio[N] = value at START of day N (before action resolves)
    # Day 0: $10,000 (initial), no action
    # Day 1: $10,000 (initial), first action taken
    # Day 2: result of Day 1's action, second action taken
    # ...
    # Day N: result of Day N-1's action, Day N's action taken

    # Portfolio: prepend initial capital twice (Day 0 and Day 1 both start at $10k)
    # Then append raw_portfolio[:-1] (shift forward by 1)
    portfolio_history = np.concatenate([
        [initial_capital, initial_capital],  # Day 0 and Day 1
        raw_portfolio[:-1]  # Days 2 onwards show previous day's result
    ])

    # Actions: Day 0 has no action (100% cash placeholder), then raw_actions
    day0_action = np.zeros((1, n_tickers + 1))
    day0_action[0, -1] = 1.0  # 100% cash (no trade)
    actions = np.concatenate([day0_action, raw_actions])

    # Rewards: prepend 0 for Day 0
    rewards = np.concatenate([[0.0], rewards])

    # Prices: prepend first row for Day 0
    prices = np.concatenate([prices[0:1], prices])

    # Update metadata
    metadata['n_steps'] = len(portfolio_history)

    # Shift dates by 1 (Day 0 = "Initial", Day 1+ = trading dates)
    if 'dates' in metadata:
        old_dates = metadata['dates']
        new_dates = {'0': 'Initial'}
        for old_key, date_val in old_dates.items():
            new_key = str(int(old_key) + 1)
            new_dates[new_key] = date_val
        metadata['dates'] = new_dates

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

    Since Day 0 is prepended with 100% cash action, this simply returns
    the action/allocation at the given step:
    - Day 0: 100% cash (initial state)
    - Day 1+: Result of trading action

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
