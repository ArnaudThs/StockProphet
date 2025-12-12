"""
Unified Trading Environment

Supports both single-ticker and multi-ticker portfolio management with:
- Single ticker: n_tickers=1, action shape (2,) -> [position_weight, cash_weight]
- Multi-ticker: n_tickers>1, action shape (n_tickers+1,) -> [pos1, pos2, ..., cash]
- Dollar-based position sizing
- Net trade transaction fees
- Short position support with borrow costs
- Bankruptcy termination
- Risk-adjusted rewards

This module consolidates all environment code into a single file:
- UnifiedTradingEnv: Core environment class
- Factory functions: create_multi_asset_env, create_train_val_test_envs, create_single_ticker_env
- Wrapper functions: make_env, make_continuous_env, create_train_test_envs, etc.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..config import (
    PPO_WINDOW_SIZE, PPO_TRAIN_RATIO, REWARD_CONFIG, VEC_NORMALIZE_PATH,
    ENV_TYPE, INITIAL_CAPITAL, CONTINUOUS_ENV_CONFIG, CONTINUOUS_ENV_VERSION,
    TREND_REWARD_MULTIPLIER, CONVICTION_REWARD, EXIT_TIMING_REWARD, PATIENCE_REWARD
)


class UnifiedTradingEnv(gym.Env):
    """
    Unified continuous trading environment for single/multi-ticker portfolios.

    Features:
    - Explicit cash allocation (agent controls cash directly)
    - Signed weights for long/short positions
    - Transaction fees on NET trades only
    - Short borrow costs (daily)
    - Bankruptcy termination (portfolio_value <= 0)
    - Risk-adjusted rewards (Sharpe-like)
    - Supports n_tickers=1 (single-ticker) and n_tickers>1 (multi-ticker)
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        prices: np.ndarray,  # shape: (n_timesteps, n_tickers) or (n_timesteps,) for single
        signal_features: np.ndarray,  # shape: (n_timesteps, n_features)
        ticker_map: Optional[Dict[int, str]] = None,  # {0: "AAPL", 1: "MSFT", ...}
        initial_capital: float = 10_000.0,
        transaction_fee_pct: float = 0.001,  # 0.1%
        short_borrow_rate: float = 0.0003,  # 0.03% per day
        window_size: int = 10,  # Not used with MlpLstmPolicy
        frame_bound: Optional[Tuple[int, int]] = None,
        reward_volatility_window: int = 30,
        df: Optional[object] = None,  # For backward compat with old envs (unused)
        fee: Optional[float] = None,  # For backward compat
        include_position_in_obs: bool = True,  # For backward compat
        render_mode: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize unified trading environment.

        Args:
            prices: Close prices. Shape (n_timesteps, n_tickers) or (n_timesteps,) for single ticker
            signal_features: All features INCLUDING close prices, shape (n_timesteps, n_features)
            ticker_map: Dict mapping array index to ticker symbol (optional)
            initial_capital: Starting capital ($)
            transaction_fee_pct: Transaction fee rate (default 0.1%)
            short_borrow_rate: Daily borrow cost for shorts (default 0.03%)
            window_size: Observation window (unused with MlpLstmPolicy)
            frame_bound: (start_idx, end_idx) for train/val/test split
            reward_volatility_window: Window for computing recent volatility
            df: Backward compatibility (unused)
            fee: Backward compatibility, overrides transaction_fee_pct if provided
            include_position_in_obs: Backward compatibility (always True for unified env)
            render_mode: Render mode (optional)
            debug: Enable debug printing
        """
        super().__init__()

        # Handle single-ticker case: convert (n_timesteps,) to (n_timesteps, 1)
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)

        assert prices.shape[0] == signal_features.shape[0], \
            "prices and signal_features must have same number of timesteps"

        self.prices = prices
        self.signal_features = signal_features
        self.initial_capital = initial_capital
        self.window_size = window_size
        self.reward_volatility_window = reward_volatility_window
        self.render_mode = render_mode
        self.debug = debug

        # Backward compatibility: use 'fee' param if provided
        if fee is not None:
            self.transaction_fee_pct = fee
        else:
            self.transaction_fee_pct = transaction_fee_pct

        self.short_borrow_rate = short_borrow_rate

        self.n_timesteps, self.n_tickers = prices.shape
        self.n_features = signal_features.shape[1]

        # Ticker map (optional, for rendering)
        if ticker_map is None:
            self.ticker_map = {i: f"TICKER_{i}" for i in range(self.n_tickers)}
        else:
            self.ticker_map = ticker_map

        # Frame bound (for train/val/test splits)
        if frame_bound is None:
            self.frame_bound = (0, self.n_timesteps - 1)
        else:
            self.frame_bound = frame_bound

        # Action space: n_tickers + 1 (positions + cash)
        # Values in [-1, 1], normalized by sum(abs) after PPO outputs
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_tickers + 1,), dtype=np.float32
        )

        # Observation space: market features + portfolio state
        # Portfolio state: n_tickers position weights + 1 cash fraction
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + self.n_tickers + 1,),
            dtype=np.float32
        )

        # State variables (initialized in reset())
        self._start_tick = None
        self._end_tick = None
        self._current_tick = None
        self._cash = None
        self._shares = None  # np.array of shape (n_tickers,)
        self._position_weights = None  # np.array of shape (n_tickers,)
        self._total_reward = 0.0
        self._total_profit = 0.0
        self._trade_count = 0
        self._first_rendering = True

        # Track recent returns for volatility calculation
        self._recent_returns = []

        # History tracking
        self._position_history = []
        self._portfolio_history = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to initial state (Gymnasium API).

        Args:
            seed: Random seed (optional, for reproducibility)
            options: Additional options (optional)

        Returns:
            Tuple of (observation, info)
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)

        self._start_tick = self.frame_bound[0]
        self._end_tick = self.frame_bound[1]
        self._current_tick = self._start_tick

        # Initialize portfolio: all cash, no positions
        self._cash = self.initial_capital
        self._shares = np.zeros(self.n_tickers, dtype=np.float32)
        self._position_weights = np.zeros(self.n_tickers, dtype=np.float32)

        self._total_reward = 0.0
        self._total_profit = 0.0
        self._trade_count = 0
        self._first_rendering = True
        self._recent_returns = []

        self._position_history = [self._position_weights.copy()]
        self._portfolio_history = [self.initial_capital]

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one trading step (Gymnasium API).

        Args:
            action: Array of shape (n_tickers+1,) with position weights + cash

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Handle wrapped actions from SB3
        if action.ndim > 1:
            action = action.flatten()

        # Normalize action (ensure sum(abs(action)) = 1.0)
        action = action / (np.sum(np.abs(action)) + 1e-8)

        # Get current prices
        current_prices = self.prices[self._current_tick]

        # Calculate portfolio value BEFORE rebalancing
        portfolio_before = self._cash + np.sum(self._shares * current_prices)

        # Handle bankruptcy (portfolio value <= 0)
        if portfolio_before <= 0:
            reward = -10.0  # Large penalty
            terminated = True
            truncated = False
            info = {
                'total_reward': self._total_reward,
                'total_profit': self._total_profit,
                'portfolio_value': 0.0,
                'bankruptcy': True,
                'trade_count': self._trade_count,
            }
            return self._get_observation(), reward, terminated, truncated, info

        # Convert action weights to target allocations
        action_weights = action[:self.n_tickers]
        cash_weight = action[-1]

        # Calculate target dollar allocations
        target_dollars_per_ticker = action_weights * portfolio_before
        target_cash_dollars = cash_weight * portfolio_before

        # Convert dollars to shares
        target_shares = target_dollars_per_ticker / (current_prices + 1e-8)

        # Calculate net trades
        net_trade_shares = target_shares - self._shares

        # Update positions
        self._shares = target_shares
        self._cash = target_cash_dollars

        # Apply transaction fees (on NET trades only)
        for i in range(self.n_tickers):
            if abs(net_trade_shares[i]) > 1e-8:
                trade_value = abs(net_trade_shares[i]) * current_prices[i]
                fee = trade_value * self.transaction_fee_pct
                self._cash -= fee
                self._trade_count += 1

        # Apply short borrow costs (daily)
        for i in range(self.n_tickers):
            if self._shares[i] < 0:  # Short position
                borrow_cost = abs(self._shares[i]) * current_prices[i] * self.short_borrow_rate
                self._cash -= borrow_cost

        # Move to next time step
        self._current_tick += 1

        # Check if episode is done (must check BEFORE accessing prices)
        terminated = self._current_tick >= self._end_tick
        truncated = False

        if not terminated:
            # Get new prices
            new_prices = self.prices[self._current_tick]

            # Calculate portfolio value AFTER all costs
            portfolio_after = self._cash + np.sum(self._shares * new_prices)

            # Check for bankruptcy AFTER rebalancing
            if portfolio_after <= 0:
                # Bankrupt! Get observation from previous tick
                self._current_tick -= 1
                obs = self._get_observation()
                self._current_tick += 1

                reward = -10.0  # Large penalty
                terminated = True
                info = {
                    'total_reward': self._total_reward,
                    'total_profit': self._total_profit,
                    'portfolio_value': 0.0,
                    'bankruptcy': True,
                    'trade_count': self._trade_count,
                }
                return obs, reward, terminated, truncated, info

            # Calculate return
            pnl = portfolio_after - portfolio_before
            pnl_pct = pnl / self.initial_capital

            # Track recent returns for volatility
            self._recent_returns.append(pnl_pct)
            if len(self._recent_returns) > self.reward_volatility_window:
                self._recent_returns.pop(0)

            # Risk-adjusted reward (Sharpe-like)
            if len(self._recent_returns) > 1:
                recent_volatility = np.std(self._recent_returns)
                reward = pnl_pct / (recent_volatility + 1e-8)
            else:
                reward = pnl_pct  # Not enough data for volatility yet

            # Update position weights (for observation)
            total_position_value = np.sum(np.abs(self._shares) * new_prices)
            total_value = self._cash + total_position_value
            if total_value > 0:
                self._position_weights = (self._shares * new_prices) / total_value
            else:
                self._position_weights = np.zeros(self.n_tickers, dtype=np.float32)

            # Track totals
            self._total_reward += reward
            self._total_profit = portfolio_after - self.initial_capital

            # Track history
            self._position_history.append(self._position_weights.copy())
            self._portfolio_history.append(portfolio_after)

        else:
            # Episode ended naturally
            final_prices = self.prices[self._current_tick - 1]
            portfolio_after = self._cash + np.sum(self._shares * final_prices)
            reward = 0.0

        # Debug output
        if self.debug:
            print(
                f"[DEBUG] tick={self._current_tick} "
                f"positions={self._position_weights} "
                f"portfolio={portfolio_after:.2f} "
                f"reward={reward:.6f}"
            )

        # Build info dict
        info = {
            'total_reward': self._total_reward,
            'total_profit': self._total_profit,
            'total_profit_pct': (self._total_profit / self.initial_capital * 100) if not terminated else 0.0,
            'portfolio_value': portfolio_after if not terminated else portfolio_before,
            'bankruptcy': False,
            'trade_count': self._trade_count,
            'tick': self._current_tick,
        }

        # Get observation (handle terminated case)
        if terminated and self._current_tick >= self._end_tick:
            # Episode ended naturally, get observation from last valid tick
            self._current_tick = self._end_tick - 1
            obs = self._get_observation()
        else:
            obs = self._get_observation()

        return obs, float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns 1D array of current market features + portfolio state.
        For MlpLstmPolicy, the policy's internal LSTM maintains temporal state.

        Returns:
            Observation array, shape (n_features + n_tickers + 1,)
        """
        # Current market features (already includes all tickers)
        market_features = self.signal_features[self._current_tick]

        # Current portfolio state (n_tickers + 1 features)
        portfolio_state = np.array([
            *self._position_weights,  # n_tickers values
            self._cash / self.initial_capital,  # 1 value (normalized)
        ], dtype=np.float32)

        return np.concatenate([market_features, portfolio_state])

    def _get_info(self) -> dict:
        """Get info dict for backward compatibility with old envs."""
        current_prices = self.prices[min(self._current_tick, self._end_tick - 1)]
        portfolio_value = self._cash + np.sum(self._shares * current_prices)

        return {
            "tick": self._current_tick,
            "position_weight": float(self._position_weights[0]) if self.n_tickers == 1 else self._position_weights.tolist(),
            "shares": float(self._shares[0]) if self.n_tickers == 1 else self._shares.tolist(),
            "cash": float(self._cash),
            "portfolio_value": float(portfolio_value),
            "total_profit": float(self._total_profit),
            "total_profit_pct": float(self._total_profit / self.initial_capital * 100),
            "trade_count": self._trade_count,
        }

    def get_portfolio_history(self) -> np.ndarray:
        """Get portfolio value history (for backward compat)."""
        return np.array(self._portfolio_history)

    def get_position_history(self) -> np.ndarray:
        """Get position weight history (for backward compat)."""
        return np.array(self._position_history)

    def render(self, mode: str = 'human'):
        """Render environment state."""
        if self._first_rendering:
            self._first_rendering = False
            print("=" * 60)
            if self.n_tickers == 1:
                print(f"Single-Ticker Trading Environment")
                print(f"Ticker: {self.ticker_map[0]}")
            else:
                print(f"Multi-Ticker Trading Environment")
                print(f"Tickers: {list(self.ticker_map.values())}")
            print(f"Initial capital: ${self.initial_capital:,.2f}")
            print("=" * 60)

        current_prices = self.prices[min(self._current_tick, self._end_tick - 1)]
        portfolio_value = self._cash + np.sum(self._shares * current_prices)

        print(f"\nStep {self._current_tick}/{self._end_tick}")
        print(f"Portfolio value: ${portfolio_value:,.2f}")
        print(f"Cash: ${self._cash:,.2f}")
        print("Positions:")
        for i, ticker in self.ticker_map.items():
            shares = self._shares[i]
            price = current_prices[i]
            value = shares * price
            print(f"  {ticker}: {shares:.2f} shares @ ${price:.2f} = ${value:,.2f}")

    def close(self):
        """Clean up resources."""
        pass


# =============================================================================
# FACTORY FUNCTIONS (Multi-Asset Environment Creation)
# =============================================================================

def create_multi_asset_env(
    prices: np.ndarray,
    signal_features: np.ndarray,
    ticker_map: dict,
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
    ticker_map: dict,
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


def create_single_ticker_env(
    prices: np.ndarray,
    signal_features: np.ndarray,
    frame_bound: Tuple[int, int] = None,
    **kwargs
) -> UnifiedTradingEnv:
    """
    Helper to create single-ticker environment (for feature selection).

    Args:
        prices: Close prices, shape (n_timesteps,) for single ticker
        signal_features: Features, shape (n_timesteps, n_features)
        frame_bound: (start_idx, end_idx)
        **kwargs: Environment parameters

    Returns:
        UnifiedTradingEnv configured for single ticker
    """
    # Reshape prices to (n_timesteps, 1) for single ticker
    if prices.ndim == 1:
        prices = prices.reshape(-1, 1)

    # Single ticker map
    ticker_map = {0: kwargs.pop('ticker', 'TICKER')}

    # Filter out env_version and V2-specific reward parameters
    # that UnifiedTradingEnv doesn't need
    kwargs.pop('env_version', None)
    kwargs.pop('trend_reward_multiplier', None)
    kwargs.pop('conviction_reward', None)
    kwargs.pop('exit_timing_reward', None)
    kwargs.pop('patience_reward', None)

    return UnifiedTradingEnv(
        prices=prices,
        signal_features=signal_features,
        ticker_map=ticker_map,
        frame_bound=frame_bound,
        **kwargs
    )


# =============================================================================
# WRAPPER FUNCTIONS (Backward Compatibility with Old Interface)
# =============================================================================

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
    DEPRECATED: Discrete environment is no longer used.
    Continuous environment (UnifiedTradingEnv) is used for all cases.
    This function is kept for backward compatibility.
    """
    raise NotImplementedError(
        "Discrete environment is deprecated. "
        "Use make_continuous_env() with ENV_TYPE='continuous' instead."
    )


def make_continuous_env(
    df: pd.DataFrame,
    prices: np.ndarray,
    signal_features: np.ndarray,
    frame_bound: tuple,
    window_size: int = PPO_WINDOW_SIZE,
    initial_capital: float = None,
    env_config: dict = None,
    env_version: str = None
):
    """
    Factory function to create UnifiedTradingEnv (continuous).

    Args:
        df: DataFrame with features (for backward compatibility, unused)
        prices: Array of close prices
        signal_features: Array of features (n_samples, n_features)
        frame_bound: Tuple of (start_tick, end_tick)
        window_size: Observation window size
        initial_capital: Starting capital in dollars
        env_config: Environment configuration dict
        env_version: Ignored (UnifiedTradingEnv uses risk-adjusted rewards)

    Returns:
        Callable that creates the environment
    """
    if initial_capital is None:
        initial_capital = INITIAL_CAPITAL
    if env_config is None:
        env_config = CONTINUOUS_ENV_CONFIG

    def _init():
        env = UnifiedTradingEnv(
            prices=prices,
            signal_features=signal_features,
            ticker_map={0: "TICKER_0"},  # Single ticker
            initial_capital=initial_capital,
            transaction_fee_pct=env_config.get('fee', 0.001),
            short_borrow_rate=env_config.get('short_borrow_rate', 0.0001),
            window_size=window_size,
            frame_bound=frame_bound,
            df=df,  # For backward compat
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
