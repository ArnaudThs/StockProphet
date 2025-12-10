"""
Multi-Asset Continuous Trading Environment

Continuous action space for multi-ticker portfolio management with:
- Explicit cash allocation (4D action for 3 tickers)
- Dollar-based position sizing
- Net trade transaction fees
- Short position support with borrow costs
- Bankruptcy termination

Action Space:
    Box(-1, 1, shape=(n_tickers+1,))
    - First n_tickers dimensions: position weights (can be negative for shorts)
    - Last dimension: cash allocation
    - Normalized by sum(abs(action)) = 1.0

Observation Space:
    Box(-inf, inf, shape=(n_features + n_tickers + 1,))
    - Market features (prices, technical indicators, RNN predictions)
    - Portfolio state: [pos_weight_1, ..., pos_weight_n, cash_frac]

Reward:
    Risk-adjusted return: (portfolio_value_change) / (recent_volatility + epsilon)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple


class MultiAssetContinuousEnv(gym.Env):
    """
    Multi-ticker continuous trading environment with dollar-based position sizing.

    Features:
    - Explicit cash allocation (agent controls cash directly)
    - Signed weights for long/short positions
    - Transaction fees on NET trades only
    - Short borrow costs (daily)
    - Bankruptcy termination (portfolio_value <= 0)
    - Risk-adjusted rewards (Sharpe-like)
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        prices: np.ndarray,  # shape: (n_timesteps, n_tickers)
        signal_features: np.ndarray,  # shape: (n_timesteps, n_features)
        ticker_map: dict,  # {0: "AAPL", 1: "MSFT", 2: "GOOGL"}
        initial_capital: float = 10_000.0,
        transaction_fee_pct: float = 0.001,  # 0.1%
        short_borrow_rate: float = 0.0003,  # 0.03% per day
        window_size: int = 10,  # Not used with MlpLstmPolicy
        frame_bound: Tuple[int, int] = None,
        reward_volatility_window: int = 30,
    ):
        """
        Initialize multi-asset continuous trading environment.

        Args:
            prices: Close prices, shape (n_timesteps, n_tickers)
            signal_features: All features INCLUDING close prices, shape (n_timesteps, n_features)
            ticker_map: Dict mapping array index to ticker symbol
            initial_capital: Starting capital ($)
            transaction_fee_pct: Transaction fee rate (default 0.1%)
            short_borrow_rate: Daily borrow cost for shorts (default 0.03%)
            window_size: Observation window (unused with MlpLstmPolicy)
            frame_bound: (start_idx, end_idx) for train/val/test split
            reward_volatility_window: Window for computing recent volatility
        """
        super().__init__()

        assert prices.shape[0] == signal_features.shape[0], \
            "prices and signal_features must have same number of timesteps"

        self.prices = prices
        self.signal_features = signal_features
        self.ticker_map = ticker_map
        self.initial_capital = initial_capital
        self.transaction_fee_pct = transaction_fee_pct
        self.short_borrow_rate = short_borrow_rate
        self.window_size = window_size  # Not used with MlpLstmPolicy
        self.reward_volatility_window = reward_volatility_window

        self.n_timesteps, self.n_tickers = prices.shape
        self.n_features = signal_features.shape[1]

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
        self._first_rendering = True

        # Track recent returns for volatility calculation
        self._recent_returns = []

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
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
        self._first_rendering = True
        self._recent_returns = []

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one trading step (Gymnasium API).

        Args:
            action: Array of shape (n_tickers+1,) with position weights + cash

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
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
                'bankruptcy': True
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
            if net_trade_shares[i] != 0:
                trade_value = abs(net_trade_shares[i]) * current_prices[i]
                fee = trade_value * self.transaction_fee_pct
                self._cash -= fee

        # Apply short borrow costs (daily)
        for i in range(self.n_tickers):
            if self._shares[i] < 0:  # Short position
                borrow_cost = abs(self._shares[i]) * current_prices[i] * self.short_borrow_rate
                self._cash -= borrow_cost

        # Move to next time step
        self._current_tick += 1

        # Check if episode is done (must check BEFORE accessing prices)
        terminated = self._current_tick >= self._end_tick
        truncated = False  # We don't use truncation in this environment

        if not terminated:
            # Get new prices (safe now, we checked bounds above)
            new_prices = self.prices[self._current_tick]

            # Calculate portfolio value AFTER all costs
            portfolio_after = self._cash + np.sum(self._shares * new_prices)

            # Check for bankruptcy AFTER rebalancing
            if portfolio_after <= 0:
                # Bankrupt! Go back one tick to get valid observation
                self._current_tick -= 1
                obs = self._get_observation()
                self._current_tick += 1  # Restore tick for consistency

                reward = -10.0  # Large penalty
                terminated = True
                info = {
                    'total_reward': self._total_reward,
                    'total_profit': self._total_profit,
                    'portfolio_value': 0.0,
                    'bankruptcy': True
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

        else:
            # Episode ended
            final_prices = self.prices[self._current_tick - 1]
            portfolio_after = self._cash + np.sum(self._shares * final_prices)
            reward = 0.0

        # Build info dict
        info = {
            'total_reward': self._total_reward,
            'total_profit': self._total_profit,
            'portfolio_value': portfolio_after if not terminated else portfolio_before,
            'bankruptcy': False
        }

        # Get observation (handle terminated case where tick may be out of bounds)
        if terminated and self._current_tick >= self._end_tick:
            # Episode ended naturally, get observation from last valid tick
            self._current_tick = self._end_tick - 1
            obs = self._get_observation()
        else:
            obs = self._get_observation()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation for MlpLstmPolicy.

        Returns 1D array of current market features + portfolio state.
        Policy's internal LSTM maintains temporal state.

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

    def render(self, mode='human'):
        """Render environment state."""
        if self._first_rendering:
            self._first_rendering = False
            print("=" * 60)
            print(f"Multi-Asset Trading Environment")
            print(f"Tickers: {list(self.ticker_map.values())}")
            print(f"Initial capital: ${self.initial_capital:,.2f}")
            print("=" * 60)

        current_prices = self.prices[self._current_tick]
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


def create_multi_asset_env(
    prices: np.ndarray,
    signal_features: np.ndarray,
    ticker_map: dict,
    initial_capital: float = 10_000.0,
    frame_bound: Tuple[int, int] = None,
    **kwargs
) -> MultiAssetContinuousEnv:
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
        MultiAssetContinuousEnv instance
    """
    return MultiAssetContinuousEnv(
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
) -> Tuple[MultiAssetContinuousEnv, MultiAssetContinuousEnv, MultiAssetContinuousEnv]:
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
