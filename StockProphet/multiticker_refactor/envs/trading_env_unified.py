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

Based on MultiAssetContinuousEnv with full backward compatibility.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict


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
