"""
Continuous Trading Environment with Position Sizing.

This environment uses a Box(-1, 1) action space where the action
represents the target position weight in the asset:
    - Positive values = long position (weight % of capital)
    - Negative values = short position (weight % of capital)
    - Magnitude = conviction/exposure level
    - Implicit cash = 1 - |position|

Example:
    action = 0.7  → 70% long, 30% cash
    action = -0.5 → 50% short, 50% cash
    action = 0.0  → 100% cash (flat)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ContinuousTradingEnv(gym.Env):
    """
    ===========================================
    CONTINUOUS TRADING ENVIRONMENT
    ===========================================

    Features:
    - Box(-1, 1) action space for continuous position sizing
    - Dollar-based portfolio tracking
    - Realistic transaction costs (proportional)
    - Implicit cash management
    - Multi-ticker ready (expand action dims)

    Reward: Simple P&L based (change in portfolio value)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        prices,
        signal_features,
        window_size=30,
        frame_bound=(30, None),
        initial_capital=10000.0,
        fee=0.001,              # 0.1% transaction fee (proportional to trade value)
        short_borrow_rate=0.0,  # Daily rate for borrowing shares (e.g., 0.0001 = 0.01%/day)
        include_position_in_obs=True,
        render_mode=None,
        debug=False,
    ):
        super().__init__()

        # ----------------------------
        # Store config
        # ----------------------------
        self.df = df
        self.prices = prices.astype(np.float64)
        self.features = signal_features.astype(np.float32)

        self.window_size = window_size
        self.frame_bound = frame_bound
        self.initial_capital = float(initial_capital)
        self.fee = float(fee)
        self.short_borrow_rate = float(short_borrow_rate)
        self.include_position_in_obs = include_position_in_obs
        self.debug = debug
        self.render_mode = render_mode

        # ----------------------------
        # Bounds
        # ----------------------------
        self._start_tick = frame_bound[0]
        self._end_tick = frame_bound[1] if frame_bound[1] else len(prices)

        # ----------------------------
        # Action Space: Continuous position weight [-1, 1]
        # ----------------------------
        # For single ticker: shape=(1,)
        # For multi-ticker: shape=(n_tickers,)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # ----------------------------
        # Observation Space
        # ----------------------------
        n_features = self.features.shape[1]
        if self.include_position_in_obs:
            n_features += 1  # Add current position weight as feature

        obs_shape = (window_size, n_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        # ----------------------------
        # Internal state
        # ----------------------------
        self._position_weight = 0.0  # Current position weight [-1, 1]
        self._shares = 0.0           # Number of shares held (can be negative for short)
        self._cash = 0.0             # Cash balance
        self._portfolio_value = 0.0  # Total portfolio value
        self._current_tick = None
        self._total_profit = 0.0
        self._trade_count = 0

        self._position_history = []
        self._portfolio_history = []

        # Risk-adjusted reward tracking
        self._recent_returns = []  # Track recent returns for volatility calculation
        self._returns_window = 20  # Window size for volatility

    # =====================================================================
    # RESET
    # =====================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._current_tick = self._start_tick
        self._position_weight = 0.0
        self._shares = 0.0
        self._cash = self.initial_capital
        self._portfolio_value = self.initial_capital
        self._total_profit = 0.0
        self._trade_count = 0

        self._position_history = [0.0]
        self._portfolio_history = [self.initial_capital]
        self._recent_returns = []  # Reset returns tracking

        obs = self._get_observation()
        return obs, self._get_info()

    # =====================================================================
    # STEP
    # =====================================================================
    def step(self, action):
        # Handle array-wrapped actions from SB3
        if isinstance(action, np.ndarray):
            action = float(action.flatten()[0])

        # Clip action to valid range
        target_weight = np.clip(action, -1.0, 1.0)

        # ---------------------------------------------------------
        # Move to next tick
        # ---------------------------------------------------------
        self._current_tick += 1
        terminated = self._current_tick >= self._end_tick - 1
        truncated = False

        # ---------------------------------------------------------
        # Get prices
        # ---------------------------------------------------------
        price_now = float(self.prices[self._current_tick])
        price_prev = float(self.prices[self._current_tick - 1])

        if price_now <= 0 or price_prev <= 0:
            raise ValueError(f"Invalid price: prev={price_prev}, now={price_now}")

        # ---------------------------------------------------------
        # Update portfolio value BEFORE rebalancing (mark-to-market)
        # ---------------------------------------------------------
        portfolio_before = self._calculate_portfolio_value(price_now)

        # ---------------------------------------------------------
        # Rebalance to target position
        # ---------------------------------------------------------
        transaction_cost = self._rebalance_to_target(target_weight, price_now)

        # ---------------------------------------------------------
        # Apply short borrow cost if holding short position
        # ---------------------------------------------------------
        borrow_cost = 0.0
        if self._position_weight < 0:
            # Cost based on absolute value of short position
            short_value = abs(self._shares * price_now)
            borrow_cost = short_value * self.short_borrow_rate
            self._cash -= borrow_cost

        # ---------------------------------------------------------
        # Calculate portfolio value AFTER all costs
        # ---------------------------------------------------------
        portfolio_after = self._calculate_portfolio_value(price_now)

        # ---------------------------------------------------------
        # Reward: Risk-adjusted P&L (Sharpe-like)
        # ---------------------------------------------------------
        # Base reward: normalized P&L
        raw_reward = (portfolio_after - portfolio_before) / self.initial_capital

        # Track recent returns for volatility calculation
        self._recent_returns.append(raw_reward)
        if len(self._recent_returns) > self._returns_window:
            self._recent_returns.pop(0)

        # Risk-adjust after sufficient history
        if len(self._recent_returns) >= 5:
            volatility = np.std(self._recent_returns) + 1e-8  # Avoid division by zero
            reward = raw_reward / volatility
        else:
            reward = raw_reward  # Use raw reward initially

        # Bankruptcy check
        if portfolio_after <= 0:
            terminated = True
            reward = -10.0  # Large penalty for bankruptcy

        # ---------------------------------------------------------
        # Update state
        # ---------------------------------------------------------
        self._portfolio_value = portfolio_after
        self._total_profit = portfolio_after - self.initial_capital
        self._position_history.append(self._position_weight)
        self._portfolio_history.append(portfolio_after)

        if self.debug:
            print(
                f"[DEBUG] tick={self._current_tick} "
                f"target_weight={target_weight:.3f} "
                f"actual_weight={self._position_weight:.3f} "
                f"shares={self._shares:.2f} "
                f"cash={self._cash:.2f} "
                f"portfolio={portfolio_after:.2f} "
                f"reward={reward:.6f}"
            )

        # ---------------------------------------------------------
        # Build outputs
        # ---------------------------------------------------------
        obs = self._get_observation()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    # =====================================================================
    # PORTFOLIO CALCULATIONS
    # =====================================================================
    def _calculate_portfolio_value(self, current_price):
        """Calculate total portfolio value (cash + position value)."""
        position_value = self._shares * current_price
        return self._cash + position_value

    def _rebalance_to_target(self, target_weight, current_price):
        """
        Rebalance portfolio to achieve target position weight.

        Returns transaction cost incurred.
        """
        # Current portfolio value
        portfolio_value = self._calculate_portfolio_value(current_price)

        # Target position value (can be negative for short)
        target_position_value = target_weight * portfolio_value

        # Target shares
        target_shares = target_position_value / current_price

        # Trade size
        shares_to_trade = target_shares - self._shares
        trade_value = abs(shares_to_trade * current_price)

        # Transaction cost (proportional to trade value)
        transaction_cost = trade_value * self.fee

        if abs(shares_to_trade) > 1e-8:  # Only count meaningful trades
            self._trade_count += 1

        # Execute trade
        # When buying: cash decreases, shares increase
        # When selling: cash increases, shares decrease
        self._cash -= shares_to_trade * current_price
        self._cash -= transaction_cost
        self._shares = target_shares

        # Update position weight
        new_portfolio_value = self._calculate_portfolio_value(current_price)
        if new_portfolio_value > 0:
            self._position_weight = (self._shares * current_price) / new_portfolio_value
        else:
            self._position_weight = 0.0

        return transaction_cost

    # =====================================================================
    # OBSERVATION
    # =====================================================================
    def _get_observation(self):
        """Build observation window with optional position feature."""
        start = self._current_tick - self.window_size + 1
        end = self._current_tick + 1
        window = self.features[start:end].copy()

        if self.include_position_in_obs:
            # Add position weight as feature (already in [-1, 1])
            position_col = np.full(
                (self.window_size, 1),
                self._position_weight,
                dtype=np.float32
            )
            window = np.concatenate([window, position_col], axis=1)

        expected_shape = self.observation_space.shape
        if window.shape != expected_shape:
            raise ValueError(
                f"Bad observation shape {window.shape}, "
                f"expected {expected_shape}"
            )

        return window.astype(np.float32)

    # =====================================================================
    # INFO DICT
    # =====================================================================
    def _get_info(self):
        """Return info dict with portfolio metrics."""
        return {
            "tick": self._current_tick,
            "position_weight": float(self._position_weight),
            "shares": float(self._shares),
            "cash": float(self._cash),
            "portfolio_value": float(self._portfolio_value),
            "total_profit": float(self._total_profit),
            "total_profit_pct": float(self._total_profit / self.initial_capital * 100),
            "trade_count": self._trade_count,
        }

    # =====================================================================
    # RENDER (Optional)
    # =====================================================================
    def _render_frame(self):
        pass

    # =====================================================================
    # UTILITY METHODS
    # =====================================================================
    def get_portfolio_history(self):
        """Return portfolio value history for analysis."""
        return np.array(self._portfolio_history)

    def get_position_history(self):
        """Return position weight history for analysis."""
        return np.array(self._position_history)
