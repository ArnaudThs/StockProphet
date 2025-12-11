"""
Continuous Trading Environment V2 - Trend-Adaptive Rewards

This version encourages:
1. Quick trend detection (bonus for early entry)
2. Position building (bonus for conviction)
3. Timely exits (bonus for closing before reversals)
4. Patience during uncertainty (bonus for staying flat)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ContinuousTradingEnvV2(gym.Env):
    """
    ===========================================
    CONTINUOUS TRADING ENVIRONMENT V2
    ===========================================

    Enhanced reward structure for trend-following behavior:
    - Rewards EARLY trend detection
    - Rewards CONVICTION (larger positions when confident)
    - Rewards EXITS before trend reversals
    - Rewards PATIENCE (staying flat when uncertain)
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
        fee=0.001,
        short_borrow_rate=0.0,
        include_position_in_obs=True,
        render_mode=None,
        debug=False,
        # V2 parameters
        trend_reward_multiplier=2.0,    # Bonus for trading WITH detected trends
        conviction_reward=0.5,          # Bonus for large positions (vs tiny positions)
        exit_timing_reward=1.0,         # Bonus for exiting before reversals
        patience_reward=0.2,            # Small reward for staying flat in noise
    ):
        super().__init__()

        # Store config
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

        # V2 reward parameters
        self.trend_reward_multiplier = trend_reward_multiplier
        self.conviction_reward = conviction_reward
        self.exit_timing_reward = exit_timing_reward
        self.patience_reward = patience_reward

        # Bounds
        self._start_tick = frame_bound[0]
        self._end_tick = frame_bound[1] if frame_bound[1] else len(prices)

        # Action Space: Continuous position weight [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation Space
        n_features = self.features.shape[1]
        if self.include_position_in_obs:
            n_features += 1

        obs_shape = (window_size, n_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        # Internal state
        self._position_weight = 0.0
        self._shares = 0.0
        self._cash = 0.0
        self._portfolio_value = 0.0
        self._current_tick = None
        self._total_profit = 0.0
        self._trade_count = 0

        self._position_history = []
        self._portfolio_history = []

        # V2 tracking
        self._recent_returns = []
        self._returns_window = 20
        self._recent_prices = []
        self._trend_window = 10
        self._previous_position_sign = 0  # +1 (long), 0 (flat), -1 (short)

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
        self._recent_returns = []
        self._recent_prices = []
        self._previous_position_sign = 0

        obs = self._get_observation()
        return obs, self._get_info()

    def step(self, action):
        # Handle array-wrapped actions from SB3
        if isinstance(action, np.ndarray):
            action = float(action.flatten()[0])

        # Clip action to valid range
        target_weight = np.clip(action, -1.0, 1.0)

        # Move to next tick
        self._current_tick += 1
        terminated = self._current_tick >= self._end_tick - 1
        truncated = False

        # Get prices
        price_now = float(self.prices[self._current_tick])
        price_prev = float(self.prices[self._current_tick - 1])

        if price_now <= 0 or price_prev <= 0:
            raise ValueError(f"Invalid price: prev={price_prev}, now={price_now}")

        # Track recent prices for trend detection
        self._recent_prices.append(price_now)
        if len(self._recent_prices) > self._trend_window:
            self._recent_prices.pop(0)

        # =====================================================================
        # PORTFOLIO UPDATE
        # =====================================================================
        portfolio_before = self._calculate_portfolio_value(price_now)
        transaction_cost = self._rebalance_to_target(target_weight, price_now)

        # Apply short borrow cost
        borrow_cost = 0.0
        if self._position_weight < 0:
            short_value = abs(self._shares * price_now)
            borrow_cost = short_value * self.short_borrow_rate
            self._cash -= borrow_cost

        portfolio_after = self._calculate_portfolio_value(price_now)

        # =====================================================================
        # V2 REWARD CALCULATION
        # =====================================================================
        reward = self._calculate_v2_reward(
            portfolio_before,
            portfolio_after,
            price_now,
            price_prev,
            target_weight
        )

        # Bankruptcy check
        if portfolio_after <= 0:
            terminated = True
            reward = -10.0

        # Update state
        self._portfolio_value = portfolio_after
        self._total_profit = portfolio_after - self.initial_capital
        self._position_history.append(self._position_weight)
        self._portfolio_history.append(portfolio_after)
        self._previous_position_sign = np.sign(self._position_weight)

        if self.debug:
            print(
                f"[DEBUG] tick={self._current_tick} "
                f"weight={self._position_weight:.3f} "
                f"portfolio={portfolio_after:.2f} "
                f"reward={reward:.6f}"
            )

        obs = self._get_observation()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def _calculate_v2_reward(self, portfolio_before, portfolio_after, price_now, price_prev, target_weight):
        """
        Calculate V2 reward with trend-adaptive bonuses.

        Components:
        1. Base P&L reward (normalized)
        2. Trend alignment bonus (reward trading WITH the trend)
        3. Conviction bonus (reward large positions when profitable)
        4. Exit timing bonus (reward reducing position before reversals)
        5. Patience bonus (reward staying flat in low-conviction periods)
        """
        # 1. BASE REWARD: Normalized P&L
        raw_pnl = (portfolio_after - portfolio_before) / self.initial_capital

        # Track returns for volatility
        self._recent_returns.append(raw_pnl)
        if len(self._recent_returns) > self._returns_window:
            self._recent_returns.pop(0)

        # Risk-adjust after sufficient history
        if len(self._recent_returns) >= 5:
            volatility = np.std(self._recent_returns) + 1e-8
            base_reward = raw_pnl / volatility
        else:
            base_reward = raw_pnl

        # 2. TREND ALIGNMENT BONUS
        trend_bonus = self._calculate_trend_bonus(price_now, price_prev)

        # 3. CONVICTION BONUS
        conviction_bonus = self._calculate_conviction_bonus(raw_pnl)

        # 4. EXIT TIMING BONUS
        exit_bonus = self._calculate_exit_bonus(price_now, price_prev)

        # 5. PATIENCE BONUS
        patience_bonus = self._calculate_patience_bonus()

        # TOTAL REWARD
        total_reward = (
            base_reward +
            trend_bonus +
            conviction_bonus +
            exit_bonus +
            patience_bonus
        )

        return total_reward

    def _calculate_trend_bonus(self, price_now, price_prev):
        """
        Bonus for trading WITH the detected trend.
        Encourages early trend detection.
        """
        if len(self._recent_prices) < 3:
            return 0.0

        # Simple trend: are recent prices rising or falling?
        recent_change = (self._recent_prices[-1] - self._recent_prices[0]) / self._recent_prices[0]
        current_return = (price_now - price_prev) / price_prev

        # Reward if position aligns with trend
        # Long position + rising prices = positive
        # Short position + falling prices = positive
        alignment = self._position_weight * current_return

        if alignment > 0:
            # Trading WITH the trend
            return self.trend_reward_multiplier * abs(alignment)
        else:
            # Trading AGAINST the trend (no penalty, just no bonus)
            return 0.0

    def _calculate_conviction_bonus(self, raw_pnl):
        """
        Bonus for taking LARGE positions when they're profitable.
        Discourages tiny positions that don't capture moves.
        """
        if raw_pnl > 0:
            # Profitable: reward conviction (large |position|)
            conviction = abs(self._position_weight)
            return self.conviction_reward * conviction * raw_pnl
        else:
            # Losing: no conviction bonus
            return 0.0

    def _calculate_exit_bonus(self, price_now, price_prev):
        """
        Bonus for REDUCING position size when trend reverses.
        Encourages timely exits.
        """
        if len(self._recent_prices) < 3:
            return 0.0

        # Detect potential reversal: recent trend vs current move
        recent_trend = (self._recent_prices[-1] - self._recent_prices[-3]) / self._recent_prices[-3]
        current_move = (price_now - price_prev) / price_prev

        # Reversal signal: trend direction changed
        reversal = recent_trend * current_move < 0

        if reversal:
            # Did we reduce position size? (good exit timing)
            if len(self._position_history) >= 2:
                prev_size = abs(self._position_history[-1])
                curr_size = abs(self._position_weight)
                position_reduction = prev_size - curr_size

                if position_reduction > 0.1:
                    # Reduced position before reversal
                    return self.exit_timing_reward * position_reduction

        return 0.0

    def _calculate_patience_bonus(self):
        """
        Small bonus for staying FLAT when there's no clear trend.
        Encourages not overtrading in noise.
        """
        if len(self._recent_prices) < 5:
            return 0.0

        # Measure recent volatility
        recent_returns = np.diff(self._recent_prices) / self._recent_prices[:-1]
        volatility = np.std(recent_returns)

        # High volatility = noisy, unclear trend
        # If we're flat (small position) in high volatility, that's good
        if volatility > 0.02 and abs(self._position_weight) < 0.2:
            return self.patience_reward

        return 0.0

    # =====================================================================
    # PORTFOLIO CALCULATIONS (unchanged from V1)
    # =====================================================================
    def _calculate_portfolio_value(self, current_price):
        position_value = self._shares * current_price
        return self._cash + position_value

    def _rebalance_to_target(self, target_weight, current_price):
        portfolio_value = self._calculate_portfolio_value(current_price)
        target_position_value = target_weight * portfolio_value
        target_shares = target_position_value / current_price
        shares_to_trade = target_shares - self._shares
        trade_value = abs(shares_to_trade * current_price)
        transaction_cost = trade_value * self.fee

        if abs(shares_to_trade) > 1e-8:
            self._trade_count += 1

        self._cash -= shares_to_trade * current_price
        self._cash -= transaction_cost
        self._shares = target_shares

        new_portfolio_value = self._calculate_portfolio_value(current_price)
        if new_portfolio_value > 0:
            self._position_weight = (self._shares * current_price) / new_portfolio_value
        else:
            self._position_weight = 0.0

        return transaction_cost

    # =====================================================================
    # OBSERVATION (unchanged)
    # =====================================================================
    def _get_observation(self):
        start = self._current_tick - self.window_size + 1
        end = self._current_tick + 1
        window = self.features[start:end].copy()

        if self.include_position_in_obs:
            position_col = np.full(
                (self.window_size, 1),
                self._position_weight,
                dtype=np.float32
            )
            window = np.concatenate([window, position_col], axis=1)

        return window.astype(np.float32)

    def _get_info(self):
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

    def _render_frame(self):
        pass

    def get_portfolio_history(self):
        return np.array(self._portfolio_history)

    def get_position_history(self):
        return np.array(self._position_history)
