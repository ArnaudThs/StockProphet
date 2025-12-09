import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlexibleTradingEnv(gym.Env):
    """
    ===========================================
    TRADING ENVIRONMENT – REALISTIC SIMULATION
    ===========================================

    This environment implements a realistic trading simulation with:

        0 = SHORT position
        1 = LONG position

    There is NO explicit HOLD action. The agent "holds" by repeating
    the same action, keeping the current position unchanged.

    -----------------------------------------------
    KEY FEATURES
    -----------------------------------------------
    1. Position included in observation (agent knows its state)
    2. Proportional transaction fees (realistic brokerage model)
    3. Double fees when flipping position (close + open)
    4. Optional short borrow cost
    5. Consistent tick handling

    -----------------------------------------------
    POSITION LOGIC
    -----------------------------------------------
    Possible positions:
        +1 → LONG
        -1 → SHORT
         0 → FLAT (only at reset)

    Action mapping:
        action 0 → desired_position = -1 (SHORT)
        action 1 → desired_position = +1 (LONG)

    -----------------------------------------------
    REWARD LOGIC
    -----------------------------------------------
    Reward is based on log-returns:

        log_ret = log(price_t / price_{t-1})
        reward = current_position * log_ret - costs

    -----------------------------------------------
    COSTS (Realistic Model)
    -----------------------------------------------
    1) Transaction fee (proportional):
       - Charged as percentage of trade value
       - FLAT → LONG/SHORT: 1x fee
       - LONG ↔ SHORT: 2x fee (close + open)

    2) Holding cost:
       - Per-step cost while in any position

    3) Short borrow cost:
       - Additional per-step cost when SHORT
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        prices,
        signal_features,
        window_size=30,
        frame_bound=(30, None),
        fee=0.0005,           # 0.05% per trade (proportional)
        holding_cost=0.0,     # per-step holding cost
        short_borrow_cost=0.0,  # additional cost for shorting
        include_position_in_obs=True,
        render_mode=None,
        debug=False,
    ):
        super().__init__()

        # ----------------------------
        # Store config
        # ----------------------------
        self.df = df
        self.prices = prices.astype(np.float32)
        self.features = signal_features.astype(np.float32)

        self.window_size = window_size
        self.frame_bound = frame_bound
        self.fee = float(fee)
        self.holding_cost = float(holding_cost)
        self.short_borrow_cost = float(short_borrow_cost)
        self.include_position_in_obs = include_position_in_obs
        self.debug = debug
        self.render_mode = render_mode

        # ----------------------------
        # Bounds (consistent handling)
        # ----------------------------
        self._start_tick = frame_bound[0]
        # End tick is exclusive (like Python range)
        self._end_tick = frame_bound[1] if frame_bound[1] else len(prices)

        # ----------------------------
        # Spaces
        # ----------------------------
        self.action_space = spaces.Discrete(2)  # 0=Short, 1=Long

        # Observation: window_size × num_features (+ optional position)
        n_features = self.features.shape[1]
        if self.include_position_in_obs:
            n_features += 1  # Add position as feature

        obs_shape = (window_size, n_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        # ----------------------------
        # Internal state variables
        # ----------------------------
        self._position = 0          # -1 short, 0 flat, +1 long
        self._current_tick = None
        self._total_reward = 0.0
        self._total_profit = 0.0
        self._trade_count = 0

        self._position_history = []

    # =====================================================================
    # RESET
    # =====================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._current_tick = self._start_tick
        self._position = 0
        self._total_reward = 0.0
        self._total_profit = 0.0
        self._trade_count = 0
        self._position_history = [self._position]

        obs = self._get_observation()

        return obs, self._get_info()

    # =====================================================================
    # STEP
    # =====================================================================
    def step(self, action):

        # SB3 sometimes passes array-wrapped actions
        if isinstance(action, (np.ndarray, list)):
            action = int(action[0])

        if action not in [0, 1]:
            raise ValueError(f"Invalid action received: {action}")

        # ---------------------------------------------------------
        # Move to next tick
        # ---------------------------------------------------------
        self._current_tick += 1
        # Episode ends when we reach end_tick - 1 (last valid index)
        terminated = self._current_tick >= self._end_tick - 1
        truncated = False

        # ---------------------------------------------------------
        # Compute reward *using previous position*
        # ---------------------------------------------------------
        new_position = self._action_to_position(action)
        reward = self._calculate_reward(new_position)

        if reward is None or np.isnan(reward) or np.isinf(reward):
            raise ValueError(f"Invalid reward: {reward}")

        # ---------------------------------------------------------
        # Update cumulative totals
        # ---------------------------------------------------------
        self._total_reward += reward
        self._total_profit += reward  # cumulative log-return

        # ---------------------------------------------------------
        # Update position AFTER applying the reward
        # ---------------------------------------------------------
        if new_position != self._position:
            self._trade_count += 1
        self._position = new_position
        self._position_history.append(self._position)

        # ---------------------------------------------------------
        # Build outputs
        # ---------------------------------------------------------
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # =====================================================================
    # REWARD LOGIC — REALISTIC MODEL
    # =====================================================================
    def _calculate_reward(self, new_position):

        tick = self._current_tick
        if tick < 1 or tick >= len(self.prices):
            raise ValueError(f"Tick {tick} out of price range [1, {len(self.prices)-1}].")

        price_now = float(self.prices[tick])
        price_prev = float(self.prices[tick - 1])

        if price_prev <= 0 or price_now <= 0:
            raise ValueError("Invalid price encountered.")

        if np.isnan(price_now) or np.isnan(price_prev):
            raise ValueError("NaN price in reward calc.")

        # -------------------------------
        # Compute log-return
        # -------------------------------
        log_ret = np.log(price_now / price_prev)

        # -------------------------------
        # BASE REWARD — USE PREVIOUS POSITION
        # -------------------------------
        reward = self._position * log_ret

        # -------------------------------
        # Transaction fee (proportional)
        # -------------------------------
        old_pos = self._position
        if new_position != old_pos:
            if old_pos == 0:
                # FLAT → LONG or SHORT: 1 trade
                reward -= self.fee
            elif new_position == 0:
                # LONG/SHORT → FLAT: 1 trade (closing)
                reward -= self.fee
            else:
                # LONG ↔ SHORT: 2 trades (close + open)
                reward -= 2 * self.fee

        # -------------------------------
        # Holding cost (any position)
        # -------------------------------
        if self._position != 0:
            reward -= self.holding_cost

        # -------------------------------
        # Short borrow cost
        # -------------------------------
        if self._position == -1:
            reward -= self.short_borrow_cost

        if self.debug:
            print(
                f"[DEBUG] tick={tick} "
                f"old_pos={old_pos} new_pos={new_position} "
                f"price_prev={price_prev:.4f} price_now={price_now:.4f} "
                f"log_ret={log_ret:.6f} reward={reward:.6f}"
            )

        return float(reward)

    # =====================================================================
    # POSITION MAPPING
    # =====================================================================
    def _action_to_position(self, action):
        return -1 if action == 0 else +1

    # =====================================================================
    # OBSERVATION
    # =====================================================================
    def _get_observation(self):
        # Get feature window ending at current tick (inclusive)
        start = self._current_tick - self.window_size + 1
        end = self._current_tick + 1
        window = self.features[start:end].copy()

        if self.include_position_in_obs:
            # Add position as an additional feature column
            # Normalized to [-1, 0, 1]
            position_col = np.full((self.window_size, 1), self._position, dtype=np.float32)
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
        return {
            "tick": self._current_tick,
            "position": self._position,
            "total_reward": float(self._total_reward),
            "total_profit": float(self._total_profit),
            "trade_count": self._trade_count,
        }

    # =====================================================================
    # RENDER (OPTIONAL)
    # =====================================================================
    def _render_frame(self):
        pass
