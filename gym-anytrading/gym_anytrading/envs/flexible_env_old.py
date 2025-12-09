import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlexibleTradingEnv(gym.Env):

    """
===========================================
TRADING ENVIRONMENT – BUSINESS LOGIC (NO HOLD)
===========================================

This environment implements the original AnyTrading-style 2-action system:

    0 = SHORT position
    1 = LONG position

There is NO explicit HOLD action. The agent “holds” simply by repeating
the same action, which keeps its current position unchanged. Example:
    - If already LONG and the agent selects LONG again → stay LONG
    - If already SHORT and selects SHORT again → stay SHORT

You cannot explicitly return to FLAT; only the first step starts flat.

-----------------------------------------------
POSITION LOGIC
-----------------------------------------------
Possible positions:
    +1 → LONG
    -1 → SHORT
     0 → FLAT  (only at reset)

The agent’s new desired position is determined by the action:
    action 0 → desired_position = -1
    action 1 → desired_position = +1

If the desired position differs from the current position, a transaction
occurs and a fee is charged. If they are the same, no fee is paid.

-----------------------------------------------
REWARD LOGIC
-----------------------------------------------
Reward is based on log-returns of price movement:

    log_ret = log(price_t / price_{t-1})
    reward = current_position * log_ret

Meaning:
    LONG  gains when price rises, loses when it falls
    SHORT gains when price falls, loses when it rises
    FLAT  produces 0 reward

-----------------------------------------------
COSTS
-----------------------------------------------
1) Transaction fee:
   Charged ONLY when switching direction:
        FLAT → LONG or SHORT
        LONG → SHORT
        SHORT → LONG

2) Holding cost:
   A small per-step penalty applied whenever:
        position != 0

-----------------------------------------------
EPISODE FLOW
-----------------------------------------------
Each step:
    1. Agent selects LONG or SHORT
    2. Env computes log-return from previous price to current price
    3. Reward = position * log-return − costs
    4. Position is updated for the next step
    5. New observation window returned
    6. Episode ends at the last price index
"""




    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        prices,
        signal_features,
        window_size=30,
        frame_bound=(30, None),
        fee=0.0003,
        holding_cost=0.0,
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
        self.debug = debug
        self.render_mode = render_mode

        # ----------------------------
        # Bounds
        # ----------------------------
        self._start_tick = frame_bound[0]
        self._end_tick = frame_bound[1] if frame_bound[1] else len(prices)

        # ----------------------------
        # Spaces
        # ----------------------------
        self.action_space = spaces.Discrete(2)  # 0=Short, 1=Long

        # Observation is window_size × num_features
        obs_shape = (window_size, self.features.shape[1])
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
        self._position_history = [self._position]
        self._end_tick = self.frame_bound[1] - 1
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
        terminated = self._current_tick >= self._end_tick
        truncated = False

        # ---------------------------------------------------------
        # Compute reward *using previous position*
        # ---------------------------------------------------------
        reward = self._calculate_reward(action)

        if reward is None or np.isnan(reward) or np.isinf(reward):
            raise ValueError(f"Invalid reward: {reward}")

        # ---------------------------------------------------------
        # Update cumulative totals (log returns)
        # ---------------------------------------------------------
        self._total_reward += reward
        self._total_profit += reward   # <-- cumulative log-return curve

        # ---------------------------------------------------------
        # Update position AFTER applying the reward
        # ---------------------------------------------------------
        self._position = self._action_to_position(action)
        self._position_history.append(self._position)

        # ---------------------------------------------------------
        # Build outputs
        # ---------------------------------------------------------
        obs = self._get_observation()
        info = self._get_info()
        info["total_profit"] = self._total_profit  # <-- ensure exists

        return obs, reward, terminated, truncated, info


    # =====================================================================
    # REWARD LOGIC — CLEAN & CORRECT
    # =====================================================================
    def _calculate_reward(self, action):

        tick = self._current_tick
        if tick < 1 or tick > len(self.prices) - 1:
            raise ValueError("Tick out of price range.")

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
        # Desired new position (for next tick)
        # -------------------------------
        new_position = self._action_to_position(action)

        # -------------------------------
        # BASE REWARD — MUST USE OLD POSITION
        # -------------------------------
        reward = self._position * log_ret

        # -------------------------------
        # Transaction fee when entering or flipping
        # -------------------------------
        if new_position != self._position:
            reward -= self.fee

        # -------------------------------
        # Holding cost while long or short
        # -------------------------------
        if self._position != 0:
            reward -= self.holding_cost

        if self.debug:
            print(
                f"[DEBUG] tick={tick} action={action} "
                f"old_pos={self._position} new_pos={new_position} "
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
        start = self._current_tick - self.window_size
        end = self._current_tick
        window = self.features[start:end]

        if window.shape != self.observation_space.shape:
            raise ValueError(
                f"Bad observation shape {window.shape}, "
                f"expected {self.observation_space.shape}"
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
        }

    # =====================================================================
    # RENDER (OPTIONAL)
    # =====================================================================
    def _render_frame(self):
        pass
