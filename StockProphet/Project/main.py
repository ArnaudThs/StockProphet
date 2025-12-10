import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque

# --- RL & Gym Imports ---
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --- TensorFlow ---
from keras.models import load_model

# --- Project Imports ---
# Ensure these match your actual folder structure
from Project.param import *
from Project.data import load_data, train_test_split_lstm
from Project.sentiment_analysis import fetch_daily_ticker_sentiment
from Project.model import LSTM_model, compile_LSTM, train_LSTM


# -------------------------
# 1. Data Loading
# -------------------------
def load_market_data(ticker: str):
    df = load_data(ticker, START_DATE, END_DATE)
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    return df


#def load_sentiment():
    try:
        df_sent = fetch_daily_ticker_sentiment(
            api_key=API_KEY_MASSIVE,
            ticker=SENTIMENT_TICKERS,
            start_date=SENTIMENT_START_DATE,
            end_date=SENTIMENT_END_DATE
        )
    except Exception as e:
        print(f"Sentiment fetch failed ({e}), filling zeros.")
        full_dates = pd.date_range(SENTIMENT_START_DATE, SENTIMENT_END_DATE, freq="D")
        df_sent = pd.DataFrame(0, index=full_dates, columns=["sentiment"])
        df_sent.index.name = "date"

    df_sent.columns = [c.lower() for c in df_sent.columns]
    df_sent.index = pd.to_datetime(df_sent.index)
    return df_sent.groupby(df_sent.index).mean()

# -------------------------
# 2. LSTM / RNN Logic
# -------------------------
def build_rnn_predictions(df_ohlc: pd.DataFrame, window_size: int = WINDOW_SIZE,
                          epochs: int = LSTM_EPOCHS, batch_size: int = BATCH_SIZE,
                          force_retrain: bool = False) -> pd.Series:

    df_local = df_ohlc[["date", "close"]].copy()
    X_train, y_train, X_test, y_test, scaler = train_test_split_lstm(df_local)

    # Build or Load Model
    if force_retrain or not os.path.exists(RNN_MODEL_PATH):
        print("Training LSTM predictor...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = LSTM_model(input_shape)
        model = compile_LSTM(model)
        train_LSTM(model, X_train, y_train, epochs=epochs, batch_size=batch_size)
        model.save(RNN_MODEL_PATH)
    else:
        print("Loading existing LSTM predictor...")
        model = load_model(RNN_MODEL_PATH)

    # Vectorized Prediction
    print("Generating predictions...")
    closes = df_local["close"].values.reshape(-1, 1)
    closes_scaled = scaler.transform(closes)

    X_all = []
    valid_indices = range(window_size, len(closes_scaled))
    for i in valid_indices:
        X_all.append(closes_scaled[i-window_size:i])
    X_all = np.array(X_all)

    preds_scaled = model.predict(X_all, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    pred_dates = df_local.loc[valid_indices, "date"]

    preds_series = pd.Series(data=preds, index=pd.to_datetime(pred_dates))
    preds_series.name = "rnn_pred_close"
    return preds_series

# -------------------------
# 3. Data Merging
# -------------------------
def build_merged_dataframe(df_ohlc: pd.DataFrame,
                           rnn_preds: pd.Series) -> pd.DataFrame: # Sentiment df
    df = df_ohlc.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    #df = df.join(df_sentiment, how="left").fillna(0.0)
    df = df.join(rnn_preds, how="left")
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    for i in range(1, 6):
        df[f"log_ret_lag_{i}"] = df["log_ret"].shift(i)

    # drop rows without enough history or without next_return
    df = df.dropna().reset_index()
    return df

# -------------------------
# 4. Gymnasium Environment
# -------------------------
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque



import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

TRANSACTION_COST_PCT = 0.0003  # REDUCED further for more trading
CASH_PENALTY_RATE = 0.0005     # INCREASED significantly
HOLDING_PENALTY = 0.15          # TRIPLED: Strong penalty for inaction

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=30):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_cash = 10_000

        # Features
        self.feature_cols = ["close", "rnn_pred_close", "log_ret"]
        self.feature_cols += [c for c in df.columns if "lag" in c]

        obs_dim = (len(self.feature_cols) * self.window_size) + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell

        # Tracking
        self.returns_window = deque(maxlen=50)
        self.portfolio_history = []
        self.trade_log = []
        self.max_portfolio_value = self.initial_cash
        self.steps_since_trade = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_cash
        self.shares = 0
        self.current_step = self.window_size
        self.portfolio_history = [self.initial_cash]
        self.trade_log = []
        self.returns_window.clear()
        self.max_portfolio_value = self.initial_cash
        self.steps_since_trade = 0
        return self._get_observation(), {}

    def _get_observation(self):
        start = self.current_step - self.window_size
        window = self.df.loc[start:self.current_step-1, self.feature_cols].values.flatten()
        obs = np.concatenate([window, [self.cash / 10000.0, self.shares / 100.0]])
        return obs.astype(np.float32)

    def step(self, action):
        price = self.df.loc[self.current_step, "close"]
        if isinstance(action, np.ndarray):
            action = action.item()

        prev_shares = self.shares
        prev_cash = self.cash

        # -- Execute Trade --
        if action == 1:  # BUY
            max_shares = int(self.cash / (price * (1 + TRANSACTION_COST_PCT)))
            shares_to_buy = min(10, max_shares)  # Buy up to 10 shares at once

            if shares_to_buy > 0:
                cost = shares_to_buy * price * (1 + TRANSACTION_COST_PCT)
                self.cash -= cost
                self.shares += shares_to_buy
                self.trade_log.append(("BUY", self.current_step, price, shares_to_buy))
                self.steps_since_trade = 0

        elif action == 2:  # SELL
            shares_to_sell = min(10, self.shares)  # Sell up to 10 shares at once

            if shares_to_sell > 0:
                proceeds = shares_to_sell * price * (1 - TRANSACTION_COST_PCT)
                self.cash += proceeds
                self.shares -= shares_to_sell
                self.trade_log.append(("SELL", self.current_step, price, shares_to_sell))
                self.steps_since_trade = 0

        # Track inactivity
        if action == 0 or (prev_shares == self.shares and prev_cash == self.cash):
            self.steps_since_trade += 1

        # Move step forward
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1

        # Cash depreciation (opportunity cost)
        depreciation_amount = self.cash * CASH_PENALTY_RATE
        self.cash -= depreciation_amount

        # --- Portfolio Value ---
        current_price = self.df.loc[self.current_step, "close"]
        portfolio_value = self.cash + self.shares * current_price
        prev_value = self.portfolio_history[-1]
        self.portfolio_history.append(portfolio_value)

        # --- Reward Engineering ---
        step_return = (portfolio_value - prev_value) / prev_value
        self.returns_window.append(step_return)

        # Base reward: portfolio change
        reward = step_return * 100

        # 1. DIRECTIONAL REWARD (predict price movement correctly)
        price_change = (current_price - price) / price

        if action == 1:  # BUY
            # Reward if price goes UP after buying
            reward += max(0, price_change * 50)
        elif action == 2:  # SELL
            # Reward if price goes DOWN after selling
            reward += max(0, -price_change * 50)

        # 2. PROFIT TAKING BONUS (sell near peaks)
        if action == 2 and self.shares >= 0:
            self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
            near_peak_factor = portfolio_value / self.max_portfolio_value
            reward += near_peak_factor * 2.0

        # 3. HOLDING PENALTY - Punish doing nothing (INCREASED)
        if action == 0:
            reward -= HOLDING_PENALTY * 2  # Double the penalty

        # 4. INACTIVITY PENALTY - Punish long periods without trading (MORE AGGRESSIVE)
        if self.steps_since_trade > 10:  # Reduced from 20
            reward -= 0.3 * (self.steps_since_trade - 10)  # Increased from 0.1

        # 5. CASH SITTING PENALTY - Punish having too much uninvested cash (STRONGER)
        cash_ratio = self.cash / portfolio_value
        if cash_ratio > 0.5:  # Lowered from 0.8
            reward -= (cash_ratio - 0.3) * 5.0  # Increased from 2.0

        # 6. POSITION HOLDING REWARD - Reward staying invested
        if self.shares > 0:
            investment_ratio = (self.shares * current_price) / portfolio_value
            reward += investment_ratio * 0.5  # Bonus for being invested

        # 6. EXPOSURE REWARD - Reward being invested when market goes up
        if self.shares > 0 and price_change > 0:
            exposure_ratio = (self.shares * current_price) / portfolio_value
            reward += exposure_ratio * price_change * 30

        # 7. DRAWDOWN PENALTY (but reduced magnitude)
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        dd = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        reward -= dd * 3  # Reduced from 7

        # 8. VOLATILITY PENALTY (but only extreme volatility)
        if len(self.returns_window) > 10:
            vol = np.std(self.returns_window)
            if vol > 0.02:  # Only penalize high volatility
                reward -= (vol - 0.02) * 50

        # 9. FINAL PORTFOLIO BONUS (encourage growth)
        if terminated:
            final_return = (portfolio_value - self.initial_cash) / self.initial_cash
            reward += final_return * 200  # Big bonus for positive final returns

        return self._get_observation(), reward, terminated, False, {}


# UPDATED TRAINING PARAMETERS IN param.py:
"""
PPO_TIMESTEPS = 200_000  # Increase to at least 200k
TRANSACTION_COST_PCT = 0.0005  # Lower transaction costs
LEARNING_RATE = 0.0001  # Add this - lower learning rate
ENT_COEF = 0.2  # Increase entropy for more exploration
"""

# UPDATED PPO TRAINING:
def train_ppo(df):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: TradingEnv(df)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0001,      # Lower learning rate
        ent_coef=0.2,               # Higher entropy = more exploration
        n_steps=2048,               # Longer episodes before update
        batch_size=64,              # Smaller batches
        n_epochs=10,                # More gradient steps per update
        gamma=0.99,                 # Discount factor
        clip_range=0.2,             # PPO clip range
    )

    model.learn(total_timesteps=200_000)  # DO NOT REDUCE THIS
    model.save("ppo_trader")
    return model

# -------------------------
# 6. Testing & Metrics (FIXED)
# -------------------------
def calculate_metrics(portfolio_values, initial_balance=10000):
    """Calculates accuracy metrics for the strategy"""
    portfolio_values = np.array(portfolio_values)

    # 1. Total Return
    total_return = (portfolio_values[-1] - initial_balance) / initial_balance * 100

    # 2. Max Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min() * 100

    # 3. Sharpe Ratio (Annualized)
    # Calculate daily returns
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if np.std(daily_returns) > 1e-9:
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    return total_return, max_drawdown, sharpe_ratio

def test_ppo(df):
    model = PPO.load(PPO_MODEL_PATH)

    # FIX: Do NOT use DummyVecEnv for testing if we want easy access to history
    # We create a single environment instance.
    env = TradingEnv(df)
    obs, _ = env.reset()
    done = False

    print("Running Backtest...")

    while not done:
        # PPO expects a batch dimension (e.g., [1, n_features])
        # We reshape our single observation to match.
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

    # --- Plotting & Accuracy ---
    history = env.portfolio_history

    # Generate Buy & Hold Comparison
    start_idx = env.window_size
    initial_price = df["close"].iloc[start_idx]
    buy_hold_shares = 10000 / initial_price

    # Get the price history from the window_size onwards
    # We must ensure lengths match perfectly
    relevant_prices = df["close"].iloc[start_idx : start_idx + len(history)].values
    buy_hold_history = relevant_prices * buy_hold_shares

    # Calculate Metrics
    strat_ret, strat_dd, strat_sharpe = calculate_metrics(history)
    bh_ret, bh_dd, bh_sharpe = calculate_metrics(buy_hold_history)

    print("\n" + "="*40)
    print("      PERFORMANCE REPORT      ")
    print("="*40)
    print(f"{'Metric':<20} | {'AI Strategy':<12} | {'Buy & Hold':<12}")
    print("-" * 50)
    print(f"{'Total Return':<20} | {strat_ret:8.2f}%    | {bh_ret:8.2f}%")
    print(f"{'Max Drawdown':<20} | {strat_dd:8.2f}%    | {bh_dd:8.2f}%")
    print(f"{'Sharpe Ratio':<20} | {strat_sharpe:8.2f}      | {bh_sharpe:8.2f}")
    print("="*40 + "\n")

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(history, label="AI Strategy", color="blue", linewidth=2)
    plt.plot(buy_hold_history, label="Buy & Hold (Benchmark)", color="gray", alpha=0.6, linestyle="--")
    plt.title(f"Backtest Results: {TICKER}")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

        # --- Plot Trades ---
    buy_x = [t[1] - start_idx for t in env.trade_log if t[0] == "BUY"]
    buy_y = [t[2] for t in env.trade_log if t[0] == "BUY"]

    sell_x = [t[1] - start_idx for t in env.trade_log if t[0] == "SELL"]
    sell_y = [t[2] for t in env.trade_log if t[0] == "SELL"]

    plt.figure(figsize=(14, 6))
    plt.plot(relevant_prices, label="Close Price", alpha=0.6)
    plt.scatter(buy_x, buy_y, marker="^", color="green", s=100, label="BUY")
    plt.scatter(sell_x, sell_y, marker="v", color="red", s=100, label="SELL")
    plt.title("Trading Actions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\nTRADE LOG SUMMARY")
    print("----------------------")
    for t in env.trade_log:
        print(f"{t[0]} at day={t[1]} price={t[2]:.2f}")


# -------------------------
# 7. Main Execution
# -------------------------
def main():
    print("--- Starting Trading Bot Pipeline ---")

    # 1. Load Data
    print(f"Loading data for {TICKER}...")
    df_ohlc = load_market_data(TICKER)

    # 2. RNN Predictions
    print("Running RNN Analysis...")
    rnn_preds = build_rnn_predictions(df_ohlc, force_retrain=False)

    # 3. Sentiment
    #print("Fetching Sentiment...")
    #df_sent = load_sentiment()

    # 4. Merge
    print("Merging Data...")
    df_merged = build_merged_dataframe(df_ohlc, rnn_preds)

    # 5. Split
    split_idx = int(len(df_merged) * TRAIN_RATIO)
    df_train = df_merged.iloc[:split_idx].reset_index(drop=True)
    df_test = df_merged.iloc[split_idx:].reset_index(drop=True)
    print(f"Train Size: {len(df_train)}, Test Size: {len(df_test)}")

    # 6. Train RL Agent
    print("Training PPO Agent...")
    train_ppo(df_train)

    # 7. Test
    print("Testing Strategy...")
    test_ppo(df_test)
    print("Done.")

if __name__ == "__main__":
    main()
