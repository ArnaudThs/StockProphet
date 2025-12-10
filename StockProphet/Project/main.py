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

# --- CONSTANTS ---
WINDOW_SIZE = 50
LSTM_EPOCHS = 10
BATCH_SIZE = 32
PPO_TIMESTEPS = 50_000
PPO_MODEL_PATH = "ppo_trader"
RNN_MODEL_PATH = "lstm_rnn.keras"
TRANSACTION_COST_PCT = 0.001
MOVEMENT_BONUS = 0.01

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
class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, window_size: int = 30):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_cash = 10_000

        # Features
        self.feature_cols = ["close", "rnn_pred_close", "log_ret"] # Add Sentiment Back After
        self.feature_cols += [c for c in df.columns if "lag" in c]

        obs_dim = (len(self.feature_cols) * self.window_size) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell

        # Metrics Tracking
        self.returns_window = deque(maxlen=50)
        self.portfolio_history = []
        self.max_portfolio_value = self.initial_cash

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_cash
        self.shares = 0
        self.current_step = self.window_size
        self.portfolio_history = [self.initial_cash]
        self.max_portfolio_value = self.initial_cash
        self.returns_window.clear()
        return self._get_observation(), {}

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        data_window = self.df.loc[start:end-1, self.feature_cols].values.flatten()
        state = np.concatenate([data_window, [self.cash / 10000.0, self.shares / 100.0]])
        return state.astype(np.float32)

    def step(self, action):
        price = self.df.loc[self.current_step, "close"]
        if isinstance(action, np.ndarray): action = action.item()

        # Execute Trade
        if action == 1: # Buy
            cost = price * (1 + TRANSACTION_COST_PCT)
            if self.cash >= cost:
                self.cash -= cost
                self.shares += 1
        elif action == 2: # Sell
            if self.shares > 0:
                proceeds = price * (1 - TRANSACTION_COST_PCT)
                self.cash += proceeds
                self.shares -= 1

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1

        # Calculate Value
        new_portfolio_value = self.cash + (self.shares * self.df.loc[self.current_step, "close"])
        prev_portfolio_value = self.portfolio_history[-1]

        # Log History
        self.portfolio_history.append(new_portfolio_value)

        # --- REWARD CALCULATION ---
        step_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.returns_window.append(step_return)

        # Base Reward
        reward = step_return * 100

        # Update Peak Value
        self.max_portfolio_value = max(self.max_portfolio_value, new_portfolio_value)

        # Calculate Drawdown
        drawdown = (self.max_portfolio_value - new_portfolio_value) / self.max_portfolio_value
        DRAWDOWN_PENALTY_COEF = 10.0 # Tune this coefficient
        if drawdown > 0.01:
            reward -= DRAWDOWN_PENALTY_COEF * (drawdown ** 2)

        # Sharpe Bonus (Risk-adjusted return reward)
        if len(self.returns_window) > 20:
            std_dev = np.std(self.returns_window)
            if std_dev > 1e-9:
                sharpe = np.mean(self.returns_window) / std_dev
                reward += (sharpe * 0.1)


        if action == 1 or action == 2: # If the agent chose Buy or Sell
            reward += MOVEMENT_BONUS

        return self._get_observation(), reward, terminated, False, {}

# -------------------------
# 5. Training Function
# -------------------------
def train_ppo(df):
    # Use DummyVecEnv for training (it's faster and standard for PPO)
    env = DummyVecEnv([lambda: TradingEnv(df)])
    model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.1)
    model.learn(total_timesteps=PPO_TIMESTEPS)
    model.save(PPO_MODEL_PATH)
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
