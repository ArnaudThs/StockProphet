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
<<<<<<< Updated upstream
from Project.model import *


def load_market_data(ticker: str):
    df = load_data(

        start_date=START_DATE,
        end_date=END_DATE
    )

    df = df.rename(columns={"Date": "date"})
    df_OHLC = df.sort_values("date").reset_index(drop=True)

    return df_OHLC


# -------------------------
# Utility: Build RNN predictions aligned to dates
# -------------------------
def build_rnn_predictions(df_ohlc: pd.DataFrame, window_size: int = WINDOW_SIZE,
                          epochs: int = LSTM_EPOCHS, batch_size: int = BATCH_SIZE,
                          force_retrain: bool = True) -> pd.Series:
    """
    Train LSTM on historical Close and produce a one-day-ahead prediction for each day
    where enough history exists. Returns a pd.Series indexed by date with predicted value
    in the same scale as the original Close (not scaled).
    """
    df_local = df_ohlc.copy()

    # Normalize date column name
    if "date" in df_local.columns:
        df_local = df_local.rename(columns={"date": "Date"})
    elif "Date" not in df_local.columns:
        raise ValueError("DataFrame must contain either 'Date' or 'date' column.")

    df_local = df_local[["Date", "Close"]].reset_index(drop=True)
    # Use the helper which returns X_train,y_train,X_test,y_test,scaler
    X_train, y_train, X_test, y_test, scaler = train_test_split_lstm(df_local)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, n_features)
    model = LSTM_model(input_shape)
    model = compile_LSTM(model)

    # Train model (if force_retrain or no saved model)
    if force_retrain or not os.path.exists(RNN_MODEL_SAVE):
        print("Training LSTM predictor...")
        train_LSTM(model, X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        model.save(RNN_MODEL_SAVE)
    else:
        print("Loading existing LSTM predictor...")
        from keras.models import load_model
        model = load_model(RNN_MODEL_SAVE)

    # Now generate sliding-window predictions across the full dataset
    closes = df_local["Close"].values.reshape(-1, 1)
    closes_scaled = scaler.transform(closes)  # use same scaler

    preds = []
    dates = []
    for end_idx in range(window_size, len(closes_scaled)):
        start_idx = end_idx - window_size
        seq = closes_scaled[start_idx:end_idx]  # shape (window_size, 1)
        seq = seq.reshape((1, seq.shape[0], seq.shape[1]))
        pred_scaled = model.predict(seq, verbose=0)
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        preds.append(pred)
        dates.append(df_local.loc[end_idx, "Date"])  # prediction aligned to day end_idx

    preds_series = pd.Series(data=preds, index=pd.to_datetime(dates))
    preds_series.name = "rnn_pred_close"
    return preds_series


# -------------------------
# Utility: Merge OHLC / Sentiment / RNN preds and create lag-features
# -------------------------
def build_merged_dataframe(df_ohlc: pd.DataFrame, df_sentiment: pd.DataFrame,
                           rnn_preds: pd.Series, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    Date, Open, High, Low, Close, Volume, sentiment, rnn_pred_close,
    close_lag_1 .. close_lag_{window_size}, next_return (reward target).
    """
    # Normalize column names & Date
=======
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
                           rnn_preds: pd.Series) -> pd.DataFrame:
>>>>>>> Stashed changes
    df = df_ohlc.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    #df = df.join(df_sentiment, how="left").fillna(0.0)
    df = df.join(rnn_preds, how="left")

<<<<<<< Updated upstream
    # Merge sentiment (df_sentiment is indexed by date)
    df_sent = df_sentiment.copy()
    if "date" in df_sent.columns or df_sent.index.name == "date":
        df_sent.index = pd.to_datetime(df_sent.index)
    df_sent = df_sent.rename(columns={df_sent.columns[0]: "sentiment"})
    df = df.join(df_sent, how="left")
    df["sentiment"] = df["sentiment"].fillna(0.0)

    # Merge RNN preds (already indexed by date)
    df = df.join(rnn_preds.rename("rnn_pred_close"), how="left")

    # compute returns (next day) for reward and intraday if needed
    df["close"] = df["Close"].astype(float)
    df["return"] = df["close"].pct_change()
    # next day return as target (reward reference)
    df["next_return"] = df["return"].shift(-1)

    # Create lag features for close (flattened)
    for i in range(1, window_size + 1):
        df[f"close_lag_{i}"] = df["close"].shift(i)
=======
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    for i in range(1, 6):
        df[f"log_ret_lag_{i}"] = df["log_ret"].shift(i)
>>>>>>> Stashed changes

    # drop rows without enough history or without next_return
    df = df.dropna().reset_index()
    return df

<<<<<<< Updated upstream

def load_sentiment():
    # Step 1: Get ticker-specific sentiment entries
    df_daily = fetch_daily_ticker_sentiment(api_key = API_KEY_MASSIVE, ticker = SENTIMENT_TICKERS, start_date = SENTIMENT_START_DATE, end_date = SENTIMENT_END_DATE)

    # Final structure: date | sentiment
    return df_daily


def merge_all(ohlcv, lstm_pred, sentiment):
    df = ohlcv.merge(lstm_pred, on="date", how="left")
    df = df.merge(sentiment, on="date", how="left")
=======
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_cash
        self.shares = 0
        self.current_step = self.window_size
        self.portfolio_history = [self.initial_cash]
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

        # Sharpe Bonus (Risk-adjusted return reward)
        if len(self.returns_window) > 20:
            std_dev = np.std(self.returns_window)
            if std_dev > 1e-9:
                sharpe = np.mean(self.returns_window) / std_dev
                reward += (sharpe * 0.1)

        return self._get_observation(), reward, terminated, False, {}

# -------------------------
# 5. Training Function
# -------------------------
def train_ppo(df):
    # Use DummyVecEnv for training (it's faster and standard for PPO)
    env = DummyVecEnv([lambda: TradingEnv(df)])
    model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.05)
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
    initial_price = df["close"].iloc[0]
    buy_hold_shares = 10000 / initial_price
    buy_hold_history = df["close"].values * buy_hold_shares
    # Trim buy_hold to match the test duration (env starts at window_size)
    # The env runs from window_size to end.
    buy_hold_history = buy_hold_history[env.window_size : env.window_size + len(history)]

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
>>>>>>> Stashed changes

    return df
