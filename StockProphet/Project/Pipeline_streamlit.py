import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from Project.param import *
from Project.data import load_data, train_test_split_lstm
from Project.model import LSTM_model, compile_LSTM, train_LSTM
from keras.models import load_model
from .main import TradingEnv, build_rnn_predictions, build_merged_dataframe

# ----------------------------------------------------
# 1. LSTM PREDICTION ONLY
# ----------------------------------------------------
def run_lstm_prediction(ticker, start_date, end_date):
    df = load_data(ticker, start_date, end_date)
    preds = build_rnn_predictions(df, force_retrain=False)

    # align predictions with actual data
    df = df.sort_values("date").reset_index(drop=True)
    preds = preds.reindex(pd.to_datetime(df["date"]), method="nearest")

    return df, preds.values


# ----------------------------------------------------
# 2. DRL SIMULATION ONLY
# ----------------------------------------------------
def run_drl_simulation(ticker, start_date, end_date):
    print("Preparing data...")

    # Load OHLC
    df_ohlc = load_data(ticker, start_date, end_date)

    # Predict using LSTM
    rnn_preds = build_rnn_predictions(df_ohlc)

    # Merge OHLC + predictions
    df_merged = build_merged_dataframe(df_ohlc, rnn_preds)

    # Train/Test split (simple split for demo)
    split = int(len(df_merged) * 0.7)
    df_train = df_merged.iloc[:split].reset_index(drop=True)
    df_test  = df_merged.iloc[split:].reset_index(drop=True)

    # Train PPO
    env_train = TradingEnv(df_train)
    model = PPO("MlpPolicy", env_train, verbose=0)
    model.learn(total_timesteps=50_000)

    # Test the strategy
    env_test = TradingEnv(df_test)
    obs, _ = env_test.reset()
    done = False

    while not done:
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action.item())
        done = terminated or truncated

    # Return portfolio history
    equity_curve = env_test.portfolio_history
    df_test_cut = df_test.iloc[env_test.window_size : env_test.window_size + len(equity_curve)]

    return df_test_cut, equity_curve


# ----------------------------------------------------
# 3. FULL PIPELINE (OPTIONAL)
# ----------------------------------------------------
def run_full_pipeline():
    """For debugging outside Streamlit."""
    from .main import main
    main()
