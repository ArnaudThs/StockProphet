import pandas as pd
from Project.data import load_data
from Project.main import build_rnn_predictions, build_merged_dataframe
from stable_baselines3 import PPO
from Project.main import TradingEnv

def normalize_ohlc(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Fix date column names
    if "date" not in df.columns:
        for c in df.columns:
            if "date" in c:
                df = df.rename(columns={c: "date"})
                break

    # Fix close column names
    if "close" not in df.columns:
        for c in df.columns:
            if "close" in c:
                df = df.rename(columns={c: "close"})
                break

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    return df


def load_data_streamlit(ticker, start_date, end_date):
    df = load_data(ticker, start_date, end_date)
    df = normalize_ohlc(df)
    return df


def prepare_dataset(df_ohlc):
    preds = build_rnn_predictions(df_ohlc, force_retrain=False)
    df_merged = build_merged_dataframe(df_ohlc, preds)
    return df_merged, preds


def run_drl(df_merged):
    split = int(len(df_merged) * 0.7)
    df_train = df_merged.iloc[:split].reset_index(drop=True)
    df_test  = df_merged.iloc[split:].reset_index(drop=True)

    # Train PPO
    env_train = TradingEnv(df_train)
    model = PPO("MlpPolicy", env_train, verbose=0)
    model.learn(total_timesteps=50_000)

    # Test PPO
    env_test = TradingEnv(df_test)
    obs, _ = env_test.reset()
    done = False

    while not done:
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action.item())
        done = terminated or truncated

    equity_curve = env_test.portfolio_history

    return df_test, equity_curve


def run_full_pipeline_streamlit(ticker, start_date, end_date):
    # 1. Load data
    df_ohlc = load_data_streamlit(ticker, start_date, end_date)

    # 2. LSTM + merge
    df_merged, preds = prepare_dataset(df_ohlc)

    # 3. DRL
    df_test, equity_curve = run_drl(df_merged)

    return df_test, df_merged, preds, equity_curve
