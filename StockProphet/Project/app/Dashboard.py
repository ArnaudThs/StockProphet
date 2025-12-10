import streamlit as st
import pandas as pd
from Project.Pipeline_streamlit import run_lstm_prediction, run_drl_simulation
from Project.sentiment_analysis import get_daily_sentiment_filled

# ------------------------------------------------
#   STREAMLIT PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="StockProphet Dashboard",
    layout="wide"
)

st.title("🤖 StockProphet – Trading Bot Dashboard")

# ------------------------------------------------
#   USER INPUTS
# ------------------------------------------------
ticker = st.selectbox("Select Ticker", ["AAPL", "MSFT", "NVDA"])
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date")
with col2:
    end_date = st.date_input("End Date")

run_button = st.button("Run Trading Simulation 🚀")

# ------------------------------------------------
#   MAIN EXECUTION
# ------------------------------------------------
if run_button:
    st.write("Running simulation... please wait.")

    # ---- 1. DRL Simulation ----
    df_test, equity_curve = run_drl_simulation(ticker, start_date, end_date)

    st.subheader("📉 Equity Curve (DRL Strategy Performance)")
    st.line_chart(equity_curve)

    # ---- 2. LSTM Prediction ----
    st.subheader("📈 Price Prediction (LSTM Model)")
    df_prices, preds = run_lstm_prediction(ticker, start_date, end_date)

    # Create a DF for Streamlit charting
    pred_df = pd.DataFrame({
        "Actual Price": df_prices["Close"].values[-len(preds):],
        "Predicted Price": preds
    })
    st.line_chart(pred_df)

    # ---- 3. Sentiment (Optional) ----
    st.subheader("📰 Daily News Sentiment")
    try:
        sentiment_df = get_daily_sentiment_filled(ticker)
        st.line_chart(sentiment_df["sentiment"])
    except Exception as e:
        st.warning(f"Sentiment unavailable: {e}")

    st.success("Simulation Completed ✔")
