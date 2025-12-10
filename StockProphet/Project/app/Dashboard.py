import sys
import os
import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# FIX PYTHON PATH FOR IMPORTS
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------------
# PIPELINE IMPORTS
# ---------------------------------------------------------
from Project.Pipeline_streamlit import run_full_pipeline_streamlit

# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="StockProphet Dashboard", layout="wide")
st.title("🤖 StockProphet – Trading Bot Dashboard")

# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------
ticker = st.selectbox("Select ticker:", ["AAPL", "MSFT", "NVDA"])

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date")
with col2:
    end_date = st.date_input("End Date")

run_button = st.button("Run Trading Simulation 🚀")

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if run_button:
    st.info("Running full simulation... please wait ⏳")

    try:
        df_test, df_merged, preds, equity_curve = run_full_pipeline_streamlit(
            ticker, start_date, end_date
        )

        # ------------------------------------------
        # 1. DRL Equity Curve
        # ------------------------------------------
        st.subheader("📉 DRL Strategy Performance (Equity Curve)")
        st.line_chart(pd.DataFrame({"Equity": equity_curve}))

        # ------------------------------------------
        # 2. LSTM Predictions
        # ------------------------------------------
        st.subheader("📈 LSTM Price Prediction vs Actual")

        pred_df = pd.DataFrame({
            "Actual Price": df_merged["close"].values[-len(preds):],
            "Predicted Price": preds
        })

        st.line_chart(pred_df)

        st.success("Simulation Completed Successfully ✔")

    except Exception as e:
        st.error(f"❌ Error: {e}")
