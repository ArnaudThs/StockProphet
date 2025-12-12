"""
Streamlit Dashboard for Multi-Ticker Trading Agent

Usage:
    streamlit run StockProphet/multiticker_refactor/streamlit_demo/app.py
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
PLAYBACK_SPEED = 0.5  # Seconds between frames (0.05 = 20 fps, 0.1 = 10 fps)
from utils import (
    load_episode_data,
    calculate_metrics,
    get_current_allocation,
    format_currency,
    format_percentage
)

# Page config
st.set_page_config(
    page_title="Multi-Ticker Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ultra-compact layout (your original unchanged)
st.markdown("""
<style>
    .main .block-container {
        padding-top: 0.3rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 1400px;
    }
    h1 {
        font-size: 0.9rem !important;
        margin-bottom: 0.2rem !important;
        margin-top: 0.2rem !important;
    }
    h2 {
        font-size: 0.75rem !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.15rem !important;
    }
    h3 {
        font-size: 0.65rem !important;
        margin-top: 0.15rem !important;
        margin-bottom: 0.1rem !important;
    }
    h4 {
        font-size: 0.7rem !important;
        margin-top: 0.15rem !important;
        margin-bottom: 0.1rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 0.75rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.55rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.5rem !important;
    }
    .stPlotlyChart {
        height: 200px !important;
        padding: 5px !important;
    }
    [data-testid="stVerticalBlock"] {
        gap: 0.3rem !important;
        padding: 5px !important;
    }
    [data-testid="stHorizontalBlock"] {
        gap: 0.5rem !important;
        padding: 5px !important;
    }
    [data-testid="column"] {
        padding: 5px !important;
    }
    .stDataFrame {
        font-size: 0.65rem !important;
    }
    .stMarkdown p {
        font-size: 0.7rem !important;
        margin-bottom: 0.2rem !important;
    }
    .stCaption {
        font-size: 0.55rem !important;
    }
    hr {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
    }
    .stSlider {
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
    }
    .stRadio > div {
        gap: 0.3rem !important;
    }
    .stRadio label {
        font-size: 0.65rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for playback
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# Load data
@st.cache_data
def load_data():
    """Load episode data (cached)."""
    try:
        return load_episode_data()
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.stop()

data = load_data()

# Extract data
metadata = data['metadata']
portfolio_history = data['portfolio_history']
actions = data['actions']
rewards = data['rewards']
prices = data['prices']

tickers = metadata['tickers']
initial_capital = metadata['initial_capital']
n_steps = metadata['n_steps']
dates = metadata.get('dates', {})  # Get date mapping if available

# Title
st.markdown(
    "<div style='font-size:40px; font-weight:700;'>Stock Prophet</div>",
    unsafe_allow_html=True
)

# Playback controls
col_play, col_slider = st.columns([1, 9])

with col_play:
    if st.button("▶️ Play" if not st.session_state.playing else "⏸️ Pause"):
        st.session_state.playing = not st.session_state.playing

# Sync slider widget state with current_step BEFORE rendering
if 'slider_widget' not in st.session_state:
    st.session_state.slider_widget = st.session_state.current_step
else:
    st.session_state.slider_widget = st.session_state.current_step

def on_slider_change():
    st.session_state.current_step = st.session_state.slider_widget

with col_slider:
    st.slider(
        "Select Trading Day",
        min_value=0,
        max_value=n_steps - 1,
        key="slider_widget",
        on_change=on_slider_change,
        help=f"Slide to view portfolio state on different days"
    )

# STATIC WIDGETS THAT MUST NOT BE IN THE DYNAMIC LOOP
st.markdown("#### Portfolio Evolution")

chart_type = st.radio(
    "Chart Type",
    options=["Portfolio Value", "Price Chart"],
    horizontal=True,
    label_visibility="collapsed"
)

if chart_type == "Price Chart":
    selected_ticker_idx = st.selectbox(
        "Ticker",
        options=range(len(tickers)),
        format_func=lambda i: tickers[i],
        label_visibility="collapsed"
    )
else:
    selected_ticker_idx = None

st.markdown("---")

# ===========================================================================
# DYNAMIC FRAME (ONLY THIS RE-RENDERS)
# ===========================================================================
frame = st.empty()

def render_dynamic(step):
    with frame.container():
        # ===========================================================
        # DATE
        # ===========================================================
        if dates and str(step) in dates:
            current_date = dates[str(step)]
            st.caption(f"**{current_date}** (Day {step + 1} of {n_steps})")
        else:
            st.caption(f"**Day {step + 1}** of {n_steps}")

        st.markdown("---")

        # ===========================================================
        # MAIN LAYOUT
        # ===========================================================
        col_left, col_right = st.columns([3, 7])

        # LEFT COLUMN
        with col_left:

            metrics = calculate_metrics(portfolio_history[:step + 1], initial_capital)

            st.metric("Portfolio Value", format_currency(metrics['final_value']))
            st.metric("Return", format_percentage(metrics['total_return']))
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

            st.markdown("---")

            st.markdown("#### Current Allocation")
            st.caption(f"**Tickers:** {', '.join(tickers)}")

            allocation = get_current_allocation(actions, step, tickers)
            allocation_df = pd.DataFrame({
                'Asset': list(allocation.keys()),
                'Percentage': list(allocation.values())
            }).query("abs(Percentage) > 0.1")

            if len(allocation_df) > 0:
                fig_pie = px.pie(
                    allocation_df,
                    values='Percentage',
                    names='Asset',
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=8)
                fig_pie.update_layout(height=200, margin=dict(l=15, r=15, t=15, b=20), showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No allocations")

        # RIGHT COLUMN
        with col_right:
            if chart_type == "Portfolio Value":
                portfolio_df = pd.DataFrame({
                    'Day': np.arange(len(portfolio_history)),
                    'Portfolio Value': portfolio_history
                })

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=portfolio_df['Day'], y=portfolio_df['Portfolio Value'], mode='lines'))
                fig.add_trace(go.Scatter(x=[step], y=[portfolio_history[step]], mode='markers'))

                fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray")
                fig.update_layout(height=220, margin=dict(l=50, r=15, t=10, b=35), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            else:
                ticker_prices = prices[:, selected_ticker_idx]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.arange(len(ticker_prices)), y=ticker_prices, mode='lines'))
                fig.add_trace(go.Scatter(x=[step], y=[ticker_prices[step]], mode='markers'))

                fig.update_layout(height=220, margin=dict(l=50, r=15, t=10, b=35), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Trades table
            st.markdown("---")
            st.markdown("#### Today's Trades")

            curr_alloc = get_current_allocation(actions, step, tickers)
            prev_alloc = get_current_allocation(actions, max(0, step - 1), tickers)

            rows = []
            portfolio_value = portfolio_history[step]
            current_prices = prices[min(step, len(prices)-1)]

            for asset, curr_pct in curr_alloc.items():
                prev_pct = prev_alloc[asset]
                change_pct = curr_pct - prev_pct
                dollar_amount = abs(change_pct / 100) * portfolio_value

                if abs(change_pct) < 0.1:
                    trade_str = f"— {curr_pct:.1f}%"
                elif change_pct > 0:
                    trade_str = f"⬆︎ +{change_pct:.1f}% (${dollar_amount:.0f})"
                else:
                    trade_str = f"⬇︎ {change_pct:.1f}% (${dollar_amount:.0f})"

                rows.append({"Asset": asset, "Trade": trade_str})

            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=200)

# INITIAL RENDER
render_dynamic(st.session_state.current_step)

# PLAYBACK LOOP
if st.session_state.playing:
    time.sleep(PLAYBACK_SPEED)
    st.session_state.current_step += 1

    if st.session_state.current_step >= n_steps:
        st.session_state.current_step = 0
        st.session_state.playing = False

    st.rerun()
