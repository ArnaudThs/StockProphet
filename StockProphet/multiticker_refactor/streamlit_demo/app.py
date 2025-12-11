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

# Custom CSS for ultra-compact layout (50% smaller, MacBook Pro 14" optimized)
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
    /* Column styling to prevent edge clipping */
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
    /* Slider styling */
    .stSlider {
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
    }
    /* Radio button styling */
    .stRadio > div {
        gap: 0.3rem !important;
    }
    .stRadio label {
        font-size: 0.65rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for playback
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'speed' not in st.session_state:
    st.session_state.speed = 0  # 0=pause, 1=normal, 5=fast

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

# Calculate metrics
metrics = calculate_metrics(portfolio_history, initial_capital)

# Title
st.markdown(
    "<div style='font-size:70px; font-weight:700;'>Stock Prophet</div>",
    unsafe_allow_html=True
)

# Playback controls
col_play, col_slider = st.columns([1, 9])

with col_play:
    if st.button("▶️ Play" if not st.session_state.playing else "⏸️ Pause"):
        st.session_state.playing = not st.session_state.playing

with col_slider:
    # Slider synced with session state
    current_step = st.slider(
        "Select Trading Day",
        min_value=0,
        max_value=n_steps - 1,
        value=st.session_state.current_step,
        key="day_slider",
        help=f"Slide to view portfolio state on different days"
    )
    # Update session state if slider moved manually
    st.session_state.current_step = current_step

# Debug: Check if dates are loaded
if not dates:
    st.warning("⚠️ Dates not found in metadata. Run evaluation again to generate date information.")

# Show date instead of day number if dates are available
if dates and str(current_step) in dates:
    current_date = dates[str(current_step)]
    st.caption(f"**{current_date}** (Day {current_step + 1} of {n_steps})")
else:
    st.caption(f"**Day {current_step + 1}** of {n_steps}")

st.markdown("---")

# ===========================================================================
# MAIN LAYOUT: LEFT SIDEBAR (30%) + RIGHT CONTENT (70%)
# ===========================================================================
col_left, col_right = st.columns([3, 7])

# ===========================================================================
# LEFT COLUMN: Metrics + Allocation
# ===========================================================================
with col_left:
    # Metrics
    st.metric("Final Value", format_currency(metrics['final_value']))
    st.metric("Return", format_percentage(metrics['total_return']))
    st.metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    st.metric("Max DD", format_percentage(metrics['max_drawdown']))
    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

    st.markdown("---")

    # Tickers + Allocation
    st.markdown("#### Current Allocation")
    st.caption(f"**Tickers:** {', '.join(tickers)}")

    allocation = get_current_allocation(actions, current_step, tickers)
    allocation_df = pd.DataFrame({
        'Asset': list(allocation.keys()),
        'Percentage': list(allocation.values())
    })
    allocation_df = allocation_df[np.abs(allocation_df['Percentage']) > 0.1]

    if len(allocation_df) > 0:
        fig_pie = px.pie(
            allocation_df,
            values='Percentage',
            names='Asset',
            hole=0.4
        )

        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=8
        )

        fig_pie.update_layout(
            height=200,
            margin=dict(l=15, r=15, t=15, b=20),
            showlegend=False,
            autosize=True
        )

        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No allocations")

# ===========================================================================
# RIGHT COLUMN: Chart + Trades
# ===========================================================================
with col_right:
    # Row 1: Chart with toggle
    st.markdown("#### Portfolio Evolution")

    chart_type = st.radio(
        "Chart Type",
        options=["Portfolio Value", "Price Chart"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if chart_type == "Portfolio Value":
        portfolio_df = pd.DataFrame({
            'Day': np.arange(len(portfolio_history)),
            'Portfolio Value': portfolio_history
        })

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=portfolio_df['Day'],
            y=portfolio_df['Portfolio Value'],
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=[current_step],
            y=[portfolio_history[current_step]],
            mode='markers',
            name='Current',
            marker=dict(size=10, color='red')
        ))

        fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray")

        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Value ($)",
            hovermode='x unified',
            height=220,
            showlegend=False,
            margin=dict(l=50, r=15, t=10, b=35),
            font=dict(size=9),
            autosize=True
        )

        st.plotly_chart(fig, use_container_width=True)

    else:  # Price Chart
        selected_ticker_idx = st.selectbox(
            "Ticker",
            options=range(len(tickers)),
            format_func=lambda i: tickers[i],
            label_visibility="collapsed"
        )

        ticker_prices = prices[:, selected_ticker_idx]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=np.arange(len(ticker_prices)),
            y=ticker_prices,
            mode='lines',
            name=tickers[selected_ticker_idx],
            line=dict(color='#2ca02c', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=[current_step],
            y=[ticker_prices[current_step]],
            mode='markers',
            name='Current',
            marker=dict(size=10, color='red')
        ))

        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=220,
            showlegend=False,
            margin=dict(l=50, r=15, t=10, b=35),
            font=dict(size=9),
            autosize=True
        )

        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Allocation table
    st.markdown("---")
    st.markdown("#### Current Allocation")

    # Show current allocation
    allocation = get_current_allocation(actions, current_step, tickers)
    allocation_data = {
        'Ticker': list(allocation.keys()),
        'Allocation': [f"{v:.1f}%" for v in allocation.values()]
    }

    allocation_df = pd.DataFrame(allocation_data)
    st.dataframe(allocation_df, use_container_width=True, height=120, hide_index=True)

# ===========================================================================
# PLAYBACK LOOP (must be at the end)
# ===========================================================================
if st.session_state.playing:
    time.sleep(0.05)  # Adjust speed (0.05 = 20 fps)
    st.session_state.current_step += 1

    # Loop back to start or stop at end
    if st.session_state.current_step >= n_steps:
        st.session_state.current_step = 0
        st.session_state.playing = False

    st.rerun()
