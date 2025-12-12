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
# PLAYBACK_SPEED is in **milliseconds** (ms)
PLAYBACK_SPEED = 400  # 400ms per frame + animation transition

# ============================
# ACCENT COLORS (NEW)
# ============================
ACCENT_GOLD    = "#FFC145"  # warm, strong contrast
ACCENT_ORANGE  = "#FF7A3D"  # energetic highlight
ACCENT_CORAL   = "#FF4F63"  # distinct soft red
ACCENT_MAGENTA = "#C95FFF"  # standout magenta

ACCENT_PALETTE = [
    ACCENT_GOLD,
    ACCENT_ORANGE,
    ACCENT_CORAL,
    ACCENT_MAGENTA,
]

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

# Custom CSS (unchanged)
st.markdown("""
<style>
.main .block-container {
    padding-top: 0.3rem;
    padding-bottom: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 1400px;
}
h1 { font-size: 0.9rem !important; margin-bottom: 0.2rem !important; margin-top: 0.2rem !important; }
h2 { font-size: 0.75rem !important; margin-top: 0.2rem !important; margin-bottom: 0.15rem !important; }
h3 { font-size: 0.65rem !important; margin-top: 0.15rem !important; margin-bottom: 0.1rem !important; }
h4 { font-size: 0.7rem !important; margin-top: 0.15rem !important; margin-bottom: 0.1rem !important; }
[data-testid="stMetricValue"] { font-size: 3rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.55rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.5rem !important; }
.stPlotlyChart { height: 200px !important; padding: 5px !important; }
[data-testid="stVerticalBlock"] { gap: 0.3rem !important; padding: 5px !important; }
[data-testid="stHorizontalBlock"] { gap: 0.5rem !important; padding: 5px !important; }
[data-testid="column"] { padding: 5px !important; }
.stDataFrame { font-size: 0.65rem !important; }
.stMarkdown p { font-size: 0.7rem !important; margin-bottom: 0.2rem !important; }
.stCaption { font-size: 0.55rem !important; }
hr { margin-top: 0.3rem !important; margin-bottom: 0.3rem !important; }
.stSlider { padding-top: 0.2rem !important; padding-bottom: 0.2rem !important; }
.stRadio > div { gap: 0.3rem !important; }
.stRadio label { font-size: 0.65rem !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# STATE
# =============================================================================

if "playing" not in st.session_state:
    st.session_state.playing = False
if "current_step" not in st.session_state:
    st.session_state.current_step = 0


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data
def load_data():
    try:
        return load_episode_data()
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.stop()

data = load_data()

metadata = data["metadata"]
portfolio_history = data["portfolio_history"]
actions = data["actions"]
prices = data["prices"]

tickers = metadata["tickers"]
initial_capital = metadata["initial_capital"]
n_steps = metadata["n_steps"]
dates = metadata.get("dates", {})

# Build consistent date axis
if dates:
    date_list = [dates[str(i)] for i in range(n_steps)]
else:
    date_list = list(range(n_steps))


# =============================================================================
# NEW COLOR MAP (ACCENT COLORS)
# =============================================================================
def generate_color_map(assets):
    color_map = {}
    for i, asset in enumerate(sorted(assets)):
        color_map[asset] = ACCENT_PALETTE[i % len(ACCENT_PALETTE)]
    return color_map

COLOR_MAP = generate_color_map(tickers + ["Cash"])


# =============================================================================
# TRACKER HELPER — red dot + vertical line
# =============================================================================

def add_tracker(fig, x_value, y_value=None, y_min=None, y_max=None):
    if y_value is not None:
        fig.add_trace(go.Scatter(
            x=[x_value],
            y=[y_value],
            mode="markers",
            marker=dict(size=10, color="red"),
            showlegend=False
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[x_value, x_value],
            y=[y_min, y_max],
            mode="lines",
            line=dict(color="red", width=2),
            showlegend=False
        ))


# =============================================================================
# HEADER + PLAYBACK CONTROLS
# =============================================================================

st.markdown("<div style='font-size:40px; font-weight:700;'>Stock Prophet</div>", unsafe_allow_html=True)

col_play, col_slider = st.columns([1, 9])

with col_play:
    if st.button("▶️ Play" if not st.session_state.playing else "⏸️ Pause"):
        st.session_state.playing = not st.session_state.playing

def on_slider_change():
    st.session_state.current_step = st.session_state.slider_widget

if "slider_widget" not in st.session_state:
    st.session_state.slider_widget = st.session_state.current_step

with col_slider:
    st.slider(
        "Select Trading Day",
        0,
        n_steps - 1,
        key="slider_widget",
        on_change=on_slider_change
    )


# =============================================================================
# CHART TYPE SELECTOR
# =============================================================================

st.markdown("#### Portfolio Evolution")

chart_type = st.radio(
    "Chart Type",
    options=["Portfolio Value", "Price Chart"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")


# =============================================================================
# STATIC COLUMNS
# =============================================================================

col_left_static, col_right_static = st.columns([3, 7])
dynamic = st.empty()


# =============================================================================
# RENDER FUNCTION
# =============================================================================

def render(step):
    with dynamic.container():

        # Date caption
        if dates and str(step) in dates:
            st.caption(f"**{dates[str(step)]}** (Day {step+1} of {n_steps})")
        else:
            st.caption(f"**Day {step+1}** of {n_steps}")

        st.markdown("---")

        col_left, col_right = col_left_static, col_right_static

        # LEFT COLUMN
        with col_left:

            metrics = calculate_metrics(portfolio_history[:step+1], initial_capital)

            st.metric("Portfolio Value", format_currency(metrics["final_value"]))
            st.metric("Return", format_percentage(metrics["total_return"]))
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

            st.markdown("---")

            st.markdown("#### Current Allocation")
            st.caption(f"**Tickers:** {', '.join(tickers)}")

            alloc = get_current_allocation(actions, step, tickers)

            df_alloc = pd.DataFrame({
                "Asset": sorted(alloc.keys()),
                "Percentage": [alloc[a] for a in sorted(alloc.keys())]
            })

            fig_alloc = go.Figure()
            fig_alloc.add_trace(go.Bar(
                x=df_alloc["Percentage"],
                y=df_alloc["Asset"],
                orientation="h",
                marker=dict(color=[COLOR_MAP[a] for a in df_alloc["Asset"]])
            ))

            fig_alloc.update_layout(
                height=220,
                margin=dict(l=15, r=15, t=15, b=15),
                transition=dict(duration=PLAYBACK_SPEED)
            )

            st.plotly_chart(fig_alloc, use_container_width=True)

        # RIGHT COLUMN
        with col_right:

            # PORTFOLIO VALUE
            if chart_type == "Portfolio Value":

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=date_list,
                    y=portfolio_history,
                    mode="lines",
                    line=dict(color="#1f77b4", width=2),
                    showlegend=False
                ))

                add_tracker(fig, x_value=date_list[step], y_value=portfolio_history[step])

                fig.add_hline(
                    y=initial_capital,
                    line_dash="dash",
                    line_color="gray"
                )

                fig.update_layout(
                    height=220,
                    margin=dict(l=50, r=15, t=10, b=35),
                    transition=dict(duration=PLAYBACK_SPEED),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

            # PRICE CHART
            else:
                selected_assets = st.multiselect(
                    "Select Tickers",
                    tickers,
                    default=tickers
                )

                fig = go.Figure()

                for asset in selected_assets:
                    idx = tickers.index(asset)
                    fig.add_trace(go.Scatter(
                        x=date_list,
                        y=prices[:, idx],
                        mode="lines",
                        line=dict(color=COLOR_MAP[asset], width=2),
                        name=asset
                    ))

                if selected_assets:
                    y_min = min(prices[:, tickers.index(a)].min() for a in selected_assets)
                    y_max = max(prices[:, tickers.index(a)].max() for a in selected_assets)
                else:
                    y_min, y_max = 0, 1

                add_tracker(fig, x_value=date_list[step], y_min=y_min, y_max=y_max)

                fig.update_layout(
                    height=250,
                    margin=dict(l=50, r=15, t=10, b=35),
                    transition=dict(duration=PLAYBACK_SPEED),
                    showlegend=False
                )

                fig.update_xaxes(type="category")

                st.plotly_chart(fig, use_container_width=True)

            # TRADES TABLE
            st.markdown("---")
            st.markdown("#### Today's Trades")

            curr = get_current_allocation(actions, step, tickers)
            prev = get_current_allocation(actions, max(step-1, 0), tickers)

            pv = portfolio_history[step]

            rows = []
            for asset in sorted(curr.keys()):
                diff = curr[asset] - prev[asset]
                amount = abs(diff / 100) * pv

                if abs(diff) < 0.1:
                    trade = f"— {curr[asset]:.1f}%"
                elif diff > 0:
                    trade = f"⬆︎ +{diff:.1f}% (${amount:.0f})"
                else:
                    trade = f"⬇︎ {diff:.1f}% (${amount:.0f})"

                rows.append({"Asset": asset, "Trade": trade})

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                height=200
            )


# INITIAL RENDER
render(st.session_state.current_step)


# =============================================================================
# PLAYBACK LOOP
# =============================================================================

if st.session_state.playing:
    time.sleep(PLAYBACK_SPEED / 1000)  # ms → seconds
    st.session_state.current_step += 1

    if st.session_state.current_step >= n_steps:
        st.session_state.current_step = 0
        st.session_state.playing = False

    st.rerun()
