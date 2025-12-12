"""
Minimal test for Streamlit playback functionality.
Run with: streamlit run multiticker_refactor/streamlit_demo/test_playback.py
"""
import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Playback Test")

# Initialize session state - use separate counter, not slider key
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

n_steps = 100

st.title("Playback Test")

# Debug info in sidebar
with st.sidebar:
    st.write("### Debug")
    st.write(f"playing: {st.session_state.playing}")
    st.write(f"current_step: {st.session_state.current_step}")

# Play button
if st.button("▶️ Play" if not st.session_state.playing else "⏸️ Pause"):
    st.session_state.playing = not st.session_state.playing

# Slider - use on_change callback to sync manual changes
def on_slider_change():
    st.session_state.current_step = st.session_state.slider_widget

current_step = st.slider(
    "Day",
    min_value=0,
    max_value=n_steps - 1,
    value=st.session_state.current_step,
    key="slider_widget",
    on_change=on_slider_change
)

# Display current step prominently
st.metric("Current Day", st.session_state.current_step)

# Simple chart that changes with step
data = np.sin(np.linspace(0, 4 * np.pi, n_steps))
st.line_chart(data[:st.session_state.current_step + 1])

# Playback loop - MUST be at the end
if st.session_state.playing:
    time.sleep(0.1)  # 10 fps
    st.session_state.current_step += 1

    if st.session_state.current_step >= n_steps:
        st.session_state.current_step = 0
        st.session_state.playing = False

    st.rerun()
