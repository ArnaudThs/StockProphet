"""
Test the playback logic without Streamlit UI.
This simulates what should happen when the play button is pressed.

Run with: python multiticker_refactor/streamlit_demo/test_playback_logic.py
"""

class MockSessionState:
    """Simulate st.session_state"""
    def __init__(self):
        self.playing = False
        self.day_slider = 0

def simulate_streamlit_run(session_state, n_steps=10):
    """
    Simulate one Streamlit script execution.
    Returns what would be displayed.
    """
    results = {
        'playing': session_state.playing,
        'day_slider_before': session_state.day_slider,
        'would_rerun': False,
        'day_slider_after': session_state.day_slider,
    }

    # This is what the playback loop does
    if session_state.playing:
        # time.sleep(0.05) - skipped in test
        session_state.day_slider += 1

        if session_state.day_slider >= n_steps:
            session_state.day_slider = 0
            session_state.playing = False

        results['would_rerun'] = True
        results['day_slider_after'] = session_state.day_slider

    return results

def test_play_button():
    """Test that clicking play advances the slider"""
    print("=" * 50)
    print("TEST: Play button functionality")
    print("=" * 50)

    state = MockSessionState()
    n_steps = 10

    print(f"\nInitial state: playing={state.playing}, day_slider={state.day_slider}")

    # Simulate clicking Play button
    print("\n[USER CLICKS PLAY BUTTON]")
    state.playing = not state.playing  # Toggle
    print(f"After click: playing={state.playing}")

    # Simulate several reruns
    print("\n[SIMULATING RERUNS]")
    for i in range(15):  # More than n_steps to test wrap-around
        result = simulate_streamlit_run(state, n_steps)
        print(f"  Run {i+1}: slider={result['day_slider_before']} -> {result['day_slider_after']}, "
              f"playing={state.playing}, would_rerun={result['would_rerun']}")

        if not result['would_rerun']:
            print(f"  [STOPPED - no more reruns]")
            break

    print(f"\nFinal state: playing={state.playing}, day_slider={state.day_slider}")

    # Verify
    assert state.playing == False, "Should stop playing at end"
    assert state.day_slider == 0, "Should reset to 0"
    print("\n✅ TEST PASSED: Play button logic works correctly")

def test_manual_slider():
    """Test that manual slider changes work"""
    print("\n" + "=" * 50)
    print("TEST: Manual slider change")
    print("=" * 50)

    state = MockSessionState()

    print(f"\nInitial: day_slider={state.day_slider}")

    # Simulate user dragging slider to 5
    state.day_slider = 5
    print(f"After manual change: day_slider={state.day_slider}")

    # Run should not rerun (not playing)
    result = simulate_streamlit_run(state, 10)
    print(f"After run: would_rerun={result['would_rerun']}, day_slider={state.day_slider}")

    assert result['would_rerun'] == False, "Should not rerun when not playing"
    assert state.day_slider == 5, "Slider should stay at 5"
    print("\n✅ TEST PASSED: Manual slider works correctly")

def test_pause():
    """Test that pause stops playback"""
    print("\n" + "=" * 50)
    print("TEST: Pause button")
    print("=" * 50)

    state = MockSessionState()
    state.playing = True
    state.day_slider = 3

    print(f"\nInitial: playing={state.playing}, day_slider={state.day_slider}")

    # One rerun while playing
    result = simulate_streamlit_run(state, 10)
    print(f"After run: day_slider={state.day_slider}")

    # User clicks pause
    print("\n[USER CLICKS PAUSE]")
    state.playing = False

    # Run should not advance
    old_slider = state.day_slider
    result = simulate_streamlit_run(state, 10)
    print(f"After run: would_rerun={result['would_rerun']}, day_slider={state.day_slider}")

    assert result['would_rerun'] == False, "Should not rerun when paused"
    assert state.day_slider == old_slider, "Slider should not advance"
    print("\n✅ TEST PASSED: Pause works correctly")

if __name__ == "__main__":
    test_play_button()
    test_manual_slider()
    test_pause()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED - Logic is correct")
    print("=" * 50)
    print("\nIf Streamlit UI doesn't work, the issue is with:")
    print("  1. Button not toggling session_state.playing")
    print("  2. st.rerun() not being called")
    print("  3. Slider key mismatch")
