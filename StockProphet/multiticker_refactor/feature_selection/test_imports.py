"""
Quick import test to verify feature selection module is properly configured.
Run this before the full pipeline to catch any import errors early.
"""

def test_imports():
    """Test all imports required by feature selection module."""
    print("Testing imports for feature selection module...")

    try:
        # Test statistical selector
        print("  [1/5] Testing StatisticalFeatureSelector...")
        from .statistical_selector import StatisticalFeatureSelector
        print("    ✓ StatisticalFeatureSelector imports successfully")

        # Test RL validator
        print("  [2/5] Testing RLFeatureValidator...")
        from .rl_validator import RLFeatureValidator
        print("    ✓ RLFeatureValidator imports successfully")

        # Test config
        print("  [3/5] Testing config...")
        from .config import (
            FEATURE_COUNTS_TO_TEST, INTERMEDIATE_COUNT,
            RL_VALIDATION_TIMESTEPS, RL_VALIDATION_SEEDS,
            RL_VALIDATION_METRIC, MIN_IMPROVEMENT_THRESHOLD
        )
        print("    ✓ Config imports successfully")
        print(f"      Feature counts to test: {FEATURE_COUNTS_TO_TEST}")
        print(f"      RL validation: {RL_VALIDATION_TIMESTEPS:,} timesteps, {RL_VALIDATION_SEEDS} seeds")
        print(f"      Primary metric: {RL_VALIDATION_METRIC}")

        # Test pipeline import
        print("  [4/5] Testing pipeline import...")
        from ..pipeline_multi import build_multi_ticker_dataset
        print("    ✓ build_multi_ticker_dataset imports successfully")

        # Test environment creation
        print("  [5/5] Testing environment import...")
        from ..envs.multi_asset_env import create_single_ticker_env
        print("    ✓ create_single_ticker_env imports successfully")

        print("\n✅ All imports successful! Feature selection module is ready to use.")
        print("\nTo run feature selection:")
        print("  cd StockProphet")
        print("  python -m multiticker_refactor.feature_selection.main --ticker AAPL --stage full")

        return True

    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("\nPlease check:")
        print("  1. You're in the StockProphet directory")
        print("  2. All dependencies are installed (stable-baselines3, sb3-contrib, sklearn, etc.)")
        return False


if __name__ == '__main__':
    test_imports()
