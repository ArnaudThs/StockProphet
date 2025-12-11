"""
Feature Selection CLI

Two-stage workflow:
1. Statistical screening (fast) - 50 features → 20 features
2. RL validation (slow) - 20 features → 12 final features

Usage:
    cd StockProphet
    python -m multiticker_refactor.feature_selection.main --ticker AAPL --stage statistical
    python -m multiticker_refactor.feature_selection.main --ticker AAPL --stage rl --input results/statistical_results.json
    python -m multiticker_refactor.feature_selection.main --ticker AAPL --stage full
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from .statistical_selector import StatisticalFeatureSelector
from .rl_validator import RLFeatureValidator
from .backward_elimination import run_backward_elimination
from .config import (
    FEATURE_COUNTS_TO_TEST, INTERMEDIATE_COUNT,
    RL_VALIDATION_TIMESTEPS, RL_VALIDATION_SEEDS,
    RL_VALIDATION_METRIC, MIN_IMPROVEMENT_THRESHOLD,
    RESULTS_DIR, BACKWARD_ELIMINATION_CONFIG
)
from ..pipeline_multi import build_multi_ticker_dataset


def run_statistical_stage(ticker: str, output_dir: str):
    """
    Stage 1: Fast statistical screening.

    Reduces ~50 features → 20 features using correlation, MI, RF importance.
    Takes ~5 minutes.
    """
    print("\n" + "="*70)
    print("STAGE 1: STATISTICAL FEATURE SCREENING")
    print("="*70)

    # Build dataset for single ticker
    print(f"\nBuilding dataset for {ticker}...")
    df, metadata = build_multi_ticker_dataset(
        tickers=[ticker],
        start_date="2020-01-01",
        end_date="2025-06-30",
        include_rnn=True,
        include_sentiment=True,  # Include sentiment in feature selection
        probabilistic_rnn=True,
        verbose=True
    )

    # Drop target column
    target_col = f"{ticker}_Close"
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found!")

    # Initialize selector
    selector = StatisticalFeatureSelector(
        df=df,
        target_col=target_col,
        train_ratio=0.8,
        verbose=True
    )

    # Compute scores
    scores_df = selector.compute_all_scores()

    # Select top features
    top_features = selector.select_top_features(n=INTERMEDIATE_COUNT)

    # Save results
    output_path = Path(output_dir) / "statistical_results.json"
    selector.save_results(str(output_path))

    # Plot
    selector.plot_feature_scores()

    print(f"\n✅ Statistical screening complete!")
    print(f"   Selected {len(top_features)} features")
    print(f"   Results saved to: {output_path}")

    return top_features, scores_df


def run_rl_stage(ticker: str, input_path: str, output_dir: str, timesteps: int = None, n_seeds: int = None):
    """
    Stage 2: RL validation with RecurrentPPO.

    Tests multiple feature counts and selects based on best PERFORMANCE.
    Uses diminishing returns principle - stops when more features don't help.
    """
    # Use provided values or defaults from config
    timesteps = timesteps or RL_VALIDATION_TIMESTEPS
    n_seeds = n_seeds or RL_VALIDATION_SEEDS

    print("\n" + "="*70)
    print("STAGE 2: RL VALIDATION WITH RECURRENTPPO")
    print("="*70)
    print(f"Strategy: Test multiple feature counts, select best {RL_VALIDATION_METRIC}")

    # Load statistical results
    with open(input_path, 'r') as f:
        statistical_results = json.load(f)

    top_features = statistical_results[f'top_{INTERMEDIATE_COUNT}']
    print(f"\nLoaded {len(top_features)} features from statistical stage")

    # Build dataset
    print(f"\nBuilding dataset for {ticker}...")
    df, metadata = build_multi_ticker_dataset(
        tickers=[ticker],
        start_date="2020-01-01",
        end_date="2025-06-30",
        include_rnn=True,
        include_sentiment=True,  # Include sentiment in feature selection
        probabilistic_rnn=True,
        verbose=True
    )

    target_col = f"{ticker}_Close"
    prices = df[target_col].values

    # Remove target from features
    feature_df = df.drop(columns=[target_col])

    # Initialize RL validator
    validator = RLFeatureValidator(
        prices=prices,
        feature_df=feature_df,
        ticker=ticker,
        timesteps=timesteps,
        n_seeds=n_seeds,
        verbose=True
    )

    # Test multiple feature counts - from large to small
    feature_sets = {}
    for count in FEATURE_COUNTS_TO_TEST:
        if count <= len(top_features):
            feature_sets[f'top_{count}'] = top_features[:count]

    print(f"\nTesting {len(feature_sets)} different feature counts:")
    print(f"  {list(feature_sets.keys())}")

    results_df = validator.compare_feature_sets(feature_sets)

    # Save results
    output_path = Path(output_dir) / "rl_validation_results.json"
    validator.save_results(str(output_path))

    # Select best configuration based on metric
    metric_col = f'{RL_VALIDATION_METRIC}_mean'
    best_idx = results_df[metric_col].idxmax()
    best_config = results_df.iloc[best_idx]
    best_features = best_config['features']

    # Check for diminishing returns
    print(f"\n{'='*70}")
    print("DIMINISHING RETURNS ANALYSIS")
    print(f"{'='*70}")

    # Sort by feature count (descending)
    results_sorted = results_df.sort_values('n_features', ascending=False)

    for i in range(len(results_sorted) - 1):
        current = results_sorted.iloc[i]
        next_row = results_sorted.iloc[i + 1]

        improvement = (current[metric_col] - next_row[metric_col]) / abs(next_row[metric_col] + 1e-8)

        print(f"  {current['n_features']} → {next_row['n_features']} features: "
              f"{current[metric_col]:.3f} → {next_row[metric_col]:.3f} "
              f"(change: {improvement*100:+.1f}%)")

        if improvement < -MIN_IMPROVEMENT_THRESHOLD:
            print(f"    ⚠️  Removing features IMPROVED performance by {abs(improvement)*100:.1f}%")

    print(f"\n✅ RL validation complete!")
    print(f"   Best configuration: {best_config['experiment']}")
    print(f"   Features: {len(best_features)}")
    print(f"   {RL_VALIDATION_METRIC.capitalize()}: {best_config[metric_col]:.3f} ± {best_config[f'{RL_VALIDATION_METRIC}_std']:.3f}")
    print(f"   Return: {best_config['test_return_mean']:+.2f}% ± {best_config['test_return_std']:.2f}%")

    # Save final selected features
    final_output = Path(output_dir) / "selected_features.json"
    with open(final_output, 'w') as f:
        json.dump({
            'ticker': ticker,
            'selection_metric': RL_VALIDATION_METRIC,
            'n_features': len(best_features),
            'features': best_features,
            'metrics': {
                'sharpe_mean': float(best_config['sharpe_mean']),
                'sharpe_std': float(best_config['sharpe_std']),
                'return_mean': float(best_config['test_return_mean']),
                'return_std': float(best_config['test_return_std']),
                'max_dd_mean': float(best_config['max_dd_mean'])
            },
            'all_results': results_df.to_dict(orient='records')
        }, f, indent=2)

    print(f"   Final features saved to: {final_output}")

    return best_features, results_df


def run_full_pipeline(ticker: str, output_dir: str, timesteps: int = None, n_seeds: int = None):
    """
    Run both stages sequentially.
    """
    print("\n" + "="*70)
    print("RUNNING FULL FEATURE SELECTION PIPELINE")
    print("="*70)
    print(f"Ticker: {ticker}")
    print(f"Output directory: {output_dir}")

    # Stage 1: Statistical
    top_features, scores_df = run_statistical_stage(ticker, output_dir)

    # Stage 2: RL validation
    statistical_results_path = Path(output_dir) / "statistical_results.json"
    final_features, results_df = run_rl_stage(ticker, str(statistical_results_path), output_dir, timesteps, n_seeds)

    print("\n" + "="*70)
    print("FEATURE SELECTION COMPLETE")
    print("="*70)
    print(f"\nFinal selected features ({len(final_features)}):")
    for i, feat in enumerate(final_features, 1):
        print(f"  {i:2d}. {feat}")

    print(f"\nAll results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Feature Selection for StockProphet")
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol')
    parser.add_argument('--stage', type=str,
                        choices=['statistical', 'rl', 'elimination', 'full'],
                        default='full',
                        help='Which stage to run')
    parser.add_argument('--input', type=str, help='Input path for RL/elimination stage (statistical_results.json)')
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR, help='Output directory')
    parser.add_argument('--timesteps', type=int, default=RL_VALIDATION_TIMESTEPS,
                        help='Training timesteps for RL validation')
    parser.add_argument('--seeds', type=int, default=RL_VALIDATION_SEEDS,
                        help='Number of random seeds for RL validation')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run requested stage
    if args.stage == 'statistical':
        run_statistical_stage(args.ticker, str(output_dir))
    elif args.stage == 'rl':
        if not args.input:
            raise ValueError("--input required for RL stage (path to statistical_results.json)")
        run_rl_stage(args.ticker, args.input, str(output_dir), args.timesteps, args.seeds)
    elif args.stage == 'elimination':
        # Run backward elimination (requires statistical results)
        if not args.input:
            # Try default path
            args.input = str(output_dir / "statistical_results.json")
            if not Path(args.input).exists():
                raise ValueError("--input required for elimination stage (path to statistical_results.json)")

        # Build config from args
        config = BACKWARD_ELIMINATION_CONFIG.copy()
        config['search_timesteps'] = args.timesteps if args.timesteps != RL_VALIDATION_TIMESTEPS else config['search_timesteps']
        config['search_seeds'] = args.seeds if args.seeds != RL_VALIDATION_SEEDS else config['search_seeds']

        run_backward_elimination(
            ticker=args.ticker,
            statistical_results_path=args.input,
            output_dir=str(output_dir),
            config=config
        )
    elif args.stage == 'full':
        run_full_pipeline(args.ticker, str(output_dir), args.timesteps, args.seeds)


if __name__ == '__main__':
    main()
