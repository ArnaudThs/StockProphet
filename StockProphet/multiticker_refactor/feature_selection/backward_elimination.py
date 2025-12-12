"""
Stratified Backward Elimination for Feature Selection

Efficiently removes features that hurt or don't improve PPO trading performance.

Three-phase approach:
1. Remove redundant pairs (r>0.95) without testing - saves time
2. Test batch removal of weak features (composite <0.20)
3. Greedy elimination with early stopping (Sharpe drop <5%)
4. Final validation with full parameters (50k timesteps, 3 seeds)

Time budget: ~90 minutes on laptop
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

from .config import BACKWARD_ELIMINATION_CONFIG, RESULTS_DIR
from .rl_validator import RLFeatureValidator
from ..pipeline import build_multi_ticker_dataset


def load_statistical_results(input_path: str) -> Tuple[List[str], pd.DataFrame]:
    """
    Load statistical screening results.

    Returns:
        features: List of feature names (ranked by composite score)
        scores_df: DataFrame with all statistical scores
    """
    with open(input_path, 'r') as f:
        results = json.load(f)

    # Get ranked feature list
    features = results['top_30']

    # Convert scores to DataFrame
    scores_df = pd.DataFrame(results['scores'])

    return features, scores_df


def phase1_remove_redundant_pairs(
    features: List[str],
    scores_df: pd.DataFrame,
    redundancy_threshold: float = 0.95
) -> Tuple[List[str], List[str]]:
    """
    Phase 1: Remove redundant feature pairs without testing.

    For pairs with redundancy > threshold, keep the higher-ranked feature.

    Returns:
        remaining_features: Features after removal
        removed_features: List of removed features
    """
    print("\n" + "="*70)
    print("PHASE 1: REMOVE REDUNDANT PAIRS")
    print("="*70)
    print(f"Redundancy threshold: {redundancy_threshold}")

    # Build feature rank map
    feature_ranks = {feat: i for i, feat in enumerate(features)}

    # Create scores lookup
    scores_dict = scores_df.set_index('feature').to_dict('index')

    # Identify redundant pairs
    redundant_pairs = []
    for feat1 in features:
        if feat1 not in scores_dict:
            continue
        redundancy = scores_dict[feat1]['redundancy']

        if redundancy > redundancy_threshold:
            # Find which feature it's redundant with
            # (In practice, this would require pairwise correlation matrix)
            # For now, we'll use known pairs from statistical analysis
            pass

    # Manually specify known redundant pairs (from statistical results)
    known_pairs = [
        ('AAPL_rnn_sigma_1d', 'AAPL_rnn_confidence_1d'),   # r=1.0
        ('AAPL_rnn_sigma_5d', 'AAPL_rnn_confidence_5d'),   # r=1.0
        ('AAPL_Open', 'AAPL_Low'),                         # r=0.999
        ('days_to_cpi', 'days_since_cpi'),                 # r=0.77 (borderline)
    ]

    removed = []
    for feat1, feat2 in known_pairs:
        if feat1 in features and feat2 in features:
            # Remove lower-ranked feature
            rank1 = feature_ranks.get(feat1, 999)
            rank2 = feature_ranks.get(feat2, 999)

            to_remove = feat2 if rank1 < rank2 else feat1
            removed.append(to_remove)
            print(f"  ❌ Removing {to_remove} (redundant with {feat1 if to_remove==feat2 else feat2})")

    remaining = [f for f in features if f not in removed]

    print(f"\n✅ Phase 1 complete:")
    print(f"   Removed: {len(removed)} features")
    print(f"   Remaining: {len(remaining)} features")
    print(f"   Time saved: 0 min (no testing required)")

    return remaining, removed


def phase2_test_weak_removal(
    features: List[str],
    scores_df: pd.DataFrame,
    weak_threshold: float,
    ticker: str,
    prices: np.ndarray,
    feature_df: pd.DataFrame,
    timesteps: int,
    n_seeds: int
) -> Tuple[List[str], Dict]:
    """
    Phase 2: Test batch removal of weak features.

    Tests baseline vs removing all features with composite < threshold.

    Returns:
        remaining_features: Features after removal (if helpful)
        phase_results: Dict with metrics
    """
    print("\n" + "="*70)
    print("PHASE 2: TEST WEAK FEATURE REMOVAL")
    print("="*70)
    print(f"Weak feature threshold: {weak_threshold}")

    # Identify weak features
    scores_dict = scores_df.set_index('feature').to_dict('index')
    weak_features = [
        f for f in features
        if f in scores_dict and scores_dict[f]['composite_score'] < weak_threshold
    ]

    print(f"\nIdentified {len(weak_features)} weak features:")
    for feat in weak_features:
        score = scores_dict[feat]['composite_score']
        print(f"  - {feat}: {score:.3f}")

    if not weak_features:
        print("\n✅ No weak features found, skipping Phase 2")
        return features, {'skipped': True}

    # Test 1: Baseline with all features
    print(f"\n[Test 1/2] Baseline with {len(features)} features...")
    start_time = time.time()

    validator = RLFeatureValidator(
        prices=prices,
        feature_df=feature_df,
        ticker=ticker,
        timesteps=timesteps,
        n_seeds=n_seeds,
        verbose=False
    )

    baseline_results = validator.compare_feature_sets({
        'baseline': features
    })

    baseline_sharpe = baseline_results.iloc[0]['sharpe_mean']
    baseline_time = time.time() - start_time

    print(f"   Sharpe: {baseline_sharpe:.3f}")
    print(f"   Time: {baseline_time/60:.1f} min")

    # Test 2: Remove weak features
    features_without_weak = [f for f in features if f not in weak_features]
    print(f"\n[Test 2/2] Without weak features ({len(features_without_weak)} features)...")
    start_time = time.time()

    test_results = validator.compare_feature_sets({
        'without_weak': features_without_weak
    })

    test_sharpe = test_results.iloc[0]['sharpe_mean']
    test_time = time.time() - start_time

    print(f"   Sharpe: {test_sharpe:.3f}")
    print(f"   Time: {test_time/60:.1f} min")

    # Decision
    sharpe_change = (test_sharpe - baseline_sharpe) / (abs(baseline_sharpe) + 1e-8)

    print(f"\n{'='*70}")
    print(f"Sharpe change: {sharpe_change*100:+.1f}%")

    if sharpe_change >= -0.05:  # If not worse than 5% drop
        print(f"✅ Removing weak features (Sharpe change within tolerance)")
        remaining = features_without_weak
        decision = 'remove'
    else:
        print(f"❌ Keeping weak features (removal hurts performance)")
        remaining = features
        decision = 'keep'

    phase_results = {
        'baseline_sharpe': float(baseline_sharpe),
        'test_sharpe': float(test_sharpe),
        'sharpe_change_pct': float(sharpe_change * 100),
        'weak_features': weak_features,
        'decision': decision,
        'remaining': len(remaining),
        'time_min': (baseline_time + test_time) / 60
    }

    print(f"\n✅ Phase 2 complete:")
    print(f"   Decision: {decision}")
    print(f"   Remaining: {len(remaining)} features")
    print(f"   Time: {phase_results['time_min']:.1f} min")

    return remaining, phase_results


def calculate_marginal_score(feature: str, scores_dict: Dict) -> float:
    """
    Calculate marginal contribution score for a feature.

    Lower score = more likely to be removed.
    """
    if feature not in scores_dict:
        return 0.0

    scores = scores_dict[feature]

    # Normalize components to [0, 1]
    corr = abs(scores['corr_with_returns'])
    mi = scores['mutual_info']
    rf = scores['rf_importance']
    redundancy = scores['redundancy']

    # Weighted combination (penalize high redundancy)
    marginal = (
        0.3 * corr +
        0.3 * mi +
        0.2 * rf +
        0.2 * (1 - redundancy)
    )

    return marginal


def phase3_greedy_elimination(
    features: List[str],
    scores_df: pd.DataFrame,
    ticker: str,
    prices: np.ndarray,
    feature_df: pd.DataFrame,
    timesteps: int,
    n_seeds: int,
    max_sharpe_drop: float = 0.05
) -> Tuple[List[str], List[Dict]]:
    """
    Phase 3: Greedy backward elimination with early stopping.

    Iteratively removes worst feature until all remaining are essential.

    Returns:
        final_features: Remaining features
        elimination_history: List of iteration results
    """
    print("\n" + "="*70)
    print("PHASE 3: GREEDY BACKWARD ELIMINATION")
    print("="*70)
    print(f"Max Sharpe drop tolerance: {max_sharpe_drop*100}%")

    scores_dict = scores_df.set_index('feature').to_dict('index')
    current_features = features.copy()
    essential_features = set()  # Features marked as essential (can't remove)
    history = []

    validator = RLFeatureValidator(
        prices=prices,
        feature_df=feature_df,
        ticker=ticker,
        timesteps=timesteps,
        n_seeds=n_seeds,
        verbose=False
    )

    # Get baseline Sharpe
    print(f"\nBaseline with {len(current_features)} features...")
    baseline_results = validator.compare_feature_sets({
        'baseline': current_features
    })
    current_sharpe = baseline_results.iloc[0]['sharpe_mean']
    print(f"   Sharpe: {current_sharpe:.3f}")

    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")

        # Find removable features (not essential)
        removable = [f for f in current_features if f not in essential_features]

        if len(removable) == 0:
            print("✅ All remaining features are essential, stopping")
            break

        if len(removable) <= 5:
            print(f"⚠️  Only {len(removable)} removable features left, stopping")
            break

        # Calculate marginal scores for removable features
        marginal_scores = {
            f: calculate_marginal_score(f, scores_dict)
            for f in removable
        }

        # Find feature with lowest marginal contribution
        worst_feature = min(marginal_scores, key=marginal_scores.get)
        worst_score = marginal_scores[worst_feature]

        print(f"\nTesting removal of: {worst_feature}")
        print(f"   Marginal score: {worst_score:.3f}")

        # Test without worst feature
        test_features = [f for f in current_features if f != worst_feature]

        print(f"   Testing with {len(test_features)} features...")
        start_time = time.time()

        test_results = validator.compare_feature_sets({
            'test': test_features
        })

        test_sharpe = test_results.iloc[0]['sharpe_mean']
        test_time = time.time() - start_time

        sharpe_drop = (current_sharpe - test_sharpe) / (abs(current_sharpe) + 1e-8)

        print(f"   Sharpe: {test_sharpe:.3f} (drop: {sharpe_drop*100:+.1f}%)")
        print(f"   Time: {test_time/60:.1f} min")

        # Decision
        if sharpe_drop <= max_sharpe_drop:
            print(f"   ✅ REMOVE (drop within {max_sharpe_drop*100}% tolerance)")
            current_features = test_features
            current_sharpe = test_sharpe
            decision = 'remove'
        else:
            print(f"   ❌ KEEP (drop exceeds {max_sharpe_drop*100}% tolerance)")
            essential_features.add(worst_feature)
            decision = 'keep_essential'

        # Record iteration
        history.append({
            'iteration': iteration,
            'tested_feature': worst_feature,
            'marginal_score': float(worst_score),
            'sharpe_before': float(current_sharpe) if decision == 'remove' else float(test_sharpe),
            'sharpe_after': float(test_sharpe) if decision == 'remove' else float(current_sharpe),
            'sharpe_drop_pct': float(sharpe_drop * 100),
            'decision': decision,
            'remaining': len(current_features),
            'time_min': test_time / 60
        })

    print(f"\n{'='*70}")
    print(f"✅ Phase 3 complete:")
    print(f"   Iterations: {iteration}")
    print(f"   Final features: {len(current_features)}")
    print(f"   Essential features: {len(essential_features)}")
    total_time = sum(h['time_min'] for h in history)
    print(f"   Total time: {total_time:.1f} min")

    return current_features, history


def phase4_final_validation(
    features: List[str],
    ticker: str,
    prices: np.ndarray,
    feature_df: pd.DataFrame,
    timesteps: int,
    n_seeds: int
) -> Dict:
    """
    Phase 4: Final validation with full parameters.

    Tests final feature set with higher timesteps and more seeds.

    Returns:
        validation_results: Dict with final metrics
    """
    print("\n" + "="*70)
    print("PHASE 4: FINAL VALIDATION")
    print("="*70)
    print(f"Features: {len(features)}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Seeds: {n_seeds}")

    start_time = time.time()

    validator = RLFeatureValidator(
        prices=prices,
        feature_df=feature_df,
        ticker=ticker,
        timesteps=timesteps,
        n_seeds=n_seeds,
        verbose=True
    )

    results = validator.compare_feature_sets({
        'final': features
    })

    validation_time = time.time() - start_time

    row = results.iloc[0]

    validation_results = {
        'final_features': features,
        'n_features': len(features),
        'sharpe_mean': float(row['sharpe_mean']),
        'sharpe_std': float(row['sharpe_std']),
        'test_return_mean': float(row['test_return_mean']),
        'test_return_std': float(row['test_return_std']),
        'max_dd_mean': float(row['max_dd_mean']),
        'time_min': validation_time / 60
    }

    print(f"\n{'='*70}")
    print(f"✅ Phase 4 complete:")
    print(f"   Sharpe: {validation_results['sharpe_mean']:.3f} ± {validation_results['sharpe_std']:.3f}")
    print(f"   Return: {validation_results['test_return_mean']:+.2f}% ± {validation_results['test_return_std']:.2f}%")
    print(f"   Max DD: {validation_results['max_dd_mean']:.2f}%")
    print(f"   Time: {validation_results['time_min']:.1f} min")

    return validation_results


def run_backward_elimination(
    ticker: str,
    statistical_results_path: str,
    output_dir: str,
    config: Dict = None
) -> Tuple[List[str], Dict]:
    """
    Run complete stratified backward elimination.

    Returns:
        final_features: Selected feature list
        full_history: Dict with all phase results
    """
    if config is None:
        config = BACKWARD_ELIMINATION_CONFIG

    print("\n" + "="*70)
    print("STRATIFIED BACKWARD ELIMINATION")
    print("="*70)
    print(f"Ticker: {ticker}")
    print(f"Search: {config['search_timesteps']:,} timesteps, {config['search_seeds']} seeds")
    print(f"Final: {config['final_timesteps']:,} timesteps, {config['final_seeds']} seeds")

    # Load statistical results
    features, scores_df = load_statistical_results(statistical_results_path)

    print(f"\nStarting with {len(features)} features from statistical stage")

    # Build dataset once (shared across all phases)
    print("\nBuilding dataset...")
    df, metadata = build_multi_ticker_dataset(
        tickers=[ticker],
        start_date="2020-01-01",
        end_date="2025-06-30",
        include_rnn=True,
        include_sentiment=True,
        probabilistic_rnn=True,
        verbose=False
    )

    target_col = f"{ticker}_Close"
    prices = df[target_col].values
    feature_df = df.drop(columns=[target_col])

    # Phase 1: Remove redundant pairs
    features, removed_redundant = phase1_remove_redundant_pairs(
        features,
        scores_df,
        config['redundancy_threshold']
    )

    # Phase 2: Test weak removal
    features, phase2_results = phase2_test_weak_removal(
        features,
        scores_df,
        config['weak_score_threshold'],
        ticker,
        prices,
        feature_df,
        config['search_timesteps'],
        config['search_seeds']
    )

    # Phase 3: Greedy elimination
    features, phase3_history = phase3_greedy_elimination(
        features,
        scores_df,
        ticker,
        prices,
        feature_df,
        config['search_timesteps'],
        config['search_seeds'],
        config['max_sharpe_drop']
    )

    # Phase 4: Final validation
    final_results = phase4_final_validation(
        features,
        ticker,
        prices,
        feature_df,
        config['final_timesteps'],
        config['final_seeds']
    )

    # Compile full history
    full_history = {
        'ticker': ticker,
        'config': config,
        'phase1_redundancy_removal': {
            'removed': removed_redundant,
            'remaining': len(features) + len(removed_redundant if not isinstance(phase2_results.get('skipped'), bool) else [])
        },
        'phase2_weak_removal': phase2_results,
        'phase3_greedy_elimination': phase3_history,
        'phase4_final_validation': final_results
    }

    # Save results
    output_path = Path(output_dir) / "elimination_history.json"
    with open(output_path, 'w') as f:
        json.dump(full_history, f, indent=2)

    final_features_path = Path(output_dir) / "selected_features.json"
    with open(final_features_path, 'w') as f:
        json.dump({
            'ticker': ticker,
            'n_features': len(features),
            'features': features,
            'metrics': final_results
        }, f, indent=2)

    print(f"\n{'='*70}")
    print("BACKWARD ELIMINATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nFinal selected features ({len(features)}):")
    for i, feat in enumerate(features, 1):
        print(f"  {i:2d}. {feat}")

    print(f"\nResults saved to:")
    print(f"  - {output_path}")
    print(f"  - {final_features_path}")

    return features, full_history


def main():
    parser = argparse.ArgumentParser(description="Backward Elimination for Feature Selection")
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to statistical_results.json')
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR,
                        help='Output directory')
    parser.add_argument('--search-timesteps', type=int,
                        default=BACKWARD_ELIMINATION_CONFIG['search_timesteps'],
                        help='Timesteps for search iterations')
    parser.add_argument('--search-seeds', type=int,
                        default=BACKWARD_ELIMINATION_CONFIG['search_seeds'],
                        help='Seeds for search iterations')
    parser.add_argument('--final-timesteps', type=int,
                        default=BACKWARD_ELIMINATION_CONFIG['final_timesteps'],
                        help='Timesteps for final validation')
    parser.add_argument('--final-seeds', type=int,
                        default=BACKWARD_ELIMINATION_CONFIG['final_seeds'],
                        help='Seeds for final validation')

    args = parser.parse_args()

    # Build config from args
    config = {
        'search_timesteps': args.search_timesteps,
        'search_seeds': args.search_seeds,
        'final_timesteps': args.final_timesteps,
        'final_seeds': args.final_seeds,
        'max_sharpe_drop': BACKWARD_ELIMINATION_CONFIG['max_sharpe_drop'],
        'redundancy_threshold': BACKWARD_ELIMINATION_CONFIG['redundancy_threshold'],
        'weak_score_threshold': BACKWARD_ELIMINATION_CONFIG['weak_score_threshold'],
    }

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run backward elimination
    run_backward_elimination(
        ticker=args.ticker,
        statistical_results_path=args.input,
        output_dir=str(output_dir),
        config=config
    )


if __name__ == '__main__':
    main()
