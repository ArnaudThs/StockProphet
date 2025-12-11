"""
Statistical Feature Selector - Stage 1

Fast feature screening using:
1. Correlation with future returns
2. Mutual information with price direction
3. Random Forest feature importance
4. Variance/redundancy checks

Goal: Reduce ~50 features → ~20 features in <5 minutes
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import json
from pathlib import Path


class StatisticalFeatureSelector:
    """
    Fast statistical feature selection without RL training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "target_close",
        train_ratio: float = 0.8,
        verbose: bool = True
    ):
        """
        Initialize selector.

        Args:
            df: DataFrame with all features
            target_col: Name of price column
            train_ratio: Fraction for training (use rest for validation)
            verbose: Print progress
        """
        self.df = df.copy()
        self.target_col = target_col
        self.train_ratio = train_ratio
        self.verbose = verbose

        # Extract features and prices
        self.feature_cols = list(df.columns.drop(target_col))
        self.prices = df[target_col].values

        # Train/test split
        split_idx = int(len(df) * train_ratio)
        self.train_df = df.iloc[:split_idx]
        self.test_df = df.iloc[split_idx:]

        # Results storage
        self.scores = {}
        self.rankings = {}

        if self.verbose:
            print(f"Initialized StatisticalFeatureSelector")
            print(f"  Total features: {len(self.feature_cols)}")
            print(f"  Train samples: {len(self.train_df)}")
            print(f"  Test samples: {len(self.test_df)}")

    def compute_all_scores(self) -> pd.DataFrame:
        """
        Compute all statistical scores for features.

        Returns:
            DataFrame with feature scores
        """
        if self.verbose:
            print("\n" + "="*60)
            print("COMPUTING STATISTICAL SCORES")
            print("="*60)

        # 1. Correlation with future returns
        if self.verbose:
            print("\n[1/5] Computing correlation with future returns...")
        corr_scores = self._compute_return_correlation()

        # 2. Mutual information with price direction
        if self.verbose:
            print("[2/5] Computing mutual information...")
        mi_scores = self._compute_mutual_information()

        # 3. Random Forest importance
        if self.verbose:
            print("[3/5] Computing Random Forest importance...")
        rf_scores = self._compute_rf_importance()

        # 4. Variance analysis
        if self.verbose:
            print("[4/5] Computing variance scores...")
        var_scores = self._compute_variance_scores()

        # 5. Redundancy check
        if self.verbose:
            print("[5/5] Checking feature redundancy...")
        redundancy_scores = self._compute_redundancy()

        # Combine into DataFrame
        results = pd.DataFrame({
            'feature': self.feature_cols,
            'corr_with_returns': [corr_scores.get(f, 0) for f in self.feature_cols],
            'mutual_info': [mi_scores.get(f, 0) for f in self.feature_cols],
            'rf_importance': [rf_scores.get(f, 0) for f in self.feature_cols],
            'variance': [var_scores.get(f, 0) for f in self.feature_cols],
            'redundancy': [redundancy_scores.get(f, 0) for f in self.feature_cols],
        })

        # Compute composite score (weighted average)
        results['composite_score'] = (
            0.3 * results['corr_with_returns'] / (results['corr_with_returns'].max() + 1e-8) +
            0.3 * results['mutual_info'] / (results['mutual_info'].max() + 1e-8) +
            0.3 * results['rf_importance'] / (results['rf_importance'].max() + 1e-8) +
            0.1 * results['variance'] / (results['variance'].max() + 1e-8) -
            0.2 * results['redundancy']  # Penalize redundant features
        )

        # Sort by composite score
        results = results.sort_values('composite_score', ascending=False).reset_index(drop=True)

        self.scores = results

        if self.verbose:
            print("\n" + "="*60)
            print("TOP 15 FEATURES BY COMPOSITE SCORE")
            print("="*60)
            print(results.head(15).to_string(index=False))

        return results

    def _compute_return_correlation(self) -> Dict[str, float]:
        """Correlation of features with future returns."""
        # Compute 1-day forward returns
        returns = np.diff(self.prices) / self.prices[:-1]

        scores = {}
        for col in self.feature_cols:
            feature_vals = self.df[col].values[:-1]  # Align with returns

            # Remove NaN
            mask = ~(np.isnan(feature_vals) | np.isnan(returns))
            if mask.sum() < 10:
                scores[col] = 0.0
                continue

            # Spearman correlation (robust to outliers)
            corr, _ = spearmanr(feature_vals[mask], returns[mask])
            scores[col] = abs(corr) if not np.isnan(corr) else 0.0

        return scores

    def _compute_mutual_information(self) -> Dict[str, float]:
        """Mutual information with next-day price direction."""
        # Binary target: next day up/down
        y = (np.diff(self.prices) > 0).astype(int)
        X = self.df[self.feature_cols].values[:-1]

        # Handle NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        if len(X) < 10:
            return {col: 0.0 for col in self.feature_cols}

        # Compute MI
        mi_scores = mutual_info_classif(X, y, random_state=42)

        return {col: score for col, score in zip(self.feature_cols, mi_scores)}

    def _compute_rf_importance(self) -> Dict[str, float]:
        """Random Forest feature importance."""
        # Binary target
        y = (np.diff(self.prices) > 0).astype(int)
        X = self.df[self.feature_cols].values[:-1]

        # Handle NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        if len(X) < 10:
            return {col: 0.0 for col in self.feature_cols}

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)

        return {col: imp for col, imp in zip(self.feature_cols, rf.feature_importances_)}

    def _compute_variance_scores(self) -> Dict[str, float]:
        """Variance of features (low variance = not informative)."""
        scores = {}
        for col in self.feature_cols:
            vals = self.df[col].values
            vals = vals[~np.isnan(vals)]
            scores[col] = np.std(vals) if len(vals) > 0 else 0.0
        return scores

    def _compute_redundancy(self) -> Dict[str, float]:
        """
        Measure redundancy - if feature is highly correlated with others,
        it's redundant.
        """
        # Compute correlation matrix
        df_clean = self.df[self.feature_cols].dropna()
        if len(df_clean) < 10:
            return {col: 0.0 for col in self.feature_cols}

        corr_matrix = df_clean.corr(method='spearman').abs()

        # For each feature, compute max correlation with other features
        redundancy = {}
        for col in self.feature_cols:
            if col not in corr_matrix.columns:
                redundancy[col] = 0.0
                continue

            # Exclude self-correlation
            other_corrs = corr_matrix[col].drop(col, errors='ignore')
            redundancy[col] = other_corrs.max() if len(other_corrs) > 0 else 0.0

        return redundancy

    def select_top_features(self, n: int = 20) -> List[str]:
        """
        Select top N features based on composite score.

        Args:
            n: Number of features to select

        Returns:
            List of selected feature names
        """
        if self.scores is None or len(self.scores) == 0:
            raise ValueError("Must call compute_all_scores() first!")

        selected = self.scores.head(n)['feature'].tolist()

        if self.verbose:
            print(f"\n" + "="*60)
            print(f"SELECTED TOP {n} FEATURES")
            print("="*60)
            for i, feat in enumerate(selected, 1):
                score = self.scores[self.scores['feature'] == feat]['composite_score'].values[0]
                print(f"  {i:2d}. {feat:<40} (score: {score:.4f})")

        return selected

    def save_results(self, output_path: str):
        """Save selection results to JSON."""
        if self.scores is None:
            raise ValueError("Must call compute_all_scores() first!")

        results = {
            'n_features_original': len(self.feature_cols),
            'scores': self.scores.to_dict(orient='records'),
            'top_30': self.select_top_features(30),
            'top_20': self.select_top_features(20),
            'top_15': self.select_top_features(15),
            'config': {
                'train_ratio': self.train_ratio,
                'target_col': self.target_col
            }
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"\nResults saved to: {output_path}")

    def plot_feature_scores(self):
        """Visualize feature scores."""
        if self.scores is None:
            raise ValueError("Must call compute_all_scores() first!")

        import matplotlib.pyplot as plt

        # Plot top 20 features
        top_20 = self.scores.head(20).copy()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Composite score
        ax = axes[0, 0]
        ax.barh(range(len(top_20)), top_20['composite_score'], color='steelblue')
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels(top_20['feature'], fontsize=8)
        ax.set_xlabel('Composite Score')
        ax.set_title('Top 20 Features - Composite Score')
        ax.grid(alpha=0.3, axis='x')
        ax.invert_yaxis()

        # Plot 2: Individual scores (heatmap)
        ax = axes[0, 1]
        score_cols = ['corr_with_returns', 'mutual_info', 'rf_importance', 'variance']
        score_matrix = top_20[score_cols].values.T

        # Normalize each row
        score_matrix = score_matrix / (score_matrix.max(axis=1, keepdims=True) + 1e-8)

        im = ax.imshow(score_matrix, aspect='auto', cmap='YlOrRd')
        ax.set_yticks(range(len(score_cols)))
        ax.set_yticklabels(score_cols, fontsize=8)
        ax.set_xticks(range(len(top_20)))
        ax.set_xticklabels(top_20['feature'], rotation=90, fontsize=6)
        ax.set_title('Score Breakdown (Normalized)')
        plt.colorbar(im, ax=ax)

        # Plot 3: Redundancy distribution
        ax = axes[1, 0]
        ax.hist(self.scores['redundancy'], bins=30, color='coral', alpha=0.7, edgecolor='white')
        ax.axvline(x=0.8, color='red', linestyle='--', label='High redundancy (>0.8)')
        ax.set_xlabel('Redundancy Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Feature Redundancy Distribution')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 4: Score components scatter
        ax = axes[1, 1]
        ax.scatter(top_20['rf_importance'], top_20['mutual_info'],
                   s=100, c=top_20['composite_score'], cmap='viridis', alpha=0.6)
        ax.set_xlabel('RF Importance')
        ax.set_ylabel('Mutual Information')
        ax.set_title('RF vs MI (sized by composite score)')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
