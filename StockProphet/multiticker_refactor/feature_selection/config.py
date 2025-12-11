"""
Configuration for feature selection experiments.
"""

# Statistical selection thresholds
MIN_CORRELATION_WITH_RETURNS = 0.01  # Minimum absolute correlation with future returns
MIN_MUTUAL_INFORMATION = 0.001       # Minimum mutual information score
MIN_FEATURE_IMPORTANCE = 0.001       # Minimum Random Forest importance

# Feature selection strategy - PERFORMANCE DRIVEN
# We test multiple feature counts and select based on best performance
FEATURE_COUNTS_TO_TEST = [30, 25, 20, 15, 12, 10, 8]  # Test these feature counts
INTERMEDIATE_COUNT = 30              # Shortlist size after statistical stage

# RL validation parameters
RL_VALIDATION_TIMESTEPS = 50_000     # Fast validation (increase to 100k for final run)
RL_VALIDATION_SEEDS = 3              # Number of random seeds for averaging
RL_VALIDATION_METRIC = 'sharpe'      # Primary metric: 'sharpe', 'return', 'calmar'

# Performance improvement threshold
MIN_IMPROVEMENT_THRESHOLD = 0.02     # Minimum improvement to keep more features (2%)
                                      # If adding features doesn't improve by 2%, use fewer

# Experiment tracking
RESULTS_DIR = './feature_selection/results'
SAVE_INTERMEDIATE_RESULTS = True

# Backward elimination configuration
BACKWARD_ELIMINATION_CONFIG = {
    'search_timesteps': 25_000,      # Fast iterations (7 min/test)
    'search_seeds': 2,               # Fast iterations
    'final_timesteps': 50_000,       # Final validation (15 min)
    'final_seeds': 3,                # Final validation
    'max_sharpe_drop': 0.05,         # 5% degradation → revert
    'redundancy_threshold': 0.95,    # Auto-remove if r>0.95
    'weak_score_threshold': 0.20,    # Features with composite < 0.20 are "weak"
}
