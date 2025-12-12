"""
Technical indicators and calendar/macro features.
Source: External dataprep.py (all technical/calendar functions)
"""
import numpy as np
import pandas as pd
import holidays

from ..config import (
    USE_HMM, FILLER, START_DATE, END_DATE
)

# =============================================================================
# FEATURE REGISTRY
# =============================================================================
feature_registry = {}


def register_feature(name: str, shift: str):
    """Register a feature in the global registry."""
    global feature_registry
    feature_registry[name] = shift


def validate_feature(df: pd.DataFrame, name: str):
    """Validate that a feature exists and has valid data."""
    assert name in df.columns, f"{name} missing from df"
    assert len(df[name]) == len(df.index), f"{name} wrong length"
    assert not df[name].isna().all(), f"{name} all NaN"
    assert not np.isinf(df[name]).any(), f"{name} contains inf"
    assert name in feature_registry, f"{name} missing in registry"


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def add_return(df: pd.DataFrame, ticker: str):
    """Add 1-day return feature."""
    close = f"{ticker}_Close"
    name = f"{ticker}_Return_1d"
    df[name] = df[close].pct_change()
    register_feature(name, "no_shift")
    validate_feature(df, name)


def add_sma(df: pd.DataFrame, ticker: str):
    """Add 10-day Simple Moving Average."""
    close = f"{ticker}_Close"
    name = f"{ticker}_SMA_10"
    df[name] = df[close].rolling(10).mean()
    register_feature(name, "no_shift")
    validate_feature(df, name)


def add_rsi(df: pd.DataFrame, ticker: str):
    """Add 14-day Relative Strength Index."""
    close = f"{ticker}_Close"
    delta = df[close].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down
    name = f"{ticker}_RSI_14"
    df[name] = 100 - (100 / (1 + rs))
    register_feature(name, "no_shift")
    validate_feature(df, name)


def add_stoch(df: pd.DataFrame, ticker: str):
    """Add Stochastic oscillator (K and D)."""
    close = f"{ticker}_Close"
    low14 = df[close].rolling(14).min()
    high14 = df[close].rolling(14).max()

    kname = f"{ticker}_StochK"
    df[kname] = 100 * (df[close] - low14) / (high14 - low14)
    register_feature(kname, "no_shift")
    validate_feature(df, kname)

    dname = f"{ticker}_StochD"
    df[dname] = df[kname].rolling(3).mean()
    register_feature(dname, "no_shift")
    validate_feature(df, dname)


def add_entropy(df: pd.DataFrame, ticker: str):
    """Add 20-day entropy (volatility proxy)."""
    ret = f"{ticker}_Return_1d"
    name = f"{ticker}_Entropy_20"
    df[name] = (df[ret] ** 2).rolling(20).sum()
    register_feature(name, "no_shift")
    validate_feature(df, name)


def add_all_technicals(df: pd.DataFrame, tickers: list, use_hmm: bool = USE_HMM) -> pd.DataFrame:
    """Add all technical indicators for given tickers."""
    for t in tickers:
        add_return(df, t)
        add_sma(df, t)
        add_rsi(df, t)
        add_stoch(df, t)
        add_entropy(df, t)
    return df


# =============================================================================
# CALENDAR & MACRO FEATURES
# =============================================================================

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features (day of week, month, etc.)."""
    idx = df.index

    cols = {
        "day_of_week": idx.dayofweek,
        "day_of_month": idx.day,
        "month": idx.month,
        "quarter": idx.quarter
    }

    for name, series in cols.items():
        df[name] = series
        register_feature(name, "no_shift")
        validate_feature(df, name)

    return df


def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Add holiday adjacency indicator."""
    us = holidays.US()
    df["is_holiday_adjacent"] = [
        int((d + pd.Timedelta(days=1) in us) or (d - pd.Timedelta(days=1) in us))
        for d in df.index
    ]
    register_feature("is_holiday_adjacent", "no_shift")
    validate_feature(df, "is_holiday_adjacent")
    return df


def add_macro_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add distances to/from macro events (CPI, NFP).
    Dynamically expands events across all years in the data.
    """
    start = df.index.min()
    end = df.index.max()

    years = range(start.year - 1, end.year + 2)

    # Event templates (month, day)
    templates = {
        "cpi": [(1, 11), (2, 13), (3, 12), (4, 10), (5, 15), (6, 12),
                (7, 11), (8, 14), (9, 13), (10, 10), (11, 14), (12, 11)],
        "nfp": [(1, 5), (2, 2), (3, 8), (4, 5), (5, 3), (6, 7),
                (7, 5), (8, 2), (9, 6), (10, 4), (11, 1), (12, 6)],
    }

    def expand_dates(md_list):
        out = []
        for y in years:
            for m, d in md_list:
                try:
                    out.append(pd.Timestamp(year=y, month=m, day=d))
                except ValueError:
                    pass  # Invalid date (e.g., Feb 30)
        s = pd.to_datetime(out)
        s = s[(s >= start - pd.Timedelta(days=365)) &
              (s <= end + pd.Timedelta(days=365))]
        return s.sort_values()

    for k, md_list in templates.items():
        ds = expand_dates(md_list)

        # Distance to next event
        def next_days(idx):
            out = []
            j = 0
            for d in idx:
                while j < len(ds) and ds[j] < d:
                    j += 1
                out.append((ds[j] - d).days if j < len(ds) else FILLER)
            return out

        # Distance since previous event
        def prev_days(idx):
            out = []
            j = 0
            for d in idx:
                while j < len(ds) and ds[j] <= d:
                    j += 1
                out.append((d - ds[j - 1]).days if j > 0 else FILLER)
            return out

        n = f"days_to_{k}"
        p = f"days_since_{k}"

        df[n] = next_days(df.index)
        df[p] = prev_days(df.index)

        register_feature(n, "no_shift")
        register_feature(p, "no_shift")
        validate_feature(df, n)
        validate_feature(df, p)

    return df


def add_calendar_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Add all calendar and macro features."""
    df = add_calendar(df)
    df = add_holidays(df)
    df = add_macro_distances(df)
    return df


# =============================================================================
# SHIFT ENGINE
# =============================================================================

def apply_shift_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shift rules from feature registry."""
    df = df.copy()
    out = {}

    for col, rule in feature_registry.items():
        if col not in df.columns:
            continue
        if rule == "no_shift":
            out[col] = df[col]
        elif rule == "shift_1":
            name = f"{col}_t-1"
            out[name] = df[col].shift(1)
        else:
            raise ValueError(f"Unknown rule {rule}")

    new_df = pd.DataFrame(out, index=df.index)
    return new_df


# =============================================================================
# DATE TRIMMING
# =============================================================================

def trim_date_range(df: pd.DataFrame, start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    """Trim DataFrame to specified date range."""
    pos_start = df.index.searchsorted(start_date)
    df = df.iloc[pos_start:].copy()

    pos_end = df.index.searchsorted(end_date, side="right")
    df = df.iloc[:pos_end].copy()
    return df


# =============================================================================
# DATA CLEANING
# =============================================================================

def clean_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Final cleaning: replace inf, drop all-NaN rows."""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="all")
    return df


def data_integrity_report(df: pd.DataFrame):
    """Print data integrity report."""
    print("--- Data Integrity Report ---")
    print(f"Rows: {len(df):,}, Columns: {df.shape[1]}")
    print(f"Index sorted: {df.index.is_monotonic_increasing}")
    print(f"Index unique: {df.index.is_unique}")
    print(f"Rows that are entirely NaN: {df.isna().all(axis=1).sum()}")
    print(f"Inf values: {np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).sum()}")

    nan_counts = df.isna().sum()
    worst = nan_counts[nan_counts > 0].sort_values(ascending=False).head(10)

    print("\nTop columns with NaNs:")
    if worst.empty:
        print("  (None)")
    else:
        for col, n in worst.items():
            print(f"  {col}: {n}")
    print("")
