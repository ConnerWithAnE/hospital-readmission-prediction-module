"""
Inventory Demand Forecasting — Data Processing & Feature Engineering
====================================================================
Merges Medicare Part D and FDA Drug Utilization data into a unified
time-series dataset, engineers features, and provides train/val/test splits
at configurable granularity (quarterly or monthly).
"""

import numpy as np
import pandas as pd

from .config import Config


# ═══════════════════════════════════════════════════════════════════════════
# 1. Per-dataset aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_medicare(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Medicare Part D from provider-level to drug x state x year."""
    agg = (
        df.groupby(["generic_name", "state", "year"], as_index=False)
        .agg(
            total_claims=("total_claims", "sum"),
            total_30day_fills=("total_30day_fills", "sum"),
            total_day_supply=("total_day_supply", "sum"),
            total_drug_cost=("total_drug_cost", "sum"),
            prescriber_count=("total_claims", "count"),  # number of prescriber rows
        )
    )
    return agg


def aggregate_fda(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate FDA utilization to drug x state x year x quarter."""
    agg = (
        df.groupby(["product_name", "state", "year", "quarter"], as_index=False)
        .agg(
            units_reimbursed=("units_reimbursed", "sum"),
            num_prescriptions=("num_prescriptions", "sum"),
            total_reimbursed=("total_reimbursed", "sum"),
        )
    )
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# 2. Merge the two datasets
# ═══════════════════════════════════════════════════════════════════════════

def _fuzzy_drug_match(fda_name: str, medicare_names: set) -> str | None:
    """Simple substring match: find a Medicare generic_name that contains
    or is contained by the FDA product_name.  Returns None if no match."""
    fda_upper = fda_name.upper().strip()
    for m in medicare_names:
        if fda_upper in m or m in fda_upper:
            return m
    return None


def merge_datasets(medicare_agg: pd.DataFrame, fda_agg: pd.DataFrame) -> pd.DataFrame:
    """Merge Medicare (annual) into FDA (quarterly) on drug + state + year.

    Medicare annual figures are spread equally across 4 quarters so that
    the merge doesn't inflate values.
    """
    # Build a drug-name mapping from FDA product_name → Medicare generic_name
    medicare_names = set(medicare_agg["generic_name"].unique())
    fda_names = fda_agg["product_name"].unique()
    name_map = {}
    for fn in fda_names:
        match = _fuzzy_drug_match(fn, medicare_names)
        if match:
            name_map[fn] = match

    fda_agg = fda_agg.copy()
    fda_agg["generic_name"] = fda_agg["product_name"].map(name_map)

    # Spread Medicare annual values across quarters
    medicare_q = medicare_agg.copy()
    for col in ["total_claims", "total_30day_fills", "total_day_supply", "total_drug_cost", "prescriber_count"]:
        if col in medicare_q.columns:
            medicare_q[col] = medicare_q[col] / 4.0

    # Cross-join Medicare with quarters so we can merge on year + quarter
    quarters = pd.DataFrame({"quarter": [1, 2, 3, 4]})
    medicare_q = medicare_q.merge(quarters, how="cross")

    merged = fda_agg.merge(
        medicare_q,
        on=["generic_name", "state", "year", "quarter"],
        how="left",
    )

    # Fill unmatched Medicare columns with 0
    for col in ["total_claims", "total_30day_fills", "total_day_supply", "total_drug_cost", "prescriber_count"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    print(f"[Merge] {len(merged):,} rows | "
          f"{merged['generic_name'].notna().sum():,} matched to Medicare")
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# 3. Granularity conversion
# ═══════════════════════════════════════════════════════════════════════════

def _quarterly_to_period(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `period` column as YYYY-Q# integer for sorting."""
    df = df.copy()
    df["period"] = df["year"] * 10 + df["quarter"]
    return df


def _quarterly_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate quarterly data to monthly by distributing volume evenly
    across the 3 months of each quarter."""
    rows = []
    volume_cols = [
        "units_reimbursed", "num_prescriptions", "total_reimbursed",
        "total_claims", "total_30day_fills", "total_day_supply", "total_drug_cost",
    ]
    for _, row in df.iterrows():
        q = int(row["quarter"])
        y = int(row["year"])
        start_month = (q - 1) * 3 + 1
        for m in range(start_month, start_month + 3):
            new_row = row.copy()
            new_row["month"] = m
            for vc in volume_cols:
                if vc in new_row.index:
                    new_row[vc] = row[vc] / 3.0
            rows.append(new_row)

    monthly = pd.DataFrame(rows)
    monthly["period"] = monthly["year"].astype(int) * 100 + monthly["month"].astype(int)
    return monthly


def convert_granularity(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Convert merged quarterly data to the requested granularity."""
    if granularity == "quarterly":
        return _quarterly_to_period(df)
    elif granularity == "monthly":
        return _quarterly_to_monthly(df)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Feature engineering
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add lag, rolling, and seasonality features for each drug x state series."""
    df = df.sort_values(["product_name", "state", "period"]).copy()

    group_cols = ["product_name", "state"]
    target = cfg.target_col

    # Map target_col to actual column name
    target_map = {"units": "units_reimbursed", "prescriptions": "num_prescriptions"}
    actual_target = target_map.get(target, "units_reimbursed")

    if actual_target not in df.columns:
        raise KeyError(f"Target column '{actual_target}' not found. Available: {list(df.columns)}")

    df["target"] = df[actual_target]

    # Lag features
    for lag in cfg.lag_periods:
        df[f"lag_{lag}"] = df.groupby(group_cols)["target"].shift(lag)

    # Rolling mean features
    for w in cfg.rolling_windows:
        df[f"rolling_mean_{w}"] = (
            df.groupby(group_cols)["target"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )

    # Rolling std
    df["rolling_std_4"] = (
        df.groupby(group_cols)["target"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=2).std())
    )

    # Year-over-year change (lag_4 for quarterly, lag_12 for monthly)
    yoy_lag = 4 if "quarter" in df.columns and "month" not in df.columns else 12
    if yoy_lag in cfg.lag_periods or True:
        yoy_col = f"lag_{yoy_lag}"
        if yoy_col in df.columns:
            df["yoy_change"] = (df["target"] - df[yoy_col]) / df[yoy_col].replace(0, np.nan)
            df["yoy_change"] = df["yoy_change"].fillna(0)

    # Seasonality indicators
    if "quarter" in df.columns:
        for q in [1, 2, 3, 4]:
            df[f"Q{q}"] = (df["quarter"] == q).astype(int)
    if "month" in df.columns:
        df["month"] = df["month"].astype(int)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Cost per unit (from Medicare)
    if "total_drug_cost" in df.columns and "total_day_supply" in df.columns:
        df["cost_per_day_supply"] = (
            df["total_drug_cost"] / df["total_day_supply"].replace(0, np.nan)
        ).fillna(0)

    # Prescriber density
    if "prescriber_count" in df.columns:
        df["prescriber_density"] = df["prescriber_count"]

    # Drop rows with NaN from lagging (early periods without enough history)
    min_lag = max(cfg.lag_periods)
    df = df.dropna(subset=[f"lag_{min_lag}"]).copy()

    print(f"[Features] {len(df):,} rows, {len(df.columns)} columns after feature engineering")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. Train / val / test split (time-based)
# ═══════════════════════════════════════════════════════════════════════════

def time_split(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time periods: everything before val as train, then val, then test."""
    periods = sorted(df["period"].unique())
    n = len(periods)

    test_start = n - cfg.test_periods
    val_start = test_start - cfg.val_periods

    train_periods = periods[:val_start]
    val_periods = periods[val_start:test_start]
    test_periods = periods[test_start:]

    train = df[df["period"].isin(train_periods)]
    val = df[df["period"].isin(val_periods)]
    test = df[df["period"].isin(test_periods)]

    print(f"[Split] Train: {len(train):,} ({len(train_periods)} periods) | "
          f"Val: {len(val):,} ({len(val_periods)} periods) | "
          f"Test: {len(test):,} ({len(test_periods)} periods)")
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════
# 6. Convenience: full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (everything except identifiers and target)."""
    exclude = {
        "product_name", "generic_name", "state", "year", "quarter", "month",
        "period", "ndc", "brand_name", "target", "suppression_used",
        "units_reimbursed", "num_prescriptions", "total_reimbursed",
    }
    return [c for c in df.columns if c not in exclude]


def prepare_data(medicare_df: pd.DataFrame, fda_df: pd.DataFrame,
                 cfg: Config, granularity: str):
    """Full pipeline: aggregate → merge → convert granularity → features → split.

    Returns (train, val, test, feature_cols).
    """
    medicare_agg = aggregate_medicare(medicare_df)
    fda_agg = aggregate_fda(fda_df)

    merged = merge_datasets(medicare_agg, fda_agg)
    converted = convert_granularity(merged, granularity)
    featured = engineer_features(converted, cfg)
    train, val, test = time_split(featured, cfg)
    feature_cols = get_feature_columns(featured)

    return train, val, test, feature_cols
