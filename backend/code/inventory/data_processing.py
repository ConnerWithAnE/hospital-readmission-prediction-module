"""
Inventory Demand Forecasting — Data Processing & Feature Engineering
====================================================================
Merges Medicare Part D and FDA Drug Utilization data into a unified
dataset for cross-sectional demand modeling.

With a single year of data, this uses a cross-sectional approach:
predict utilization volume from drug characteristics, state, and
Medicare prescribing patterns. Quarter is treated as a feature, not
a time axis for lag-based forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config


# ═══════════════════════════════════════════════════════════════════════════
# 1. Per-dataset aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_medicare(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Medicare Part D from provider-level to drug x state."""
    group = ["generic_name", "state"]
    if "year" in df.columns:
        group.append("year")

    agg = (
        df.groupby(group, as_index=False)
        .agg(
            total_claims=("total_claims", "sum"),
            total_30day_fills=("total_30day_fills", "sum"),
            total_day_supply=("total_day_supply", "sum"),
            total_drug_cost=("total_drug_cost", "sum"),
            prescriber_count=("total_claims", "count"),
        )
    )
    return agg


def aggregate_fda(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate FDA utilization to drug x state x quarter."""
    group = ["product_name", "state", "quarter"]
    if "year" in df.columns:
        group.append("year")

    agg = (
        df.groupby(group, as_index=False)
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
    """Merge Medicare (annual) into FDA (quarterly) on drug + state.

    Medicare annual figures are spread equally across quarters present
    in the FDA data so the merge doesn't inflate values.
    """
    # Build drug-name mapping: FDA product_name → Medicare generic_name
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
    n_quarters = fda_agg["quarter"].nunique()
    medicare_q = medicare_agg.copy()
    for col in ["total_claims", "total_30day_fills", "total_day_supply", "total_drug_cost", "prescriber_count"]:
        if col in medicare_q.columns:
            medicare_q[col] = medicare_q[col] / max(n_quarters, 1)

    # Cross-join Medicare with the quarters that exist in FDA data
    quarters = pd.DataFrame({"quarter": fda_agg["quarter"].unique()})
    medicare_q = medicare_q.merge(quarters, how="cross")

    merged = fda_agg.merge(
        medicare_q,
        on=["generic_name", "state", "quarter"],
        how="left",
    )

    # Fill unmatched Medicare columns with 0
    for col in ["total_claims", "total_30day_fills", "total_day_supply", "total_drug_cost", "prescriber_count"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    matched = merged["generic_name"].notna().sum()
    print(f"[Merge] {len(merged):,} rows | {matched:,} matched to Medicare")
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# 3. Granularity conversion
# ═══════════════════════════════════════════════════════════════════════════

def _to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `period` column from quarter for sorting."""
    df = df.copy()
    df["quarter"] = df["quarter"].astype(int)
    df["period"] = df["quarter"]
    return df


def _quarterly_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate quarterly data to monthly by distributing volume evenly
    across the 3 months of each quarter."""
    volume_cols = [
        "units_reimbursed", "num_prescriptions", "total_reimbursed",
        "total_claims", "total_30day_fills", "total_day_supply", "total_drug_cost",
    ]
    rows = []
    for _, row in df.iterrows():
        q = int(row["quarter"])
        start_month = (q - 1) * 3 + 1
        for m in range(start_month, start_month + 3):
            new_row = row.copy()
            new_row["month"] = m
            for vc in volume_cols:
                if vc in new_row.index:
                    new_row[vc] = row[vc] / 3.0
            rows.append(new_row)

    monthly = pd.DataFrame(rows)
    monthly["month"] = monthly["month"].astype(int)
    monthly["period"] = monthly["month"]
    return monthly


def convert_granularity(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if granularity == "quarterly":
        return _to_quarterly(df)
    elif granularity == "monthly":
        return _quarterly_to_monthly(df)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Feature engineering (cross-sectional)
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Engineer cross-sectional features for demand prediction.

    With only one year of data, we focus on:
    - Medicare prescribing volume/cost as demand signals
    - State-level relative rankings
    - Seasonality (quarter/month indicators)
    - Drug-level summary statistics across states
    """
    df = df.sort_values(["product_name", "state", "period"]).copy()

    # Map target column
    target_map = {"units": "units_reimbursed", "prescriptions": "num_prescriptions"}
    actual_target = target_map.get(cfg.target_col, "units_reimbursed")
    if actual_target not in df.columns:
        raise KeyError(f"Target column '{actual_target}' not found. Available: {list(df.columns)}")
    df["target"] = df[actual_target]

    # ── Seasonality indicators ──────────────────────────────────────────
    if "quarter" in df.columns:
        df["quarter"] = df["quarter"].astype(int)
        for q in [1, 2, 3, 4]:
            df[f"Q{q}"] = (df["quarter"] == q).astype(int)
    if "month" in df.columns:
        df["month"] = df["month"].astype(int)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Cost efficiency features (from Medicare) ────────────────────────
    if "total_drug_cost" in df.columns and "total_day_supply" in df.columns:
        df["cost_per_day_supply"] = (
            df["total_drug_cost"] / df["total_day_supply"].replace(0, np.nan)
        ).fillna(0)

    if "total_drug_cost" in df.columns and "total_claims" in df.columns:
        df["cost_per_claim"] = (
            df["total_drug_cost"] / df["total_claims"].replace(0, np.nan)
        ).fillna(0)

    if "total_day_supply" in df.columns and "total_claims" in df.columns:
        df["days_per_claim"] = (
            df["total_day_supply"] / df["total_claims"].replace(0, np.nan)
        ).fillna(0)

    # ── Prescriber density ──────────────────────────────────────────────
    if "prescriber_count" in df.columns:
        df["prescriber_density"] = df["prescriber_count"]

    # ── Drug-level stats (how does this drug behave across all states) ──
    drug_stats = df.groupby("product_name")["target"].agg(
        drug_mean="mean", drug_std="std", drug_median="median"
    ).reset_index()
    drug_stats["drug_std"] = drug_stats["drug_std"].fillna(0)
    df = df.merge(drug_stats, on="product_name", how="left")

    # ── State-level stats (overall demand level for this state) ─────────
    state_stats = df.groupby("state")["target"].agg(
        state_mean="mean", state_total="sum"
    ).reset_index()
    df = df.merge(state_stats, on="state", how="left")

    # ── Relative features ───────────────────────────────────────────────
    df["drug_vs_state_ratio"] = (
        df["target"] / df["state_mean"].replace(0, np.nan)
    ).fillna(0)

    # ── Prescription-to-units ratio ─────────────────────────────────────
    if "num_prescriptions" in df.columns and "units_reimbursed" in df.columns:
        df["units_per_rx"] = (
            df["units_reimbursed"] / df["num_prescriptions"].replace(0, np.nan)
        ).fillna(0)

    # ── Log-transform skewed volume columns ─────────────────────────────
    for col in ["total_claims", "total_30day_fills", "total_day_supply", "prescriber_density"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # Drop rows with NaN target
    df = df.dropna(subset=["target"]).copy()

    print(f"[Features] {len(df):,} rows, {len(df.columns)} columns after feature engineering")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. Train / val / test split
# ═══════════════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data for training. Strategy depends on available periods.

    With many periods: time-based split.
    With few periods (single year): random split by drug×state groups
    to avoid data leakage.
    """
    periods = sorted(df["period"].unique())
    n_periods = len(periods)

    if n_periods >= (cfg.test_periods + cfg.val_periods + 2):
        # Enough periods for time-based split
        test_start = n_periods - cfg.test_periods
        val_start = test_start - cfg.val_periods
        train_periods = periods[:val_start]
        val_periods = periods[val_start:test_start]
        test_periods_list = periods[test_start:]

        train = df[df["period"].isin(train_periods)]
        val = df[df["period"].isin(val_periods)]
        test = df[df["period"].isin(test_periods_list)]
        split_type = "time-based"
    else:
        # Not enough periods — split by drug×state groups
        groups = df.groupby(["product_name", "state"]).ngroup()
        unique_groups = groups.unique()

        train_groups, temp_groups = train_test_split(
            unique_groups, test_size=(cfg.val_size + cfg.test_size),
            random_state=cfg.seed,
        )
        val_groups, test_groups = train_test_split(
            temp_groups, test_size=0.5, random_state=cfg.seed,
        )

        train = df[groups.isin(train_groups)]
        val = df[groups.isin(val_groups)]
        test = df[groups.isin(test_groups)]
        split_type = "group-based"

    print(f"[Split ({split_type})] Train: {len(train):,} | "
          f"Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════
# 6. Helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (everything except identifiers and target)."""
    exclude = {
        "product_name", "generic_name", "state", "year", "quarter", "month",
        "period", "ndc", "brand_name", "target", "suppression_used",
        "units_reimbursed", "num_prescriptions", "total_reimbursed",
        "drug_vs_state_ratio",  # derived from target, would leak
    }
    return [c for c in df.columns if c not in exclude]


def prepare_data(medicare_df: pd.DataFrame, fda_df: pd.DataFrame,
                 cfg: Config, granularity: str):
    """Full pipeline: aggregate → merge → convert granularity → features → split.

    Returns (train, val, test, feature_cols).
    """
    print(f"  Aggregating Medicare...")
    medicare_agg = aggregate_medicare(medicare_df)
    print(f"  Aggregating FDA...")
    fda_agg = aggregate_fda(fda_df)

    merged = merge_datasets(medicare_agg, fda_agg)
    converted = convert_granularity(merged, granularity)
    featured = engineer_features(converted, cfg)
    train, val, test = split_data(featured, cfg)
    feature_cols = get_feature_columns(featured)

    return train, val, test, feature_cols
