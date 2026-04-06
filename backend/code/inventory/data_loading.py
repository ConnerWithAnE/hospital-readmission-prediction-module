"""
Inventory Demand Forecasting — Data Loading
============================================
Loaders for Medicare Part D and FDA State Drug Utilization datasets.

Expected directory layout:
    datasets/inventory/
    ├── medicare_part_d/
    │   ├── MUP_DPR_RY22_P04_V10_DY20_Geo.csv   (or similar annual files)
    │   └── ...
    └── fda_drug_utilization/
        ├── State_Drug_Utilization_Data_2020.csv
        └── ...
"""

from pathlib import Path

import pandas as pd


# ── Medicare Part D ─────────────────────────────────────────────────────────

# CMS renames columns across years; map common variants to stable names.
_MEDICARE_COL_MAP = {
    # Provider geography
    "Prscrbr_State_Abrvtn": "state",
    "PRSCRBR_STATE_ABRVTN": "state",
    # Drug identity
    "Brnd_Name": "brand_name",
    "BRND_NAME": "brand_name",
    "Gnrc_Name": "generic_name",
    "GNRC_NAME": "generic_name",
    # Volume & cost
    "Tot_Clms": "total_claims",
    "TOT_CLMS": "total_claims",
    "Tot_30day_Fills": "total_30day_fills",
    "TOT_30DAY_FILLS": "total_30day_fills",
    "Tot_Day_Suply": "total_day_supply",
    "TOT_DAY_SUPLY": "total_day_supply",
    "Tot_Drug_Cst": "total_drug_cost",
    "TOT_DRUG_CST": "total_drug_cost",
    # Year (some files embed it, others require filename parsing)
    "Year": "year",
    "YEAR": "year",
}

_MEDICARE_KEEP = [
    "state", "brand_name", "generic_name",
    "total_claims", "total_30day_fills", "total_day_supply", "total_drug_cost",
    "year",
]


def _sniff_usecols(path: Path, col_map: dict) -> list[str]:
    """Read just the header row and return only the columns we can map."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    return [c for c in header if c in col_map]


def load_medicare_part_d(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all Medicare Part D CSVs in *data_dir*.

    Returns a DataFrame with one row per provider-drug record, with
    standardised column names.  Only reads the columns we actually need.
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in csv_files:
        usecols = _sniff_usecols(f, _MEDICARE_COL_MAP)
        print(f"  Reading {f.name} ({len(usecols)} cols)...")
        df = pd.read_csv(f, usecols=usecols, dtype=str)
        df.rename(columns=_MEDICARE_COL_MAP, inplace=True)

        # If year column is missing, parse from filename
        if "year" not in df.columns:
            for token in f.stem.split("_"):
                # "DY20" style (CMS shorthand)
                if token.startswith("DY") and token[2:].isdigit():
                    yr = int(token[2:])
                    df["year"] = 2000 + yr if yr < 100 else yr
                    break
                # Plain 4-digit year like "2023"
                if token.isdigit() and len(token) == 4:
                    df["year"] = int(token)
                    break

        # Coerce numeric columns
        for col in ["total_claims", "total_30day_fills", "total_day_supply", "total_drug_cost"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["generic_name"] = combined["generic_name"].str.upper().str.strip()
    combined["brand_name"] = combined["brand_name"].str.upper().str.strip()
    print(f"[Medicare Part D] Loaded {len(combined):,} rows from {len(csv_files)} file(s)")
    return combined


# ── FDA State Drug Utilization ──────────────────────────────────────────────

_FDA_COL_MAP = {
    "State": "state",
    "Year": "year",
    "Quarter": "quarter",
    "Product Name": "product_name",
    "Units Reimbursed": "units_reimbursed",
    "Number of Prescriptions": "num_prescriptions",
    "Total Amount Reimbursed": "total_reimbursed",
    "Medicaid Amount Reimbursed": "medicaid_reimbursed",
    "Non Medicaid Amount Reimbursed": "non_medicaid_reimbursed",
    "Suppression Used": "suppression_used",
    "NDC": "ndc",
    # Alternate casing seen in some files
    "STATE": "state",
    "YEAR": "year",
    "QUARTER": "quarter",
    "PRODUCT_NAME": "product_name",
    "UNITS_REIMBURSED": "units_reimbursed",
    "NUMBER_OF_PRESCRIPTIONS": "num_prescriptions",
    "TOTAL_AMOUNT_REIMBURSED": "total_reimbursed",
}

_FDA_KEEP = [
    "state", "year", "quarter", "product_name", "ndc",
    "units_reimbursed", "num_prescriptions", "total_reimbursed",
    "suppression_used",
]


def load_fda_utilization(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all FDA State Drug Utilization CSVs in *data_dir*.

    Returns a DataFrame with one row per state-drug-quarter record.
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in csv_files:
        usecols = _sniff_usecols(f, _FDA_COL_MAP)
        print(f"  Reading {f.name} ({len(usecols)} cols)...")
        df = pd.read_csv(f, usecols=usecols, dtype=str)
        df.rename(columns=_FDA_COL_MAP, inplace=True)

        # Coerce numeric columns
        for col in ["units_reimbursed", "num_prescriptions", "total_reimbursed"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop suppressed rows (no usable volume data)
        if "suppression_used" in df.columns:
            df = df[df["suppression_used"] != "Y"].copy()
            df.drop(columns=["suppression_used"], inplace=True)

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["product_name"] = combined["product_name"].str.upper().str.strip()
    print(f"[FDA Utilization] Loaded {len(combined):,} rows from {len(csv_files)} file(s)")
    return combined
