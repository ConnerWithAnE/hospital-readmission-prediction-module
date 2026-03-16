
"""Comorbidity index computation using comorbidipy.

Computes both Charlson Comorbidity Index (CCI) and Elixhauser Comorbidity Index
from ICD-9 diagnosis codes using validated mappings (Quan variant).
"""

import pandas as pd
import polars as pl
from comorbidipy import comorbidity

# ---------------------------------------------------------------------------
# comorbidipy short column names → readable names
# ---------------------------------------------------------------------------

CHARLSON_COL_MAP = {
    "ami": "myocardial_infarction",
    "chf": "congestive_heart_failure",
    "pvd": "peripheral_vascular_disease",
    "cevd": "cerebrovascular_disease",
    "dementia": "dementia",
    "copd": "chronic_pulmonary_disease",
    "rheumd": "rheumatic_disease",
    "pud": "peptic_ulcer_disease",
    "mld": "mild_liver_disease",
    "diab": "diabetes_uncomplicated",
    "diabwc": "diabetes_complicated",
    "hp": "hemiplegia_paraplegia",
    "rend": "renal_disease",
    "canc": "cancer",
    "msld": "moderate_severe_liver_disease",
    "metacanc": "metastatic_cancer",
    "aids": "aids_hiv",
}

ELIXHAUSER_COL_MAP = {
    "chf": "congestive_heart_failure",
    "carit": "cardiac_arrhythmias",
    "valv": "valvular_disease",
    "pcd": "pulmonary_circulation_disorders",
    "pvd": "peripheral_vascular_disease",
    "hypunc": "hypertension_uncomplicated",
    "hypc": "hypertension_complicated",
    "para": "paralysis",
    "ond": "other_neurological_disorders",
    "cpd": "chronic_pulmonary_disease",
    "diabunc": "diabetes_uncomplicated",
    "diabc": "diabetes_complicated",
    "hypothy": "hypothyroidism",
    "rf": "renal_failure",
    "ld": "liver_disease",
    "pud": "peptic_ulcer_disease",
    "aids": "aids_hiv",
    "lymph": "lymphoma",
    "metacanc": "metastatic_cancer",
    "solidtum": "solid_tumor",
    "rheumd": "rheumatic_disease",
    "coag": "coagulopathy",
    "obes": "obesity",
    "wloss": "weight_loss",
    "fed": "fluid_electrolyte_disorders",
    "blane": "blood_loss_anemia",
    "dane": "deficiency_anemias",
    "alcohol": "alcohol_abuse",
    "drug": "drug_abuse",
    "psycho": "psychoses",
    "depre": "depression",
}

# ---------------------------------------------------------------------------
# Weight tables for manual score computation (used by prediction endpoint)
# ---------------------------------------------------------------------------

CHARLSON_WEIGHTS = {
    "myocardial_infarction": 1,
    "congestive_heart_failure": 1,
    "peripheral_vascular_disease": 1,
    "cerebrovascular_disease": 1,
    "dementia": 1,
    "chronic_pulmonary_disease": 1,
    "rheumatic_disease": 1,
    "peptic_ulcer_disease": 1,
    "mild_liver_disease": 1,
    "diabetes_uncomplicated": 1,
    "diabetes_complicated": 2,
    "hemiplegia_paraplegia": 2,
    "renal_disease": 2,
    "cancer": 2,
    "moderate_severe_liver_disease": 3,
    "metastatic_cancer": 6,
    "aids_hiv": 6,
}

ELIXHAUSER_VW_WEIGHTS = {
    "congestive_heart_failure": 7,
    "cardiac_arrhythmias": 5,
    "valvular_disease": -1,
    "pulmonary_circulation_disorders": 4,
    "peripheral_vascular_disease": 2,
    "hypertension_uncomplicated": 0,
    "hypertension_complicated": 0,
    "paralysis": 7,
    "other_neurological_disorders": 6,
    "chronic_pulmonary_disease": 3,
    "diabetes_uncomplicated": 0,
    "diabetes_complicated": 0,
    "hypothyroidism": 0,
    "renal_failure": 5,
    "liver_disease": 11,
    "peptic_ulcer_disease": 0,
    "aids_hiv": 0,
    "lymphoma": 9,
    "metastatic_cancer": 12,
    "solid_tumor": 4,
    "rheumatic_disease": 0,
    "coagulopathy": 3,
    "obesity": -4,
    "weight_loss": 6,
    "fluid_electrolyte_disorders": 5,
    "blood_loss_anemia": -2,
    "deficiency_anemias": -2,
    "alcohol_abuse": 0,
    "drug_abuse": -7,
    "psychoses": 0,
    "depression": -3,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_long_format(df, diag_columns):
    """Melt wide-format diagnosis columns into long format for comorbidipy."""
    df_temp = df[diag_columns].copy()
    df_temp["_row_id"] = range(len(df))

    long = df_temp.melt(
        id_vars="_row_id",
        value_vars=diag_columns,
        var_name="_diag_col",
        value_name="code",
    )

    # Drop missing / invalid codes
    long = long.dropna(subset=["code"])
    long["code"] = long["code"].astype(str).str.strip()
    long = long[~long["code"].isin(["?", "", "nan", "None"])]

    # Strip dots — comorbidipy expects ICD-9 codes without dots
    long["code"] = long["code"].str.replace(".", "", regex=False)

    # Build Polars DataFrame directly to avoid pyarrow dependency
    return pl.DataFrame({
        "_row_id": long["_row_id"].to_numpy(),
        "code": long["code"].to_list(),
    })


def _rename_and_select(result_pd, col_map, score_col, prefix):
    """Rename comorbidipy short column names and select relevant columns."""
    rename = {"comorbidity_score": score_col}
    for short, readable in col_map.items():
        if short in result_pd.columns:
            rename[short] = f"{prefix}{readable}"
    result_pd = result_pd.rename(columns=rename)

    keep = ["_row_id", score_col] + [
        f"{prefix}{readable}"
        for short, readable in col_map.items()
        if f"{prefix}{readable}" in result_pd.columns
    ]
    return result_pd[[c for c in keep if c in result_pd.columns]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_comorbidities_to_dataframe(df, diag_columns=None):
    """
    Add Charlson and Elixhauser comorbidity scores and binary flags.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ICD-9 diagnosis columns.
    diag_columns : list[str], optional
        Columns containing ICD-9 codes. Defaults to ['diag_1', 'diag_2', 'diag_3'].

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - ``cci_score``  — Charlson Comorbidity Index (Charlson weighting)
        - ``cci_<category>`` — binary flags for each Charlson category
        - ``elixhauser_score`` — Elixhauser index (van Walraven weighting)
        - ``elix_<category>`` — binary flags for each Elixhauser category
    """
    if diag_columns is None:
        diag_columns = ["diag_1", "diag_2", "diag_3"]

    missing = [col for col in diag_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    long_pl = _to_long_format(df, diag_columns)

    df = df.copy()
    n_rows = len(df)
    df["_row_id"] = range(n_rows)

    # Guard: if no valid diagnosis codes exist, return zero-filled columns
    if long_pl.height == 0:
        for readable in CHARLSON_COL_MAP.values():
            df[f"cci_{readable}"] = 0
        df["cci_score"] = 0
        for readable in ELIXHAUSER_COL_MAP.values():
            df[f"elix_{readable}"] = 0
        df["elixhauser_score"] = 0
        return df.drop(columns=["_row_id"])

    # --- Charlson ---
    charlson_pd = comorbidity(
        long_pl,
        id_col="_row_id",
        code_col="code",
        score="charlson",
        icd="icd9",
        variant="quan",
        weighting="charlson",
    ).to_pandas()
    charlson_pd = _rename_and_select(charlson_pd, CHARLSON_COL_MAP, "cci_score", "cci_")

    # --- Elixhauser ---
    elix_pd = comorbidity(
        long_pl,
        id_col="_row_id",
        code_col="code",
        score="elixhauser",
        icd="icd9",
        variant="quan",
        weighting="van_walraven",
    ).to_pandas()
    elix_pd = _rename_and_select(elix_pd, ELIXHAUSER_COL_MAP, "elixhauser_score", "elix_")

    # --- Merge back ---
    df = df.merge(charlson_pd, on="_row_id", how="left")
    df = df.merge(elix_pd, on="_row_id", how="left")

    # Fill NaN for rows with no valid ICD codes
    fill_cols = [c for c in df.columns if c.startswith(("cci_", "elix_", "elixhauser_"))]
    df[fill_cols] = df[fill_cols].fillna(0).astype(int)

    return df.drop(columns=["_row_id"])


# Backward-compatible alias
add_cci_to_dataframe = add_comorbidities_to_dataframe