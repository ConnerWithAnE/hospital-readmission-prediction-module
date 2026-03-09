import pandas as pd

CHARLSON_CATEGORIES = {
    "myocardial_infarction": {
        "weight": 1,
        "icd9_ranges": [(410, 412)],
        "icd9_exact": ["410", "411", "412"],
    },
    "congestive_heart_failure": {
        "weight": 1,
        "icd9_ranges": [(428, 428)],
        "icd9_prefixes": ["428"],
    },
    "peripheral_vascular_disease": {
        "weight": 1,
        "icd9_ranges": [(440, 441)],
        "icd9_prefixes": ["440", "441", "443.9", "785.4"],
        "icd9_exact": ["V43.4"],
    },
    "cerebrovascular_disease": {
        "weight": 1,
        "icd9_ranges": [(430, 438)],
        "icd9_prefixes": [],
    },
    "dementia": {
        "weight": 1,
        "icd9_ranges": [(290, 290)],
        "icd9_prefixes": ["290"],
    },
    "chronic_pulmonary_disease": {
        "weight": 1,
        "icd9_ranges": [(490, 496), (500, 505)],
        "icd9_prefixes": ["506.4"],
    },
    "rheumatic_connective_tissue": {
        "weight": 1,
        "icd9_prefixes": ["710.0", "710.1", "710.4", "714.0", "714.1", "714.2", "714.81", "725"],
        "icd9_ranges": [],
    },
    "peptic_ulcer_disease": {
        "weight": 1,
        "icd9_ranges": [(531, 534)],
        "icd9_prefixes": [],
    },
    "mild_liver_disease": {
        "weight": 1,
        "icd9_prefixes": ["571.2", "571.4", "571.5", "571.6"],
        "icd9_ranges": [],
    },
    "diabetes_no_complications": {
        "weight": 1,
        "icd9_prefixes": ["250.0", "250.1", "250.2", "250.3"],
        "icd9_ranges": [],
    },
    "diabetes_with_complications": {
        "weight": 2,
        "icd9_prefixes": ["250.4", "250.5", "250.6", "250.7", "250.8", "250.9"],
        "icd9_ranges": [],
    },
    "hemiplegia_paraplegia": {
        "weight": 2,
        "icd9_prefixes": ["342", "344.1"],
        "icd9_ranges": [(342, 342)],
    },
    "renal_disease": {
        "weight": 2,
        "icd9_prefixes": ["582", "583", "585", "586", "588"],
        "icd9_ranges": [(582, 583), (585, 586)],
        "icd9_exact": ["V42.0", "V45.1", "V56"],
    },
    "cancer_malignancy": {
        "weight": 2,
        "icd9_ranges": [(140, 172), (174, 195), (200, 208)],
        "icd9_prefixes": [],
    },
    "moderate_severe_liver_disease": {
        "weight": 3,
        "icd9_prefixes": ["572.2", "572.3", "572.4", "572.5", "572.6", "572.7", "572.8"],
        "icd9_ranges": [],
    },
    "metastatic_solid_tumor": {
        "weight": 6,
        "icd9_ranges": [(196, 199)],
        "icd9_prefixes": [],
    },
    "aids_hiv": {
        "weight": 6,
        "icd9_ranges": [(42, 44)],
        "icd9_prefixes": ["042", "043", "044"],
    },
}

def clean_icd9_code(code):
    """
    Clean and standardize an ICD-9 code from the UCI dataset.
    """
    if pd.isna(code) or str(code).strip() in ("?", "", "nan", "None"):
        return None

    code = str(code).strip()
    return code


def get_numeric_prefix(code):
    """
    Extract the numeric prefix of an ICD-9 code for range matching.
    Returns the integer portion before any decimal point.
    Returns None for V-codes and E-codes (handled separately).
    """
    if code is None:
        return None

    if code.startswith(("V", "v", "E", "e")):
        return None

    try:
        # Extract the integer part before the decimal
        numeric_part = code.split(".")[0]
        return int(numeric_part)
    except (ValueError, IndexError):
        return None


def check_code_in_category(code, category_def):
    """
    Check if a single ICD-9 code falls into a Charlson category.

    Matching logic:
    1. Check exact matches first (for V-codes and special cases)
    2. Check prefix matches (e.g., "250.4" matches codes starting with "250.4")
    3. Check numeric range matches (e.g., 410-412 matches 410, 411, 412 and subcodes)
    """
    if code is None:
        return False

    # 1. Exact matches (handles V-codes, E-codes, special cases)
    exact_codes = category_def.get("icd9_exact", [])
    if code in exact_codes or code.upper() in [c.upper() for c in exact_codes]:
        return True

    # 2. Prefix matches
    prefixes = category_def.get("icd9_prefixes", [])
    for prefix in prefixes:
        if code.startswith(prefix):
            return True

    # 3. Numeric range matches
    numeric_val = get_numeric_prefix(code)
    if numeric_val is not None:
        ranges = category_def.get("icd9_ranges", [])
        for range_start, range_end in ranges:
            if range_start <= numeric_val <= range_end:
                return True

    return False

def compute_cci_for_codes(codes):
    """
    Compute the Charlson Comorbidity Index for a list of ICD-9 codes.

    Rules:
    - Each category is counted at most once (even if multiple codes match)
    - If both 'diabetes_no_complications' and 'diabetes_with_complications'
      are flagged, only the higher-weighted one counts
    - Same hierarchy for mild vs moderate/severe liver disease
    - Same hierarchy for cancer vs metastatic tumor
    """
    matched_categories = set()

    for code in codes:
        cleaned = clean_icd9_code(code)
        if cleaned is None:
            continue

        for category_name, category_def in CHARLSON_CATEGORIES.items():
            if check_code_in_category(cleaned, category_def):
                matched_categories.add(category_name)

    # Apply hierarchical rules:
    # If patient has complicated diabetes, remove uncomplicated
    if "diabetes_with_complications" in matched_categories:
        matched_categories.discard("diabetes_no_complications")

    # If patient has moderate/severe liver disease, remove mild
    if "moderate_severe_liver_disease" in matched_categories:
        matched_categories.discard("mild_liver_disease")

    # If patient has metastatic tumor, remove non-metastatic cancer
    if "metastatic_solid_tumor" in matched_categories:
        matched_categories.discard("cancer_malignancy")

    # Sum the weights
    total_cci = sum(
        CHARLSON_CATEGORIES[cat]["weight"] for cat in matched_categories
    )

    return total_cci, matched_categories

def add_cci_to_dataframe(df, diag_columns=None):
    """
    Add CCI score and individual category flags to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The UCI Diabetes dataset (or any DataFrame with ICD-9 diagnosis columns)
    diag_columns : list of str, optional
        Column names containing ICD-9 codes. Defaults to ['diag_1', 'diag_2', 'diag_3']

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - 'cci_score': integer CCI score
        - 'cci_<category_name>': binary flags for each Charlson category
    """
    if diag_columns is None:
        diag_columns = ["diag_1", "diag_2", "diag_3"]

    # Verify columns exist
    missing = [col for col in diag_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Compute CCI for each row
    cci_scores = []
    category_flags = {cat: [] for cat in CHARLSON_CATEGORIES}

    for _, row in df.iterrows():
        codes = [row[col] for col in diag_columns]
        score, matched = compute_cci_for_codes(codes)

        cci_scores.append(score)
        for cat in CHARLSON_CATEGORIES:
            category_flags[cat].append(1 if cat in matched else 0)

    # Add CCI score column
    df = df.copy()
    df["cci_score"] = cci_scores

    # Add individual category flag columns
    for cat in CHARLSON_CATEGORIES:
        df[f"cci_{cat}"] = category_flags[cat]

    return df