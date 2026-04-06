age_map = {
    "[0-10)": 0,
    "[10-20)": 1,
    "[20-30)": 2,
    "[30-40)": 3,
    "[40-50)": 4,
    "[50-60)": 5,
    "[60-70)": 6,
    "[70-80)": 7,
    "[80-90)": 8,
    "[90-100)": 9,
}

admission_type_map = {
    1: "emergency",
    7: "emergency",

    2: "urgent",

    3: "elective",

    4: "birth",

    5: "unknown",
    6: "unknown",
    8: "unknown"
}

discharge_disposition_map = {
    # Home
    1: "Home",
    6: "Home",
    8: "Home",

    # Transfers to hospitals / institutions
    2: "transfer",
    5: "transfer",
    9: "transfer",
    10: "transfer",
    15: "transfer",
    16: "transfer",
    17: "transfer",
    27: "transfer",
    28: "transfer",
    29: "transfer",
    30: "transfer",

    # Skilled nursing / rehab / long-term care
    3: "care_facility",
    4: "care_facility",
    22: "care_facility",
    23: "care_facility",
    24: "care_facility",

    # Hospice or death
    11: "hospice_death",
    13: "hospice_death",
    14: "hospice_death",
    19: "hospice_death",
    20: "hospice_death",
    21: "hospice_death",

    # Other / unknown
    7: "Other",
    12: "Other",
    18: "Other",
    25: "Other",
    26: "Other"
}

admission_source_map = {
    # Physician / clinic referrals
    1: "physician_referral",
    2: "physician_referral",
    3: "physician_referral",

    # Transfers from facilities
    4: "transfer",
    5: "transfer",
    6: "transfer",
    10: "transfer",
    22: "transfer",
    25: "transfer",

    # Emergency
    7: "emergency",

    # Legal
    8: "legal",

    # Birth related
    11: "birth",
    12: "birth",
    13: "birth",
    14: "birth",
    23: "birth",
    24: "birth",

    # Home health
    18: "home_health",
    19: "home_health",

    # Hospice
    26: "hospice",

    # Unknown / invalid
    9: "unknown",
    15: "unknown",
    17: "unknown",
    20: "unknown",
    21: "unknown"
}

keep = ["age", "gender", "race", "time_in_hospital", "admission_source_id", "num_lab_procedures",
          "num_procedures", "num_medications", "number_diagnoses", "admission_type_id", "discharge_disposition_id",
          "number_inpatient", "number_outpatient", "number_emergency", "cci_score", "elixhauser_score",
          # Individual comorbidity flags — clinically strong readmission predictors
          "cci_congestive_heart_failure", "cci_renal_disease", "cci_chronic_pulmonary_disease",
          "elix_depression", "elix_fluid_electrolyte_disorders", "elix_renal_failure",
          "elix_coagulopathy", "elix_weight_loss",
          # Diabetes-specific clinical columns
          "A1Cresult", "max_glu_serum", "diabetesMed", "change",
          # Individual medication columns (high-variation only)
          "insulin", "metformin",
          # Primary diagnosis code for category grouping
          "diag_1",
          "readmitted"]

# All 23 medication columns used for aggregate feature computation
med_columns = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def icd9_to_category(code):
    """Map an ICD-9 code string to a clinical category."""
    if not isinstance(code, str) or code in ("?", "", "nan", "None"):
        return "unknown"
    code = code.strip().replace(".", "")
    # E/V codes
    if code.startswith("E"):
        return "injury_external"
    if code.startswith("V"):
        return "supplementary"
    try:
        numeric = float(code[:3])
    except ValueError:
        return "unknown"
    if 1 <= numeric <= 139:
        return "infectious"
    if 140 <= numeric <= 239:
        return "neoplasms"
    if 240 <= numeric <= 279:
        return "endocrine"  # includes diabetes (250)
    if 280 <= numeric <= 289:
        return "blood"
    if 290 <= numeric <= 319:
        return "mental"
    if 320 <= numeric <= 389:
        return "nervous"
    if 390 <= numeric <= 459:
        return "circulatory"
    if 460 <= numeric <= 519:
        return "respiratory"
    if 520 <= numeric <= 579:
        return "digestive"
    if 580 <= numeric <= 629:
        return "genitourinary"
    if 630 <= numeric <= 679:
        return "pregnancy"
    if 680 <= numeric <= 709:
        return "skin"
    if 710 <= numeric <= 739:
        return "musculoskeletal"
    if 740 <= numeric <= 759:
        return "congenital"
    if 760 <= numeric <= 779:
        return "perinatal"
    if 780 <= numeric <= 799:
        return "symptoms"
    if 800 <= numeric <= 999:
        return "injury"
    return "unknown"