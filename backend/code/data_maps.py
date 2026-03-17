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
          "readmitted"]