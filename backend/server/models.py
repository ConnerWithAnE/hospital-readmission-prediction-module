from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field


class GenderEnum(str, Enum):
    male = "male"
    female = "female"


class AdmissionTypeEnum(str, Enum):
    emergency = "emergency"
    urgent = "urgent"
    elective = "elective"
    unknown = "unknown"


class DischargeGroupEnum(str, Enum):
    home = "Home"
    transfer = "transfer"
    care_facility = "care_facility"
    hospice_death = "hospice_death"
    other = "Other"


class RaceEnum(str, Enum):
    african_american = "AfricanAmerican"
    asian = "Asian"
    caucasian = "Caucasian"
    hispanic = "Hispanic"
    other = "Other"
    unknown = "unknown"


class AdmissionSourceEnum(str, Enum):
    physician_referral = "physician_referral"
    emergency = "emergency"
    transfer = "transfer"
    birth = "birth"
    legal = "legal"
    unknown = "unknown"


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


class CharlsonCategoryEnum(str, Enum):
    myocardial_infarction = "myocardial_infarction"
    congestive_heart_failure = "congestive_heart_failure"
    peripheral_vascular_disease = "peripheral_vascular_disease"
    cerebrovascular_disease = "cerebrovascular_disease"
    dementia = "dementia"
    chronic_pulmonary_disease = "chronic_pulmonary_disease"
    rheumatic_connective_tissue = "rheumatic_connective_tissue"
    peptic_ulcer_disease = "peptic_ulcer_disease"
    mild_liver_disease = "mild_liver_disease"
    diabetes_no_complications = "diabetes_no_complications"
    diabetes_with_complications = "diabetes_with_complications"
    hemiplegia_paraplegia = "hemiplegia_paraplegia"
    renal_disease = "renal_disease"
    cancer_malignancy = "cancer_malignancy"
    moderate_severe_liver_disease = "moderate_severe_liver_disease"
    metastatic_solid_tumor = "metastatic_solid_tumor"
    aids_hiv = "aids_hiv"


class PatientInput(BaseModel):
    age: int = Field(ge=0, le=120)
    gender: GenderEnum
    race: RaceEnum
    time_in_hospital: int = Field(ge=0)
    admission_type: AdmissionTypeEnum
    admission_source: AdmissionSourceEnum
    discharge_group: DischargeGroupEnum
    num_lab_procedures: int = Field(ge=0)
    num_procedures: int = Field(ge=0)
    num_medications: int = Field(ge=0)
    number_diagnoses: int = Field(ge=0)
    number_inpatient: int = Field(ge=0)
    number_outpatient: int = Field(ge=0)
    number_emergency: int = Field(ge=0)
    charlson_categories: list[CharlsonCategoryEnum] = Field(default_factory=list)

    def compute_cci(self) -> int:
        selected = set(c.value for c in self.charlson_categories)

        # Apply hierarchical rules
        if "diabetes_with_complications" in selected:
            selected.discard("diabetes_no_complications")
        if "moderate_severe_liver_disease" in selected:
            selected.discard("mild_liver_disease")
        if "metastatic_solid_tumor" in selected:
            selected.discard("cancer_malignancy")

        return sum(CHARLSON_CATEGORIES[cat]["weight"] for cat in selected)

    @staticmethod
    def get_fields():
        def enum_options(enum_cls):
            return [{"value": m.value, "label": m.name.replace("_", " ").title()} for m in enum_cls]

        return [
            {"name": "age", "label": "Age", "type": "number", "min": 0, "max": 120},
            {"name": "gender", "label": "Gender", "type": "select", "options": enum_options(GenderEnum)},
            {"name": "race", "label": "Race", "type": "select", "options": enum_options(RaceEnum)},
            {"name": "time_in_hospital", "label": "Time in Hospital (days)", "type": "number", "min": 1},
            {"name": "admission_type", "label": "Admission Type", "type": "select", "options": enum_options(AdmissionTypeEnum)},
            {"name": "admission_source", "label": "Admission Source", "type": "select", "options": enum_options(AdmissionSourceEnum)},
            {"name": "discharge_group", "label": "Discharge Disposition", "type": "select", "options": enum_options(DischargeGroupEnum)},
            {"name": "num_lab_procedures", "label": "Number of Lab Procedures", "type": "number", "min": 0},
            {"name": "num_procedures", "label": "Number of Procedures", "type": "number", "min": 0},
            {"name": "num_medications", "label": "Number of Medications", "type": "number", "min": 0},
            {"name": "number_diagnoses", "label": "Number of Diagnoses", "type": "number", "min": 0},
            {"name": "number_inpatient", "label": "Prior Inpatient Visits", "type": "number", "min": 0},
            {"name": "number_outpatient", "label": "Prior Outpatient Visits", "type": "number", "min": 0},
            {"name": "number_emergency", "label": "Prior Emergency Visits", "type": "number", "min": 0},
            {"name": "charlson_categories", "label": "Comorbidities (Charlson Index)", "type": "checkbox_group",
             "options": [{"value": m.value, "label": m.name.replace("_", " ").title()} for m in CharlsonCategoryEnum]},
        ]

    def to_raw_df(self) -> pd.DataFrame:
        """Convert to a DataFrame with categorical columns ready for the encoder."""
        row = {
            "age": self.age,
            "gender": 1 if self.gender == GenderEnum.male else 0,
            "time_in_hospital": self.time_in_hospital,
            "num_lab_procedures": self.num_lab_procedures,
            "num_procedures": self.num_procedures,
            "num_medications": self.num_medications,
            "number_diagnoses": self.number_diagnoses,
            "number_inpatient": self.number_inpatient,
            "number_outpatient": self.number_outpatient,
            "number_emergency": self.number_emergency,
            "cci_score": self.compute_cci(),
            # Engineered features (same logic as training)
            "total_prior_visits": self.number_inpatient + self.number_outpatient + self.number_emergency,
            "has_prior_inpatient": int(self.number_inpatient > 0),
            "meds_per_day": self.num_medications / (self.time_in_hospital + 1),
            "diag_per_day": self.number_diagnoses / (self.time_in_hospital + 1),
            # Categorical columns — the saved encoder will one-hot encode these
            "race": self.race.value,
            "discharge_group": self.discharge_group.value,
            "admission_type": self.admission_type.value,
            "admission_source": self.admission_source.value,
        }
        return pd.DataFrame([row])