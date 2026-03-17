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


class A1CResultEnum(str, Enum):
    not_measured = "not_measured"
    norm = "Norm"
    gt7 = ">7"
    gt8 = ">8"


class GluSerumEnum(str, Enum):
    not_measured = "not_measured"
    norm = "Norm"
    gt200 = ">200"
    gt300 = ">300"


class MedStatusEnum(str, Enum):
    no = "No"
    steady = "Steady"
    down = "Down"
    up = "Up"


class DiagCategoryEnum(str, Enum):
    infectious = "infectious"
    neoplasms = "neoplasms"
    endocrine = "endocrine"
    blood = "blood"
    mental = "mental"
    nervous = "nervous"
    circulatory = "circulatory"
    respiratory = "respiratory"
    digestive = "digestive"
    genitourinary = "genitourinary"
    skin = "skin"
    musculoskeletal = "musculoskeletal"
    symptoms = "symptoms"
    injury = "injury"
    injury_external = "injury_external"
    supplementary = "supplementary"
    unknown = "unknown"


# ---------------------------------------------------------------------------
# Unified comorbidity weights: (charlson_weight, elixhauser_van_walraven_weight)
# None = condition is not part of that index.
# ---------------------------------------------------------------------------
COMORBIDITY_WEIGHTS = {
    # Shared between Charlson and Elixhauser
    "congestive_heart_failure":        (1, 7),
    "peripheral_vascular_disease":     (1, 2),
    "chronic_pulmonary_disease":       (1, 3),
    "diabetes_uncomplicated":          (1, 0),
    "diabetes_complicated":            (2, 0),
    "paralysis":                       (2, 7),
    "renal_disease":                   (2, 5),
    "peptic_ulcer_disease":            (1, 0),
    "rheumatic_disease":               (1, 0),
    "aids_hiv":                        (6, 0),
    "metastatic_cancer":               (6, 12),
    "cancer":                          (2, 4),
    "liver_disease":                   (3, 11),
    "mild_liver_disease":              (1, None),
    # Charlson-only
    "myocardial_infarction":           (1, None),
    "cerebrovascular_disease":         (1, None),
    "dementia":                        (1, None),
    # Elixhauser-only
    "cardiac_arrhythmias":             (None, 5),
    "valvular_disease":                (None, -1),
    "pulmonary_circulation_disorders": (None, 4),
    "hypertension_uncomplicated":      (None, 0),
    "hypertension_complicated":        (None, 0),
    "other_neurological_disorders":    (None, 6),
    "hypothyroidism":                  (None, 0),
    "lymphoma":                        (None, 9),
    "coagulopathy":                    (None, 3),
    "obesity":                         (None, -4),
    "weight_loss":                     (None, 6),
    "fluid_electrolyte_disorders":     (None, 5),
    "blood_loss_anemia":               (None, -2),
    "deficiency_anemias":              (None, -2),
    "alcohol_abuse":                   (None, 0),
    "drug_abuse":                      (None, -7),
    "psychoses":                       (None, 0),
    "depression":                      (None, -3),
}


class ComorbidityEnum(str, Enum):
    # Shared
    congestive_heart_failure = "congestive_heart_failure"
    peripheral_vascular_disease = "peripheral_vascular_disease"
    chronic_pulmonary_disease = "chronic_pulmonary_disease"
    diabetes_uncomplicated = "diabetes_uncomplicated"
    diabetes_complicated = "diabetes_complicated"
    paralysis = "paralysis"
    renal_disease = "renal_disease"
    peptic_ulcer_disease = "peptic_ulcer_disease"
    rheumatic_disease = "rheumatic_disease"
    aids_hiv = "aids_hiv"
    metastatic_cancer = "metastatic_cancer"
    cancer = "cancer"
    liver_disease = "liver_disease"
    mild_liver_disease = "mild_liver_disease"
    # Charlson-only
    myocardial_infarction = "myocardial_infarction"
    cerebrovascular_disease = "cerebrovascular_disease"
    dementia = "dementia"
    # Elixhauser-only
    cardiac_arrhythmias = "cardiac_arrhythmias"
    valvular_disease = "valvular_disease"
    pulmonary_circulation_disorders = "pulmonary_circulation_disorders"
    hypertension_uncomplicated = "hypertension_uncomplicated"
    hypertension_complicated = "hypertension_complicated"
    other_neurological_disorders = "other_neurological_disorders"
    hypothyroidism = "hypothyroidism"
    lymphoma = "lymphoma"
    coagulopathy = "coagulopathy"
    obesity = "obesity"
    weight_loss = "weight_loss"
    fluid_electrolyte_disorders = "fluid_electrolyte_disorders"
    blood_loss_anemia = "blood_loss_anemia"
    deficiency_anemias = "deficiency_anemias"
    alcohol_abuse = "alcohol_abuse"
    drug_abuse = "drug_abuse"
    psychoses = "psychoses"
    depression = "depression"


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
    comorbidities: list[ComorbidityEnum] = Field(default_factory=list)
    a1c_result: A1CResultEnum = A1CResultEnum.not_measured
    max_glu_serum: GluSerumEnum = GluSerumEnum.not_measured
    diabetes_med: bool = False
    med_change: bool = False
    insulin: MedStatusEnum = MedStatusEnum.no
    metformin: MedStatusEnum = MedStatusEnum.no
    diag_category: DiagCategoryEnum = DiagCategoryEnum.unknown
    n_active_meds: int = Field(ge=0, default=0)
    n_med_changes: int = Field(ge=0, default=0)

    def compute_cci(self):
        """Compute Charlson Comorbidity Index from selected comorbidities."""
        selected = set(c.value for c in self.comorbidities)

        # Charlson hierarchical rules
        if "diabetes_complicated" in selected:
            selected.discard("diabetes_uncomplicated")
        if "liver_disease" in selected:
            selected.discard("mild_liver_disease")
        if "metastatic_cancer" in selected:
            selected.discard("cancer")

        total = 0
        for cat in selected:
            weights = COMORBIDITY_WEIGHTS.get(cat)
            if weights and weights[0] is not None:
                total += weights[0]
        return total

    def compute_elixhauser(self):
        """Compute Elixhauser Comorbidity Index (van Walraven weighting)."""
        selected = set(c.value for c in self.comorbidities)

        # Elixhauser hierarchical rules
        if "diabetes_complicated" in selected:
            selected.discard("diabetes_uncomplicated")
        if "hypertension_complicated" in selected:
            selected.discard("hypertension_uncomplicated")

        total = 0
        for cat in selected:
            weights = COMORBIDITY_WEIGHTS.get(cat)
            if weights and weights[1] is not None:
                total += weights[1]
        return total

    @staticmethod
    def get_fields():
        def enum_options(enum_cls):
            return [{"value": m.value, "label": m.name.replace("_", " ").title()} for m in enum_cls]

        return [
            {"name": "age", "label": "Age", "type": "number", "min": 0, "max": 120},
            {"name": "gender", "label": "Gender", "type": "select", "options": enum_options(GenderEnum)},
            {"name": "race", "label": "Race", "type": "select", "options": enum_options(RaceEnum)},
            {"name": "time_in_hospital", "label": "Time in Hospital (days)", "type": "number", "min": 1},
            {"name": "admission_type", "label": "Admission Type", "type": "select",
             "options": enum_options(AdmissionTypeEnum)},
            {"name": "admission_source", "label": "Admission Source", "type": "select",
             "options": enum_options(AdmissionSourceEnum)},
            {"name": "discharge_group", "label": "Discharge Disposition", "type": "select",
             "options": enum_options(DischargeGroupEnum)},
            {"name": "num_lab_procedures", "label": "Number of Lab Procedures", "type": "number", "min": 0},
            {"name": "num_procedures", "label": "Number of Procedures", "type": "number", "min": 0},
            {"name": "num_medications", "label": "Number of Medications", "type": "number", "min": 0},
            {"name": "number_diagnoses", "label": "Number of Diagnoses", "type": "number", "min": 0},
            {"name": "number_inpatient", "label": "Prior Inpatient Visits", "type": "number", "min": 0},
            {"name": "number_outpatient", "label": "Prior Outpatient Visits", "type": "number", "min": 0},
            {"name": "number_emergency", "label": "Prior Emergency Visits", "type": "number", "min": 0},
            {"name": "comorbidities", "label": "Comorbidities (Charlson + Elixhauser)", "type": "checkbox_group",
             "options": [{"value": m.value, "label": m.name.replace("_", " ").title()} for m in ComorbidityEnum]},
            {"name": "a1c_result", "label": "A1C Result", "type": "select",
             "options": [{"value": "not_measured", "label": "Not Measured"},
                         {"value": "Norm", "label": "Normal"},
                         {"value": ">7", "label": "> 7%"},
                         {"value": ">8", "label": "> 8%"}]},
            {"name": "max_glu_serum", "label": "Max Glucose Serum", "type": "select",
             "options": [{"value": "not_measured", "label": "Not Measured"},
                         {"value": "Norm", "label": "Normal"},
                         {"value": ">200", "label": "> 200 mg/dL"},
                         {"value": ">300", "label": "> 300 mg/dL"}]},
            {"name": "diabetes_med", "label": "On Diabetes Medication", "type": "boolean"},
            {"name": "med_change", "label": "Medication Changed During Visit", "type": "boolean"},
            {"name": "insulin", "label": "Insulin Status", "type": "select",
             "options": enum_options(MedStatusEnum)},
            {"name": "metformin", "label": "Metformin Status", "type": "select",
             "options": enum_options(MedStatusEnum)},
            {"name": "diag_category", "label": "Primary Diagnosis Category", "type": "select",
             "options": enum_options(DiagCategoryEnum)},
            {"name": "n_active_meds", "label": "Number of Active Diabetes Medications", "type": "number", "min": 0},
            {"name": "n_med_changes", "label": "Number of Medication Changes", "type": "number", "min": 0},
        ]

    def to_raw_df(self) -> pd.DataFrame:
        """Convert to a DataFrame with categorical columns ready for the encoder."""
        # Encode medication status ordinally (same mapping as training)
        med_ordinal = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}
        insulin_val = med_ordinal[self.insulin.value]
        metformin_val = med_ordinal[self.metformin.value]

        # Encode A1C and glucose (same mapping as training)
        a1c_map = {"not_measured": -1, "Norm": 0, ">7": 1, ">8": 2}
        glu_map = {"not_measured": -1, "Norm": 0, ">200": 1, ">300": 2}
        a1c_val = a1c_map[self.a1c_result.value]
        glu_val = glu_map[self.max_glu_serum.value]

        # Individual comorbidity flags
        selected = set(c.value for c in self.comorbidities)

        total_prior = self.number_inpatient + self.number_outpatient + self.number_emergency
        any_dose_up = int(insulin_val == 3 or metformin_val == 3 or self.n_med_changes > 0)

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
            "elixhauser_score": self.compute_elixhauser(),
            # Individual comorbidity flags
            "cci_congestive_heart_failure": int("congestive_heart_failure" in selected),
            "cci_renal_disease": int("renal_disease" in selected),
            "cci_chronic_pulmonary_disease": int("chronic_pulmonary_disease" in selected),
            "elix_depression": int("depression" in selected),
            "elix_fluid_electrolyte_disorders": int("fluid_electrolyte_disorders" in selected),
            "elix_renal_failure": int("renal_disease" in selected),
            "elix_coagulopathy": int("coagulopathy" in selected),
            "elix_weight_loss": int("weight_loss" in selected),
            # Diabetes-specific clinical columns
            "A1Cresult": a1c_val,
            "max_glu_serum": glu_val,
            "diabetesMed": int(self.diabetes_med),
            "change": int(self.med_change),
            # Individual medication status
            "insulin": insulin_val,
            "metformin": metformin_val,
            # Aggregate medication features
            "n_active_meds": self.n_active_meds,
            "n_med_changes": self.n_med_changes,
            "any_dose_up": any_dose_up,
            # Engineered features (same logic as training)
            "total_prior_visits": total_prior,
            "has_prior_inpatient": int(self.number_inpatient > 0),
            "meds_per_day": self.num_medications / (self.time_in_hospital + 1),
            "diag_per_day": 0 if self.number_diagnoses == 0 else self.number_diagnoses / (self.time_in_hospital or 1),
            "lab_proc_ratio": self.num_lab_procedures / (self.num_procedures + 1),
            "emergency_ratio": self.number_emergency / (total_prior + 1),
            "insulin_x_a1c": insulin_val * a1c_val,
            # Categorical columns — the saved encoder will one-hot encode these
            "race": self.race.value,
            "discharge_group": self.discharge_group.value,
            "admission_type": self.admission_type.value,
            "admission_source": self.admission_source.value,
            "diag_category": self.diag_category.value,
        }
        return pd.DataFrame([row])