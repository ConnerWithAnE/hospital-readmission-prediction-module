# Feature Improvement Plan — Readmission Prediction

## High Impact

- [ ] **Include individual comorbidity flags**
  - `add_comorbidities_to_dataframe` already computes ~45 binary flags but `keep` drops them all
  - Add clinically significant ones: `cci_congestive_heart_failure`, `cci_renal_disease`, `cci_chronic_pulmonary_disease`, `elix_depression`, `elix_fluid_electrolyte_disorders`, `elix_renal_failure`, `elix_coagulopathy`, `elix_weight_loss`
  - These conditions are individually strong readmission predictors that get diluted in aggregate scores
  - Files: `data_maps.py` (update `keep`), `models.py` (if needed for prediction input)

- [ ] **Include diabetes-specific clinical columns**
  - `A1Cresult`, `max_glu_serum`, `diabetesMed`, `change` exist in the UCI dataset but are currently dropped
  - Medication changes during hospitalization and poor glycemic control are strong readmission signals
  - Encode `A1Cresult` ordinally (None < Norm < >7 < >8) and `max_glu_serum` similarly
  - `change` and `diabetesMed` are binary Yes/No
  - Files: `data_maps.py` (add to `keep`, add encoding maps), `data_processing.py` (encoding logic)

## Medium Impact

- [ ] **Age-adjusted CCI score**
  - Original Charlson index adds 1 point per decade over 40; comorbidipy does not include this
  - `df["age_adjusted_cci"] = df["cci_score"] + np.maximum(0, df["age"] - 4)`
  - Files: `data_processing.py`

- [ ] **Age x utilization interaction features (frailty proxies)**
  - Without a formal frailty instrument, interactions approximate frailty risk:
    - `age_x_inpatient = age * number_inpatient`
    - `age_x_medications = age * num_medications` (polypharmacy in elderly)
    - `age_x_cci = age * cci_score`
  - Files: `data_processing.py`

- [ ] **Non-linear age thresholds**
  - Linear 0-9 encoding misses non-linear risk jumps at older ages
  - `age_over_70 = (age >= 7).astype(int)`
  - `age_over_80 = (age >= 8).astype(int)`
  - Files: `data_processing.py`

## Low Impact

- [ ] **Discharge × comorbidity interaction**
  - Patients discharged to care facility with high CCI have elevated readmission risk
  - `discharge_care_x_cci = (discharge_group == "care_facility") * cci_score`
  - Files: `data_processing.py`

## Notes

- After feature changes, retrain the model and compare AUROC/AUPRC against current baseline
- Update `PatientInput` in `models.py` and the frontend form if new user-facing fields are added
- Individual comorbidity flags do NOT require user input — they are derived from ICD codes during training; for prediction they come from the comorbidity scores the user already provides