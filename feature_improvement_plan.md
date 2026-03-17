# Feature Improvement Plan — Readmission Prediction

## High Impact

- [x] **Include individual comorbidity flags**
  - `add_comorbidities_to_dataframe` already computes ~45 binary flags but `keep` drops them all
  - Add clinically significant ones: `cci_congestive_heart_failure`, `cci_renal_disease`, `cci_chronic_pulmonary_disease`, `elix_depression`, `elix_fluid_electrolyte_disorders`, `elix_renal_failure`, `elix_coagulopathy`, `elix_weight_loss`
  - These conditions are individually strong readmission predictors that get diluted in aggregate scores
  - Files: `data_maps.py` (update `keep`), `models.py` (if needed for prediction input)

- [x] **Include diabetes-specific clinical columns**
  - `A1Cresult`, `max_glu_serum`, `diabetesMed`, `change` exist in the UCI dataset but are currently dropped
  - Medication changes during hospitalization and poor glycemic control are strong readmission signals
  - Encode `A1Cresult` ordinally (None < Norm < >7 < >8) and `max_glu_serum` similarly
  - `change` and `diabetesMed` are binary Yes/No
  - Files: `data_maps.py` (add to `keep`, add encoding maps), `data_processing.py` (encoding logic)

- [x] **Medication-level features** ✓ improved macro f1 0.56→0.58, accuracy 0.59→0.60
  - Only medications with meaningful variation are useful: insulin (~54k), metformin (~20k), glipizide (~12.7k), glyburide (~10.6k), pioglitazone (~7.3k), rosiglitazone (~6.4k), glimepiride (~5.2k)
  - Individual features:
    - `insulin` — ordinal encode No=0, Steady=1, Down=2, Up=3. Strongest signal; dosage changes indicate severity
    - `metformin` — same encoding. Discontinuation can signal worsening renal function
  - Aggregate features (computed across all 23 medication columns):
    - `n_active_meds` — count of medications not "No". Diabetes-specific polypharmacy, distinct from `num_medications`
    - `n_med_changes` — count of medications with Up or Down. Treatment instability = poorly controlled diabetes
    - `any_dose_up` — binary: any medication escalated. Suggests inadequate control
  - Files: `data_maps.py` (add insulin/metformin to `keep`), `data_processing.py` (encoding + aggregate features)

## Medium Impact (tested — no improvement, reverted)

- [x] **Age-adjusted CCI score**
  - Original Charlson index adds 1 point per decade over 40; comorbidipy does not include this
  - `df["age_adjusted_cci"] = df["cci_score"] + np.maximum(0, df["age"] - 4)`
  - Files: `data_processing.py`

- [x] **Age x utilization interaction features (frailty proxies)**
  - Without a formal frailty instrument, interactions approximate frailty risk:
    - `age_x_inpatient = age * number_inpatient`
    - `age_x_medications = age * num_medications` (polypharmacy in elderly)
    - `age_x_cci = age * cci_score`
  - Files: `data_processing.py`

- [x] **Non-linear age thresholds**
  - Linear 0-9 encoding misses non-linear risk jumps at older ages
  - `age_over_70 = (age >= 7).astype(int)`
  - `age_over_80 = (age >= 8).astype(int)`
  - Files: `data_processing.py`

## Feature Engineering — Round 2 ✓ Readmit f1 0.66→0.67

- [x] **Primary diagnosis category grouping**
  - `diag_1` ICD-9 codes are currently only used for CCI/Elixhauser computation, not as a direct feature
  - Group into clinical categories by ICD-9 range: circulatory (390–459), respiratory (460–519), diabetes (250.xx), digestive (520–579), injury (800–999), etc.
  - One-hot encode the primary diagnosis category — reason for admission is clinically meaningful
  - Files: `data_maps.py` (add ICD-9 category map, add `diag_1` to `keep`), `data_processing.py` (mapping + add to encoder)

- [x] **Lab-to-procedure ratio**
  - `lab_proc_ratio = num_lab_procedures / (num_procedures + 1)`
  - High labs relative to procedures signals diagnostic uncertainty or monitoring-heavy stays
  - Files: `data_processing.py`

- [x] **Emergency frequency ratio**
  - `emergency_ratio = number_emergency / (total_prior_visits + 1)`
  - Patients whose prior visits are disproportionately emergencies have different risk profiles
  - Files: `data_processing.py`

- [x] **Insulin × A1C interaction**
  - `insulin_x_a1c = insulin * A1Cresult`
  - Patients on escalating insulin with high A1C are poorly controlled despite treatment — compounding risk
  - Files: `data_processing.py`

## Low Impact

- [ ] **Discharge × comorbidity interaction**
  - Patients discharged to care facility with high CCI have elevated readmission risk
  - `discharge_care_x_cci = (discharge_group == "care_facility") * cci_score`
  - Files: `data_processing.py`

## Skipped — insufficient data

- **`weight`** — 97% null
- **`payer_code`** — 40% null, 17 categories, weak proxy
- **`medical_specialty`** — 49% null, 72 categories, would need heavy grouping

---

# Model Tuning Plan

Current setup: StackingClassifier with LR + LGBM + RF base models, LR meta-learner, 5-fold CV.
Current best: accuracy 0.60, macro f1 0.58, Readmit f1 0.67.

## 1. Scale features for Logistic Regression

- [x] **Add StandardScaler to the LR base model via a Pipeline** ✗ worse, reverted
  - LR is the only base model that benefits from scaling, but currently gets unscaled data
  - Tree models (LGBM, RF) are scale-invariant so they're fine
  - Wrap LR in `Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(...))])`
  - This lets the stacker feed raw data to trees and scaled data to LR
  - Files: `data_processing.py` (train method)

## 2. LGBM hyperparameter tuning

- [x] **Increase model capacity and regularization** ✗ worse, reverted
  - Current: `n_estimators=300, num_leaves=31, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`
  - Try: `n_estimators=500, num_leaves=63, learning_rate=0.03, min_child_samples=50, reg_alpha=0.1, reg_lambda=1.0`
  - More trees + lower learning rate = better generalization
  - `min_child_samples=50` prevents overfitting to small leaf groups
  - Files: `data_processing.py`

## 3. Random Forest tuning

- [x] **Increase depth and add min_samples_leaf** ✗ worse, reverted
  - Current: `n_estimators=200, max_depth=12, class_weight='balanced'`
  - Try: `n_estimators=300, max_depth=16, min_samples_leaf=20, class_weight='balanced'`
  - Files: `data_processing.py`

## 4. Meta-learner improvements

- [x] **Try class-weighted meta-learner** ✗ worse (double-corrects imbalance), reverted
  - Current meta-learner is bare `LogisticRegression()` — no class weighting
  - Try: `LogisticRegression(class_weight='balanced', C=0.5)` to let the meta-learner account for imbalance
  - Files: `data_processing.py`

- [ ] **Try passthrough=True**
  - Currently `passthrough=False` — meta-learner only sees base model probabilities
  - With `passthrough=True`, meta-learner also sees raw features, which can help if base models miss different signals
  - Trade-off: more features for meta-learner = risk of overfitting
  - Files: `data_processing.py`

## 5. Threshold optimization

- [ ] **Use PR-curve optimal threshold instead of default 0.5**
  - Already computed in `get_model_stats()` as `best_f1_threshold` but not used in `predict()`
  - Store the best threshold during training and apply it in the prediction endpoint
  - Files: `data_processing.py` (save threshold), `predict()` method

## 6. Target encoding refinement

- [ ] **Separate <30 day vs >30 day readmission**
  - Currently both map to 1. Consider training on <30 only (clinically more actionable) or using ordinal encoding (NO=0, >30=1, <30=2)
  - Trade-off: smaller positive class = harder to learn, but more clinically precise
  - Files: `data_processing.py` (get_split method)

## Notes

- After feature changes, retrain the model and compare AUROC/AUPRC against current baseline
- Update `PatientInput` in `models.py` and the frontend form if new user-facing fields are added
- Individual comorbidity flags do NOT require user input — they are derived from ICD codes during training; for prediction they come from the comorbidity scores the user already provides
- Tune one thing at a time to isolate what helps vs hurts
- Items 1–4 can be tested together as a single "tuned stacker" run if preferred