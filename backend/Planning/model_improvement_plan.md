# Model Improvement Plan

## Context
The hospital readmission model (UCI Diabetes 130 dataset) sits at AUROC 0.671, AUPRC 0.224 — already at the published literature ceiling (0.65-0.70) for this dataset. The goal is to squeeze out remaining gains through medication features, secondary diagnoses, model architecture tweaks, and optionally a NN comparison. The dataset fundamentally lacks social determinants, vitals, and post-discharge data, so expectations should be modest (+0.01-0.02 AUROC realistic).

---

## Phase 1: Medication & Derived Features (no frontend changes)

**Files:** `data_processing.py` (`refine_dataset`, `to_raw_df` in models.py), `data_maps.py`

### 1a. Medication class groupings
Group the 23 med columns into pharmacological classes:
- **Sulfonylureas**: glimepiride, glipizide, glyburide, chlorpropamide, tolbutamide, tolazamide, acetohexamide
- **TZDs**: pioglitazone, rosiglitazone, troglitazone
- **Meglitinides**: repaglinide, nateglinide
- **Alpha-glucosidase inhibitors**: acarbose, miglitol

New features: `sulfonylurea_active`, `tzd_active`, `n_drug_classes` (count of distinct active classes)

### 1b. Combination therapy flag
`insulin_plus_oral` = insulin active AND (metformin or sulfonylurea active) — signals failed monotherapy

### 1c. Dose direction features
- `any_dose_down` (binary)
- `n_dose_ups`, `n_dose_downs` (counts)
- `net_dose_direction` = ups - downs

### 1d. Polypharmacy threshold
`diabetes_polypharmacy` = (n_active_meds >= 5)

### 1e. Additional interactions
- `inpatient_x_emergency` = number_inpatient * number_emergency

All derivable from training data — mirror in `to_raw_df()` using existing API fields. Retrain and evaluate.

---

## Phase 2: Secondary Diagnosis Features

**Files:** `data_maps.py` (add diag_2, diag_3 to keep), `data_processing.py`, `models.py`, frontend manual-entry

### Compact approach (avoids 34 one-hot columns):
- Apply `icd9_to_category()` to diag_2 and diag_3
- Create binary flags: `has_circulatory_secondary`, `has_respiratory_secondary`, `has_endocrine_secondary`
- `diag_diversity` = count of distinct categories across diag_1/2/3

**Frontend**: Add `diag_2_category` and `diag_3_category` selects to PatientInput and manual-entry form. Update scenario inputs.

---

## Phase 3: Model Architecture Tweaks

**Files:** `data_processing.py` (`train` method)

### 3a. Add XGBoost as 4th base model
Already imported but unused. Adds diversity to the ensemble:
```
('xgb', XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8))
```

### 3b. Test passthrough=True
One-line change — meta-learner sees base predictions + original features. Revert if worse.

### 3c. Feature pruning
After training, extract LightGBM feature importances, drop bottom 10-15% zero/near-zero importance features, retrain.

---

## Phase 4: Optional — Optuna Tuning

**Files:** `data_processing.py` (new method)

Run Bayesian hyperparameter search on LGBM params (n_estimators, num_leaves, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda). ~50-100 trials, optimize AUROC on 3-fold CV. Apply best params to the stacking ensemble.

---

## Phase 5: Optional — Neural Network Comparison

**Files:** New `backend/code/uci_nn.py`, adapt from existing `mimic_readmission_nn.py`

### Architecture
Small feedforward MLP: Linear(input, 128) → BatchNorm → ReLU → Dropout(0.3) → Linear(128, 64) → BatchNorm → ReLU → Dropout(0.3) → Linear(64, 1)

- BCEWithLogitsLoss with pos_weight for imbalance
- AdamW + cosine annealing + early stopping on val AUROC
- StandardScaler on all features

### Explainability
Replace LightGBM `pred_contrib` with Captum IntegratedGradients for per-prediction feature attributions. Fast enough for real-time API.

### Expected outcome
AUROC 0.64-0.67 — likely matches or slightly underperforms the tree ensemble. Valuable as a comparison point for the report, not as a replacement.

---

## Execution Order
1. Phase 1 → retrain → evaluate (if better, keep; if worse, revert individually)
2. Phase 2 → retrain → evaluate
3. Phase 3a-3c → retrain → evaluate
4. Phase 4 if time permits
5. Phase 5 if desired for report/learning

## Verification
After each phase: run `python -m backend.code.data_processing` and compare AUROC, AUPRC, and F1 threshold against baseline (0.671 / 0.224 / 0.152). Check scenario spread on the frontend after rebuilding with `bun run build.ts`.

## Realistic Expectations
- Phases 1-3 combined: AUROC 0.675-0.685
- All phases: AUROC 0.68-0.69
- The dataset's information ceiling is the binding constraint, not the model.