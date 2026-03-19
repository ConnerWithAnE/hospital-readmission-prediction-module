# Neural Network Integration Plan

## Context
The current production model is a 3-model stacking ensemble (LR + LGBM + RF) with LR meta-learner, SMOTE oversampling, feature pruning, and isotonic calibration. Baseline: AUROC ~0.670, AUPRC ~0.223 on UCI Diabetes 130 dataset.

This plan explores two approaches: (A) adding a small NN as a 4th base model in the stack, and (B) replacing the LR meta-learner with a NN meta-learner. A standalone NN comparison is also covered for reporting purposes.

---

## Option A: NN as a Base Model in the Stack

### Architecture
A small feedforward MLP added as the 4th estimator in the StackingClassifier:

```
Linear(n_features, 128) -> BatchNorm -> ReLU -> Dropout(0.3)
-> Linear(128, 64) -> BatchNorm -> ReLU -> Dropout(0.3)
-> Linear(64, 1)
```

### Implementation
- Wrap the PyTorch model in a scikit-learn compatible estimator (inherit `BaseEstimator`, `ClassifierMixin`)
- Must implement `fit(X, y)`, `predict(X)`, `predict_proba(X)` so StackingClassifier can use it
- StandardScaler applied internally (the tree models don't need scaling, but the NN does)
- BCEWithLogitsLoss with `pos_weight` matching the class ratio
- AdamW optimizer, cosine annealing LR scheduler
- Early stopping on validation loss (carve 15% of training data as validation)
- Training: ~50-100 epochs, batch size 256-512

### Pros
- Adds a fundamentally different inductive bias (gradient-based vs tree-based) — more diversity than XGBoost added
- The LR meta-learner can learn optimal weighting between tree and NN predictions
- NN may capture smooth nonlinear interactions that trees approximate with step functions

### Cons
- Slower training (especially inside 5-fold CV for stacking)
- More hyperparameters to tune
- Scikit-learn wrapper adds complexity
- Risk of overfitting on tabular data with only ~80k training samples
- XGBoost (also a tree model) added zero lift — a NN might fare better due to different inductive bias, but gains are not guaranteed

### Expected Impact
- AUROC +0.005 to +0.015 if the NN captures complementary patterns
- Training time: ~3-5x slower due to NN training in each CV fold

---

## Option B: NN as the Meta-Learner

### Architecture
Replace `final_estimator=LogisticRegression()` with a small NN that takes the 3 base model probability outputs as input:

```
Linear(3, 16) -> ReLU -> Dropout(0.2)
-> Linear(16, 8) -> ReLU
-> Linear(8, 1)
```

Very small network since the input is only 3 features (one probability per base model).

### Implementation
- Same scikit-learn wrapper approach as Option A
- The StackingClassifier passes base model predictions (3 values per sample) to the meta-learner
- The NN meta-learner learns a nonlinear combination of the base predictions
- With `passthrough=True`, it also sees the original features (3 + 65 = 68 inputs), which would justify a slightly larger network

### Variant: passthrough=True
If passthrough is enabled, the meta-learner sees both base predictions AND original features:
```
Linear(68, 64) -> BatchNorm -> ReLU -> Dropout(0.3)
-> Linear(64, 32) -> ReLU -> Dropout(0.2)
-> Linear(32, 1)
```
This lets the NN meta-learner correct base model errors using raw features.

### Pros
- LR meta-learner can only learn linear combinations of base predictions — a NN can learn nonlinear blending
- With passthrough, the meta-learner can learn feature-dependent weighting (e.g., trust LGBM more for certain patient profiles)
- Very fast since the meta-learner input is tiny (3 or 68 features)

### Cons
- Risk of overfitting the meta-learner on the stacking CV outputs (only ~16k samples per fold)
- More complex than LR for minimal expected gain — LR meta-learner is already near-optimal for probability blending
- Harder to interpret than LR coefficients

### Expected Impact
- AUROC +0.000 to +0.005 — the meta-learner sees very little data and a LR is usually sufficient
- With passthrough: possibly +0.005 to +0.010, but risk of overfitting increases

---

## Option C: Standalone NN Comparison (for report)

### Purpose
Train a standalone NN on the same data and compare metrics side-by-side with the ensemble. Useful for the report even if it doesn't beat the trees.

### Architecture
```
Linear(65, 128) -> BatchNorm -> ReLU -> Dropout(0.3)
-> Linear(128, 64) -> BatchNorm -> ReLU -> Dropout(0.3)
-> Linear(64, 1)
```

### Training
- Same train/test split as ensemble
- SMOTE on training data (or use pos_weight instead)
- BCEWithLogitsLoss, AdamW, cosine annealing
- Early stopping on val AUROC (patience 10-15 epochs)
- StandardScaler on all features

### Explainability
- Replace LGBM `pred_contrib` with Captum IntegratedGradients for per-prediction feature attributions
- Fast enough for real-time API (~50ms per prediction)
- Output format identical to current contributing_factors list

### Integration with API
- Add model selection to the predict endpoint (query param or config toggle)
- Load NN weights alongside the ensemble model
- Feature attribution via Captum instead of LGBM pred_contrib

### Expected Outcome
- AUROC 0.64-0.67 — likely matches or slightly underperforms the tree ensemble
- Valuable as a comparison data point for the report

---

## Recommendation

**Start with Option A** (NN as base model). It has the highest potential uplift and tests whether a fundamentally different model architecture adds diversity to the ensemble. If it shows no improvement (like XGBoost), fall back to **Option C** (standalone comparison for the report).

Option B (NN meta-learner) is lower priority — the meta-learner's job is simple probability blending, and LR handles that well.

## Dependencies
- PyTorch (`torch`)
- Captum (for explainability, only needed for Option C API integration)
- skorch (optional — provides scikit-learn compatible PyTorch wrapper, avoids writing custom wrapper)

## Files to Modify
- `backend/code/data_processing.py` — add NN base model wrapper, modify train()
- `backend/code/uci_nn.py` (new) — standalone NN for Option C
- `backend/server/models.py` — add model selection if Option C is integrated into API
- `backend/server/main.py` — add endpoint or config for model switching