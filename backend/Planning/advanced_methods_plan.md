# Advanced Model Improvement Methods

## Current State
3-model stacking (LR + LGBM + RF), SMOTE, feature pruning (65 features), isotonic calibration. AUROC ~0.670, AUPRC ~0.223. Optuna tuning in progress.

---

## 1. CatBoost (High Priority)

### What It Is
CatBoost (Categorical Boosting) is Yandex's gradient boosting library, purpose-built for datasets with categorical features. Unlike LGBM/XGB which require one-hot encoding or label encoding, CatBoost handles categoricals natively using **ordered target statistics** — a technique that computes running target means with added noise to prevent target leakage.

### Why It's Promising for This Dataset
- **5 categorical columns** (race, admission_type, admission_source, discharge_group, diag_category) are currently one-hot encoded into ~40+ sparse binary columns. CatBoost processes these directly, avoiding the curse of dimensionality.
- **Ordered boosting** — CatBoost uses a permutation-based approach that reduces prediction shift (a form of overfitting specific to gradient boosting). This is especially helpful with imbalanced datasets.
- **Symmetric (oblivious) decision trees** — CatBoost uses the same splitting condition at each depth level across the entire tree. This acts as regularization and makes predictions faster.
- **In published benchmarks**, CatBoost consistently outperforms LGBM/XGB on datasets with mixed feature types (numerical + categorical) without extensive preprocessing.

### Implementation Plan

#### Step 1: CatBoost as LGBM Replacement
Replace LGBM in the stacking ensemble with CatBoost. Pass categorical column indices so CatBoost uses native handling instead of one-hot encoded features.

```python
from catboost import CatBoostClassifier

# Identify categorical column indices in the NON-encoded dataframe
cat_features = ['race', 'discharge_group', 'admission_type', 'admission_source', 'diag_category']

catboost_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3,               # L2 regularization
    border_count=128,            # number of splits for numerical features
    cat_features=cat_feature_indices,
    auto_class_weights='Balanced',  # built-in class balancing
    eval_metric='AUC',
    random_seed=42,
    verbose=0,
)
```

**Key change**: To use CatBoost's native categorical handling, we need to modify `refine_dataset()` to NOT one-hot encode the categorical columns. Instead, pass them as strings/integers and let CatBoost handle them internally. This means creating a separate data path or making encoding conditional.

#### Step 2: CatBoost + LGBM Ensemble
Keep both CatBoost and LGBM as base models (they have different inductive biases — CatBoost uses symmetric trees, LGBM uses leaf-wise growth). The meta-learner can weight them optimally.

```python
base_models = [
    ('lr', LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs')),
    ('catboost', CatBoostClassifier(iterations=500, depth=6, ...)),
    ('lgbm', LGBMClassifier(n_estimators=300, ...)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=12)),
]
```

**Caveat**: CatBoost inside StackingClassifier requires the scikit-learn API wrapper, which works but doesn't support `cat_features` per fold. Workaround: use the one-hot encoded data for stacking (CatBoost still performs well on encoded data), or manually implement the stacking with CatBoost's native API.

#### Step 3: CatBoost with Native Categoricals (Standalone Comparison)
Train CatBoost standalone on the pre-encoded data (categoricals as strings) and compare against the full ensemble. If it matches or beats the ensemble alone, it could simplify the entire pipeline.

### Hyperparameters to Tune
| Parameter | Range | Notes |
|-----------|-------|-------|
| iterations | 300-1000 | Number of boosting rounds |
| depth | 4-10 | Tree depth (6-8 typical for tabular) |
| learning_rate | 0.01-0.1 | Step size |
| l2_leaf_reg | 1-10 | L2 regularization on leaf values |
| border_count | 64-255 | Binning resolution for numericals |
| random_strength | 0-5 | Randomness for scoring splits |
| bagging_temperature | 0-1 | Bayesian bootstrap temperature |
| min_data_in_leaf | 1-50 | Minimum samples per leaf |

### Expected Impact
- As LGBM replacement: AUROC +0.005 to +0.015 (categorical handling advantage)
- In ensemble with LGBM: AUROC +0.003 to +0.010 (diversity gain)
- Standalone: potentially matches full ensemble with simpler pipeline

### Dependencies
```
pip install catboost
```

---

## 2. SMOTE Variants

### Problem with Vanilla SMOTE
SMOTE generates synthetic samples by interpolating between random minority class neighbors. This can:
- Create noisy samples in overlapping regions where majority/minority classes mix
- Oversample easy-to-classify regions that don't help the decision boundary
- Produce unrealistic feature combinations

### Alternatives

#### BorderlineSMOTE
Only generates synthetic samples from minority instances that are near the decision boundary (have majority class neighbors). Focuses oversampling where it matters most.

```python
from imblearn.over_sampling import BorderlineSMOTE
smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_res, y_res = smote.fit_resample(X_train, y_train)
```

#### SMOTE-ENN (SMOTE + Edited Nearest Neighbors)
Applies SMOTE first, then removes samples (both synthetic and original) whose class label differs from the majority of their k nearest neighbors. Cleans the overlapping region.

```python
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_res, y_res = smote_enn.fit_resample(X_train, y_train)
```

#### ADASYN (Adaptive Synthetic Sampling)
Generates more synthetic samples in regions where the minority class is harder to learn (higher density of majority neighbors). Adapts the oversampling density.

```python
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X_res, y_res = adasyn.fit_resample(X_train, y_train)
```

#### SMOTE-Tomek
Combines SMOTE with Tomek link removal — deletes pairs of nearest-neighbor samples from opposite classes. Cleans the boundary more aggressively than ENN.

```python
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
```

### Implementation
Drop-in replacement — swap the SMOTE line in `train()`. Compare all variants on the same train/test split.

### Expected Impact
- AUROC: +0.000 to +0.005 (small but measurable)
- Calibration: potentially better (cleaner synthetic samples = less probability distortion)
- SMOTE-ENN typically performs best in benchmarks on medical datasets

---

## 3. Target Encoding

### What It Is
Replaces each categorical value with the smoothed mean of the target variable for that category. For example, if `admission_type=emergency` has a 15% readmission rate, it becomes 0.15. Smoothing blends with the global mean to prevent overfitting on rare categories.

### Why It Helps
- Reduces one-hot encoding from ~40 sparse columns to 5 dense columns
- Encodes predictive signal directly (rare categories with few samples get shrunk toward global mean)
- Tree models can split on a single target-encoded column instead of scanning 10+ one-hot columns
- Scikit-learn's `TargetEncoder` (added in 1.3) handles cross-validation internally to prevent target leakage

### Implementation
```python
from sklearn.preprocessing import TargetEncoder

cat_cols = ['race', 'discharge_group', 'admission_type', 'admission_source', 'diag_category']
target_enc = TargetEncoder(smooth='auto', cv=5)
encoded = target_enc.fit_transform(df[cat_cols], df['readmitted_binary'])
```

### Considerations
- Must fit on training data only, transform test data — same as OneHotEncoder
- Need to update `_prepare_input()` and `save_model()` to use the target encoder for predictions
- Can use both target encoding AND one-hot encoding as separate features (let the model decide)
- Interacts with CatBoost: if using CatBoost with native categoricals, target encoding is redundant

### Expected Impact
- AUROC: +0.000 to +0.010 (depends on how well tree models utilized the one-hot columns)
- Feature count: 65 -> ~30 (dramatic reduction in dimensionality)
- May eliminate need for feature pruning entirely

---

## 4. TabPFN v2

### What It Is
A pre-trained transformer (Prior-Data Fitted Network) that has been trained on millions of synthetic classification tasks. Given a new dataset, it performs Bayesian inference in a single forward pass — no training or hyperparameter tuning needed.

### How It Works
- The model learns a prior over all possible datasets during pre-training
- At inference, it conditions on the training data as context and predicts test labels
- Effectively performs in-context learning (like GPT but for tabular data)
- v2 (2024) supports larger datasets and more features than v1

### Limitations for This Dataset
- Designed for datasets with <10k rows and <100 features
- Our dataset has ~80k rows — would need subsampling or ensembling over subsets
- No GPU required for inference but pre-trained weights are ~1GB
- Not easily integrated into scikit-learn pipelines

### Implementation
```python
from tabpfn import TabPFNClassifier

model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
```

For large datasets, ensemble over subsampled training sets:
```python
predictions = []
for i in range(10):
    subset = X_train.sample(n=3000, random_state=i)
    model.fit(subset, y_train.loc[subset.index])
    predictions.append(model.predict_proba(X_test)[:, 1])
y_proba = np.mean(predictions, axis=0)
```

### Expected Impact
- AUROC: 0.64-0.67 (competitive but unlikely to beat tuned ensemble on this dataset size)
- Value: comparison point for report, zero-tuning baseline

### Dependencies
```
pip install tabpfn
```

---

## 5. FT-Transformer

### What It Is
Feature Tokenizer + Transformer. Each feature (numerical or categorical) is converted into a d-dimensional token via learned embedding layers, then standard transformer self-attention is applied across feature tokens.

### Why It's Interesting
- Self-attention can model arbitrary feature interactions without explicit engineering
- Often outperforms tree models on datasets where interactions between distant features matter
- Published results show it competitive with GBDT on many tabular benchmarks

### Architecture
```
Input features -> Feature Tokenizer (per-feature linear embedding)
-> [CLS] token prepended
-> N Transformer blocks (self-attention + FFN)
-> [CLS] output -> Linear head -> prediction
```

Typical config: d_token=192, n_blocks=3, attention_heads=8, FFN multiplier=4/3, dropout=0.1

### Implementation
Would use the `rtdl` (Revisiting Deep Learning Models for Tabular Data) library or a custom PyTorch implementation.

```python
import rtdl

model = rtdl.FTTransformer.make_default(
    n_num_features=n_numerical,
    cat_cardinalities=cat_cardinalities,  # list of category counts
    d_out=1,  # binary classification
)
```

### Considerations
- Needs GPU for reasonable training time
- Requires careful learning rate scheduling (warmup + cosine decay)
- Prone to overfitting on small datasets — our 80k rows is in the sweet spot
- Would need custom scikit-learn wrapper for stacking integration

### Expected Impact
- AUROC: 0.66-0.68 standalone
- As ensemble member: potentially +0.005 to +0.010 (different inductive bias from trees)

### Dependencies
```
pip install rtdl torch
```

---

## 6. GRANDE (2024)

### What It Is
GRAdient-based Neural Decision Ensembles. Learns an ensemble of differentiable oblique decision trees end-to-end via gradient descent. Each tree uses soft (differentiable) splits instead of hard axis-aligned splits, allowing optimization with backpropagation.

### Why It's Interesting
- Bridges trees and neural networks — gets the interpretability benefits of trees with the optimization power of gradient descent
- Oblique splits (linear combinations of features) are strictly more expressive than axis-aligned splits
- Very recent (2024 paper), shown to be competitive with GBDT on many benchmarks
- Built-in feature selection via learned split weights

### Implementation
```python
from grande import GRANDE

model = GRANDE(
    depth=5,
    n_estimators=2048,
    learning_rate_weights=0.005,
    learning_rate_index=0.01,
    learning_rate_values=0.01,
    learning_rate_leaf=0.01,
    optimizer='SWA',
    cosine_decay_steps=0,
    normalizer='T',
    temperature=0.25,
    dropout=0.0,
)
model.fit(X_train, y_train, n_epochs=1000)
```

### Expected Impact
- AUROC: 0.66-0.69 (competitive with GBDT)
- Newer and less battle-tested than CatBoost/LGBM

### Dependencies
```
pip install GRANDE
```

---

## 7. Multi-Level Stacking

### What It Is
Instead of one level of base models -> meta-learner, add an intermediate level. Level 1 base models generate out-of-fold predictions, level 2 models train on those, and a final meta-learner combines level 2 outputs.

### Architecture
```
Level 1: LR, LGBM, RF, CatBoost (out-of-fold predictions)
Level 2: LGBM, ExtraTrees (trained on level 1 predictions + optional raw features)
Level 3: LR meta-learner (combines level 2 predictions)
```

### Implementation
Manual implementation required (scikit-learn's StackingClassifier doesn't support multiple levels natively).

```python
# Level 1: generate OOF predictions
oof_preds = np.zeros((len(X_train), n_level1_models))
for i, (name, model) in enumerate(level1_models):
    for train_idx, val_idx in kfold.split(X_train, y_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof_preds[val_idx, i] = model.predict_proba(X_train.iloc[val_idx])[:, 1]

# Level 2: train on OOF predictions
level2_input = pd.DataFrame(oof_preds, columns=[m[0] for m in level1_models])
# optionally: level2_input = pd.concat([level2_input, X_train.reset_index(drop=True)], axis=1)

# Repeat OOF process for level 2 models...
```

### Considerations
- Significant overfitting risk with each additional level
- Diminishing returns after level 2
- Much slower training (each level requires full CV)
- Kaggle competition technique — less common in production

### Expected Impact
- AUROC: +0.002 to +0.008 over single-level stacking
- Heavy engineering effort for marginal gain

---

## 8. Beta Calibration / Venn-ABERS

### Beta Calibration
Fits a 3-parameter beta distribution to map raw model outputs to calibrated probabilities. More flexible than Platt scaling (sigmoid, 2 params) but less prone to overfitting than isotonic (non-parametric).

```python
from betacal import BetaCalibration
bc = BetaCalibration()
bc.fit(y_scores_train, y_train)
calibrated = bc.predict(y_scores_test)
```

### Venn-ABERS Calibration
Provides calibrated prediction intervals with guaranteed coverage. For each test sample, it produces two probabilities (one assuming the true label is 0, one assuming 1) and combines them. Provides valid probabilities under exchangeability.

```python
from venn_abers import VennAbersCalibrator
va = VennAbersCalibrator()
va.fit(y_scores_cal, y_cal)
p0, p1 = va.predict_proba(y_scores_test)
```

### Expected Impact
- Calibration improvement: potentially significant
- AUROC: unchanged (calibration is monotone — doesn't affect ranking)
- AUPRC: slight improvement possible due to better probability estimates

---

## 9. FLAML (Fast AutoML)

### What It Is
Microsoft's lightweight AutoML library. Searches over model types (LGBM, XGBoost, RF, ExtraTrees, CatBoost, LR) AND their hyperparameters simultaneously using cost-frugal optimization.

### Why It's Interesting
- Extremely fast — often finds good configurations in minutes
- Cost-aware: spends more time on promising configurations
- May find model types or parameter combinations humans wouldn't try
- Easy to use as a black-box or extract the best model for manual tuning

### Implementation
```python
from flaml import AutoML

automl = AutoML()
automl.fit(
    X_train, y_train,
    task='classification',
    metric='roc_auc',
    time_budget=300,  # seconds
    estimator_list=['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree'],
)
print(automl.best_estimator)
print(automl.best_config)
```

### Expected Impact
- May find better hyperparameters than manual Optuna search
- Could identify that CatBoost or ExtraTrees outperforms LGBM for this dataset
- AUROC: +0.000 to +0.010

### Dependencies
```
pip install flaml[automl]
```

---

## 10. LightGBM DART Mode

### What It Is
DART (Dropouts meet Multiple Additive Regression Trees) applies dropout to boosting. At each iteration, a random subset of existing trees is "dropped" (excluded), and the new tree is fitted to the residual of the remaining trees. This prevents individual trees from dominating.

### Implementation
One-line change in the existing LGBM config:

```python
LGBMClassifier(
    boosting_type='dart',
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    drop_rate=0.1,        # fraction of trees to drop
    max_drop=50,          # max number of trees to drop per iteration
    skip_drop=0.5,        # probability of skipping dropout
    ...
)
```

### Considerations
- Significantly slower training than GBDT (no early stopping shortcut)
- Better regularization — often improves generalization on noisy datasets
- The readmission dataset is inherently noisy (many confounders not captured), so DART may help

### Expected Impact
- AUROC: +0.000 to +0.005
- Training time: ~2-3x slower

---

## Priority Order

| Priority | Method | Expected AUROC Gain | Effort | Risk |
|----------|--------|-------------------|--------|------|
| 1 | CatBoost (replace or add to ensemble) | +0.005 to +0.015 | Medium | Low |
| 2 | SMOTE-ENN / BorderlineSMOTE | +0.000 to +0.005 | Low | Low |
| 3 | Target Encoding | +0.000 to +0.010 | Medium | Low |
| 4 | LGBM DART mode | +0.000 to +0.005 | Very Low | Low |
| 5 | FLAML | +0.000 to +0.010 | Low | Low |
| 6 | Beta Calibration | calibration only | Low | Low |
| 7 | FT-Transformer | +0.000 to +0.010 | High | Medium |
| 8 | Multi-level Stacking | +0.002 to +0.008 | High | Medium |
| 9 | GRANDE | +0.000 to +0.010 | Medium | Medium |
| 10 | TabPFN v2 | comparison only | Low | Low |

## Realistic Combined Ceiling
With the best combination of these methods, the realistic ceiling for this dataset remains ~0.68-0.70 AUROC. The dataset fundamentally lacks social determinants, vitals, medication adherence, and post-discharge data — no model architecture can recover signal that isn't in the features.