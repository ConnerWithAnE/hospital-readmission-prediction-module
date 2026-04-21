# Inventory Demand Model — Stats Explained

A plain-language walkthrough of every number on the Inventory Model dashboard.

Values shown are from the current trained run (GBM (XGBoost), monthly, 2023 Medicare Part D + FDA Drug Utilization).

---

## Top headline cards

### Best Model — GBM (XGBoost) @ monthly
Of everything that trained (XGBoost + LightGBM at both quarterly and monthly), XGBoost at monthly granularity had the lowest validation MAE, so it's the winner. The `@ monthly` tag means predictions are per drug × state × **month**, not quarter.

### MAE = 14,735 units
**Mean Absolute Error** — on average, the model's predicted units-dispensed is off by ~14.7k units from the actual value, per drug-state-month row. It's in the same units as the target, so it's the most intuitive metric. Lower = better. Whether 14.7k is "good" depends entirely on the typical row value:

- For a tiny drug dispensing 500 units/month → terrible (28× off).
- For a high-volume drug dispensing 500,000 units/month → great (~3% off).

The dataset has both extremes, which is exactly why the *single-number* MAE can be misleading — see MAPE below.

### RMSE = 238,553
**Root Mean Squared Error**. Same units as MAE, but squares errors before averaging, so it punishes big misses much harder. RMSE being ~16× larger than MAE is the telltale sign that **a small number of rows have enormous errors** — almost certainly the highest-volume drug/state combos (think blockbuster generics in CA/TX/FL). The bulk of rows are off by ~14k; a handful are off by millions.

### MAPE = 1,226%
**Mean Absolute Percentage Error**. On average across all rows, the prediction is off by 12× the actual value. That sounds catastrophic, but it's almost certainly dominated by denominator-near-zero rows: drugs that dispensed 1–10 units in a state for a month, where even a prediction of 100 units = 1000%+ error. MAPE is effectively useless on inventory/utilization data with many rare drugs.

**Rule of thumb:** trust MAE for the common case; trust RMSE to spot tail blowouts; largely ignore MAPE here.

---

## Results by Granularity table

| Granularity | MAE     | RMSE      | MAPE   |
|-------------|---------|-----------|--------|
| quarterly   | 59,575  | 1,060,307 | 1,649% |
| **monthly** (best) | **14,735** | **238,553** | **1,226%** |

Monthly wins across the board — which is unusual. Normally longer horizons are easier to predict because noise averages out. The reason quarterly is worse: at quarterly granularity each row's target is ~3× larger (sum of 3 months), so the *absolute* errors compound. If you divide each quarterly metric by ~3 to compare them on a per-month basis, they actually land close to monthly — quarterly isn't really worse, it's just predicting a bigger number. The model recommender picks monthly because that's where the raw MAE is smallest.

---

## Model Comparison chart

The bar chart visualizes the same MAE / RMSE / MAPE across the candidate model × granularity combinations (e.g. GBM-monthly vs. GBM-quarterly). It's the visual version of the table above; the shorter the bar, the better.

---

## Summary cards

- **11,413 Unique Drugs** — distinct `product_name` values in the held-out test set.
- **53 Unique States** — 50 states + DC + territories (PR, etc.).
- **588,209 Test Rows** — total drug × state × month records the model predicted on. Roughly: 11,413 drugs × 53 states × 12 months = ~7.2M possible, but most drug-state-month combos are empty. 588k is what actually had observations.

---

## Bottom line

The model learns the broad structure well (MAE ≈ 14.7k on rows averaging who-knows-what), but two structural issues are visible in the numbers:

1. **Heavy-tailed errors** — RMSE/MAE ≈ 16 means a few huge-volume rows dominate total squared error. If the inventory use case cares about high-volume drugs, that matters; if it cares about rare ones, less so.
2. **MAPE is uninformative** on this data shape because zero/low-volume rows explode the ratio.

For better intuition on model quality, the *Predictions* tab (filter by drug/state, look at predicted vs. actual side by side) will tell you more than any single aggregate metric.