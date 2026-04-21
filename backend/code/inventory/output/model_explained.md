# Inventory Demand Model — What It's Doing

A plain-language description of what the inventory model predicts, what inputs it uses, and what it is *not* doing.

---

## The task in one sentence

**Given a drug × state × month combination, predict how many units of that medication will be dispensed (reimbursed) there.**

---

## The target variable

`units_reimbursed` from the **FDA State Drug Utilization Data** (Medicaid). It's the actual physical quantity of medication (tablets, mL, grams — depends on the drug's unit-of-measure) paid for by Medicaid in a given state for a given drug in a given period.

This is what the dashboard calls **"units"**.

---

## The inputs (features it learns from)

For each drug-state-month row, the model gets:

### From Medicare Part D Prescriber Public Use File
- `total_claims` — how many prescriptions were filled
- `total_30day_fills` — normalized prescription count
- `total_day_supply` — total days of therapy prescribed
- `total_drug_cost` — aggregate cost
- `prescriber_count` — how many doctors prescribed it

### Time indicators
- `Q1`, `Q2`, `Q3`, `Q4` — quarter flags
- `month_sin`, `month_cos` — cyclical month encoding so December (12) and January (1) are treated as adjacent, not 11 apart

### Engineered ratios
- `cost_per_day_supply`, `cost_per_claim`, `days_per_claim` — cost/intensity signals
- `prescriber_density` — how many prescribers per claim
- `units_per_rx` — typical pack size for that drug

### Historical context (per drug, per state)
- `drug_mean`, `drug_std`, `drug_median` — what this drug typically looks like nationwide
- `state_mean`, `state_total` — what this state typically consumes

### Log-transformed versions
Log transforms of the skewed count features (so a drug with 10,000,000 claims doesn't drown out one with 1,000).

---

## What it's *not* doing

- **Not time-series forecasting.** Only 2023 data exists, so there's no "predict next year from last 5 years." It's cross-sectional: the model learns patterns *across* drugs/states within a single year and fills in the missing drug-state-month cells that were held out for testing.
- **Not patient-level.** It has no idea about individual patients, diagnoses, or outcomes. It operates purely at the aggregate (drug × geography × time) level.
- **Not predicting the future.** It's predicting held-out rows from the *same* 2023 dataset — useful to validate that the model *could* generalize, but for actual forecasting you'd need multiple years of history.

---

## Why this is useful (the business framing)

Pharmacy inventory = "how much of each drug should we stock, where, and when?" If the model can reliably predict dispensing volume given cost/prescriber signals and historical patterns, it answers questions like:

- *"Metformin in Texas in March — expected demand?"*
- *"Which drug × state × month combos are likely to spike?"*
- *"If prescriber count for drug X in state Y jumps, how much more stock do we need?"*

---

## The honest caveat

With one year of data, the model can't learn seasonal trends or year-over-year growth. Feeding it 2022 + 2023 + 2024 would turn it from "interpolate missing cells in 2023" into "forecast 2025" — a much more useful job.