# Medication Impact Analysis — "What-If" Tool

## Context
The readmission model tracks 23 diabetes medications but there's no way to see how changing a medication affects risk. This feature lets clinicians ask "what happens to this patient's risk if we switch metformin from No to Steady?" by running the model with every medication variant and showing the deltas.

Key insight: only `insulin` and `metformin` are direct model features. The other 21 affect predictions only through aggregates (`n_active_meds`, `n_med_changes`, `any_dose_up`), so their individual impact will be smaller.

---

## Backend

### New endpoint in `backend/server/main.py` (3 lines)
```python
@app.post("/api/predict/med-impact")
async def predict_med_impact(patient_data: PatientInput):
    return prediction_model.predict_med_impact(patient_data)
```

### New method in `backend/code/data_processing.py`
Add `predict_med_impact(self, patient_input)` to `PredictionModel`:

1. Run baseline prediction to get `baseline_risk`
2. For each of 23 medication fields, for each of 4 statuses (skip current):
   - `variant = patient_input.model_copy(update={field: MedStatusEnum(status)})`
   - `_prepare_input(variant)` to get encoded row
3. Concatenate all encoded rows into one DataFrame (~69 rows)
4. Single `self.model.predict_proba(batch)[:, 1]` call (~200-300ms total)
5. Return structured response:

```python
{
    "baseline_risk": float,
    "baseline_category": str,
    "medications": [
        {
            "name": "metformin",         # field name
            "label": "Metformin",        # display name
            "current_status": "No",
            "scenarios": [
                {"status": "Steady", "risk_score": 0.28, "delta": -0.07},
                {"status": "Down", "risk_score": 0.33, "delta": -0.02},
                {"status": "Up", "risk_score": 0.30, "delta": -0.05}
            ],
            "max_reduction": -0.07,
            "max_increase": 0.0,
            "best_status": "Steady"
        },
        ...
    ]
}
```

Use Pydantic field names (underscores: `glyburide_metformin`) when creating variants. `to_raw_df()` already recomputes aggregates from scratch, so ripple effects are handled automatically.

---

## Frontend

### Score Window tabs (`score-window.tsx`)
Already imports `Tabs`/`TabsContent`/`TabsList`/`TabsTrigger` but doesn't use them. Wrap existing content in tabs:

- **Tab 1: "Risk Overview"** — All existing content unchanged
- **Tab 2: "Medication Impact"** — New `MedImpactView` component

### New component: `frontend/src/components/patient-assessment/med-impact-view.tsx`

Props: `data: MedImpactResult | null`, `loading: boolean`, `onRequest: () => void`, `hasAssessment: boolean`

**Layout (top to bottom):**

1. **"Run Analysis" button** — triggers API call, disabled until an assessment has been run
2. **Summary banner** — "Most impactful: switching [med] from [current] to [best] would change risk from X% to Y%"
3. **Tornado chart** — Top 8 medications by `|max_reduction|`. Green bars left for risk decrease, red bars right for increase. Hand-rolled SVG (same pattern as existing `SvgPieChart` and `BarChart`)
4. **Detail table** — All 23 medications. Columns: Medication, Current Status, No, Steady, Down, Up. Each cell shows `risk% (delta%)` with green/red coloring. Current status cell is grayed out. Sorted by `|max_reduction|` descending.

### Page state (`patient-assesment.tsx`)
- Add `medImpact` state + `MedImpactResult`/`MedImpact` type exports
- `handleResult` clears `medImpact` (stale after re-assessment)
- `handleClear` clears `medImpact`
- Pass `medImpact`, `loading`, `onRequestMedImpact` to `ScoreWindow`

---

## Files

| File | Change |
|------|--------|
| `backend/code/data_processing.py` | Add `predict_med_impact()` method |
| `backend/server/main.py` | Add `/api/predict/med-impact` endpoint |
| `frontend/src/pages/patient-assesment.tsx` | Add types, state, handler for med impact |
| `frontend/src/components/patient-assessment/score-window.tsx` | Wrap in Tabs, add Medication Impact tab |
| `frontend/src/components/patient-assessment/med-impact-view.tsx` | **New** — tornado chart + detail table |

## Verification
1. Start server, run a patient assessment
2. Switch to "Medication Impact" tab, click "Run Analysis"
3. Verify tornado chart shows top medications by impact
4. Verify detail table shows all 23 meds with correct deltas
5. Change a form value, re-run assessment — med impact should clear
6. `bun run build.ts` succeeds