"""
API endpoints for visualising the Inventory Demand Forecasting model.

Serves the artifacts produced by backend/code/inventory/train.py:
    - results_summary.csv (model metrics per granularity)
    - feature_importance_gbm_*.csv
    - model_comparison.png
    - best_model_meta.json
    - test_predictions.csv (sampled for display)
"""
import base64
import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/inventory/model", tags=["inventory-model"])

INVENTORY_DIR = Path(__file__).resolve().parent.parent / "code" / "inventory"
OUTPUT_DIR = INVENTORY_DIR / "output"
MODELS_DIR = INVENTORY_DIR / "saved_models"

# ── Cache for the large test_predictions.csv ───────────────────────────────
_predictions_cache: dict = {"mtime": None, "df": None}


def _load_predictions() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "test_predictions.csv"
    if not path.exists():
        return None

    mtime = path.stat().st_mtime
    if _predictions_cache["mtime"] != mtime:
        cols = ["product_name", "state", "quarter", "month", "target", "predicted"]
        # Only read columns that exist in the file
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [c for c in cols if c in header]
        df = pd.read_csv(path, usecols=usecols)
        if "target" in df.columns and "predicted" in df.columns:
            df["error"] = (df["predicted"] - df["target"]).abs()
        _predictions_cache["df"] = df
        _predictions_cache["mtime"] = mtime

    return _predictions_cache["df"]


def _read_csv_or_empty(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


@router.get("/stats")
def get_inventory_model_stats():
    """Return best model metadata + results summary across granularities."""
    meta_path = MODELS_DIR / "best_model_meta.json"
    summary_path = OUTPUT_DIR / "results_summary.csv"

    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No trained inventory model found. Run backend/code/inventory/train.py first.",
        )

    with open(meta_path) as f:
        meta = json.load(f)

    summary = _read_csv_or_empty(summary_path)

    return {
        "best_model": meta["best_model"],
        "best_type": meta["best_type"],
        "granularity": meta["granularity"],
        "MAE": meta["MAE"],
        "RMSE": meta["RMSE"],
        "MAPE": meta["MAPE"],
        "feature_count": len(meta.get("feature_cols", [])),
        "results_summary": summary,
    }


@router.get("/feature-importance")
def get_feature_importance(granularity: str = Query("monthly")):
    """Return the feature importance list for the given granularity."""
    path = OUTPUT_DIR / f"feature_importance_gbm_{granularity}.csv"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No feature importance for granularity '{granularity}'.",
        )
    records = _read_csv_or_empty(path)
    return {"granularity": granularity, "features": records}


@router.get("/comparison-chart")
def get_comparison_chart():
    """Return the model comparison PNG as base64."""
    path = OUTPUT_DIR / "model_comparison.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Comparison chart not found.")
    data = path.read_bytes()
    return {"png_base64": base64.b64encode(data).decode("ascii")}


@router.get("/predictions")
def get_predictions(
    mode: str = Query("top", pattern="^(top|bottom|random|largest_error)$"),
    limit: int = Query(50, ge=1, le=500),
    state: str | None = Query(None),
    drug: str | None = Query(None),
):
    """Return a sample of test predictions.

    Modes:
        top            — highest actual volume
        bottom         — lowest actual volume
        random         — random sample
        largest_error  — rows with biggest prediction error
    """
    df = _load_predictions()
    if df is None:
        raise HTTPException(
            status_code=404,
            detail="No test predictions found. Run training first.",
        )

    # Filter
    filtered = df
    if state and "state" in filtered.columns:
        filtered = filtered[filtered["state"].str.upper() == state.upper()]
    if drug and "product_name" in filtered.columns:
        filtered = filtered[filtered["product_name"].str.contains(drug, case=False, na=False)]

    if len(filtered) == 0:
        return {"mode": mode, "count": 0, "total_rows": len(df), "rows": []}

    if mode == "top":
        result = filtered.nlargest(limit, "target")
    elif mode == "bottom":
        result = filtered.nsmallest(limit, "target")
    elif mode == "random":
        result = filtered.sample(n=min(limit, len(filtered)), random_state=42)
    elif mode == "largest_error":
        result = filtered.nlargest(limit, "error")

    return {
        "mode": mode,
        "count": len(result),
        "total_rows": len(df),
        "rows": result.fillna("").to_dict(orient="records"),
    }


@router.get("/summary")
def get_prediction_summary():
    """Return high-level aggregates for the dashboard header cards."""
    df = _load_predictions()
    if df is None:
        raise HTTPException(status_code=404, detail="No test predictions found.")

    unique_drugs = df["product_name"].nunique() if "product_name" in df.columns else 0
    unique_states = df["state"].nunique() if "state" in df.columns else 0
    total_rows = len(df)

    predicted_total = float(df["predicted"].sum()) if "predicted" in df.columns else 0.0
    actual_total = float(df["target"].sum()) if "target" in df.columns else 0.0

    top_drugs = []
    if "product_name" in df.columns and "predicted" in df.columns:
        top_drugs = (
            df.groupby("product_name", as_index=False)["predicted"]
            .sum()
            .nlargest(10, "predicted")
            .rename(columns={"predicted": "total_predicted"})
            .to_dict(orient="records")
        )

    top_states = []
    if "state" in df.columns and "predicted" in df.columns:
        top_states = (
            df.groupby("state", as_index=False)["predicted"]
            .sum()
            .nlargest(10, "predicted")
            .rename(columns={"predicted": "total_predicted"})
            .to_dict(orient="records")
        )

    return {
        "unique_drugs": unique_drugs,
        "unique_states": unique_states,
        "total_test_rows": total_rows,
        "predicted_total": predicted_total,
        "actual_total": actual_total,
        "top_drugs": top_drugs,
        "top_states": top_states,
    }