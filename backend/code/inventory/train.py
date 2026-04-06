"""
Inventory Demand Forecasting — Training Pipeline
=================================================
Trains all three model types (GBM, LSTM, Hybrid) at each configured
granularity, compares metrics, and saves results + best model.

Usage:
    python -m backend.code.inventory.train
"""

import json
import sys
from pathlib import Path

# Allow running directly: python train.py
if __name__ == "__main__" and __package__ is None:
    _backend = str(Path(__file__).resolve().parent.parent.parent)
    if _backend not in sys.path:
        sys.path.insert(0, _backend)
    __package__ = "backend.code.inventory"

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .data_loading import load_fda_utilization, load_medicare_part_d
from .data_processing import prepare_data
from .model_gbm import GBMForecaster


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save_comparison_chart(results: list[dict], output_dir: Path):
    """Bar chart comparing MAE across all model × granularity combos."""
    df = pd.DataFrame(results)
    df["label"] = df["model"] + "\n(" + df["granularity"] + ")"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
        vals = df[metric].values
        colors = ["#2ecc71" if v == np.nanmin(vals) else "#3498db" for v in vals]
        ax.barh(df["label"], vals, color=colors)
        ax.set_xlabel(metric)
        ax.set_title(metric)
        ax.invert_yaxis()
    plt.suptitle("Model Comparison — Demand Forecasting", fontsize=14)
    plt.tight_layout()
    chart_path = output_dir / "model_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nComparison chart saved to {chart_path}")


def run(cfg: Config | None = None):
    if cfg is None:
        cfg = Config()
    _set_seed(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load raw data ───────────────────────────────────────────────────
    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    medicare_df = load_medicare_part_d(cfg.medicare_path)
    fda_df = load_fda_utilization(cfg.fda_path)

    all_results = []
    best_overall = {"MAE": float("inf")}

    for granularity in cfg.granularities:
        print(f"\n{'=' * 60}")
        print(f"Granularity: {granularity.upper()}")
        print(f"{'=' * 60}")

        train, val, test, feature_cols = prepare_data(
            medicare_df, fda_df, cfg, granularity
        )

        if len(train) == 0:
            print(f"  No training data at {granularity} granularity, skipping.")
            continue

        # ── GBM ─────────────────────────────────────────────────────────
        print(f"\n--- GBM ({granularity}) ---")
        gbm = GBMForecaster(cfg)
        gbm.fit(train, val, feature_cols)
        gbm_metrics = gbm.evaluate(test, feature_cols)
        gbm_metrics["granularity"] = granularity
        all_results.append(gbm_metrics)
        print(f"  Test → MAE: {gbm_metrics['MAE']:,.1f} | "
              f"RMSE: {gbm_metrics['RMSE']:,.1f} | MAPE: {gbm_metrics['MAPE']:.1f}%")

        # Save feature importance
        fi = gbm.feature_importance(feature_cols)
        if len(fi):
            fi.to_csv(cfg.output_dir / f"feature_importance_gbm_{granularity}.csv", index=False)

        if gbm_metrics["MAE"] < best_overall["MAE"]:
            best_overall = {**gbm_metrics, "obj": gbm, "type": "gbm", "feature_cols": feature_cols}

        # NOTE: LSTM and Hybrid disabled — need multi-year data for temporal depth.
        # Re-enable when more years of data are available.

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    results_df.to_csv(cfg.output_dir / "results_summary.csv", index=False)
    print(f"\nResults saved to {cfg.output_dir / 'results_summary.csv'}")

    # Save best model
    if "obj" in best_overall:
        best_type = best_overall["type"]
        best_model = best_overall["obj"]
        print(f"\nBest model: {best_overall['model']} @ {best_overall['granularity']} "
              f"(MAE={best_overall['MAE']:,.1f})")

        if best_type == "gbm":
            joblib.dump(best_model.best_model, cfg.models_dir / "best_model_gbm.joblib")

        # Save metadata
        meta = {
            "best_model": best_overall["model"],
            "best_type": best_type,
            "granularity": best_overall["granularity"],
            "MAE": best_overall["MAE"],
            "RMSE": best_overall["RMSE"],
            "MAPE": best_overall["MAPE"],
            "feature_cols": best_overall["feature_cols"],
        }
        with open(cfg.models_dir / "best_model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # Comparison chart
    if all_results:
        _save_comparison_chart(all_results, cfg.output_dir)

    # ── Sample predictions ──────────────────────────────────────────────
    if "obj" in best_overall and best_overall["type"] == "gbm":
        best_model = best_overall["obj"]
        feat_cols = best_overall["feature_cols"]

        # Re-run prepare_data for the best granularity to get the test set
        _, _, test_best, _ = prepare_data(
            medicare_df, fda_df, cfg, best_overall["granularity"]
        )

        test_best = test_best.copy()
        test_best["predicted"] = best_model.predict(test_best, feat_cols)
        test_best["error"] = (test_best["predicted"] - test_best["target"]).abs()

        # Show top predictions by volume
        sample = (
            test_best
            .nlargest(30, "target")
            [["product_name", "state", "quarter", "target", "predicted", "error"]]
            .copy()
        )
        sample.columns = ["Drug", "State", "Qtr", "Actual", "Predicted", "Error"]
        for col in ["Actual", "Predicted", "Error"]:
            sample[col] = sample[col].apply(lambda x: f"{x:,.0f}")

        print(f"\n{'=' * 60}")
        print("SAMPLE PREDICTIONS (top 30 by actual volume)")
        print(f"{'=' * 60}")
        print(sample.to_string(index=False))

        # Also save full test predictions
        test_best.to_csv(cfg.output_dir / "test_predictions.csv", index=False)
        print(f"\nFull test predictions saved to {cfg.output_dir / 'test_predictions.csv'}")

    return results_df


if __name__ == "__main__":
    run()
