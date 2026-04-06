"""
Inventory Demand Forecasting — Prediction / Recommendation
===========================================================
Loads the best trained model and generates ordering recommendations:
    - What to order (which drugs)
    - How much (predicted demand vs current stock)
    - When (lead time + predicted stockout date)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Config


@dataclass
class OrderRecommendation:
    drug_name: str
    state: str
    predicted_demand: float
    current_stock: float
    recommended_order_qty: float
    urgency: str  # "critical", "soon", "routine"
    granularity: str
    confidence_note: str


def load_best_model(cfg: Config) -> tuple[object, dict]:
    """Load the best model and its metadata."""
    meta_path = cfg.models_dir / "best_model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No trained model found. Run train.py first.\n"
            f"Expected: {meta_path}"
        )

    with open(meta_path) as f:
        meta = json.load(f)

    best_type = meta["best_type"]

    if best_type == "gbm":
        model = joblib.load(cfg.models_dir / "best_model_gbm.joblib")
    elif best_type == "lstm":
        import torch
        from .model_lstm import _LSTMNet
        input_size = len(meta["feature_cols"])
        model = _LSTMNet(
            input_size=input_size,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.lstm_dropout,
        )
        model.load_state_dict(torch.load(cfg.models_dir / "best_model_lstm.pt", weights_only=True))
        model.eval()
    elif best_type == "hybrid":
        import torch
        from .model_hybrid import _LSTMEncoder
        input_size = len(meta["feature_cols"])
        encoder = _LSTMEncoder(
            input_size=input_size,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.lstm_dropout,
        )
        encoder.load_state_dict(torch.load(cfg.models_dir / "best_model_hybrid_encoder.pt", weights_only=True))
        encoder.eval()
        gbm = joblib.load(cfg.models_dir / "best_model_hybrid_gbm.joblib")
        model = {"encoder": encoder, "gbm": gbm}
    else:
        raise ValueError(f"Unknown model type: {best_type}")

    return model, meta


def generate_recommendations(
    df: pd.DataFrame,
    current_stock: dict[str, float] | None = None,
    cfg: Config | None = None,
    safety_factor: float = 1.2,
) -> list[OrderRecommendation]:
    """Generate ordering recommendations from the latest data.

    Args:
        df: Processed feature DataFrame (output of data_processing.prepare_data).
        current_stock: Optional dict mapping drug_name → current quantity on hand.
            If None, uses the latest period's actual values as a proxy.
        cfg: Config instance.
        safety_factor: Multiply predicted demand by this to add a safety buffer.

    Returns:
        List of OrderRecommendation sorted by urgency then predicted demand.
    """
    if cfg is None:
        cfg = Config()

    model, meta = load_best_model(cfg)
    feature_cols = meta["feature_cols"]
    best_type = meta["best_type"]
    granularity = meta["granularity"]

    # Get the latest period data for each drug × state
    latest_period = df["period"].max()
    latest = df[df["period"] == latest_period].copy()

    if len(latest) == 0:
        return []

    # Predict next-period demand
    if best_type == "gbm":
        predictions = model.predict(latest[feature_cols].values)
    elif best_type == "lstm":
        import torch
        from sklearn.preprocessing import StandardScaler
        scaler = joblib.load(cfg.models_dir / "best_model_lstm_scaler.joblib")
        scaled = scaler.transform(latest[feature_cols].values)
        # For single-step, use available features as a 1-step sequence
        X = torch.FloatTensor(scaled).unsqueeze(1)
        with torch.no_grad():
            predictions = model(X).numpy()
    elif best_type == "hybrid":
        import torch
        scaler = joblib.load(cfg.models_dir / "best_model_hybrid_scaler.joblib")
        scaled = scaler.transform(latest[feature_cols].values)
        X_seq = torch.FloatTensor(scaled).unsqueeze(1)
        with torch.no_grad():
            embeddings, _ = model["encoder"](X_seq)
        combined = np.hstack([scaled, embeddings.numpy()])
        predictions = model["gbm"].predict(combined)

    predictions = np.maximum(predictions, 0)  # demand can't be negative

    recommendations = []
    for i, (_, row) in enumerate(latest.iterrows()):
        drug = row.get("product_name", row.get("generic_name", "Unknown"))
        state = row.get("state", "Unknown")
        predicted = float(predictions[i]) * safety_factor

        stock = 0.0
        if current_stock and drug in current_stock:
            stock = current_stock[drug]
        elif "target" in row:
            stock = float(row["target"])  # use last known as proxy

        deficit = predicted - stock
        if deficit <= 0:
            urgency = "routine"
        elif deficit > predicted * 0.5:
            urgency = "critical"
        else:
            urgency = "soon"

        order_qty = max(0, deficit)

        recommendations.append(OrderRecommendation(
            drug_name=drug,
            state=state,
            predicted_demand=round(predicted, 1),
            current_stock=round(stock, 1),
            recommended_order_qty=round(order_qty, 1),
            urgency=urgency,
            granularity=granularity,
            confidence_note=f"Based on {meta['best_model']} (MAE={meta['MAE']:.1f})",
        ))

    # Sort: critical first, then by order quantity descending
    urgency_order = {"critical": 0, "soon": 1, "routine": 2}
    recommendations.sort(key=lambda r: (urgency_order[r.urgency], -r.recommended_order_qty))

    return recommendations


def recommendations_to_df(recs: list[OrderRecommendation]) -> pd.DataFrame:
    """Convert recommendations to a DataFrame for display or export."""
    return pd.DataFrame([
        {
            "Drug": r.drug_name,
            "State": r.state,
            "Predicted Demand": r.predicted_demand,
            "Current Stock": r.current_stock,
            "Order Qty": r.recommended_order_qty,
            "Urgency": r.urgency,
            "Granularity": r.granularity,
            "Note": r.confidence_note,
        }
        for r in recs
    ])
