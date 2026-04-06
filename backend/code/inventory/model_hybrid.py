"""
Inventory Demand Forecasting — Hybrid Model
============================================
Combines LSTM sequence encoding with gradient boosting on tabular features.

Architecture:
    1. An LSTM encodes the recent sequence history into a learned embedding.
    2. That embedding is concatenated with hand-crafted tabular features.
    3. A gradient boosting model (XGBoost) makes the final prediction from
       the combined feature vector.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from .config import Config


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── LSTM Encoder (no prediction head — just returns embeddings) ─────────────

class _LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # Also train with a simple regression head so the encoder learns useful representations
        self.aux_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        embedding = lstm_out[:, -1, :]           # (batch, hidden_size)
        aux_pred = self.aux_head(embedding).squeeze(-1)  # (batch,)
        return embedding, aux_pred


# ── Sequence builder (same as model_lstm but returns aligned tabular rows) ──

def _build_sequences_with_tabular(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_seq, X_tabular, y).

    X_seq: sequences for LSTM encoder (n, seq_len, features)
    X_tabular: the tabular feature row at prediction time (n, features)
    y: target values (n,)
    """
    X_seqs, X_tabs, y_vals = [], [], []
    for _, group in df.groupby(["product_name", "state"]):
        group = group.sort_values("period")
        features = group[feature_cols].values
        targets = group["target"].values
        for i in range(len(group) - seq_len):
            X_seqs.append(features[i : i + seq_len])
            X_tabs.append(features[i + seq_len])
            y_vals.append(targets[i + seq_len])

    # Fall back to single-step if no sequences could be built
    if not X_seqs:
        X_all = df[feature_cols].values
        y_all = df["target"].values
        return X_all[:, np.newaxis, :], X_all, y_all  # (n, 1, feat), (n, feat), (n,)

    return np.array(X_seqs), np.array(X_tabs), np.array(y_vals)


# ── Hybrid Forecaster ──────────────────────────────────────────────────────

class HybridForecaster:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.scaler = StandardScaler()
        self.encoder = None
        self.gbm = None

    def _train_encoder(self, X_seq: np.ndarray, y: np.ndarray,
                       X_val_seq: np.ndarray, y_val: np.ndarray):
        """Pre-train the LSTM encoder with an auxiliary regression objective."""
        cfg = self.cfg
        input_size = X_seq.shape[2]

        self.encoder = _LSTMEncoder(
            input_size=input_size,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.lstm_dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lstm_lr)
        criterion = nn.MSELoss()

        X_t = torch.FloatTensor(np.array(X_seq, copy=True)).to(self.device)
        y_t = torch.FloatTensor(np.array(y, copy=True)).to(self.device)
        X_val_t = torch.FloatTensor(np.array(X_val_seq, copy=True)).to(self.device)
        y_val_t = torch.FloatTensor(np.array(y_val, copy=True)).to(self.device)

        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=cfg.lstm_batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
        patience_counter = 0
        epochs = cfg.lstm_epochs // 2  # fewer epochs since this is just for embeddings

        for epoch in range(1, epochs + 1):
            self.encoder.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                _, aux_pred = self.encoder(xb)
                loss = criterion(aux_pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                optimizer.step()

            self.encoder.eval()
            with torch.no_grad():
                _, val_aux = self.encoder(X_val_t)
                val_loss = criterion(val_aux, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= cfg.lstm_patience:
                print(f"    Encoder early stopping at epoch {epoch}")
                break

        self.encoder.load_state_dict(best_state)

    def _get_embeddings(self, X_seq: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.array(X_seq, copy=True)).to(self.device)
            embeddings, _ = self.encoder(X_t)
        return embeddings.cpu().numpy()

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str]):
        cfg = self.cfg

        # Scale
        self.scaler.fit(train[feature_cols].values)
        train_scaled = train.copy()
        val_scaled = val.copy()
        train_scaled[feature_cols] = self.scaler.transform(train[feature_cols].values)
        val_scaled[feature_cols] = self.scaler.transform(val[feature_cols].values)

        # Build sequences + tabular
        X_seq_train, X_tab_train, y_train = _build_sequences_with_tabular(
            train_scaled, feature_cols, cfg.lstm_seq_len
        )
        X_seq_val, X_tab_val, y_val = _build_sequences_with_tabular(
            val_scaled, feature_cols, cfg.lstm_seq_len
        )

        if len(X_seq_train) == 0:
            print("  Hybrid: Not enough data for sequences, skipping.")
            return

        # Step 1: Train LSTM encoder
        print("  Training LSTM encoder...")
        self._train_encoder(X_seq_train, y_train, X_seq_val, y_val)

        # Step 2: Extract embeddings
        emb_train = self._get_embeddings(X_seq_train)
        emb_val = self._get_embeddings(X_seq_val)

        if np.isnan(emb_train).any() or np.isnan(emb_val).any():
            print("  Hybrid: encoder producing NaN embeddings — not enough temporal depth.")
            self.encoder = None
            return

        # Step 3: Concatenate embeddings with tabular features
        X_combined_train = np.hstack([X_tab_train, emb_train])
        X_combined_val = np.hstack([X_tab_val, emb_val])

        # Step 4: Train GBM on combined features
        print("  Training GBM on combined features...")
        self.gbm = XGBRegressor(
            n_estimators=cfg.hybrid_gbm_n_estimators,
            max_depth=cfg.hybrid_gbm_max_depth,
            learning_rate=cfg.gbm_learning_rate,
            random_state=cfg.seed,
            n_jobs=-1,
            verbosity=0,
        )
        self.gbm.fit(
            X_combined_train, y_train,
            eval_set=[(X_combined_val, y_val)],
            verbose=False,
        )

        val_pred = self.gbm.predict(X_combined_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        print(f"  Hybrid val MAE: {val_mae:,.1f}")

    def evaluate(self, test: pd.DataFrame, feature_cols: list[str]) -> dict:
        if self.encoder is None or self.gbm is None:
            return {"model": "Hybrid", "MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}

        test_scaled = test.copy()
        test_scaled[feature_cols] = self.scaler.transform(test[feature_cols].values)

        X_seq, X_tab, y_test = _build_sequences_with_tabular(
            test_scaled, feature_cols, self.cfg.lstm_seq_len
        )
        if len(X_seq) == 0:
            return {"model": "Hybrid", "MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}

        emb = self._get_embeddings(X_seq)
        X_combined = np.hstack([X_tab, emb])
        y_pred = self.gbm.predict(X_combined)

        return {
            "model": "Hybrid (LSTM+XGB)",
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAPE": _mape(y_test, y_pred),
        }
