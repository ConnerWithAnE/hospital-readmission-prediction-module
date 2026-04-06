"""
Inventory Demand Forecasting — LSTM Time-Series Model
=====================================================
PyTorch LSTM that takes a sequence of past periods and predicts the next
period's demand.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import Config


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── Network ─────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # take last timestep
        return self.head(last_hidden).squeeze(-1)


# ── Sequence builder ────────────────────────────────────────────────────────

def _build_sequences(df: pd.DataFrame, feature_cols: list[str],
                     seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) sequences from a sorted DataFrame.

    Groups by (product_name, state) and creates overlapping windows of
    length `seq_len`, with the target being the value at the next step.

    Falls back to single-step mode (each row as a length-1 sequence)
    when groups don't have enough periods for the requested seq_len.
    """
    X_seqs, y_vals = [], []
    for _, group in df.groupby(["product_name", "state"]):
        group = group.sort_values("period")
        features = group[feature_cols].values
        targets = group["target"].values
        for i in range(len(group) - seq_len):
            X_seqs.append(features[i : i + seq_len])
            y_vals.append(targets[i + seq_len])

    # Fall back to single-step if no sequences could be built
    if not X_seqs:
        X_single = df[feature_cols].values
        y_single = df["target"].values
        return X_single[:, np.newaxis, :], y_single  # (n, 1, features)

    return np.array(X_seqs), np.array(y_vals)


# ── Forecaster ──────────────────────────────────────────────────────────────

class LSTMForecaster:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str]):
        cfg = self.cfg

        # Scale features
        self.scaler.fit(train[feature_cols].values)
        train_scaled = train.copy()
        val_scaled = val.copy()
        train_scaled[feature_cols] = self.scaler.transform(train[feature_cols].values)
        val_scaled[feature_cols] = self.scaler.transform(val[feature_cols].values)

        # Scale target (raw units can be huge → MSE explodes → NaN gradients)
        self.target_scaler = StandardScaler()
        y_train_raw = train["target"].values.reshape(-1, 1)
        self.target_scaler.fit(y_train_raw)

        # Build sequences
        X_train, y_train = _build_sequences(train_scaled, feature_cols, cfg.lstm_seq_len)
        X_val, y_val = _build_sequences(val_scaled, feature_cols, cfg.lstm_seq_len)

        if len(X_train) == 0:
            print("  LSTM: Not enough data to build sequences, skipping.")
            return

        # Scale targets
        y_train_scaled = self.target_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).ravel()

        # Tensors — .copy() to ensure writable arrays
        X_train_t = torch.FloatTensor(np.array(X_train, copy=True)).to(self.device)
        y_train_t = torch.FloatTensor(np.array(y_train_scaled, copy=True)).to(self.device)
        X_val_t = torch.FloatTensor(np.array(X_val, copy=True)).to(self.device)
        y_val_t = torch.FloatTensor(np.array(y_val_scaled, copy=True)).to(self.device)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=cfg.lstm_batch_size, shuffle=True)

        # Model
        input_size = X_train.shape[2]
        self.model = _LSTMNet(
            input_size=input_size,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.lstm_dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lstm_lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        patience_counter = 0

        for epoch in range(1, cfg.lstm_epochs + 1):
            # Train
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                if torch.isnan(loss):
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_ds)

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_pred_scaled = self.model(X_val_t)
                val_loss = criterion(val_pred_scaled, y_val_t).item()

            if not np.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 20 == 0:
                print(f"    Epoch {epoch:3d} | train loss: {epoch_loss:.4f} | val loss: {val_loss:.4f}")

            if patience_counter >= cfg.lstm_patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        # Restore best and compute val MAE in original scale
        self.model.load_state_dict(best_state)
        self.model.eval()
        with torch.no_grad():
            val_pred_scaled = self.model(X_val_t).cpu().numpy()

        if np.isnan(val_pred_scaled).any():
            print("  LSTM: model producing NaN — not enough temporal depth for LSTM.")
            self.model = None
            return

        val_pred = self.target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
        val_mae = mean_absolute_error(y_val, val_pred)
        print(f"  LSTM val MAE: {val_mae:,.1f}")

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        if self.model is None:
            return np.full(len(df), np.nan)

        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.transform(df[feature_cols].values)

        X, _ = _build_sequences(df_scaled, feature_cols, self.cfg.lstm_seq_len)
        if len(X) == 0:
            return np.full(len(df), np.nan)

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.array(X, copy=True)).to(self.device)
            preds_scaled = self.model(X_t).cpu().numpy()
        return self.target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

    def evaluate(self, test: pd.DataFrame, feature_cols: list[str]) -> dict:
        if self.model is None:
            return {"model": "LSTM", "MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}

        test_scaled = test.copy()
        test_scaled[feature_cols] = self.scaler.transform(test[feature_cols].values)

        X_test, y_test = _build_sequences(test_scaled, feature_cols, self.cfg.lstm_seq_len)
        if len(X_test) == 0:
            return {"model": "LSTM", "MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.array(X_test, copy=True)).to(self.device)
            y_pred_scaled = self.model(X_t).cpu().numpy()
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        return {
            "model": "LSTM",
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAPE": _mape(y_test, y_pred),
        }
