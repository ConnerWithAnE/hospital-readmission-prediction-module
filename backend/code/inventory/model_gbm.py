"""
Inventory Demand Forecasting — Gradient Boosting Models
=======================================================
XGBoost and LightGBM regressors for tabular demand forecasting.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .config import Config


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class GBMForecaster:
    """Wraps both XGBoost and LightGBM, trains both, and picks the best."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.xgb = XGBRegressor(
            n_estimators=cfg.gbm_n_estimators,
            max_depth=cfg.gbm_max_depth,
            learning_rate=cfg.gbm_learning_rate,
            random_state=cfg.seed,
            n_jobs=-1,
            verbosity=0,
        )
        self.lgbm = LGBMRegressor(
            n_estimators=cfg.gbm_n_estimators,
            max_depth=cfg.gbm_max_depth,
            learning_rate=cfg.gbm_learning_rate,
            random_state=cfg.seed,
            n_jobs=-1,
            verbose=-1,
        )
        self.best_model = None
        self.best_name = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str]):
        X_train = train[feature_cols].values
        y_train = train["target"].values
        X_val = val[feature_cols].values
        y_val = val["target"].values

        # Train XGBoost
        self.xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        xgb_pred = self.xgb.predict(X_val)
        xgb_mae = mean_absolute_error(y_val, xgb_pred)

        # Train LightGBM
        self.lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[],
        )
        lgbm_pred = self.lgbm.predict(X_val)
        lgbm_mae = mean_absolute_error(y_val, lgbm_pred)

        print(f"  XGBoost  val MAE: {xgb_mae:,.1f}")
        print(f"  LightGBM val MAE: {lgbm_mae:,.1f}")

        if xgb_mae <= lgbm_mae:
            self.best_model = self.xgb
            self.best_name = "XGBoost"
        else:
            self.best_model = self.lgbm
            self.best_name = "LightGBM"

        print(f"  → Best GBM: {self.best_name}")

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        return self.best_model.predict(df[feature_cols].values)

    def evaluate(self, test: pd.DataFrame, feature_cols: list[str]) -> dict:
        y_true = test["target"].values
        y_pred = self.predict(test, feature_cols)

        return {
            "model": f"GBM ({self.best_name})",
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE": _mape(y_true, y_pred),
        }

    def feature_importance(self, feature_cols: list[str]) -> pd.DataFrame:
        if hasattr(self.best_model, "feature_importances_"):
            imp = self.best_model.feature_importances_
            return (
                pd.DataFrame({"feature": feature_cols, "importance": imp})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        return pd.DataFrame()
